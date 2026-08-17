#!/usr/bin/env python3
"""Collect a video memory, then train the reasoning layer over it.

    .venv-libero/bin/python brigade/bench/train_reasoner.py --collect --episodes 2
    .venv-libero/bin/python brigade/bench/train_reasoner.py --train

COLLECT runs the robot through the goal set with the memory camera streaming
into the store, exactly as it will at serve. Segments are cut on a clock, so an
episode contributes several, and each is tagged with which behaviour was
running. **That tag lives in a training file on disk, never in the database** —
the store keeps clips and vectors, and this is the supervision, which is used
once here and never at serve.

TRAIN fits the head from `agent/latent.py`: RelMo clip vector + the request's
embedding -> weights over frozen instruction prototypes. Two checks decide
whether the result is worth anything, and they matter more than the accuracy:

  ABLATE CLIP     zero the clip vector. If accuracy holds, the head learned to
                  ignore memory and read the request alone — the system would
                  work with an empty database, and the whole claim is dead.
  ABLATE REQUEST  zero the request. Tells us how much the clips carry on their
                  own.

A head that scores well and survives the clip ablation is a head that is not
using memory. The number to want is high accuracy that COLLAPSES when the clip
is removed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

TRAIN = "../eval_logs/reasoner"
GOALS = list(range(10))          # libero_goal: one kitchen, ten behaviours

# AMBIGUOUS REQUESTS: the reason the ablation means anything.
#
# The first head scored 1.000 held-out and still scored 0.875 with the clip
# vector zeroed, and I wrongly called that "memory is doing the work". It was
# not: each behaviour had exactly one instruction, so request -> behaviour was a
# 1:1 map the head could memorise, and the clips were never needed. A benchmark
# whose inputs are individually sufficient cannot measure which one is used.
#
# So the head is also trained on what a person actually says. "Put the bowl
# away" is true of four different behaviours; the words cannot pick between
# them and nothing in the sentence ever will. What picks is where the bowl has
# been going, which is in the video and nowhere else. This is not a trick to
# make the ablation look good — it IS the demo: same words, different history,
# different action.
FAMILIES = {
    "put the bowl away": (1, 3, 4, 8),
    "tidy up the bowl": (1, 3, 4, 8),
    "put the wine bottle away": (2, 9),
    "put the bottle back": (2, 9),
    "get the stove ready": (5, 7),
    "open a drawer": (0, 3),
}


# ------------------------------------------------------------------- collect

def collect(episodes: int, seed0: int = 10000) -> int:
    """Fill the store with video of each behaviour, then fit its basis.

    Collection runs the goals in BLOCKS: several episodes of one behaviour back
    to back, with the camera never stopping, then a session break before the
    next goal. Two reasons, both learned the hard way.

    A block, not one episode: the store indexes a sliding 15 s span and a single
    LIBERO episode is 10-15 s of video, so an episode alone would cut at most
    one span. Consecutive same-goal episodes give the sliding window something
    to slide over.

    A break between goals, not within one: a span that straddles two DIFFERENT
    behaviours has no single label. Inside a block every span shows the same
    behaviour whether or not it crosses an episode boundary, so the labels are
    clean by construction rather than by filtering.
    """
    from brigade.agent.pilot import Pilot
    from brigade.memory.relmo import RelMoSidecar
    from brigade.memory.store import HOP_S, SPAN_S, VideoStore

    os.makedirs(TRAIN, exist_ok=True)
    relmo = RelMoSidecar()
    print("starting RelMo (~85s) ...", flush=True)
    if not relmo.start():
        print(f"RelMo unavailable: {relmo.error}")
        return 1
    store = VideoStore(relmo=relmo)
    store.setup()
    store.start()
    print(f"store: {SPAN_S:.0f}s spans on a {HOP_S:.0f}s hop", flush=True)

    p = Pilot(device="mps")
    p.load()

    rows: list[dict] = []
    for goal in GOALS:
        p.open_scene("libero_goal", goal)
        instruction = p.default_instruction
        # A new scene is a genuine discontinuity in the video, so the ring
        # starts empty; a span bridging two kitchens never existed.
        store.new_session()
        block: list[str] = []
        ok = 0
        for k in range(episodes):
            def stream(_buf, _step, _s=store, _c=block, _p=p):
                # The memory camera streams on its own clock, independent of
                # whether an episode is running. Every other control step at
                # 20 Hz gives the store's 10 fps.
                if _step % 2 == 0:
                    cid = _s.write(_p.memory_frame())
                    if cid:
                        _c.append(cid)

            # INDEPENDENT episodes on purpose. Carrying the world across all
            # twenty scrambled the kitchen — persistence costs roughly half the
            # policy's success rate (RESULTS.md) — and every later goal failed,
            # so the training set was all failures. The continuous kitchen
            # belongs at serve time; training wants clean examples.
            ep = p.run(instruction, keep_frames=False, on_frame=stream,
                       seed=seed0 + k)
            ok += bool(ep.success)
            print(f"  goal {goal} ep {k}: {'ok' if ep.success else 'fail'}  "
                  f"{ep.steps} steps  (store {store.n_written} written, "
                  f"{store.pending()} pending)", flush=True)
        tail = store.new_session()
        if tail:
            block.append(tail)
        # The block is labelled if the behaviour was demonstrated at all. A
        # per-episode success filter cannot apply here: a span may cover the
        # end of a failed attempt and the whole of a successful one.
        for cid in block:
            rows.append(dict(clip_id=cid, goal=goal, instruction=instruction,
                             successes=ok, episodes=episodes))
        print(f"  goal {goal}: {ok}/{episodes} succeeded, {len(block)} spans",
              flush=True)

    print("\nwaiting for the encoder to drain ...", flush=True)
    store.drain()
    # APPEND. A second round with a different --seed0 adds episodes to the same
    # store, and overwriting here would throw away the labels for round one
    # while its clips stayed in the database — supervision silently detached
    # from the video it describes.
    path = os.path.join(TRAIN, "segments.json")
    prior = json.load(open(path)) if os.path.exists(path) else []
    seen = {r["clip_id"] for r in prior}
    rows = prior + [r for r in rows if r["clip_id"] not in seen]
    json.dump(rows, open(path, "w"), indent=1)
    print(f"wrote {len(rows)} labelled spans to {path} "
          f"({len(rows) - len(prior)} new)")

    # THE BASIS. Whitening removes the variance a corpus shares, and RelMo fits
    # it on the corpus being searched for exactly that reason. Until now the
    # store borrowed RoboCasa's, which does not share this kitchen's camera,
    # room or lighting. Measured on the 5 s corpus: 1-NN 0.300 -> 0.400.
    print(f"backfilled the motion view for {store.backfill_motion()} spans")
    print("\nfitting this kitchen's own whitening basis ...", flush=True)
    print(f"  {store.refit_basis()}")
    print(f"store: {store.stats()}")
    relmo.stop()
    return 0


# --------------------------------------------------------------------- train

def train(epochs: int = 400, use_request: bool = True) -> int:
    import torch
    import torch.nn as nn

    from brigade.agent.latent import InstructionHead, PrototypeBank
    from brigade.agent.pilot import Pilot
    from brigade.memory.store import HOP_S, SPAN_S

    rows = json.load(open(os.path.join(TRAIN, "segments.json")))

    # ---- inputs: RelMo's TRACE, reduced. Not the indexed column. ----
    # The database column is a mean over the trace, which is what a stage-1
    # prefilter has to be and is measured at 0.305 on this question. The head
    # is not bound to that shape — it reads the trace file the write path
    # already saved and takes the per-channel temporal spread, 0.712 on the
    # same measurement. See memory/traces.py for the whole table.
    from brigade.memory import traces as TR

    X, kept = TR.features([r["clip_id"] for r in rows])
    keep = set(kept)
    rows = [r for r in rows if r["clip_id"] in keep]
    if not rows:
        print("no encoded spans; run --collect first")
        return 1
    y = np.array([r["goal"] for r in rows], dtype=np.int64)
    print(f"{len(rows)} spans, {len(set(y.tolist()))} behaviours, "
          f"feature {TR.BLOCKS} dim {X.shape[1]}")

    # ---- the request encoder ----
    # SigLIP2's text tower needs transformers 4.57, which is RelMo's
    # interpreter, not pi0.5's (pinned to 4.53.2). Same split as everywhere
    # else in this project: the other environment computes it, this one reads
    # the file. `text_vectors.py` writes the cache.
    cache = os.path.join(TRAIN, "text_vectors.npz")
    if not os.path.exists(cache):
        print(f"missing {cache} — run:\n"
              f"  myenv/bin/python brigade/bench/text_vectors.py")
        return 1
    z = np.load(cache, allow_pickle=True)
    lut = {k: np.asarray(v, np.float32) for k, v in zip(z["texts"].tolist(), z["vecs"])}

    # Every behaviour is trained under its own instruction AND under every
    # ambiguous phrase that is true of it. See FAMILIES.
    phrasings: dict[int, list[str]] = {}
    for r in rows:
        phrasings.setdefault(int(r["goal"]), [r["instruction"]])
    for text, goals in FAMILIES.items():
        for g in goals:
            if g in phrasings and text not in phrasings[g]:
                phrasings[g].append(text)
    missing = sorted({t for v in phrasings.values() for t in v if t not in lut})
    if missing:
        print(f"{len(missing)} phrasings not in the cache; re-run text_vectors.py:")
        for t in missing[:6]:
            print(f"  {t!r}")
        return 1
    R = np.stack([lut[r["instruction"]] for r in rows]).astype(np.float32)

    # ---- the frozen prototype bank, through pi0.5's own embedding path ----
    p = Pilot(device="mps")
    p.load()
    p.open_scene("libero_goal", 0)
    order = sorted(set(y.tolist()))
    texts = {g: next(r["instruction"] for r in rows if r["goal"] == g) for g in order}

    pwe = p.policy.model.paligemma_with_expert
    orig = pwe.embed_language_tokens
    grabbed = {}

    def grab(tokens):
        grabbed["t"] = tokens.detach().clone()
        return orig(tokens)

    protos = []
    for g in order:
        pwe.embed_language_tokens = grab
        p.env.envs[0].task_description = texts[g]
        p.run(texts[g], keep_frames=False, max_steps=2)
        pwe.embed_language_tokens = orig
        with torch.no_grad():
            protos.append(orig(grabbed["t"]).squeeze(0).cpu())
    bank = PrototypeBank(torch.stack(protos))
    print(f"prototype bank: {tuple(bank.bank.shape)}  ({bank.k} behaviours)")

    # ---- the split: BY TIME, WITH A GUARD BAND ----
    # Spans overlap by SPAN_S - HOP_S = 10 s, so two consecutive rows share two
    # thirds of their video and a random split would put the same seconds on
    # both sides of it. Per behaviour: the last two spans are held out, and the
    # two before them are DROPPED — that is exactly the overlap depth, so no
    # training span shares a frame with a test span.
    by_goal: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_goal.setdefault(int(r["goal"]), []).append(i)
    guard = int(SPAN_S / HOP_S) - 1
    tr, te, thin = [], [], []
    for g, idxs in sorted(by_goal.items()):
        n_te = max(1, len(idxs) // 4)
        if len(idxs) < n_te + guard + 1:
            # Too few spans to hold any out AND keep a clean guard band. Putting
            # one in the test set anyway would mean testing a behaviour with no
            # training example of it — a guaranteed error dressed up as a
            # measurement. It trains and is reported as untested instead.
            tr += idxs
            thin.append(g)
            continue
        te += idxs[-n_te:]
        tr += idxs[:len(idxs) - n_te - guard]
    tested = sorted(set(int(rows[i]["goal"]) for i in te))
    print(f"split: {len(tr)} train / {len(te)} held out "
          f"({len(rows) - len(tr) - len(te)} dropped as the overlap guard)")
    if thin:
        print(f"  behaviours with too few spans to test cleanly: {thin} "
              f"(trained, not tested)")
    print(f"  held-out set covers {len(tested)}/{len(by_goal)} behaviours: {tested}")

    remap = {g: i for i, g in enumerate(order)}

    def batch(idxs, ambiguous_only=False):
        """-> (clip, request, label). One row per (span, phrasing) pair."""
        cx, cr, cy = [], [], []
        for i in idxs:
            g = int(rows[i]["goal"])
            for text in phrasings[g]:
                if ambiguous_only and text == rows[i]["instruction"]:
                    continue
                cx.append(X[i]); cr.append(lut[text]); cy.append(remap[g])
        if not cx:                       # e.g. no held-out goal has a family
            return (torch.zeros(0, X.shape[1]), torch.zeros(0, R.shape[1]),
                    torch.zeros(0, dtype=torch.int64))
        return (torch.tensor(np.stack(cx)), torch.tensor(np.stack(cr)),
                torch.tensor(np.array(cy, dtype=np.int64)))

    Xt, Rt, yt = batch(tr)
    Xv, Rv, yv = batch(te)
    Xa, Ra, ya = batch(te, ambiguous_only=True)
    print(f"examples: {len(yt)} train, {len(yv)} held out "
          f"({len(ya)} of them under an ambiguous request)")

    # ---- THREE HEADS, not one head and two ablations ----
    #
    # Zeroing an input is a bad control. A head trained on both inputs and then
    # shown a zero vector is not the head you would have trained on one input:
    # its biases still fire, and it still knows the label distribution. Measured
    # here, that reads 0.536 for "the request alone" where a request-only MODEL
    # measures much lower, and the difference is not information the words
    # carry, it is the prior.
    #
    # So each arm is trained from scratch on exactly the inputs it is allowed:
    #
    #   clips + words   the system
    #   words only      the ceiling on what language alone can decide. Under
    #                   "put the bowl away" four behaviours fit, so this is
    #                   capped by the ambiguity no matter how good the model is.
    #   clips only      what the video alone decides, with nobody speaking.
    def fit_head(use_clip: bool, use_req: bool, tag: str):
        h = InstructionHead(k=bank.k, clip_dim=X.shape[1], req_dim=R.shape[1],
                            use_request=use_req)
        o = torch.optim.AdamW(h.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        xt = Xt if use_clip else torch.zeros_like(Xt)
        for e in range(epochs):
            h.train()
            o.zero_grad()
            lossf(h(xt, Rt if use_req else None), yt).backward()
            o.step()
        h.eval()
        return h

    def acc(h, use_clip, use_req, C, Rq, Y):
        if not len(Y):
            return float("nan")
        with torch.no_grad():
            c = C if use_clip else torch.zeros_like(C)
            return (h(c, Rq if use_req else None).argmax(1) == Y).float().mean().item()

    arms = {"clips + words": (True, True), "words only": (False, True),
            "clips only": (True, False)}
    got = {}
    for tag, (uc, ur) in arms.items():
        h = fit_head(uc, ur, tag)
        got[tag] = dict(head=h, use_clip=uc, use_req=ur,
                        all=acc(h, uc, ur, Xv, Rv, yv),
                        amb=acc(h, uc, ur, Xa, Ra, ya))
    head = got["clips + words"]["head"]
    use_request = True

    chance = 1.0 / bank.k
    print(f"\n{'=' * 70}")
    print(f"  {'trained on':<16}{'held out':>10}{'ambiguous only':>18}")
    for tag, g in got.items():
        print(f"  {tag:<16}{g['all']:>10.3f}{g['amb']:>18.3f}")
    print(f"  {'chance':<16}{chance:>10.3f}{chance:>18.3f}")

    full, words, clips = (got["clips + words"]["all"], got["words only"]["all"],
                          got["clips only"]["all"])
    amb_full, amb_words = got["clips + words"]["amb"], got["words only"]["amb"]
    print(f"\nThe ambiguous column is the one that decides it. Those requests are\n"
          f"true of several behaviours each, so a words-only model cannot exceed\n"
          f"the ambiguity however well it is trained — measured, {amb_words:.3f}.\n"
          f"With the retrieved video it reads {amb_full:.3f}. The gap is what the\n"
          f"memory contributed, and nothing else in the system could have.")
    if amb_full <= amb_words + 0.05:
        print("\nWARNING: the video adds nothing over the words. MEMORY IS NOT\n"
              "LOAD-BEARING in this model.")
    no_clip, no_req = words, clips

    os.makedirs(TRAIN, exist_ok=True)
    torch.save(dict(head=head.state_dict(), bank=bank.bank,
                    use_request=use_request, clip_dim=int(X.shape[1]),
                    req_dim=int(R.shape[1]), blocks=list(TR.BLOCKS),
                    goals=order, instructions=[texts[g] for g in order]),
               os.path.join(TRAIN, "reasoner.pt"))
    json.dump(dict(n_spans=len(rows), n_train=len(yt), n_test=len(yv), k=bank.k,
                   feature=list(TR.BLOCKS), chance=chance,
                   arms={t: dict(held_out=g["all"], ambiguous=g["amb"])
                         for t, g in got.items()}),
              open(os.path.join(TRAIN, "train.json"), "w"), indent=1)
    print(f"saved {TRAIN}/reasoner.pt")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--no-request", action="store_true",
                    help="head reads ONLY the clip — the real memory test")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=10000,
                    help="episode seeds start here; a second collection\nround must move it or it replays the same episodes")
    a = ap.parse_args()
    if a.collect:
        return collect(a.episodes, a.seed0)
    if a.train:
        return train(use_request=not a.no_request)
    ap.error("--collect or --train")


if __name__ == "__main__":
    sys.exit(main())
