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


# ------------------------------------------------------------------- collect

def collect(episodes: int) -> int:
    from brigade.agent.pilot import Pilot
    from brigade.memory.relmo import RelMoSidecar
    from brigade.memory.store import VideoStore

    os.makedirs(TRAIN, exist_ok=True)
    relmo = RelMoSidecar()
    print("starting RelMo (~85s) ...", flush=True)
    if not relmo.start():
        print(f"RelMo unavailable: {relmo.error}")
        return 1
    store = VideoStore(relmo=relmo)
    store.setup()
    store.start()

    p = Pilot(device="mps")
    p.load()

    rows = []
    for goal in GOALS:
        p.open_scene("libero_goal", goal)
        instruction = p.default_instruction
        for k in range(episodes):
            cut: list[str] = []

            def stream(_buf, _step, _s=store, _c=cut, _p=p):
                # The memory camera streams on its own clock, independent of
                # whether an episode is running. Every other control step at
                # 20 Hz gives the store's 10 fps.
                if _step % 2 == 0:
                    cid = _s.write(_p.memory_frame())
                    if cid:
                        _c.append(cid)

            # Collection runs INDEPENDENT episodes on purpose. Carrying the
            # world across all twenty scrambled the kitchen — persistence costs
            # roughly half the policy's success rate (RESULTS.md) — and every
            # later goal failed, so the training set was all failures. The
            # continuous kitchen belongs at serve time; training wants clean
            # examples of each behaviour.
            ep = p.run(instruction, keep_frames=False, on_frame=stream,
                       seed=10000 + k)
            # Only SUCCESSFUL episodes teach. A failed attempt's video shows
            # the robot not achieving the behaviour, and labelling it with that
            # behaviour teaches the head the wrong thing.
            if ep.success:
                for cid in cut:
                    rows.append(dict(clip_id=cid, goal=goal,
                                     instruction=instruction, success=True))
            print(f"  goal {goal} ep {k}: {'ok' if ep.success else 'fail'}  "
                  f"{len(cut)} segments  (store {store.n_written} written, "
                  f"{store.pending()} pending)", flush=True)

    print("\nwaiting for the encoder to drain ...", flush=True)
    while store.pending() and relmo.ready:
        time.sleep(2)
    time.sleep(3)
    json.dump(rows, open(os.path.join(TRAIN, "segments.json"), "w"), indent=1)
    print(f"store: {store.stats()}")
    print(f"wrote {len(rows)} labelled segments to {TRAIN}/segments.json")
    relmo.stop()
    return 0


# --------------------------------------------------------------------- train

def train(epochs: int = 400, use_request: bool = True) -> int:
    import torch
    import torch.nn as nn

    from brigade.agent.latent import InstructionHead, PrototypeBank, Reasoner
    from brigade.agent.pilot import Pilot
    from brigade.memory.store import VideoStore

    rows = json.load(open(os.path.join(TRAIN, "segments.json")))
    store = VideoStore()

    # ---- inputs: RelMo vectors of the segments, straight from the store ----
    ids = [r["clip_id"] for r in rows]
    have = store.db.query(
        "SELECT clip_id, embedding FROM clips WHERE clip_id = ANY(%s) "
        "AND embedding IS NOT NULL", (ids,))
    from brigade.memory.store import _parse_vec

    vec = {r["clip_id"]: _parse_vec(r["embedding"]) for r in have}
    rows = [r for r in rows if r["clip_id"] in vec]
    if not rows:
        print("no encoded segments; run --collect first")
        return 1
    X = np.stack([vec[r["clip_id"]] for r in rows]).astype(np.float32)
    y = np.array([r["goal"] for r in rows], dtype=np.int64)
    print(f"{len(rows)} segments, {len(set(y.tolist()))} behaviours, "
          f"clip dim {X.shape[1]}")

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
    lut = {k: v for k, v in zip(z["texts"].tolist(), z["vecs"])}
    missing = [r["instruction"] for r in rows if r["instruction"] not in lut]
    if missing:
        print(f"{len(set(missing))} instructions not in the cache; re-run text_vectors.py")
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

    # ---- fit ----
    idx = np.random.RandomState(0).permutation(len(rows))
    cut = int(len(idx) * 0.75)
    tr, te = idx[:cut], idx[cut:]
    Xt, Rt, yt = map(torch.tensor, (X[tr], R[tr], y[tr]))
    Xv, Rv, yv = map(torch.tensor, (X[te], R[te], y[te]))
    remap = {g: i for i, g in enumerate(order)}
    yt = torch.tensor([remap[int(v)] for v in yt])
    yv = torch.tensor([remap[int(v)] for v in yv])

    head = InstructionHead(k=bank.k, clip_dim=X.shape[1], req_dim=R.shape[1],
                           use_request=use_request)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    for e in range(epochs):
        head.train()
        opt.zero_grad()
        loss = lossf(head(Xt, Rt if use_request else None), yt)
        loss.backward()
        opt.step()
        if (e + 1) % 100 == 0:
            head.eval()
            with torch.no_grad():
                acc = (head(Xv, Rv if use_request else None).argmax(1) == yv).float().mean().item()
            print(f"  epoch {e+1:4d}  loss {loss.item():.3f}  held-out {acc:.3f}",
                  flush=True)

    # ---- the ablations that decide whether memory is doing the work ----
    head.eval()
    with torch.no_grad():
        rq = Rv if use_request else None
        full = (head(Xv, rq).argmax(1) == yv).float().mean().item()
        no_clip = (head(torch.zeros_like(Xv), rq).argmax(1) == yv).float().mean().item()
        no_req = (head(Xv, torch.zeros_like(Rv)).argmax(1) == yv).float().mean().item() \
            if use_request else full
    chance = 1.0 / bank.k
    print(f"\n{'='*66}\nheld-out accuracy   full {full:.3f}   "
          f"clip-ablated {no_clip:.3f}   request-ablated {no_req:.3f}   "
          f"chance {chance:.3f}")
    # A 0.125 gap is not a collapse. The first run read full 1.000 /
    # clip-ablated 0.875 and printed "memory is doing the work", which was
    # wrong: at training the request IS the instruction, a 1:1 map the head can
    # memorise, so it never needed the clips. Half is the honest bar.
    if no_clip >= full * 0.5:
        print("WARNING: accuracy survives removing the clip. The head is reading\n"
              "the request alone and MEMORY IS NOT LOAD-BEARING in this model.")
    else:
        print("clip ablation collapses the head -> memory is doing the work.")

    os.makedirs(TRAIN, exist_ok=True)
    torch.save(dict(head=head.state_dict(), bank=bank.bank,
                    use_request=use_request,
                    goals=order, instructions=[texts[g] for g in order]),
               os.path.join(TRAIN, "reasoner.pt"))
    json.dump(dict(n=len(rows), k=bank.k, full=full, clip_ablated=no_clip,
                   request_ablated=no_req, chance=chance),
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
    a = ap.parse_args()
    if a.collect:
        return collect(a.episodes)
    if a.train:
        return train(use_request=not a.no_request)
    ap.error("--collect or --train")


if __name__ == "__main__":
    sys.exit(main())
