"""TIER-1 cross-encoder: a VLM scores query vs candidate JOINTLY.

Why this and not more embedding compute: measured (native/rerank.py)
that re-comparing the SAME stored evidence more carefully moves the
mean DOWN (0.41 -> 0.36). Recall is not the problem (recall@200 =
0.61 mean, 0.84-1.00 on needles); ORDERING is. Ordering can only
improve if something reads the query and the candidate TOGETHER, with
the query in hand - which is what a cross-encoder does and what no
write-time vector can do, because a stored vector must commit to its
distinctions before the query exists.

Nothing is stored: the VLM never touches the write path, sees only the
~100 survivors of the index, and produces no artifact. This is the
query-conditioned read tier.

    python native/vlmrank.py --time              # one-call timing
    python native/vlmrank.py --q 1,2,7 --pool 100
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

MID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
NF = 6                      # frames sampled per clip
# SAMPLED digits are useless here: the model answers "9" to almost
# everything (measured AUC 0.11 - the classic positive bias). The
# usable signal is the TOKEN PROBABILITY of the answer, which is
# continuous even when the argmax never changes (measured AUC 0.77).
PROMPT = (
    "Row A: frames from a reference clip. Row B: frames from a "
    "candidate clip. Is the SAME action performed on the SAME kind "
    "of object in both? Answer Yes or No.")


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def clip_frames(db, ep_row, n=NF):
    """n evenly spaced RGB frames of one episode."""
    import chain_delta as cd
    ts, F = cd.decode_view(db, ep_row)
    idx = np.linspace(0, len(F) - 1, n).round().astype(int)
    return [F[i] for i in idx]


def to_pil(frames, w=224):
    from PIL import Image
    out = []
    for f in frames:
        im = Image.fromarray(np.asarray(f, np.uint8))
        im.thumbnail((w, w))
        out.append(im)
    return out


_TOK = {}


def score_pair(model, proc, cfg, q_imgs, c_imgs):
    """P(Yes) from the answer token's logprobs - continuous score."""
    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    if not _TOK:
        tk = proc.tokenizer if hasattr(proc, "tokenizer") else proc
        _TOK["yes"] = tk.encode("Yes")[0]
        _TOK["no"] = tk.encode("No")[0]
    imgs = q_imgs + c_imgs
    prompt = apply_chat_template(proc, cfg, PROMPT,
                                 num_images=len(imgs))
    for r in stream_generate(model, proc, prompt, image=imgs,
                             max_tokens=1):
        lp = r.logprobs
        if lp is None:
            return 0.5
        a = np.array(lp).ravel()
        ey, en = np.exp(a[_TOK["yes"]]), np.exp(a[_TOK["no"]])
        return float(ey / (ey + en + 1e-9))
    return 0.5


def main():
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    from elidedb import Store
    import chain_delta as cd
    import pyarrow.parquet as pq
    from mem import load_channels, build_reps
    from memfuse import per_seed_scores, combine

    db = Store.open(str(ROOT / "lake/fresh_bench"))
    views = {v[0]: v[2] for v in cd.episode_views(db)}
    print(f"loading {MID} ...", flush=True)
    model, proc = load(MID)
    cfg = load_config(MID)

    if "--time" in sys.argv:
        eps = sorted(views)[:2]
        a = to_pil(clip_frames(db, views[eps[0]]))
        b = to_pil(clip_frames(db, views[eps[1]]))
        t0 = time.time()
        s = score_pair(model, proc, cfg, a, b)
        dt = time.time() - t0
        t0 = time.time()
        s2 = score_pair(model, proc, cfg, a, b)
        dt2 = time.time() - t0
        print(f"one call: {dt:.1f}s (warm {dt2:.1f}s)  score {s}/{s2}")
        print(f"-> 100 candidates = {dt2*100/60:.0f} min per query-group")
        return

    POOL = arg("--pool", 100, int)
    NGRP = arg("--groups", 1, int)
    NREF = arg("--refs", 2, int)
    want = [int(x) for x in arg("--q", "1,2,7", str).split(",")]
    reps = build_reps(load_channels(db, "fresh_bench"))
    eps = sorted({e for R in reps.values() for e in R["pool"]})
    ep = db.table("episodes").scan()
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet") \
        .to_pydict()
    G, sup = {}, {}
    have = set(eidx)
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        ei = int(ei)
        G[(int(q), ei)] = int(v)
        if ei in have:
            sup[int(q)] = sup.get(int(q), 0) + int(v)

    frame_cache = {}

    def imgs_of(e):
        if e not in frame_cache:
            frame_cache[e] = to_pil(clip_frames(db, views[e]))
        return frame_cache[e]

    print(f"{'q':<5}{'sup':<6}{'tier0':<8}{'vlm':<8}{'delta':<8}"
          f"{'calls':<7}{'min':<6}")
    b_all, v_all = [], []
    for qi in want:
        s = sup.get(qi, 0)
        if not s:
            continue
        truths = [e for e in eidx if G.get((qi, e)) == 1]
        ns = min(5, len(truths))
        if ns < 2 or len(truths) - ns < 1:
            continue
        kb = int(np.ceil(s * 1.5))
        # THE POOL MUST EXCEED k, or the rerank is capped by pool size
        # rather than model quality (measured: q4 kb=248 vs pool=100
        # scored 0.36, i.e. recall@100, and read as a model failure).
        pool_n = max(POOL, 2 * kb)
        rs = np.random.RandomState(0)
        b_runs, v_runs, calls, mins = [], [], 0, 0.0
        for _ in range(NGRP):
            sd = sorted(set(int(x) for x in
                            rs.choice(truths, ns, replace=False)))
            S = per_seed_scores(reps, eps, sd)
            tot = combine(S, sd, eps, "mean", "all_rrf")
            pos = {e: i for i, e in enumerate(eps)}
            for x in sd:
                if x in pos:
                    tot[pos[x]] = -1e9
            order = np.argsort(-tot)
            rem = s - len(sd)
            base = [eps[i] for i in order[:kb]]
            b_runs.append(sum(1 for e in base
                              if G.get((qi, e)) == 1) / max(rem, 1))
            cand = [eps[i] for i in order[:pool_n]]
            refs = [imgs_of(x) for x in sd[:NREF] if x in views]
            t0 = time.time()
            sc = []
            for c in cand:
                if c not in views:
                    sc.append(-1)
                    continue
                sc.append(max(score_pair(model, proc, cfg, r,
                                         imgs_of(c)) for r in refs))
            mins += (time.time() - t0) / 60
            calls += len(cand) * len(refs)
            # BLEND, never replace: the index ordering already encodes
            # six encoders' evidence; a reranker with pairwise AUC
            # 0.56-0.77 that OVERWRITES it measured -0.10 mean. Rank
            # fusion keeps the retriever's prior and lets the
            # cross-encoder perturb it.
            v_arr = np.array(sc, np.float64)
            r_vlm = np.empty(len(cand))
            r_vlm[np.argsort(-v_arr)] = np.arange(len(cand))
            r_idx = np.arange(len(cand))            # index order
            W = arg("--w", 0.5, float)
            fused = (1 - W) / (60.0 + r_idx) + W / (60.0 + r_vlm)
            rk = np.argsort(-fused)
            top = [cand[i] for i in rk[:kb]]
            v_runs.append(sum(1 for e in top
                              if G.get((qi, e)) == 1) / max(rem, 1))
        b, v = float(np.mean(b_runs)), float(np.mean(v_runs))
        b_all.append(b)
        v_all.append(v)
        print(f"q{qi:<4}{s:<6}{b:<8.2f}{v:<8.2f}{v-b:+.2f}    "
              f"{calls:<7}{mins:<6.0f}", flush=True)
    if b_all:
        print(f"{'MEAN':<11}{np.mean(b_all):<8.2f}{np.mean(v_all):<8.2f}"
              f"{np.mean(v_all)-np.mean(b_all):+.2f}")


if __name__ == "__main__":
    main()
