"""TIER-1 rerank: cheap-recall index -> careful comparison of the few.

Measured precondition (native, 2026-08-04): the index's RECALL@200 is
0.61 mean / 0.84-1.00 on needle queries, while its yield at
k=1.5*support is 0.41. The answers are already in the pool; only the
ordering is wrong. That is the classic bi-encoder -> cross-encoder
situation, and it is also the elision thesis: prune 99%, then read the
survivors carefully.

This is the cheapest possible Tier-1 - the SAME frozen encoders, but
compared without the shortcuts the index takes for speed:
  * full-resolution sequences (the index resamples to 12 steps)
  * per-window late interaction (every query window finds its best
    match in the candidate, and vice versa - symmetric chamfer)
  * all channels z-fused, no selection
No new model, no text, no domain knowledge. If this recovers headroom,
a heavier Tier-1 (VLM cross-encoder) is justified on the same wiring.

    python native/rerank.py --store lake/fresh_bench [--pool 200]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from mem import load_channels, build_reps, arg, norm            # noqa: E402
from memfuse import per_seed_scores, combine, _z                # noqa: E402


def late_interaction(A, B):
    """Symmetric chamfer over FULL-resolution window sequences."""
    C = A @ B.T
    return 0.5 * (C.max(1).mean() + C.max(0).mean())


def rerank_scores(chans, cand, seeds):
    """z-fused late-interaction score per candidate, all channels."""
    tot = np.zeros(len(cand), np.float64)
    used = 0
    for name, seqs in chans.items():
        qs = [seqs[s] for s in seeds if s in seqs and len(seqs[s]) >= 2]
        if not qs:
            continue
        v = np.full(len(cand), -1e9, np.float64)
        for i, c in enumerate(cand):
            B = seqs.get(c)
            if B is None or len(B) < 2:
                continue
            v[i] = max(late_interaction(q, B) for q in qs)
        if (v > -1e8).sum() < 5:
            continue
        tot += _z(v)
        used += 1
    return tot, used


def main():
    from elidedb import Store
    import pyarrow.parquet as pq
    store = ROOT / arg("--store", "lake/fresh_bench")
    POOL = arg("--pool", 200, int)
    db = Store.open(str(store))
    chans = load_channels(db, store.name)
    reps = build_reps(chans)
    eps = sorted({e for R in reps.values() for e in R["pool"]})
    print(f"{len(eps)} episodes, channels {sorted(chans)}, "
          f"pool={POOL}", flush=True)

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

    b_all, r_all = [], []
    print(f"{'q':<5}{'sup':<6}{'tier0':<8}{'tier1':<8}{'delta':<8}")
    for qi in range(11):
        s = sup.get(qi, 0)
        if not s:
            continue
        truths = [e for e in eidx if G.get((qi, e)) == 1]
        ns = min(5, len(truths))
        if ns < 2 or len(truths) - ns < 1:
            continue
        kb = int(np.ceil(s * 1.5))
        rs = np.random.RandomState(0)
        b_runs, r_runs = [], []
        for _ in range(5):
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
            b_runs.append(sum(1 for e in base if G.get((qi, e)) == 1)
                          / max(rem, 1))
            cand = [eps[i] for i in order[:POOL]]
            rsc, used = rerank_scores(chans, cand, sd)
            top = [cand[i] for i in np.argsort(-rsc)[:kb]]
            r_runs.append(sum(1 for e in top if G.get((qi, e)) == 1)
                          / max(rem, 1))
        b, r = float(np.mean(b_runs)), float(np.mean(r_runs))
        b_all.append(b)
        r_all.append(r)
        print(f"q{qi:<4}{s:<6}{b:<8.2f}{r:<8.2f}{r-b:+.2f}", flush=True)
    print(f"{'MEAN':<11}{np.mean(b_all):<8.2f}{np.mean(r_all):<8.2f}"
          f"{np.mean(r_all)-np.mean(b_all):+.2f}")


if __name__ == "__main__":
    main()
