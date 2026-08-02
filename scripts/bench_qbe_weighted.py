"""QbE with SEED-COHERENCE weighting, and the variants it was chosen over.

The fused QbE set scored BELOW its best single channel on every query -
q04 fused 0.37 against mot alone 0.675 - because seven channels vote
equally while their QbE quality ranges from AUC 0.50 to 0.91. That is the
same dilution the text path had, and the text path's fix (corroboration)
cannot be reused: it can veto an inverted channel but never identify the
good one, because the good channel is the one that disagrees.

QbE has a signal text does not: THE SEEDS ARE KNOWN TO BE THE SAME KIND.
The user supplied them, or a first pass did. So a channel that pulls the
seeds tightly together, relative to how tightly it holds the corpus in
general, is measuring whatever the seeds share. A channel that spreads
them is not. No truthset is consulted - the seeds are the query, not the
answer key.

    coherence(c) = (mean pairwise cosine among SEEDS
                    - mean cosine of seeds to the corpus) / pooled sd

which is Cohen's d computed from the query itself. Measured against the
truthset AFTERWARDS (never fitted on) it ranks iv2 first on q03 (d 1.91,
AUC 0.853) and mot first on q04 (d 2.21, AUC 0.747) - the right channel
for an appearance query and for a direction query respectively, chosen by
the same rule.

Variants compared so the shipped one is the measured winner, not the
first idea: uniform (today), coherence at three sharpnesses, and
best-single-by-coherence.

    python scripts/bench_qbe_weighted.py --seeds 5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                     # noqa: E402
from elidedb.setpath import confidence_cut                    # noqa: E402
from elidedb.qbe import spaces                                # noqa: E402


def coherence(A, ok, seeds):
    """Cohen's d of seed-to-seed cosine against seed-to-corpus cosine.
    Uses only the seeds and the corpus - no labels."""
    S = A[seeds]
    W = S @ S.T
    iu = np.triu_indices(len(seeds), 1)
    if not len(iu[0]):
        return 0.0
    within = W[iu]
    other = np.setdiff1d(np.where(ok)[0], seeds)
    if len(other) < 10:
        return 0.0
    B = (S @ A[other].T).ravel()
    sd = np.sqrt((within.var() + B.var()) / 2)
    if sd <= 0:
        return 0.0
    return float((within.mean() - B.mean()) / sd)


def fuse(M, seeds, weights, k_max, rrf_k=60.0):
    """Weighted RRF over each channel's cosine-to-seed-centroid."""
    tot = None
    for c, (A, ok) in M.items():
        w = weights.get(c, 0.0)
        if w <= 0:
            continue
        sd = [i for i in seeds if ok[i]]
        if not sd:
            continue
        cen = A[sd].mean(0)
        n = np.linalg.norm(cen)
        if n <= 0:
            continue
        sc = A @ (cen / n)
        sc[~ok] = -np.inf
        sc[seeds] = -np.inf
        r = np.empty(len(sc))
        r[np.argsort(-sc)] = np.arange(len(sc))
        v = w / (rrf_k + r)
        tot = v if tot is None else tot + v
    if tot is None:
        return np.array([], int)
    order = np.argsort(-tot)
    return order[:confidence_cut(tot[order], 0.0, k_max)]


def main():
    argv = sys.argv
    n_seed = int(argv[argv.index("--seeds") + 1] if "--seeds" in argv else 5)
    db = Store.open(str(ROOT / "lake/fresh_bench"))
    ep = db.table("episodes").scan()
    keys = [(str(s), int(a)) for s, a in
            zip(ep.column("stream").to_pylist(), ep.column("ts").to_pylist())]
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
    pos = {k: i for i, k in enumerate(keys)}

    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    G = {}
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        G[(int(q), int(ei))] = int(v)

    # channels discovered by the live path - see bench_qbe for why the
    # bench must never keep its own channel list
    skeys, SM = spaces(db, drop_pc=0)
    spos = {k: i for i, k in enumerate(skeys)}
    take = np.array([spos[k] for k in keys])
    M = {}
    for c, (A, _, _) in SM.items():
        A = np.asarray(A, np.float32)[take]
        M[c] = (A, np.abs(A).sum(1) > 0)

    VARIANTS = ("uniform", "coh^1", "coh^2", "coh^4", "best-single")
    from tqdm import tqdm
    res = {v: [] for v in VARIANTS}
    detail = []
    for qi in tqdm((3, 4, 5), desc="variants", unit="q", dynamic_ncols=True):
        truths = [i for i, e in enumerate(eidx) if G.get((qi, e)) == 1]
        judged = {i: G[(qi, eidx[i])] for i in range(len(eidx))
                  if (qi, eidx[i]) in G}
        sup = len(truths)
        k_max = int(np.ceil(sup * 1.5))
        # disjoint seed groups so no run reuses another's seeds
        rs = np.random.RandomState(0)
        groups = [rs.choice(truths, n_seed, replace=False) for _ in range(5)]
        for v in VARIANTS:
            ys, ps, pgs = [], [], []
            for seeds in groups:
                seeds = np.asarray(sorted(set(int(x) for x in seeds)))
                coh = {c: max(coherence(A, ok, seeds), 0.0)
                       for c, (A, ok) in M.items()}
                if v == "uniform":
                    w = {c: 1.0 for c in M}
                elif v == "best-single":
                    b = max(coh, key=coh.get)
                    w = {c: (1.0 if c == b else 0.0) for c in M}
                else:
                    p = float(v.split("^")[1])
                    w = {c: coh[c] ** p for c in M}
                chosen = fuse(M, seeds, w, k_max)
                lab = [judged.get(int(i)) for i in chosen]
                tr = sum(1 for x in lab if x == 1)
                ng = sum(1 for x in lab if x is not None)
                ys.append(tr / sup)
                ps.append(tr / max(len(chosen), 1))
                if ng:
                    pgs.append(tr / ng)
                if v == "coh^2":
                    detail.append({"q": qi, "weights":
                                   {c: round(coh[c], 2) for c in coh}})
            res[v].append({"q": qi, "yield": float(np.mean(ys)),
                           "prec": float(np.mean(ps)),
                           "prec_g": float(np.mean(pgs)) if pgs else None})

    print(f"\n{'variant':<13}" + "".join(f"{'q%02d yld' % q:>10}" for q in (3,4,5))
          + "".join(f"{'q%02d prc_g' % q:>12}" for q in (3,4,5)) + f"{'mean yld':>10}")
    for v in VARIANTS:
        r = {x["q"]: x for x in res[v]}
        print(f"  {v:<11}"
              + "".join(f"{r[q]['yield']:>10.2f}" for q in (3,4,5))
              + "".join(f"{(r[q]['prec_g'] or 0):>12.2f}" for q in (3,4,5))
              + f"{np.mean([r[q]['yield'] for q in (3,4,5)]):>10.2f}")
    print("\nseed-coherence weights (coh^2 run, one example per query):")
    seen = set()
    for d in detail:
        if d["q"] in seen:
            continue
        seen.add(d["q"])
        top = sorted(d["weights"].items(), key=lambda kv: -kv[1])
        print(f"  q{d['q']:02d}  " + "  ".join(f"{c}={x}" for c, x in top))
    (ROOT / "bench" / "bench_qbe_weighted.json").write_text(
        json.dumps(res, indent=1))
    print("\nwrote bench/bench_qbe_weighted.json")


if __name__ == "__main__":
    main()
