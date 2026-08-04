"""Retrieval-layer maximization over the channel pool.

Extraction is measured to its plateau (stages 0.85-0.91, every decoder
flat). What is NOT yet exhausted is the retrieval layer above the
channels - the machinery that made fresh_bench: rank fusion, seed-LOO
weighting, and pseudo-relevance feedback (PRF anchors - the banked
compliant pattern: the query's own top results re-query). Everything
here is query-time computation on the seeds; no truth touches any
scoring path. DEV picks the recipe; HOLDOUT runs it frozen.

    python scripts/chain_fuse.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
    "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
    "scratchpad")
DEV = ("swap", "precarious", "push_then_build", "build_unstack_move")
HOLD = ("relocate_build", "two_sites_merge")


def load():
    z = np.load(SCRATCH / "chain_channels.npz")
    eps = [int(e) for e in z["eps"]]
    mats = {k[2:]: z[k] for k in z.files if k.startswith("S_")}
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    return eps, mats, tmpl


def loo_weights(mats, si, temp=0.08):
    """Softmax over per-channel seed-LOO quality (query-time only)."""
    names = sorted(mats)
    sc = []
    for nm in names:
        S = mats[nm]
        rr = []
        for h in si:
            rest = [x for x in si if x != h]
            s = S[rest].max(0).copy()
            s[rest] = -1e9
            rr.append(1.0 / (1.0 + int((s > s[h]).sum())))
        sc.append(float(np.mean(rr)))
    sc = np.array(sc)
    w = np.exp((sc - sc.max()) / temp)
    return dict(zip(names, w / w.sum())), dict(zip(names, sc))


def zscore(sc):
    m = sc > -1e8
    mu, sd = float(sc[m].mean()), float(sc[m].std()) + 1e-8
    out = (sc - mu) / sd
    out[~m] = -1e9
    return out


def fused_scores(mats, si, kind, temp=0.08, topk=2):
    names = sorted(mats)
    w, raw = loo_weights(mats, si, temp)
    if kind == "rrf":
        z = np.zeros(len(next(iter(mats.values()))), np.float64)
        for nm in names:
            sc = mats[nm][si].max(0).copy()
            sc[si] = -1e9
            r = np.empty_like(sc)
            r[np.argsort(-sc)] = np.arange(len(sc))
            z += 1.0 / (60.0 + r)
    elif kind == "rrf_top":
        best = sorted(names, key=lambda nm: -raw[nm])[:topk]
        z = np.zeros(len(next(iter(mats.values()))), np.float64)
        for nm in best:
            sc = mats[nm][si].max(0).copy()
            sc[si] = -1e9
            r = np.empty_like(sc)
            r[np.argsort(-sc)] = np.arange(len(sc))
            z += 1.0 / (60.0 + r)
    elif kind == "soft":
        z = np.zeros(len(next(iter(mats.values()))), np.float64)
        for nm in names:
            sc = mats[nm][si].max(0).copy()
            sc[si] = -1e9
            z += w[nm] * zscore(sc)
    else:                                   # topk z (the baseline)
        best = sorted(names, key=lambda nm: -raw[nm])[:topk]
        z = np.zeros(len(next(iter(mats.values()))), np.float64)
        for nm in best:
            sc = mats[nm][si].max(0).copy()
            sc[si] = -1e9
            z += zscore(sc)
    z[si] = -1e9
    return z


def evaluate(eps, mats, tmpl, targets, kind, prf=0, prf_m=4,
             label="", temp=0.08, topk=2):
    pos = {e: i for i, e in enumerate(eps)}
    rs = np.random.RandomState(0)
    ys, ps = [], []
    for target in targets:
        pool = sorted(e for e, tm in tmpl.items()
                      if tm == target and e in pos)
        seeds = sorted(int(x) for x in rs.choice(pool, 5,
                                                 replace=False))
        support = len(pool) - len(seeds)
        k = math.ceil(1.5 * support)
        si = [pos[e] for e in seeds]
        z = fused_scores(mats, si, kind, temp, topk)
        for _ in range(prf):
            add = list(np.argsort(-z)[:prf_m])
            z = fused_scores(mats, si + add, kind, temp, topk)
        got = [eps[i] for i in np.argsort(-z)[:k]]
        true = sum(1 for e in got if tmpl.get(e) == target)
        ys.append(true / support)
        ps.append(true / len(got))
    print(f"   {label:<34} yield {np.mean(ys):.3f}  prec "
          f"{np.mean(ps):.3f}  ({' '.join(f'{y:.2f}' for y in ys)})")
    return float(np.mean(ys))


def main():
    eps, mats, tmpl = load()
    print(f"channels: {sorted(mats)}")
    print("-- DEV recipe search")
    results = {}
    for kind in ("top", "rrf", "rrf_top", "soft"):
        for prf in (0, 1, 2):
            for m in ((4,) if prf else (0,)):
                lab = f"{kind} prf{prf}m{m}"
                results[(kind, prf, m)] = evaluate(
                    eps, mats, tmpl, DEV, kind, prf, m, "DEV " + lab)
    best = max(results, key=results.get)
    print(f"-- frozen best recipe on HOLDOUT: {best}")
    kind, prf, m = best
    evaluate(eps, mats, tmpl, HOLD, kind, prf, m,
             f"HOLDOUT {kind} prf{prf}m{m}")
    # also show holdout across recipes for the record (not selection)
    print("-- HOLDOUT full table (record only)")
    for (kind, prf, m), _ in sorted(results.items()):
        evaluate(eps, mats, tmpl, HOLD, kind, prf, m,
                 f"HOLDOUT {kind} prf{prf}m{m}")


if __name__ == "__main__":
    main()
