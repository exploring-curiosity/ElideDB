"""Evaluate v4: three content channels x two matchers, WITH intervals.

Every comparison in this project so far has been a point estimate. I told the
owner a 0.573-vs-0.517 gap was "well outside the noise" without ever computing
an interval, and they were right to reject that. Nothing here is reported
without one.

BOOTSTRAP is over QUERIES, not over returned items. The unit of independence is
the episode: one query contributes ~90 correlated returns, so resampling
returns would give an interval several times too narrow. 2000 resamples of the
query set, percentile interval on the pooled precision sum(correct)/sum(support).
Paired differences between arms use the SAME resample indices, which is what
makes a difference-interval meaningful - the arms share queries, so their errors
are correlated and an unpaired comparison overstates the uncertainty.

CHANNELS (v4 caches all three; they differ only in what is subtracted)
    error       pred(t+k) - actual(t+k)
    obs_change  actual(t+k) - actual(t)
    pred_change pred(t+k)   - actual(t)

MATCHERS
    cosine    mean over t of cos(A_t,B_t) - assumes a shared clock
    span-dtw  best sub-alignment, free start and end - returns a SPAN and
              absorbs speed differences

    python -m relmo.vjeval4 --dataset rcasa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjeval import REC, group_key, l2, parse  # noqa: E402
from relmo.vjrec4 import CHANNELS, OUT4  # noqa: E402


def subseq_batch(Q, B, pen=0.05):
    """Best-span cost of Q inside each of B. Free start AND end."""
    S, n = Q.shape[0], B.shape[1]
    C = 1.0 - np.einsum("sd,nkd->nsk", Q, B)
    D = C[:, 0, :].astype(np.float64).copy()
    INF = 1e18
    for i in range(1, S):
        diag = np.concatenate([np.full((len(B), 1), INF), D[:, :-1]], 1)
        run = C[:, i, :] + np.minimum(diag, D + pen)
        for j in range(1, n):
            run[:, j] = np.minimum(run[:, j], run[:, j - 1] + pen)
        D = run
    return D.min(1) / S


def boot(corr, sup, idx):
    """Pooled precision under one resample of the query set."""
    return corr[idx].sum() / max(sup[idx].sum(), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--boots", type=int, default=2000)
    a = ap.parse_args()

    d4 = OUT4 / f"{a.dataset}_L{a.layer}"
    files = sorted(d4.glob("*.npz"))
    if len(files) < 100:
        raise SystemExit(f"only {len(files)} v4 records - let vjrec4 finish")
    meta = [parse(p.stem) for p in files]
    Z = [np.load(p) for p in files]
    ch = {k: l2(np.stack([z[k] for z in Z])) for k in CHANNELS}
    # scene-only baseline comes from the v1 cache
    d1 = REC / a.dataset
    FF = l2(np.stack([np.load(d1 / f"{p.stem}.npz")["f_first"] for p in files]))
    epid = np.array([f"{m['task']}#{m['epnum']}" for m in meta])
    G = np.array([group_key(m) for m in meta])
    N = len(files)
    print(f"{N} records, {len(set(G))} groups\n")

    arms = {"scene-only": ("flat", FF)}
    for k in CHANNELS:
        arms[f"{k}/cosine"] = ("cos", ch[k])
        arms[f"{k}/span"] = ("dtw", ch[k])

    corr = {k: [] for k in arms}
    sup, rnd = [], []
    fam = {k: {} for k in arms}
    for i in range(N):
        full = np.where(epid != epid[i])[0]
        sv = G[full] == G[i]
        s_ = int(sv.sum())
        if s_ < 5:
            continue
        sup.append(s_)
        rnd.append(s_ * s_ / len(full))
        for k, (mode, M) in arms.items():
            if mode == "flat":
                sc = M[full] @ M[i]
            elif mode == "cos":
                sc = np.einsum("sd,nsd->n", M[i], M[full]) / M.shape[1]
            else:
                sc = -subseq_batch(M[i], M[full])
            c = int(sv[np.argsort(-sc)[:s_]].sum())
            corr[k].append(c)
            f = fam[k].setdefault(meta[i]["task"], [0, 0, 0])
            f[0] += c
            f[1] += s_
            f[2] += 1
    sup = np.array(sup, float)
    rnd = np.array(rnd, float)
    corr = {k: np.array(v, float) for k, v in corr.items()}
    base = rnd.sum() / sup.sum()
    rs = np.random.default_rng(0)
    B = np.stack([rs.integers(0, len(sup), len(sup)) for _ in range(a.boots)])

    print(f"{len(sup)} queries | k=support | group-aware | chance {base:.3f}")
    print(f"{'arm':22s} {'true/returned':>15s} {'prec':>7s} {'95% CI':>16s} "
          f"{'lift':>6s}")
    print("-" * 72)
    pt, dist = {}, {}
    for k in arms:
        p = corr[k].sum() / sup.sum()
        ds = np.array([boot(corr[k], sup, B[b]) for b in range(a.boots)])
        pt[k], dist[k] = p, ds
        lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"{k:22s} {f'{corr[k].sum():.0f}/{sup.sum():.0f}':>15s} "
              f"{p:7.3f} {f'[{lo:.3f},{hi:.3f}]':>16s} {p/base:5.2f}x")

    # PAIRED differences against the incumbent
    ref = "error/cosine"
    print(f"\npaired difference vs {ref} (same resamples; CI excluding 0 = real)")
    for k in arms:
        if k == ref:
            continue
        dd = dist[k] - dist[ref]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        verdict = "REAL" if lo > 0 or hi < 0 else "not separated"
        print(f"  {k:22s} {pt[k]-pt[ref]:+.4f}  "
              f"[{lo:+.4f},{hi:+.4f}]  {verdict}")

    best = max(pt, key=pt.get)
    print(f"\nper-family, {best}")
    print(f"{'task family':26s} {'n':>5s} {'true/returned':>15s} {'prec':>7s}")
    print("-" * 58)
    for kk in sorted(fam[best], key=lambda x: -fam[best][x][0] / fam[best][x][1]):
        c, n, q = fam[best][kk]
        print(f"{kk:26s} {q:5d} {f'{c}/{n}':>15s} {c/n:7.3f}")
    R.log("vjeval4", dataset=a.dataset, layer=a.layer, records=N,
          queries=len(sup), chance=round(base, 4), best=best,
          **{k.replace("/", "_"): round(float(v), 4) for k, v in pt.items()})


if __name__ == "__main__":
    main()
