"""Late-interaction matching over multi-region records, against the v4 baseline.

v4 emits ONE vector per timestep, so a hand and a door doing different things
are averaged into a direction describing neither. v5 emits K, and this matches
them as SETS.

MaxSim (ColBERT-style), per aligned timestep pair:

    sim(i, j) = mean over query regions k of  max over reference regions l
                cos( Q[i,k], B[j,l] )

Max over reference regions is what makes region identity irrelevant: the hand
need not be region 0 in both clips, and the number of genuinely distinct
motions differs between clips anyway. Mean over query regions rather than max
so a single well-matched region cannot carry the whole score.

That cost matrix then feeds the same subsequence DTW as v4 - free start and
end, so the result is still a span and the comparison against v4 is a change of
descriptor only, not of matcher.

BASELINE ARMS, all on the identical query set:
    v4 single    the current shipped descriptor
    v5 mean      v5's regions averaged back into one vector. This is the
                 control that matters: if it matches v5-maxsim, the gain came
                 from the extra pooling rather than from the set structure.

    python -m relmo.vjeval5 --regions 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjeval import REC, group_key, l2, parse  # noqa: E402
from relmo.vjrec4 import OUT4  # noqa: E402
from relmo.vjrec5 import OUT5  # noqa: E402


def dtw_from_cost(C, pen=0.05):
    """Subsequence DTW over a precomputed (N,S,S) cost matrix. Free ends."""
    n_ref = C.shape[2]
    D = C[:, 0, :].astype(np.float64).copy()
    INF = 1e18
    for i in range(1, C.shape[1]):
        diag = np.concatenate([np.full((len(C), 1), INF), D[:, :-1]], 1)
        run = C[:, i, :] + np.minimum(diag, D + pen)
        for j in range(1, n_ref):
            run[:, j] = np.minimum(run[:, j], run[:, j - 1] + pen)
        D = run
    return D.min(1) / C.shape[1]


def cost_single(Q, B):
    return 1.0 - np.einsum("sd,nkd->nsk", Q, B)


def cost_maxsim(Q, B):
    """Q (S,K,D), B (N,S,K,D) -> (N,S,S) cost. All L2-normalised."""
    sim = np.einsum("ikd,njld->nijkl", Q, B)
    return 1.0 - sim.max(-1).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--regions", type=int, default=3)
    a = ap.parse_args()

    d5 = OUT5 / f"{a.dataset}_L{a.layer}_K{a.regions}"
    d4 = OUT4 / f"{a.dataset}_L{a.layer}"
    files = [p for p in sorted(d5.glob("*.npz"))
             if (d4 / f"{p.stem}.npz").exists()]
    if len(files) < 50:
        raise SystemExit(f"only {len(files)} paired records - let vjrec5 run")
    meta = [parse(p.stem) for p in files]
    M5 = np.stack([np.load(p)["multi"] for p in files])          # (N,S,K,D)
    M4 = np.stack([np.load(d4 / f"{p.stem}.npz")["pred_change"] for p in files])
    V5 = l2(M5)
    V5m = l2(M5.mean(2))                                          # regions -> 1
    V4 = l2(M4)
    epid = np.array([f"{m['task']}#{m['epnum']}" for m in meta])
    G = np.array([group_key(m) for m in meta])
    N = len(files)
    print(f"{N} paired records | K={a.regions}\n")

    arms = {"v4 single": ("s", V4), "v5 mean": ("s", V5m),
            "v5 maxsim": ("m", V5)}
    tot = {k: [0, 0] for k in list(arms) + ["random"]}
    fam = {k: {} for k in arms}
    for i in range(N):
        full = np.where(epid != epid[i])[0]
        sv = G[full] == G[i]
        sup = int(sv.sum())
        if sup < 5:
            continue
        for k, (mode, M) in arms.items():
            C = cost_single(M[i], M[full]) if mode == "s" else \
                cost_maxsim(M[i], M[full])
            s = -dtw_from_cost(C)
            c = int(sv[np.argsort(-s)[:sup]].sum())
            tot[k][0] += c
            tot[k][1] += sup
            f = fam[k].setdefault(meta[i]["task"], [0, 0, 0])
            f[0] += c
            f[1] += sup
            f[2] += 1
        tot["random"][0] += sup * sup / len(full)
        tot["random"][1] += sup
    r = tot["random"][0] / tot["random"][1]
    print(f"k=support | group-aware | chance {r:.3f}")
    print(f"{'arm':14s} {'true/returned':>15s} {'prec':>7s} {'lift':>7s}")
    print("-" * 48)
    for k in ["random"] + list(arms):
        c, n = tot[k]
        print(f"{k:14s} {f'{c:.0f}/{n}':>15s} {c/n:7.3f} {(c/n)/r:6.2f}x")
    best = max(arms, key=lambda k: tot[k][0] / tot[k][1])
    print(f"\nper-family: {best}   (v4 single in brackets)")
    print(f"{'task family':26s} {'n':>4s} {'prec':>7s} {'v4':>8s} {'delta':>8s}")
    print("-" * 58)
    for kk in sorted(fam[best], key=lambda x: -fam[best][x][0] / fam[best][x][1]):
        c, n, q = fam[best][kk]
        c4, n4, _ = fam["v4 single"][kk]
        print(f"{kk:26s} {q:4d} {c/n:7.3f} {c4/n4:8.3f} {c/n-c4/n4:+8.3f}")
    R.log("vjeval5", dataset=a.dataset, K=a.regions, records=N, chance=round(r, 4),
          **{k.replace(" ", "_"): round(tot[k][0] / tot[k][1], 4) for k in arms})


if __name__ == "__main__":
    main()
