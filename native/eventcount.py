"""Counting events is a DIFFERENT problem from locating them.

Debugging found the real bottleneck. The six sim templates are near
identical as action sequences (every pair >= 0.57 similar; two differ by
a single inserted unstack), and what separates them is HOW MANY events
occur - deterministic per template, 5/6/6/7/7/8. Measured:

    TRUE event count      AUC 0.905 separating same-template pairs
    CPD-estimated count   AUC 0.520   (chance)

So the single most discriminative feature in the corpus arrives at
chance. CPD gets the count exactly right 13% of the time, |error| 1.97.

That also explains sens2.py's cliff at last: only PERFECT boundaries
help because only perfect boundaries give the right COUNT.

The insight this file acts on: **you do not need boundaries in the
right place, you need the right NUMBER**, and that is a strictly easier
and better-posed question. RepNet (Dwibedi et al., CVPR 2020) makes the
same move for repetition counting - it forces prediction through a
temporal self-similarity bottleneck, and that constraint is what buys
class-agnostic generalisation.

We already build that matrix: `affinity(frame_features(F))` is a
temporal self-similarity matrix. graphgebd asked it WHERE to cut. Nobody
asked it HOW MANY, and the classical answer is the EIGENGAP of the
normalised Laplacian: for a matrix with k well-separated blocks, k
eigenvalues sit near zero and the (k+1)-th jumps. Choosing k by the
largest gap is standard spectral clustering theory, needs no training,
no threshold, and no labels.

Estimators compared, all label-free:
  eigengap   largest gap in the Laplacian spectrum
  eigenrole  count of eigenvalues below the spectrum's own Otsu cut
  cpd        the incumbent (slope-heuristic change points)
  novelty    Foote checkerboard-kernel peak count

    python native/eventcount.py --limit 150
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import arg                                       # noqa: E402

KMAX = 16


def laplacian_spectrum(W):
    """Symmetric normalised Laplacian eigenvalues, ascending."""
    d = W.sum(1)
    dm = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    L = np.eye(len(W)) - (W * dm[:, None]) * dm[None, :]
    ev = np.linalg.eigvalsh((L + L.T) * 0.5)
    return np.clip(ev, 0.0, None)


def count_eigengap(W, kmax=KMAX):
    ev = laplacian_spectrum(W)[:kmax + 1]
    if len(ev) < 3:
        return 1
    gaps = np.diff(ev)
    # k = index of the largest gap; +1 because a gap after the k-th
    # eigenvalue means k blocks
    return int(np.argmax(gaps[1:]) + 2)


def count_eigenotsu(W, kmax=KMAX):
    """How many eigenvalues are 'small'? Otsu on the spectrum itself,
    so the cut is fitted from this media's own distribution."""
    ev = laplacian_spectrum(W)[:kmax + 1]
    x = np.sort(ev)
    if len(x) < 3:
        return 1
    best, thr = -1.0, x[0]
    for i in range(1, len(x)):
        a, b = x[:i], x[i:]
        s = (len(a) * len(b) / len(x) ** 2) * (a.mean() - b.mean()) ** 2
        if s > best:
            best, thr = s, (a[-1] + b[0]) / 2
    return int(max((ev <= thr).sum(), 1))


def count_novelty(W, kern=8):
    """Foote checkerboard novelty: peaks = boundaries, +1 = segments."""
    n = len(W)
    if n < 2 * kern + 3:
        return 1
    k = np.ones((2 * kern, 2 * kern))
    k[:kern, kern:] = -1
    k[kern:, :kern] = -1
    nov = np.zeros(n)
    for t in range(kern, n - kern):
        nov[t] = (W[t - kern:t + kern, t - kern:t + kern] * k).sum()
    nov = nov[kern:n - kern]
    if len(nov) < 3:
        return 1
    z = (nov - nov.mean()) / (nov.std() + 1e-8)
    peaks = sum(1 for i in range(1, len(z) - 1)
                if z[i] > 1.0 and z[i] >= z[i - 1] and z[i] >= z[i + 1])
    return int(peaks + 1)


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return float((r[y == 1].sum() - len(pos) * (len(pos) - 1) / 2)
                 / (len(pos) * len(neg)))


def pair_auc(vals, labels):
    pos, neg = [], []
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            (pos if labels[i] == labels[j] else neg).append(
                -abs(vals[i] - vals[j]))
    return auc(np.array(pos), np.array(neg))


def main():
    import collections
    import encode as E
    import pyarrow.parquet as pq
    from tqdm import tqdm
    from cpd import fit
    from graphgebd import affinity, frame_features
    from segment import spans as to_spans
    limit = arg("--limit", 150, int)

    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    nev = collections.Counter()
    for e in t["episode"]:
        nev[int(e)] += 1

    CACHE = Path("/private/tmp/claude-501/u6retr")
    rows = []
    dirs = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                  if p.is_dir() and p.name.startswith("ep"))[:limit]
    for d in tqdm(dirs, desc="count", unit="ep"):
        ei = int(d.name[2:])
        if ei not in tmpl:
            continue
        cam = sorted(d.glob("cam*.mp4"))[0]
        dur = E.probe_duration(cam)
        F = E.decode(cam, fps=4.0, w=256)
        W = affinity(frame_features(F))
        fp = CACHE / f"cpd_{ei}_{len(F)}.npy"
        if fp.exists():
            ncpd = len([tuple(r) for r in np.load(fp)])
        else:
            bk = fit(frame_features(F), kind="slope", model="rbf")
            ncpd = len(to_spans([b / 4.0 for b in bk], dur))
        rows.append(dict(tm=tmpl[ei], true=nev[ei], cpd=ncpd,
                         eig=count_eigengap(W),
                         otsu=count_eigenotsu(W),
                         nov=count_novelty(W)))

    print(f"\nEVENT COUNT estimators, {len(rows)} episodes")
    print(f"{'estimator':<12}{'AUC':<9}{'exact%':<9}{'±1%':<9}"
          f"{'|err|':<8}{'mean pred (true 6.5)'}")
    tms = [r["tm"] for r in rows]
    tr = np.array([r["true"] for r in rows])
    for key in ("true", "eig", "otsu", "nov", "cpd"):
        v = np.array([r[key] for r in rows], float)
        e = v - tr
        print(f"{key:<12}{pair_auc(v, tms):<9.3f}"
              f"{np.mean(e == 0) * 100:<9.0f}"
              f"{np.mean(np.abs(e) <= 1) * 100:<9.0f}"
              f"{np.abs(e).mean():<8.2f}{v.mean():.2f}", flush=True)
    print("\ntrue-count AUC 0.905 is the ceiling; cpd 0.520 is chance.")


if __name__ == "__main__":
    main()
