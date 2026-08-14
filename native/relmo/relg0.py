"""G0 for the RELATIONAL target: is it predictable from PIXELS?

relsep proved the relational channel carries the distinction motion-R2
cannot: on trajectory-matched windows, articulation separates
articulated manipulation from transport at AUC 0.997 while the best
trajectory feature manages 0.771. But that used ORACLE relations read
out of sim state. The world model does not get those at serve time - it
gets video.

So this asks the only question that matters before building a relational
head: can the target be predicted from what the model can actually see?
Same discipline as do_gate_g0 - establish the target is learnable BEFORE
spending a run on it.

STRICTLY 2D TRACKS. Features come from `xy` and `vis` only, which are
CoTracker's output on raw video and are available on any clip at deploy.
Deliberately NOT used: xpos/xquat/qpos/contact_pairs (sim state), gxy /
gvis / gdist (lifted through sim), bid / ident / seg (privileged body
ids), pdepth (rendered depth). If the answer here is yes, it is yes for
a system that only ever sees pixels.

The features are the observable correlates of "a hinged part swinging
through its range" versus "a free body being carried":

  moving_frac    how much of the scene moves at all
  n_clusters     is the motion one coherent thing or several
  rigidity       spread of pairwise distances within the moving set -
                 a rigid part holds them, a carried object plus a hand
                 does not
  rot_ratio      rotation of the moving set's principal axis vs its
                 translation - a hinge rotates, a carry translates
  decel          does speed fall toward the end of the window - a joint
                 limit stops motion, a carry does not
  path_curv      straightness of the moving centroid's path
  spread_change  does the moving set grow/shrink (grasp and release)

Scored by training on TRAIN episodes and testing on TEST episodes, so
the number is generalisation, not fit. Oracle articulation supplies the
label at scoring time only.

    python -m relmo.relg0 --dataset rcasa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import splits as SP  # noqa: E402
from relmo.relsep import auc, rel_feats  # noqa: E402

MOVE_Q = 0.75          # a point is "moving" if above this quantile of speed


def pixel_feats(z, t0, W):
    """2D tracks only. Nothing here exists outside the video."""
    xy, vs = z["xy"].astype(np.float64), z["vis"]
    T = len(xy)
    a, b = t0, min(t0 + W, T)
    if b - a < 4:
        return None
    X, V = xy[a:b], vs[a:b]
    ok = V.all(0)                                   # visible all window
    if ok.sum() < 12:
        return None
    X = X[:, ok]
    d = np.linalg.norm(X[-1] - X[0], axis=-1)
    thr = np.quantile(d, MOVE_Q)
    mv = d >= max(thr, 1e-6)
    if mv.sum() < 4:
        return None
    M = X[:, mv]                                    # (W, P, 2)
    cen = M.mean(1)
    step = np.linalg.norm(np.diff(cen, axis=0), axis=-1)
    net = np.linalg.norm(cen[-1] - cen[0])
    path = step.sum() + 1e-9
    # rigidity: how constant are pairwise distances inside the moving set
    i0, i1 = np.triu_indices(M.shape[1], 1)
    if len(i0) > 400:
        sel = np.random.default_rng(0).choice(len(i0), 400, replace=False)
        i0, i1 = i0[sel], i1[sel]
    pd = np.linalg.norm(M[:, i0] - M[:, i1], axis=-1)      # (W, npair)
    rig = float(np.median(pd.std(0) / (pd.mean(0) + 1e-9)))
    # rotation of the principal axis vs translation
    ang = []
    for t in (0, len(M) - 1):
        c = M[t] - M[t].mean(0)
        w, vec = np.linalg.eigh(c.T @ c)
        ang.append(np.arctan2(vec[1, -1], vec[0, -1]))
    dang = abs(np.arctan2(np.sin(ang[1] - ang[0]), np.cos(ang[1] - ang[0])))
    spread = float(np.linalg.norm(M[0] - M[0].mean(0), axis=-1).mean())
    rot_ratio = float(dang * spread / (net + 1e-6))
    h = max(len(step) // 3, 1)
    decel = float((step[:h].mean() + 1e-9) / (step[-h:].mean() + 1e-9))
    # how many coherent motion groups? cheap proxy: spread of per-point
    # displacement directions
    dv = (M[-1] - M[0])
    dv = dv / (np.linalg.norm(dv, axis=-1, keepdims=True) + 1e-9)
    coh = float(np.linalg.norm(dv.mean(0)))          # 1 = one direction
    s0 = np.linalg.norm(M[0] - M[0].mean(0), axis=-1).mean()
    s1 = np.linalg.norm(M[-1] - M[-1].mean(0), axis=-1).mean()
    rest = d[~mv]
    contrast = float(d[mv].mean() / (rest.mean() + 1e-6)) if rest.size else 1.0
    return dict(motion_contrast=contrast, rigidity=rig,
                rot_ratio=rot_ratio, decel=decel, coherence=coh,
                path_curv=float(net / path),
                spread_change=float(s1 / (s0 + 1e-9)))


def fit_probe(tr, te, F, return_scores=False):
    """Logistic probe fit on TRAIN rows, scored on held-out rows.

    Factored out of __main__ so the 3D probe (relmo/relg3.py) scores its
    arms with the SAME procedure that produced the 0.816 reference. Two
    probes with two normalisation pipelines cannot be compared, and the
    normalisation here is not incidental - each step below was added to
    fix a measured failure."""
    Xtr = np.array([[r[k] for k in F] for r in tr], np.float64)
    ytr = np.array([r["y"] for r in tr])
    Xte = np.array([[r[k] for k in F] for r in te], np.float64)
    yte = np.array([r["y"] for r in te])
    # rot_ratio / decel / spread_change are unbounded ratios; raw, they
    # overflowed the probe (inf weights, nan scores, an AUC that meant
    # nothing). log1p then winsorise to the TRAIN quantiles - fitted on
    # train only, so the held-out episodes cannot leak through the
    # normalisation either.
    Xtr, Xte = np.log1p(np.abs(Xtr)) * np.sign(Xtr), \
        np.log1p(np.abs(Xte)) * np.sign(Xte)
    lo, hi = np.quantile(Xtr, 0.02, axis=0), np.quantile(Xtr, 0.98, axis=0)
    Xtr, Xte = np.clip(Xtr, lo, hi), np.clip(Xte, lo, hi)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    # A near-constant column divided by ~0 std produces enormous inputs and
    # overflows the probe. Floor the scale and drop columns that carry no
    # variance at all, rather than letting them dominate.
    keep = sd > 1e-6
    Xtr, Xte, mu = Xtr[:, keep], Xte[:, keep], mu[keep]
    sd = np.maximum(sd[keep], 1e-3)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    assert np.isfinite(Ztr).all() and np.isfinite(Zte).all()
    # sklearn's solver, not a hand-rolled loop: the hand-rolled one
    # overflowed on unbounded ratio features and produced an AUC computed
    # from non-finite scores. A tested solver with an explicit convergence
    # criterion is the right tool; class_weight balances 121 positives
    # against 301 negatives so the probe cannot win by predicting "no".
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    clf.fit(Ztr, ytr)
    # A diverged solver still returns coefficients and still prints a
    # believable AUC. This project has already published one number that
    # came out of a non-converged fit; refuse it here instead.
    if int(np.max(clf.n_iter_)) >= 5000:
        raise RuntimeError("probe did NOT converge in 5000 iterations - "
                           "its AUC would be meaningless")
    s = clf.decision_function(Zte)
    assert np.isfinite(s).all(), "probe produced non-finite scores"
    a = auc(s[yte > 0.5], s[yte < 0.5])
    # per-row scores let a caller bootstrap the AUC by EPISODE. With 28
    # held-out episodes and 31 positives, a 0.07 AUC gap is inside one
    # standard error, so any claim about a gap needs an interval - not
    # a point estimate quoted to three decimals.
    return (a, s, yte) if return_scores else a


def collect(dataset, W=24):
    man = R.read_manifest(dataset)
    by = {e["id"]: e for e in man["episodes"]}
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    part = SP.partition(files, dataset)
    which = {}
    for nm, key in (("train", SP.TRAIN), ("val", SP.VAL), ("test", SP.TEST)):
        for f in part[key]:
            which[f.stem] = nm
    rows = []
    for f in files:
        if f.stem not in by:
            continue
        z = np.load(f)
        if "contact_pairs" not in z.files or "window_starts" not in z.files:
            continue
        ws = z["window_starts"]
        for t0 in ws[:6]:
            px = pixel_feats(z, int(t0), W)
            if px is None:
                continue
            lab = rel_feats(z, int(t0), W)        # ORACLE, label only
            rows.append(dict(id=f.stem, split=which.get(f.stem, "train"),
                             task=f.stem.split("_episode_")[0],
                             y=float(lab["articulation"] > 0.05), **px))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--win", type=int, default=24)
    a = ap.parse_args()
    rows = collect(a.dataset, a.win)
    F = ("motion_contrast", "rigidity", "rot_ratio", "decel", "coherence",
         "path_curv", "spread_change")
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] in ("test", "val")]
    print(f"{a.dataset}: {len(rows)} windows  train {len(tr)}  test {len(te)}")
    print(f"  positives (articulated): train {sum(r['y'] for r in tr):.0f}  "
          f"test {sum(r['y'] for r in te):.0f}\n")
    if not tr or not te or len(set(r["y"] for r in te)) < 2:
        raise SystemExit("not enough of both classes to score")
    print(f"{'pixel feature':16s} {'AUC(test)':>10s}")
    print("-" * 28)
    best = 0.0
    for k in F:
        v = auc([r[k] for r in te if r["y"] > 0.5],
                [r[k] for r in te if r["y"] < 0.5])
        best = max(best, abs(v - 0.5))
        print(f"{k:16s} {v:10.3f}")
    # a linear probe FIT ON TRAIN ONLY, scored on held-out episodes
    pa = fit_probe(tr, te, F)
    print(f"\nlinear probe (fit TRAIN, scored on held-out episodes): "
          f"AUC {pa:.3f}")
    print(f"best single pixel feature: AUC {0.5 + best:.3f}")
    print("\nreference: ORACLE relation separated the same families at 0.997 "
          "on trajectory-matched windows;\n           the best TRAJECTORY "
          "feature managed 0.771.")
    R.log("rel_g0", dataset=a.dataset, windows=len(rows), n_train=len(tr),
          n_test=len(te), probe_auc=round(float(pa), 4),
          best_single=round(0.5 + best, 4))
