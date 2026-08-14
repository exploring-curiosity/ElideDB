"""What does 'near perfect' actually mean for motion-R2 on this data?

R2=1 is perfect and R2=0 is predict-stillness, both by construction.
Neither tells you what a good model should be aiming at, because the
future is not a deterministic function of 16 frames of point tracks -
a controller decides part of it. So this measures a LADDER of oracles
on the real targets, from trivial to omniscient, using exactly the
trainer's protocol (TC/HZ, canonicalisation, moving-point mask, POOLED
sse/sst - never averaged ratios).

    stillness         predict no motion                       -> 0 exactly
    const-vel         last-frame velocity, extrapolated       (G0 baseline)
    linear-fit-vel    velocity from a least-squares fit over
                      the last 4 frames - same model, quieter estimator
    const-accel       second difference too
    ORACLE parts+screw   TRUE body ids, each body's SE(3) step fitted on
                      the context and repeated. "Perfect part discovery
                      plus constant screw velocity, no future knowledge."
                      THIS is the number the architecture should chase.
    ORACLE rigid+future  TRUE body ids AND the true future: per-body
                      Kabsch onto the actual future frame. The ceiling
                      of a per-part rigid decoder - anything below 1.0
                      here is motion the architecture CANNOT express.

The gap between the last two is the part of the future that is
genuinely unpredictable from the context. That gap, not 1.0, is what
"near perfection" means here.
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
from relmo.train_wm2 import HZ, TC, lift3d  # noqa: E402


def kabsch_np(A, B):
    """Rigid transform mapping A -> B (both (N,3)). Closed form."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Rm, cb - Rm @ ca


def apply_rt(Rm, t, P):
    return P @ Rm.T + t


def run(dataset="rcasa_v1", split="val", n=200, seed=0, min_pts=3):
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    part = SP.partition(files, dataset)
    files = part[SP.VAL if split == "val" else SP.TRAIN]
    rng = np.random.default_rng(seed)
    names = ["stillness", "const_vel", "linear_fit_vel", "const_accel",
             "oracle_parts_screw", "oracle_rigid_future"]
    sse = {k: np.zeros(HZ) for k in names}
    sst = np.zeros(HZ)
    used = 0
    for f in files[:n]:
        z = np.load(f)
        if "gdist" not in z.files or "gbody" not in z.files:
            continue
        X = lift3d(z)
        V = z["gvis"]
        gb = z["gbody"]
        T = X.shape[0]
        if T < TC + HZ:
            continue
        t0 = int(rng.integers(0, T - TC - HZ + 1))
        X, V = X[t0:t0 + TC + HZ], V[t0:t0 + TC + HZ]
        s = max(np.linalg.norm(
            X[:TC] - X[:TC].mean((0, 1)), axis=-1).mean(), 1e-6)
        Xn = (X - X[:TC].mean((0, 1))) / s
        base = Xn[TC - 1]
        tgt = Xn[TC:]
        m = V[TC:] & V[TC - 1][None]
        disp = tgt - base[None]
        mag = np.linalg.norm(disp, axis=-1)
        if not m.any():
            continue
        thr = max(mag[m].mean(), 1e-6) * 0.5
        mv = m & (mag > thr)
        if not mv.any():
            continue
        used += 1

        v1 = Xn[TC - 1] - Xn[TC - 2]
        # least-squares slope over the last 4 context frames
        w = np.arange(4) - 1.5
        vlf = (w[:, None, None] * Xn[TC - 4:TC]).sum(0) / (w ** 2).sum()
        acc = Xn[TC - 1] - 2 * Xn[TC - 2] + Xn[TC - 3]

        # ---- oracle predictions on TRUE bodies
        bodies = [b for b in np.unique(gb) if b >= 0]
        # Default is STILLNESS (the point stays at base), not zero. Zero
        # means "this point moves to the scene origin", which for any
        # point the oracle cannot fit is a colossal fabricated error -
        # measured: it drove the oracle to R2 -144 and briefly made the
        # omniscient predictor look worse than predicting nothing.
        screw = np.repeat(base[None], HZ, 0).copy()
        fut = np.repeat(base[None], HZ, 0).copy()
        for b in bodies:
            sel = (gb == b) & V[TC - 1] & V[TC - 2]
            if sel.sum() < min_pts:
                continue
            Rm, t = kabsch_np(Xn[TC - 2][sel], Xn[TC - 1][sel])
            cur = base[sel].copy()
            for h in range(HZ):
                cur = apply_rt(Rm, t, cur)          # repeat the step
                screw[h][sel] = cur
            for h in range(HZ):
                s2 = sel & V[TC + h]
                if s2.sum() < min_pts:
                    fut[h][sel] = base[sel]
                    continue
                R2m, t2 = kabsch_np(base[s2], Xn[TC + h][s2])
                fut[h][sel] = apply_rt(R2m, t2, base[sel])

        for h in range(HZ):
            sel = mv[h]
            if not sel.any():
                continue
            d = disp[h][sel]
            sst[h] += (d ** 2).sum()
            k = h + 1
            preds = dict(
                stillness=np.zeros_like(d),
                const_vel=(v1 * k)[sel],
                linear_fit_vel=(vlf * k)[sel],
                const_accel=(v1 * k + 0.5 * acc * k * k)[sel],
                oracle_parts_screw=(screw[h] - base)[sel],
                oracle_rigid_future=(fut[h] - base)[sel])
            for nm, p in preds.items():
                sse[nm][h] += ((p - d) ** 2).sum()

    r2 = {k: [round(float(1 - sse[k][h] / max(sst[h], 1e-12)), 4)
              for h in range(HZ)] for k in names}
    # The trainer reports ONE val_r2 pooled over every horizon, so give
    # the ladder in the same units - comparing a pooled model number
    # against a per-horizon oracle would flatter or damn it arbitrarily.
    pooled = {k: round(float(1 - sse[k].sum() / max(sst.sum(), 1e-12)), 4)
              for k in names}
    rep = dict(dataset=dataset, split=split, episodes_used=used,
               horizons=list(range(1, HZ + 1)), r2=r2, pooled=pooled)
    R.log("ceiling", **rep)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa_v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    rep = run(a.dataset, a.split, a.n)
    print(f"{a.dataset}/{a.split}  episodes={rep['episodes_used']}\n")
    print(f"{'predictor':22s}" + "".join(f"  h={h}" for h in rep["horizons"]))
    for k, v in rep["r2"].items():
        print(f"{k:22s}" + "".join(f"{x:+6.3f}" for x in v))
