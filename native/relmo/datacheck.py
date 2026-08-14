"""Is the data good? Measure it, before touching the model again.

Answers, with numbers, replicating EXACTLY the trainer's sampling
(load / norm_ctx / moving-point mask from train_wm.step_loss):

  1. MOTION  what fraction of sampled windows contain motion at all,
             and how many points per window pass the moving mask.
  2. SIGNAL/NOISE  the target is a displacement; xy is stored float16,
             so quantization + tracking jitter put a floor under it.
             Compare moving-point magnitudes to the static-point
             "motion" (which should be zero and is pure noise).
  3. LEARNABILITY  constant-velocity extrapolation from the last two
             context frames - the dumbest possible physics. Its R2 on
             the same masked targets is the floor the model must beat
             and a direct proof the target is (or is not) predictable.

    python -m relmo.datacheck --dataset physgen_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import splits as SP  # noqa: E402

TC, HZ = 16, 8


def norm_ctx_np(xy, tc):
    c = xy[:tc].mean(axis=(0, 1), keepdims=True)
    s = max(xy[:tc].reshape(-1, 2).std(0).mean(), 1e-3)
    return (xy - c) / s, s


def window(z, rng):
    xy = z["xy"].astype(np.float32)
    vis = z["vis"]
    T = xy.shape[0]
    if T < TC + HZ:
        return None
    t0 = int(rng.integers(0, T - TC - HZ + 1))
    return xy[t0:t0 + TC + HZ], vis[t0:t0 + TC + HZ], z


def main(dataset, n=400, seed=0):
    files = sorted((R.TRACKS / dataset).glob("ep*.npz"))
    part = SP.partition(files)
    print(f"{dataset}: {len(files)} eps  "
          f"train/val/test {len(part[SP.TRAIN])}/{len(part[SP.VAL])}/"
          f"{len(part[SP.TEST])}")
    rng = np.random.default_rng(seed)
    tr = part[SP.TRAIN]

    no_motion = 0
    mv_frac, mv_mag, static_mag, cv_r2, mv_count = [], [], [], [], []
    sse_cv = sst_all = 0.0
    for _ in range(n):
        w = window(np.load(rng.choice(tr)), rng)
        if w is None:
            continue
        xy, vis, z = w
        xyn, s = norm_ctx_np(xy, TC)
        base = xyn[TC - 1]                        # (P,2)
        tgt = xyn[TC:] - base[None]               # (H,P,2)
        tgt = tgt.transpose(1, 0, 2)              # (P,H,2)
        m = vis[TC:].T[..., None] & vis[TC - 1][:, None, None]
        mag = np.linalg.norm(tgt, axis=-1, keepdims=True)
        if not m.any():
            no_motion += 1
            continue
        thr = max(mag[m[..., 0]].mean(), 1e-4) * 0.5
        mv = m & (mag > thr)
        if not mv.any():
            no_motion += 1
            continue
        mv_frac.append(mv[..., 0].any(1).mean())  # pts with any moving t
        mv_count.append(int(mv.sum()))
        mv_mag.append(float(mag[mv[..., 0]].mean() * s))    # px
        st = m & ~mv
        if st.any():
            static_mag.append(float(mag[st[..., 0]].mean() * s))
        # constant velocity from the last two context frames
        v = (xyn[TC - 1] - xyn[TC - 2])           # (P,2)
        steps = np.arange(1, HZ + 1, dtype=np.float32)
        pred = v[:, None, :] * steps[None, :, None]
        e = np.broadcast_to(mv, tgt.shape)
        sse = float(((pred - tgt)[e] ** 2).sum())
        sst = float((tgt[e] ** 2).sum())
        sse_cv += sse
        sst_all += sst
        cv_r2.append(1.0 - sse / max(sst, 1e-9))

    cv = np.array(cv_r2)
    print(f"\nwindows sampled            {n}")
    print(f"windows with NO motion     {no_motion}  "
          f"({no_motion / n:.0%})  <- these give the trainer nothing")
    print(f"moving pts per window      med {np.median(mv_count):.0f}  "
          f"(of {384 * HZ} pt-frames)")
    print(f"frac of pts ever moving    med {np.median(mv_frac):.2f}")
    print(f"|tgt| moving  (px, orig)   med {np.median(mv_mag):.2f}")
    if static_mag:
        print(f"|tgt| static  (px, orig)   med {np.median(static_mag):.2f}"
              "   <- noise floor: fp16 quant + jitter")
        print(f"signal/noise ratio         "
              f"{np.median(mv_mag) / max(np.median(static_mag), 1e-6):.1f}x")
    print(f"\nconst-velocity R2 per-win  mean {cv.mean():+.3f}  "
          f"med {np.median(cv):+.3f}  frac>0 {(cv > 0).mean():.2f}")
    print(f"const-velocity R2 pooled   {1.0 - sse_cv / max(sst_all, 1e-9):+.3f}")
    print("\nreading: pooled CV R2 >> 0 means the target is predictable"
          "\nby dumb extrapolation and the MODEL is the problem;"
          "\nCV R2 <= 0 means the target itself is noise-dominated or"
          "\nchaotic at this frame rate and the DATA is the problem.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="physgen_v2")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    main(a.dataset, a.n)
