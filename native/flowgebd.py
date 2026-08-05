"""STEP 4, take 4: MOTION CUES, per the unsupervised GEBD literature.

Everything I tried before scored a frozen video model's LATENT
trajectory (finite differences, multi-scale change statistics, and the
model's own predictor) and all three landed at ~0.6 per-media AUC.
The published unsupervised state of the art does not use semantic
embeddings at all - it uses OPTICAL FLOW:

  FlowGEBD (WACV 2024) - non-parametric, unsupervised, two algorithms
  over optical flow; F1@0.05 = 0.713 on Kinetics-GEBD, 0.623 on TAPOS,
  +31.7 points absolute over the unsupervised baseline.
  GraphGEBD - zero-shot, graph + normalised cut; F1@0.05 = 0.732.
  UBoCo - temporal self-similarity matrix + recursive kernel matching.

Implemented here, faithfully:
  PT  Pixel Tracking - Shi-Tomasi key points, Lucas-Kanade sparse
      flow; a boundary is where the fraction of still-tracking points
      collapses (|P_current| / |P_base| < theta1). Framewise and
      patchwise.
  FN  Flow Normalisation - dense Farneback flow, frame split into a
      grid, PatchFlow = max displacement inside a patch, accumulated
      per patch and normalised; boundary where the normalised value
      spikes.

Evaluation is the field's: F1 at relative distance 0.05 (|pred - gt| /
duration), plus F1 at an absolute tolerance for long media where
relative distance is meaningless.

    python native/flowgebd.py --corpus sim
    python native/flowgebd.py --corpus oxford
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

FPS = 8.0            # analysis rate for flow
GRID = 4             # patch grid (GRID x GRID) for FN and patchwise PT


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


# ---------------------------------------------------------------- PT
def pixel_tracking(gray, grid=1, max_pts=200):
    """Ratio of still-tracked points per frame. Low ratio = boundary.
    grid=1 -> framewise; grid>1 -> patchwise (min ratio over patches)."""
    import cv2
    T = len(gray)
    H, W = gray[0].shape
    score = np.zeros(T)
    cells = [(y, x) for y in range(grid) for x in range(grid)]
    hs, ws = H // grid, W // grid
    state = {}
    for c in cells:
        y, x = c
        sub = gray[0][y * hs:(y + 1) * hs, x * ws:(x + 1) * ws]
        p = cv2.goodFeaturesToTrack(sub, maxCorners=max_pts // len(cells),
                                    qualityLevel=0.01, minDistance=5)
        state[c] = (p, len(p) if p is not None else 0)
    for t in range(1, T):
        ratios = []
        for c in cells:
            y, x = c
            p0, base = state[c]
            if p0 is None or base == 0:
                ratios.append(1.0)
                continue
            a = gray[t - 1][y * hs:(y + 1) * hs, x * ws:(x + 1) * ws]
            b = gray[t][y * hs:(y + 1) * hs, x * ws:(x + 1) * ws]
            p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
            if p1 is None:
                ratios.append(0.0)
                state[c] = (None, base)
                continue
            ok = st.ravel() == 1
            moved = np.linalg.norm((p1 - p0).reshape(-1, 2), axis=1) > 0.5
            keep = ok & moved
            ratios.append(float(keep.sum()) / max(base, 1))
            nxt = p1[keep] if keep.any() else None
            if nxt is None or len(nxt) < 4:
                sub = gray[t][y * hs:(y + 1) * hs, x * ws:(x + 1) * ws]
                nxt = cv2.goodFeaturesToTrack(
                    sub, maxCorners=max_pts // len(cells),
                    qualityLevel=0.01, minDistance=5)
                state[c] = (nxt, len(nxt) if nxt is not None else 0)
            else:
                state[c] = (nxt, base)
        score[t] = 1.0 - float(np.min(ratios))    # collapse = boundary
    return score


# ---------------------------------------------------------------- FN
def flow_normalization(gray, grid=GRID):
    """Dense Farneback flow; PatchFlow = max displacement per patch,
    accumulated per patch and normalised; score = spike across patches."""
    import cv2
    T = len(gray)
    H, W = gray[0].shape
    hs, ws = H // grid, W // grid
    PF = np.zeros((T, grid * grid))
    for t in range(1, T):
        fl = cv2.calcOpticalFlowFarneback(gray[t - 1], gray[t], None,
                                          0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(fl, axis=2)
        k = 0
        for y in range(grid):
            for x in range(grid):
                PF[t, k] = mag[y * hs:(y + 1) * hs,
                               x * ws:(x + 1) * ws].max()
                k += 1
    # per-patch normalisation over time, then aggregate
    denom = PF.sum(0, keepdims=True) + 1e-8
    PFn = PF / denom
    return PFn.max(1)


def peaks(score, min_gap):
    """Local maxima, ranked; returned as indices."""
    idx = []
    for i in range(1, len(score) - 1):
        if score[i] >= score[i - 1] and score[i] > score[i + 1]:
            idx.append(i)
    idx.sort(key=lambda i: -score[i])
    out = []
    for i in idx:
        if all(abs(i - j) >= min_gap for j in out):
            out.append(i)
    return out


def f1_at(pred_t, gt_t, tol):
    if not gt_t:
        return float("nan"), 0, 0
    used = set()
    tp = 0
    for p in pred_t:
        best, bj = tol, None
        for j, g in enumerate(gt_t):
            if j in used:
                continue
            d = abs(p - g)
            if d <= best:
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            tp += 1
    prec = tp / max(len(pred_t), 1)
    rec = tp / len(gt_t)
    return (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
            len(pred_t), len(gt_t))


def media_jobs(corpus, limit):
    import pyarrow.parquet as pq
    jobs = []
    if corpus == "sim":
        t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
            .to_pydict()
        bnd = {}
        for e, a, b in zip(t["episode"], t["t0"], t["t1"]):
            bnd.setdefault(int(e), set()).update(
                [round(float(a), 2), round(float(b), 2)])
        for d in sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                        if p.is_dir() and p.name.startswith("ep"))[:limit]:
            jobs.append((d.name, sorted(d.glob("cam*.mp4"))[0],
                         sorted(bnd.get(int(d.name[2:]), []))))
    else:
        import oxford as OX
        bs, rel, sp, yr = OX.truth_boundaries()
        jobs.append(("drive", OX.MP4, bs))
    return jobs


def main():
    import cv2
    import encode as E
    corpus = arg("--corpus", "sim")
    limit = arg("--limit", 12, int)
    jobs = media_jobs(corpus, limit)
    print(f"{corpus}: {len(jobs)} media, flow at {FPS} fps, "
          f"grid {GRID}x{GRID}", flush=True)

    res = {}
    for name, path, gt in jobs:
        dur = E.probe_duration(path)
        F = E.decode(path, fps=FPS, w=256)
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in F]
        times = np.arange(len(gray)) / FPS
        sigs = {
            "PT_frame": pixel_tracking(gray, grid=1),
            "PT_patch": pixel_tracking(gray, grid=GRID),
            "FN": flow_normalization(gray, grid=GRID),
        }
        tol_rel = 0.05 * dur
        for k, s in sigs.items():
            # predict as many boundaries as truth has (count-matched,
            # so F1 measures PLACEMENT not calibration)
            pk = peaks(s, min_gap=int(0.5 * FPS))[:max(len(gt), 1)]
            pt = [float(times[i]) for i in pk]
            f_rel, npd, ngt = f1_at(pt, gt, tol_rel)
            f_abs, _, _ = f1_at(pt, gt, 1.0)
            res.setdefault(k, []).append((f_rel, f_abs))
        print(f"  {name:<8} dur {dur:5.1f}s gt {len(gt):2d}  " +
              "  ".join(f"{k} {res[k][-1][0]:.2f}" for k in sigs),
              flush=True)
    print()
    for k, v in res.items():
        fr = np.nanmean([a for a, _ in v])
        fa = np.nanmean([b for _, b in v])
        print(f"STEP 4 flow/{k:<9} F1@0.05rel {fr:.3f}   F1@1s {fa:.3f}"
              f"   (n={len(v)})")


if __name__ == "__main__":
    main()
