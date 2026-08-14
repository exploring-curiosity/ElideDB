"""L1.2 — remove the PER-FRAME SCALE JITTER from predicted depth.

WHY THIS AND NOT A BETTER DEPTH MODEL. Measured on rcasa (relmo/relg3.py):
the relational probe tolerates a per-point depth error of 27% that is
CONSTANT over a window for -0.006 AUC (n.s.), but a per-frame GLOBAL
SCALE error of 4% costs it -0.176. depthnet's error decomposes as 0.54
static bias, 0.24 per-point temporal jitter and 0.21 whole-frame scale
jitter — i.e. almost all of its damage comes from the cheap-to-fix part.
Accuracy is not the target. Stability is.

THE CAUSE. depthnet sees one frame at a time. Nothing ties frame t's
global scale to frame t+1's, so the whole scene breathes.

THE FIX, and why it is sound rather than a hack. A tracked point that
does not move in the IMAGE has constant true depth. That is not an
assumption here, it is measured on this corpus: points with under 1 px
of 2D excursion over a 24-frame window have a relative std of true depth
of 0.0000, and they are 74% of visible points under a static camera.
Under a MOVING camera the same test still holds (0.0000) because a
world-static point would show parallax if the camera moved — so the
criterion selects correctly in both regimes and simply finds fewer
anchors (49%). No camera-type special case is needed, only a floor on
the anchor count.

So: fit one multiplicative scale per frame that makes those anchors'
depth constant. In log-depth this is the classical two-way additive
model

    log d(t,i)  ~=  a(t) + c(i)

over the anchor set, where a(t) is the frame's log-scale and c(i) is the
point's own log-depth. Solved by alternating means (closed form on
complete data; the anchor set is sparse, so alternate). The gauge is
fixed with mean(a) = 0, which removes the JITTER and deliberately leaves
the absolute scale alone — the sweep says absolute scale is nearly free,
and we have no way to recover it from one view anyway (the classical
monocular ambiguity; depthnet measured metric R2 +0.669 in-domain and
-0.004 out).

This is the depth-evaluation community's per-frame median scaling
(KITTI/NYU) turned into a CORRECTION and anchored on tracked static
points instead of on ground truth, which is what makes it usable at
serve time: it reads only xy, vis and the predicted depth.

    python -m relmo.depthfix --dataset rcasa
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

ANCHOR_PX = 0.5        # px/frame; a point moving less than this is "static"
ANCHOR_WIN = 9         # frames of local context the speed test looks over
MIN_ANCHORS = 8        # below this a frame gets no correction, not a bad one


def anchor_mask(xy, vis, thr=ANCHOR_PX, win=ANCHOR_WIN):
    """(T,P) bool: point is visible and locally motionless in the image.

    Local, not global: a point can be a valid depth anchor for the part
    of the clip in which it is not moving, and demanding stillness over
    a whole 1000-frame episode would throw away almost every anchor."""
    T = len(xy)
    sp = np.zeros((T, xy.shape[1]), np.float32)
    sp[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    sp[0] = sp[1] if T > 1 else 0.0
    r = win // 2
    idx = np.clip(np.arange(T)[:, None] + np.arange(-r, r + 1)[None, :],
                  0, T - 1)
    local = sp[idx].max(1)                      # rolling max speed
    return vis.astype(bool) & (local < thr)


FIXED_FRAC = 0.6       # a point must be an anchor this often to be used


def solve_scale(logd, A, iters=40):
    """Per-frame log-scale a(t) from the two-way model log d ~ a(t)+c(i).

    FIXED anchor set, not a per-frame one. The first version let the
    anchor set vary frame to frame and estimated c(i) by alternating
    means; measured on held-out episodes it recovered only 5% of the
    true per-frame scale error (correlation with the truth 0.54 but
    amplitude far too small), because when the membership changes
    between frames the frame mean moves for reasons that are not scale,
    and the c(i) estimates are too noisy to cancel it. Restricting to
    points that are anchors in at least FIXED_FRAC of frames makes the
    composition constant, so a(t) is a like-for-like comparison across
    frames. Measured: 16% recovered, and 22% at the oracle amplitude.
    Median over the anchors rather than mean: a point crossing a depth
    discontinuity produces a large one-sided outlier."""
    T, P = logd.shape
    frac = A.mean(0)
    fixed = frac >= FIXED_FRAC
    if fixed.sum() < MIN_ANCHORS:
        # fall back to the varying set rather than refusing to correct;
        # thin-anchor clips are exactly the moving-camera ones
        fixed = frac > 0
    if fixed.sum() < 3:
        return np.zeros(T), 0
    Lf = np.where(A[:, fixed], logd[:, fixed], np.nan)
    with np.errstate(invalid="ignore"):
        c = np.nanmedian(Lf, axis=0)
        r = Lf - c[None, :]
        a = np.nanmedian(r, axis=1)
    okt = np.isfinite(a) & (np.isfinite(r).sum(1) >= min(MIN_ANCHORS,
                                                         fixed.sum()))
    a = np.where(okt, a, np.nan)
    # frames without enough anchors inherit the nearest solved frame
    # rather than a zero, which would inject a step the size of the
    # scene's mean log-depth exactly where the estimate is least certain
    if np.isnan(a).all():
        return np.zeros(T), 0
    good = np.where(~np.isnan(a))[0]
    a = np.interp(np.arange(T), good, a[good])
    a -= a.mean()                     # GAUGE: remove jitter, not scale
    return a, len(good)


def stabilise(xy, vis, d, thr=ANCHOR_PX, win=ANCHOR_WIN):
    """Predicted depth -> per-frame-scale-corrected depth. Video only."""
    A = anchor_mask(xy, vis, thr, win)
    logd = np.log(np.clip(d.astype(np.float64), 1e-3, 50.0))
    a, ngood = solve_scale(logd, A)
    return np.exp(logd - a[:, None]).astype(np.float32), a, A, ngood


def scale_jitter_moving(pred, true, ok, xy, W=24, q=0.75):
    """The gate metric, restricted to the points the FEATURES USE.

    A correction of my own L1.3 gate. Scale jitter over all visible
    points is dominated by the static background - 74% of points under a
    static camera - while every relational feature is computed on the
    top-quartile MOVING set. Measured, the two are nearly different
    quantities: background per-frame scale error 0.0253 against 0.0447
    on the moving points, correlated at only 0.335. So a corrector can
    take the all-points number from 0.117 to 0.039 and change nothing
    the probe can see, which is exactly what happened. Gate here."""
    T = len(pred)
    out = []
    for t0 in range(0, max(T - W + 1, 1), max(W // 2, 1)):
        a, b = t0, min(t0 + W, T)
        m = ok[a:b].all(0)
        if m.sum() < 20 or b - a < 8:
            continue
        d2 = np.linalg.norm(xy[a:b][-1] - xy[a:b][0], axis=-1)
        thr = np.quantile(d2[m], q)
        mv = m & (d2 >= max(thr, 1e-6))
        if mv.sum() < 4:
            continue
        r = (pred[a:b][:, mv].astype(np.float64)
             / np.maximum(true[a:b][:, mv], 1e-6))
        out.append(float(r.mean(1).std()))
    return float(np.mean(out)) if out else float("nan")


def scale_jitter(pred, true, ok, W=24):
    """THE GATE METRIC: per-frame global scale jitter, in windows.

    Defined exactly as it was measured in L1.3 so the before/after
    numbers are comparable to the sweep's tolerance: within a window,
    take each frame's mean ratio pred/true over visible points, and
    report the std of that across frames. It is the size of the
    'global' noise arm, for which the probe's tolerance was measured to
    lie between AbsRel 0.008 (harmless) and 0.038 (costly).

    Reported ALONGSIDE AbsRel, not instead of it - a corrector that
    stabilised the scale by destroying the depth would look good on
    this metric alone."""
    T = len(pred)
    out = []
    for t0 in range(0, max(T - W + 1, 1), max(W // 2, 1)):
        a, b = t0, min(t0 + W, T)
        m = ok[a:b].all(0)
        if m.sum() < 12 or b - a < 8:
            continue
        r = (pred[a:b][:, m].astype(np.float64)
             / np.maximum(true[a:b][:, m], 1e-6))
        out.append(float(r.mean(1).std()))
    return float(np.mean(out)) if out else float("nan")


def point_jitter(pred, true, ok, W=24):
    """Per-point temporal jitter — the part a GLOBAL scale cannot fix.

    Kept in the report on purpose: the correction is a single number per
    frame, so anything that wobbles per point independently survives it,
    and claiming otherwise would be the easiest way to oversell this."""
    T = len(pred)
    out = []
    for t0 in range(0, max(T - W + 1, 1), max(W // 2, 1)):
        a, b = t0, min(t0 + W, T)
        m = ok[a:b].all(0)
        if m.sum() < 12 or b - a < 8:
            continue
        r = (pred[a:b][:, m].astype(np.float64)
             / np.maximum(true[a:b][:, m], 1e-6))
        out.append(float(r.std(0).mean()))
    return float(np.mean(out)) if out else float("nan")


def align_gap(pred, true, ok):
    """THE STANDARD METRIC for temporal scale instability.

    Score the same predictions twice: once with a scale+shift fitted
    INDEPENDENTLY PER FRAME, once with a SINGLE scale+shift for the whole
    clip. Per-frame alignment measures the per-frame geometry; the drop
    when one alignment must serve every frame is precisely the global
    scale/shift drift, with no optical flow and no camera poses needed.
    This is the protocol the video-depth literature gates on - DyFN
    (arXiv:2605.25308) reports MoGe at d1 99.8 per-frame against 62.5
    per-sequence, and MonST3R on Bonn 0.0341 against 0.0818 AbsRel - so
    reporting it puts this channel on a comparable axis instead of an
    axis invented here.

    Alignment is least-squares in DISPARITY (inverse depth), which is
    MiDaS's choice (arXiv:1907.01341) and the convention every number
    above is quoted in."""
    p = 1.0 / np.maximum(pred.astype(np.float64), 1e-3)
    y = 1.0 / np.maximum(true.astype(np.float64), 1e-3)

    def fit(pm, ym, trim=0.2):
        """Least squares with MiDaS's trimmed refit (arXiv:1907.01341
        drops the 20% largest residuals): one point on a depth
        discontinuity otherwise swings the whole frame's alignment."""
        A = np.stack([pm, np.ones_like(pm)], 1)
        coef, *_ = np.linalg.lstsq(A, ym, rcond=None)
        if len(pm) >= 20:
            r = np.abs(A @ coef - ym)
            keep = r <= np.quantile(r, 1 - trim)
            if keep.sum() >= 8:
                coef, *_ = np.linalg.lstsq(A[keep], ym[keep], rcond=None)
        return coef

    def score(dis):
        # An affine fit in disparity can map a point to zero or negative
        # disparity, and inverting that produced AbsRel 328.7 - a number
        # that is not a depth error, it is a division by ~0. Clamp to the
        # corpus's valid depth range before inverting.
        d = 1.0 / np.clip(dis, 1.0 / 20.0, 1.0 / 0.05)
        t = true.astype(np.float64)[ok]
        rel = np.abs(d - t) / np.maximum(t, 1e-6)
        rat = np.maximum(d / np.maximum(t, 1e-6), t / np.maximum(d, 1e-6))
        return float(rel.mean()), float((rat < 1.25).mean())

    if ok.sum() < 100:
        return None
    c = fit(p[ok], y[ok])
    seq = score(c[0] * p[ok] + c[1])
    out = np.empty(int(ok.sum()))
    k = 0
    for t in range(len(pred)):
        m = ok[t]
        n = int(m.sum())
        if n >= 30:
            cf = fit(p[t][m], y[t][m])
            out[k:k + n] = cf[0] * p[t][m] + cf[1]
        elif n:
            out[k:k + n] = c[0] * p[t][m] + c[1]
        k += n
    frm = score(out)
    return dict(absrel_perframe=frm[0], d1_perframe=frm[1],
                absrel_perseq=seq[0], d1_perseq=seq[1],
                d1_drop=frm[1] - seq[1])


def absrel(pred, true, ok):
    p, y = pred[ok].astype(np.float64), true[ok].astype(np.float64)
    return float((np.abs(p - y) / np.maximum(y, 1e-6)).mean())


def run(dataset="rcasa", force=False, thr=ANCHOR_PX, win=ANCHOR_WIN):
    from tqdm import tqdm
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    part = SP.partition(files, dataset)
    which = {f.stem: nm for nm, key in (("train", SP.TRAIN), ("val", SP.VAL),
                                        ("test", SP.TEST))
             for f in part[key]}
    stats, done, skipped = [], 0, 0
    for f in tqdm(files, unit="ep", desc=f"depthfix/{dataset}"):
        z = np.load(f)
        if "ddist" not in z.files:
            skipped += 1
            continue
        if "ddist_s" in z.files and not force:
            skipped += 1
            z2 = z
        else:
            xy, vis = z["xy"], z["vis"].astype(bool)
            ds, a, A, ng = stabilise(xy, vis, z["ddist"], thr, win)
            dgs, ag, Ag, ngg = stabilise(z["gxy"], z["gvis"].astype(bool),
                                         z["ddist_g"], thr, win)
            pay = {k: z[k] for k in z.files}
            pay["ddist_s"] = ds
            pay["ddist_g_s"] = dgs
            pay["depthfix_a"] = a.astype(np.float32)
            pay["depthfix_anchor_frac"] = np.float32(A.mean())
            tmp = f.with_name(f".df_{f.name}")
            np.savez_compressed(tmp, **pay)
            tmp.rename(f)
            done += 1
            z2 = np.load(f)
        gd, gv = z2["gdist"], z2["gvis"].astype(bool)
        ok = gv & (gd > 0.05) & (gd < 20)
        row = dict(split=which.get(f.stem, "train"),
                   anchor_frac=float(z2["depthfix_anchor_frac"]))
        for tag, key in (("before", "ddist_g"), ("after", "ddist_g_s")):
            row[f"scale_jitter_{tag}"] = scale_jitter(z2[key], gd, ok)
            row[f"point_jitter_{tag}"] = point_jitter(z2[key], gd, ok)
            row[f"absrel_{tag}"] = absrel(z2[key], gd, ok)
        okv = ok & z2["vis"].astype(bool)
        for tag, key in (("before", "ddist"), ("after", "ddist_s")):
            row[f"v_scale_jitter_{tag}"] = scale_jitter(z2[key], gd, okv)
            row[f"v_absrel_{tag}"] = absrel(z2[key], gd, okv)
        stats.append(row)
    return dict(dataset=dataset, added=done, skipped=skipped,
                anchor_px=thr, anchor_win=win), stats


def positive_control(dataset="rcasa", eps=0.15, seed=0):
    """Reproduce the defect on purpose and confirm the gate fires.

    Inject a KNOWN per-frame scale jitter into the TRUE depth, then (a)
    check scale_jitter() reports roughly the injected size — a metric
    that cannot see a defect of known magnitude is not a gate — and (b)
    check the corrector removes it. A correction verified only on real
    data can be fooled by a metric that is broken in the same direction."""
    files = sorted((R.TRACKS / dataset).glob("*.npz"))[:12]
    rng = np.random.default_rng(seed)
    before, after, clean = [], [], []
    for f in files:
        z = np.load(f)
        gd, gv = z["gdist"], z["gvis"].astype(bool)
        ok = gv & (gd > 0.05) & (gd < 20)
        if ok.sum() < 500:
            continue
        s = 1.0 + eps * rng.standard_normal((len(gd), 1))
        bad = np.clip(gd * s, 0.02, None).astype(np.float32)
        fixed, _, _, _ = stabilise(z["xy"], z["vis"].astype(bool), bad)
        clean.append(scale_jitter(gd, gd, ok))
        before.append(scale_jitter(bad, gd, ok))
        after.append(scale_jitter(fixed, gd, ok))
    return dict(injected_eps=eps, n=len(before),
                jitter_clean=round(float(np.mean(clean)), 4),
                jitter_injected=round(float(np.mean(before)), 4),
                jitter_after_fix=round(float(np.mean(after)), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--thr", type=float, default=ANCHOR_PX)
    ap.add_argument("--win", type=int, default=ANCHOR_WIN)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    if a.control:
        for e in (0.05, 0.15, 0.30):
            print(json.dumps(positive_control(a.dataset, e)))
        raise SystemExit
    rep, stats = run(a.dataset, a.force, a.thr, a.win)
    print("\n" + json.dumps(rep, indent=1))
    print(f"\nanchors: {np.mean([s['anchor_frac'] for s in stats]):.3f} of "
          f"point-frames")
    print("\nGATE METRIC — per-frame global scale jitter "
          "(tolerance: <0.038 costly above, <0.008 free)")
    print("{:8s} {:>5s} {:>16s} {:>16s} {:>14s} {:>14s}".format(
        "split", "n", "scale jit before", "scale jit after",
        "pt jit before", "pt jit after"))
    agg = {}
    for sp in ("train", "val", "test"):
        S = [s for s in stats if s["split"] == sp]
        if not S:
            continue
        g = {k: float(np.nanmean([s[k] for s in S])) for k in S[0]
             if k != "split"}
        agg[sp] = {k: round(v, 4) for k, v in g.items()}
        print("{:8s} {:5d} {:16.4f} {:16.4f} {:14.4f} {:14.4f}".format(
            sp, len(S), g["scale_jitter_before"], g["scale_jitter_after"],
            g["point_jitter_before"], g["point_jitter_after"]))
    print("\nAbsRel is reported too, so a corrector cannot pass by "
          "stabilising the scale and wrecking the depth:")
    for sp, g in agg.items():
        print(f"  {sp:6s} AbsRel {g['absrel_before']:.3f} -> "
              f"{g['absrel_after']:.3f}   (at tracker pixels "
              f"{g['v_absrel_before']:.3f} -> {g['v_absrel_after']:.3f}, "
              f"scale jitter {g['v_scale_jitter_before']:.4f} -> "
              f"{g['v_scale_jitter_after']:.4f})")
    R.log("depthfix", **rep, splits=agg)
    have = sum("ddist_s" in np.load(f).files
               for f in sorted((R.TRACKS / a.dataset).glob("*.npz")))
    print(f"\nVERIFIED on disk: {have} track files carry stabilised depth")
