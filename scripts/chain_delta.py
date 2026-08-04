"""PERSISTENT-CHANGE chains: the detection fix, route 3.

Every failed route shared one primitive - seeing the block WHILE IT
MOVES (14-40% reliable everywhere measured), through a proposal soup
that under-covers resting blocks. This route needs neither. A
manipulation leaves a PERSISTENT mark on the workspace: pixels at the
pickup spot change to what lies beneath and STAY changed; pixels at
the set-down spot change to the block and stay changed. The arm's
sweeps are transient and vanish under a temporal median. So:

    change sites   |median(after) - median(before)| over a time grid,
                   union across the episode, connected components.
                   A site is any place whose stable appearance ever
                   changed - exactly the pickup/set-down footprints.
    events         per site, a 1-D changepoint trace (distance of the
                   site's appearance to its own episode median) gives
                   the times the site's state flipped.
    crops          before/after crops at each event, embedded by the
                   store's KIND ENCODER. All identity lives there -
                   the encoder distinguishes a green block from a red
                   block (the vision-native rule's point). Junk is
                   whatever matches the AGENT gallery (crops sampled
                   at transient-motion peaks = the sweeping arm) or
                   BACKGROUND gallery (crops far from every site) at
                   fitted bars.
    manipulations  departure of cluster c (c on the BEFORE side) pairs
                   with the next arrival of c (c on the AFTER side):
                   object permanence as event pairing. A push shows as
                   one event whose before AND after are the same block
                   cluster - a self-pair. The carry is never observed
                   and never needs to be.

Everything fitted from the corpus (diff cut, step cut, gallery bars:
Otsu after distribution inspection; quantile edges: terciles). No
truth, no text, no hand appearance features, no camera meta.

    python scripts/chain_delta.py [--store lake/sim_chains]
                                  [--limit N] [--probe]
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import pyarrow.compute as pc                                   # noqa: E402

from elidedb import Store                                      # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402
import chain_qbe                                               # noqa: E402
from chain_moves import bench, otsu                            # noqa: E402
from chain_ledger import embed_crops                           # noqa: E402

DS = 2            # spatial downsample for detection (crops are full-res)
WIN_S = 3.0       # stable-state median window: the arm holds a pose
                  # ~1-2s during grasp; a 3s median needs >1.5s of
                  # presence to admit it (measured: 1.2s windows let
                  # arm poses through and the sweep envelope became one
                  # 45%-of-frame site)
GUARD_S = 0.35    # guard band around a step (the arm lingers at contact)
GRID_S = 0.5      # detection grid
MIN_WIN_FRAMES = 15   # clamped edge windows: 0.8s let the arm's end-
                      # of-episode park into the medians (measured: a
                      # junk factory at t>19s); 1.5s matches the WIN_S
                      # half-presence bar
MIN_EVENT_SEP_S = 1.2


def k_means_1d(vals, k):
    """1-D k-means centroids: robust to mass imbalance where an otsu
    valley drifts into the heavy mode's interior (measured twice)."""
    v = np.asarray(vals, np.float32)
    c = np.quantile(v, np.linspace(0.05, 0.95, k)).astype(np.float64)
    for _ in range(64):
        d = np.abs(v[:, None] - c[None, :])
        a = d.argmin(1)
        n = np.array([v[a == i].mean() if (a == i).any() else c[i]
                      for i in range(k)])
        if np.abs(n - c).sum() < 1e-7:
            break
        c = n
    return np.sort(c)


def two_means(vals):
    c0, c1 = k_means_1d(vals, 2)
    return (c0 + c1) / 2, c0, c1


def k_means_rgb(px, k=3, iters=32, seed=0):
    """Tiny k-means over RGB pixels - the episode's dominant SURFACES
    (table, floor, void). Background-subtraction machinery: decides
    only whether a region is a bare surface, never which object it is."""
    rng = np.random.default_rng(seed)
    c = px[rng.choice(len(px), k, replace=False)].astype(np.float64)
    for _ in range(iters):
        d = ((px[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        n = np.array([px[a == i].mean(0) if (a == i).any() else c[i]
                      for i in range(k)])
        if np.abs(n - c).sum() < 1e-6:
            break
        c = n
    return c.astype(np.float32)


def surf_angle(v, cents):
    """Angle (x1000) of a state's colour to the nearest dominant
    surface - same metric as the change detector, so a shadow (scaled
    surface) reads as surface."""
    best = 2000.0
    for c in cents:
        num = float((v * c).sum())
        den = float(np.sqrt((v * v).sum() * (c * c).sum())) + 1e-6
        best = min(best, (1.0 - num / den) * 1000.0)
    return best


def surf_ratio_std(v, cents):
    """Channel-ratio NON-UNIFORMITY vs the nearest surface: a shadow
    scales every channel equally (std ~0); a table-HUED object (orange
    on tan - measured at angle 42, barely 2x the surface cut, junking
    half the corpus's arrivals) shifts channel ratios wildly. Still
    photometric detection machinery: occupied-or-not, never which
    object."""
    best_a, best_c = 2000.0, cents[0]
    for c in cents:
        num = float((v * c).sum())
        den = float(np.sqrt((v * v).sum() * (c * c).sum())) + 1e-6
        a = (1.0 - num / den) * 1000.0
        if a < best_a:
            best_a, best_c = a, c
    r = np.log((np.asarray(v, np.float64) + 1.0)
               / (np.asarray(best_c, np.float64) + 1.0))
    return float(r.std())


def obj_arr(lst):
    a = np.empty(len(lst), object)
    for i, x in enumerate(lst):
        a[i] = x
    return a


def episode_views(db):
    """[(episode, stream, frame-rows)] - one entry per recorded view,
    grouped by source segment (one segment = one episode-view by
    construction of the write)."""
    ft = db.table("frames").scan()
    ft = ft.take(pc.sort_indices(ft, sort_keys=[("source", "ascending"),
                                                ("ts", "ascending")]))
    src = ft.column("source").to_pylist()
    out, start = [], 0
    for i in range(1, len(src) + 1):
        if i == len(src) or src[i] != src[start]:
            sl = ft.slice(start, i - start)
            out.append((int(sl.column("episode_index")[0].as_py()),
                        str(sl.column("stream")[0].as_py()), sl))
            start = i
    return out


def decode_view(db, sl):
    dec = FrameSet(db, "frames", sl).decode()
    dec.sort()
    ts = np.array([t for t, _ in dec], np.int64)
    F = np.stack([im for _, im in dec])
    return ts, F


def stable_median(Fs, i0, i1):
    """Median over a strided slice - robust to the arm sweeping through."""
    sub = Fs[i0:i1:2] if i1 - i0 > 6 else Fs[i0:i1]
    return np.median(sub, axis=0)


def color_change(a, b):
    """ILLUMINATION-INVARIANT change: angle between the RGB vectors,
    scaled x1000. A shadow scales a surface's RGB vector (direction
    kept, angle ~0); a different surface changes the direction. The
    arm's table-wide moving shadow dominated every |diff| threshold
    and swallowed the small blocks into giant components (measured:
    222x232 shadow-field crops). Photometric change GEOMETRY only -
    the encoder alone judges what each change is."""
    num = (a * b).sum(-1)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + 1e-6
    d = (1.0 - num / den) * 1000.0
    d[(a.sum(-1) + b.sum(-1)) < 30.0] = 0.0     # black void guard
    return d


def grid_params(ts, F):
    T = len(F)
    fps = 1e9 * (T - 1) / max(ts[-1] - ts[0], 1)
    w = max(int(WIN_S * fps), 3)
    g = max(int(GUARD_S * fps), 1)
    # ga once carried an asymmetric 1.2s after-guard (hypothesis: the
    # arm's retreat over a fresh set-down contaminates the after
    # state). MEASURED NEGATIVE: arrivals stayed 439 vs 440 and DEV
    # dropped 0.30 -> 0.24, so the guard is symmetric again and the
    # arrival deficit lives elsewhere (association/persistence at
    # set-down moments - the open front).
    ga = g
    step = max(int(GRID_S * fps), 1)
    return T, fps, w, g, ga, step


def grid_components(ts, F, strong):
    """Per-GRID-TIME change components. A global union mask merged the
    arm's park envelope with every block spot it visits (measured:
    one 45%-of-frame site). At a single grid time only what changed
    AROUND THEN lights up - block spots and arm poses are small,
    separate blobs. Windows clamp at episode edges (the first pick and
    last set-down live there). Returns [(c, cx, cy, area, mag, bbox)]
    in downsampled coords + the params."""
    from scipy import ndimage
    T, fps, w, g, ga, step = grid_params(ts, F)
    Fs = F[:, ::DS, ::DS, :].astype(np.float32)
    comps = []
    lo = MIN_WIN_FRAMES + g
    for c in range(lo, T - MIN_WIN_FRAMES - ga, step):
        b = stable_median(Fs, max(c - w - g, 0), c - g)
        a = stable_median(Fs, c + ga, min(c + ga + w, T))
        dC = color_change(a, b)
        # 2x2 opening: 3x3 eroded small stack-arrival components (the
        # new block's visible sliver) below the area floor
        m = ndimage.binary_opening(dC > strong, np.ones((2, 2), bool))
        lab, n = ndimage.label(m)
        for k in range(1, n + 1):
            ys, xs = np.where(lab == k)
            if len(ys) < 10:
                continue
            # CORE = the strongest-change quartile: the object's own
            # pixels. The full component drags the displaced shadow in
            # and state colours read as surface (measured: arrivals
            # 429 vs departures 635 corpus-wide - half the arrivals
            # junked as shadow)
            v = dC[ys, xs]
            thr = max(2.0 * strong, float(np.percentile(v, 75)))
            k_ = v >= thr
            core = (ys[k_], xs[k_]) if int(k_.sum()) >= 4 else (ys, xs)
            top = (int(ys[int(v.argmax())]), int(xs[int(v.argmax())]))
            comps.append((c, float(xs.mean()), float(ys.mean()),
                          len(ys), float(v.mean()),
                          (int(ys.min()), int(xs.min()),
                           int(ys.max()), int(xs.max())), (ys, xs),
                          core, top))
    return comps, (Fs, fps, w, g, ga, step)


def associate(comps, step, max_r):
    """Chain per-grid components into EVENTS: same place, adjacent
    grid times = the same change seen through sliding windows. A quiet
    grid gap at a spot separates that spot's successive events."""
    comps = sorted(comps, key=lambda co: (co[0], co[1], co[2]))
    groups = []
    open_ = []                       # [last_c, cx, cy, members]
    for co in comps:
        c, cx, cy, area, mag, bb = co[:6]
        hit = None
        for gr in open_:
            if c - gr[0] <= 2 * step and \
                    np.hypot(cx - gr[1], cy - gr[2]) < max_r:
                hit = gr
                break
        if hit is None:
            open_.append([c, cx, cy, [co]])
        else:
            n = len(hit[3])
            hit[0] = c
            hit[1] = (hit[1] * n + cx) / (n + 1)
            hit[2] = (hit[2] * n + cy) / (n + 1)
            hit[3].append(co)
    return [gr[3] for gr in open_]


def persists(Fs, bb, c_ev, w, g, ga, strong, T,
             prev_c=None, next_c=None):
    """EVENT-BOUNDED state persistence: the after-state must hold
    until the NEXT event at this spot, the before-state since the
    PREVIOUS one - that is the true physical claim (a fixed 1.5w span
    straddled the neighbouring stack at busy spots and rejected 30% of
    real set-downs, measured stage-by-stage). The floor keeps the arm
    out: its same-spot neighbours are always closer than 0.8w, so a
    pose can never prove persistence, while stack cadence (~5s) can.
    MEDIAN over the span keeps a transiting arm from failing a real
    event."""
    y0, x0, y1, x1 = bb
    box = Fs[:, y0:y1 + 1, x0:x1 + 1, :]
    span = int(1.5 * w)

    def check(b_lim, a_lim, floor):
        a0, a1 = c_ev + ga, min(c_ev + ga + w, a_lim)
        b0, b1 = max(c_ev - g - w, b_lim), c_ev - g
        if a1 - a0 < 3 or b1 - b0 < 3:
            return False
        A = np.median(box[a0:a1:2], 0)
        B = np.median(box[b0:b1:2], 0)
        cb0 = max(c_ev - g - span, b_lim)
        ca1 = min(c_ev + ga + span, a_lim)
        if min((c_ev - g) - cb0, ca1 - (c_ev + ga)) < floor:
            return False
        da = np.median(color_change(box[a0:ca1:2],
                                    A[None]).mean((1, 2)))
        db = np.median(color_change(box[cb0:b1:2],
                                    B[None]).mean((1, 2)))
        return da < 0.5 * strong and db < 0.5 * strong

    # ADDITIVE: the full-span test (as always) OR the event-bounded
    # test. Bounding alone LOST recall (0.70 -> 0.53 measured): junk
    # candidates near a spot truncate real events' spans below any
    # floor. As an OR it can only add: busy-spot set-downs whose full
    # span straddles a neighbour pass the bounded test; arm churn
    # fails both (its same-spot neighbours are closer than 0.4w).
    if check(0, T, 0):
        return True
    if prev_c is None and next_c is None:
        return False
    return check(0 if prev_c is None else max(prev_c + g, 0),
                 T if next_c is None else min(max(next_c - g, 0), T),
                 int(0.4 * w))


def stable_crop(F, bb, yx, i0, i1):
    """Temporal-MEDIAN crop of the component's bbox, MASKED to the
    changed pixels: everything outside the component becomes the
    outside's own median colour. Unmasked crops were dominated by the
    shared table context - a block and bare table sat 0.37 apart in
    encoder space and every event self-paired (measured). Returns
    (crop, mask): the mask rides along so the encoder pools only the
    object's own patch tokens."""
    y0, x0, y1, x1 = bb
    pad = max(2, (y1 - y0) // 4, (x1 - x0) // 4)
    py0 = max(y0 - pad, 0)
    px0 = max(x0 - pad, 0)
    py1 = min(y1 + pad + 1, F.shape[1] // DS)
    px1 = min(x1 + pad + 1, F.shape[2] // DS)
    i0 = max(i0, 0)
    i1 = min(i1, len(F))
    if i1 - i0 < 1 or (py1 - py0) < 4 or (px1 - px0) < 4:
        return None
    win = F[i0:i1:3, py0 * DS:py1 * DS, px0 * DS:px1 * DS]
    c = np.median(win, axis=0)
    m = np.zeros((py1 - py0, px1 - px0), bool)
    if yx is not None:
        ys, xs = yx
        m[ys - py0, xs - px0] = True
        from scipy import ndimage
        m = ndimage.binary_dilation(m, iterations=1)
    m_full = np.repeat(np.repeat(m, DS, 0), DS, 1)
    out = ~m_full
    if out.any() and m_full.any():
        # NEUTRAL fill, not the outside's own median: a table-coloured
        # fill correlated every crop with the bare-table crops and
        # dragged whole colour groups onto the table cluster (measured:
        # cyan ~ table at 0.82)
        c[out] = 128.0
    if c.size < 3 * 10 * 10:
        return None
    return c.astype(np.uint8), m_full, (py0, px0)


MAE_MID = "facebook/vit-mae-base"
_MAE = {}


def embed_masked(crops, masks, bs=48):
    """MAE (pixel-reconstruction) features, MASK-POOLED over patch
    tokens. Chosen by measurement, not taste: BOTH store DINOv3
    encoders are colour-blind by augmentation design (cyan-blue cubes
    0.85-0.90 similarity ACROSS colours vs 0.73-0.84 for the same
    object) - they cannot hold the sharpened rule's promise that the
    vision model distinguishes a green block from a red one. MAE's
    objective is pixel reconstruction: colour survives by
    construction (measured: same-red 0.78, red-cyan -0.10). Naive
    patch-mean drowned in the flat fill (bare-table crops all ~1.0),
    so tokens pool under the component's own mask. Vision-only, no
    text anywhere in the model or the path."""
    import cv2
    import torch
    from transformers import AutoImageProcessor, ViTMAEModel
    if not _MAE:
        print(f"loading appearance encoder {MAE_MID}...", flush=True)
        _MAE["proc"] = AutoImageProcessor.from_pretrained(MAE_MID)
        m = ViTMAEModel.from_pretrained(MAE_MID, mask_ratio=0.0).eval()
        _MAE["dev"] = "mps" if torch.backends.mps.is_available() \
            else "cpu"
        _MAE["model"] = m.to(_MAE["dev"])
    proc, model, dev = _MAE["proc"], _MAE["model"], _MAE["dev"]
    from tqdm import tqdm
    out = []
    for i0 in tqdm(range(0, len(crops), bs), desc="mae-embed",
                   unit="batch", mininterval=5):
        ims = crops[i0:i0 + bs]
        with torch.no_grad():
            inp = proc(images=ims, return_tensors="pt").to(dev)
            hs = model(**inp).last_hidden_state[:, 1:, :].cpu().numpy()
        for h_, m_ in zip(hs, masks[i0:i0 + bs]):
            side = int(round(math.sqrt(h_.shape[0])))
            pm = cv2.resize(m_.astype(np.float32), (side, side),
                            interpolation=cv2.INTER_AREA).reshape(-1)
            if pm.max() > 0.05:
                wgt = pm / pm.sum()
            else:
                wgt = np.full(h_.shape[0], 1.0 / h_.shape[0],
                              np.float32)
            out.append((h_ * wgt[:, None]).sum(0))
    E = np.stack(out).astype(np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True),
                          1e-8)


def extract(db, views, probe=False):
    """Full pass: decode -> sites -> events -> crops. Returns events
    (with raw crops) + transient-motion agent crops + far background
    crops, everything needed downstream, frames dropped."""
    from tqdm import tqdm
    rng = np.random.default_rng(0)
    all_events = []          # (ep, stream, t_ns, cx, cy, area, bi, ai)
    crops, masks = [], []    # raw crop images + masks, indexed bi/ai
    agent_crops, bg_crops = [], []
    agent_masks, bg_masks = [], []
    D_pool, step_pool = [], []
    area_pool = []

    # PASS A on a sample: pool the diff distribution, INSPECT, fit cuts
    sample = views[:: max(len(views) // 24, 1)][:24]
    for ep, sv, sl in tqdm(sample, desc="fit-cuts", unit="view"):
        ts, F = decode_view(db, sl)
        T, fps, w, g, ga, step = grid_params(ts, F)
        Fs = F[:, ::DS, ::DS, :].astype(np.float32)
        lo = MIN_WIN_FRAMES + g
        for c in range(lo, T - MIN_WIN_FRAMES - ga, step * 4):
            b = stable_median(Fs, max(c - w - g, 0), c - g)
            a = stable_median(Fs, c + ga, min(c + ga + w, T))
            D_pool.append(color_change(a, b)[::4, ::4].ravel())
    Dv = np.concatenate(D_pool)
    lo, hi = np.percentile(Dv, [50, 99.9])
    Dl = np.log10(Dv + 0.01)
    # THREE-MODE fit: the per-grid diff mass is noise (~0), weak change
    # (moved shadows, AA edges, ~30-95) and strong change (block <->
    # table, >126). Otsu drifted to the zero spike (measured: cut 4,
    # 120 events/view); 3-means in log space lands one centroid per
    # mode and the strong cut is the upper midpoint.
    ck = k_means_1d(Dl[Dv > 0], 3)
    diff_cut = 10 ** ((ck[0] + ck[1]) / 2) - 0.01
    strong_cut = 10 ** ((ck[1] + ck[2]) / 2) - 0.01
    h, e = np.histogram(Dl, bins=24)
    print(f"diff distribution (angle x1000): p50 {lo:.2f}  "
          f"p99.9 {hi:.1f}  cut {diff_cut:.2f}  "
          f"strong {strong_cut:.2f}", flush=True)
    print("  log-hist " + " ".join(f"{10**c-0.01:.2f}:{v}" for c, v in
          zip((e[:-1] + e[1:]) / 2, h) if v), flush=True)

    # PASS B: full extraction
    for ep, sv, sl in tqdm(views, desc="events", unit="view"):
        ts, F = decode_view(db, sl)
        comps, (Fs, fps, w, g, ga, step) = grid_components(ts, F,
                                                           strong_cut)
        if not comps:
            continue
        area_pool += [c_[3] for c_ in comps]
        S_med = math.sqrt(np.median([c_[3] for c_ in comps]))
        max_r = max(1.2 * S_med, 6.0)
        groups = associate(comps, step, max_r)
        # dominant surfaces of this view (table / floor / void):
        # direction and shadow-junk read against these, in the same
        # angle metric as the detector
        med5 = np.median(Fs[::5], axis=0)
        px = med5.reshape(-1, 3)
        cents = k_means_rgb(px[rng.choice(len(px), 4000)], 3)
        ev_pos = []
        cand = []
        for members in groups:
            if len(members) < 2:               # single-grid flicker
                continue
            peak = max(members, key=lambda m: m[4])
            c_ev = int(round(np.average([m[0] for m in members],
                                        weights=[m[4] for m in
                                                 members])))
            cand.append((c_ev, peak))
        cand.sort(key=lambda cp: cp[0])
        for ci, (c_ev, peak) in enumerate(cand):
            # nearest candidate events at the SAME spot bound the
            # persistence claim on each side
            prev_c = next_c = None
            for c2, p2 in cand:
                if c2 == c_ev and p2 is peak:
                    continue
                if np.hypot(p2[1] - peak[1], p2[2] - peak[2]) \
                        >= 2.0 * max_r:
                    continue
                if c2 < c_ev:
                    prev_c = c2 if prev_c is None else max(prev_c, c2)
                elif c2 > c_ev and (next_c is None or c2 < next_c):
                    next_c = c2
            # PERSISTENCE IS EVIDENCE, NOT A GATE. The hard gate
            # rejected real arrivals whenever junk candidates crowded
            # the spot and collapsed the neighbour bounds (measured:
            # true-colour purple arrivals with pers=False). Every
            # candidate is emitted; the seriality parse weighs the
            # flag globally.
            pers = persists(Fs, peak[5], c_ev, w, g, ga, strong_cut,
                            len(F), prev_c, next_c)
            cb = stable_crop(F, peak[5], peak[6], c_ev - w - g,
                             c_ev - g)
            ca = stable_crop(F, peak[5], peak[6], c_ev + ga,
                             c_ev + ga + w)
            if cb is None or ca is None:
                continue
            cx, cy = peak[1] * DS, peak[2] * DS
            # State colours read DIRECTLY from the frames at the core
            # pixels (the crop-mapped read landed on mask fill and
            # junked half the corpus's set-downs, measured), plus a
            # PEAK-PIXEL second opinion: the quartile core can dilute
            # a small block with edge pixels; the strongest-change
            # pixel's 3x3 is the purest object sample.
            ys_c, xs_c = peak[7]
            b0s, b1s = max(c_ev - g - w, 0), c_ev - g
            a0s, a1s = c_ev + ga, min(c_ev + ga + w, len(F))
            vb = np.median(np.median(
                Fs[b0s:b1s:2][:, ys_c, xs_c, :], 0), 0) \
                .astype(np.float32)
            va = np.median(np.median(
                Fs[a0s:a1s:2][:, ys_c, xs_c, :], 0), 0) \
                .astype(np.float32)
            ty, tx = peak[8]
            sl_y = slice(max(ty - 1, 0), ty + 2)
            sl_x = slice(max(tx - 1, 0), tx + 2)
            vb2 = np.median(Fs[b0s:b1s:2, sl_y, sl_x, :]
                            .reshape(-1, 3), 0).astype(np.float32)
            va2 = np.median(Fs[a0s:a1s:2, sl_y, sl_x, :]
                            .reshape(-1, 3), 0).astype(np.float32)
            all_events.append([ep, sv, int(ts[min(c_ev, len(ts) - 1)]),
                               cx, cy, float(peak[3]), len(crops),
                               len(crops) + 1,
                               surf_angle(vb, cents),
                               surf_angle(va, cents),
                               surf_ratio_std(vb, cents),
                               surf_ratio_std(va, cents),
                               int(pers),
                               surf_angle(vb2, cents),
                               surf_angle(va2, cents)])
            crops += [cb[0], ca[0]]
            masks += [cb[1], ca[1]]
            ev_pos.append((peak[1], peak[2]))
        # agent gallery: crops inside the TOP-ACTIVITY envelope - the
        # arm re-traverses its workspace all episode long, a carried
        # block passes any pixel once, so high-count pixels are arm
        # (sampling at single transient peaks grabbed carried blocks
        # and the gallery junked real events - measured 88/119)
        from scipy import ndimage
        if len(agent_crops) < 200:
            a3 = Fs[::3]
            fd = color_change(a3[1:], a3[:-1])
            act = (fd > strong_cut).sum(0)
            hi = act > np.quantile(act[act > 0], 0.85) \
                if (act > 0).any() else act > 0
            ys_h, xs_h = np.where(hi)
            for _ in range(4):
                if not len(ys_h):
                    break
                j = int(rng.integers(len(ys_h)))
                y0a, x0a = int(ys_h[j]), int(xs_h[j])
                # a frame where this pixel is currently in motion
                fr = np.where(fd[:, y0a, x0a] > strong_cut)[0]
                if not len(fr):
                    continue
                i = int(fr[rng.integers(len(fr))])
                m = fd[i]
                lab, _ = ndimage.label(m > strong_cut)
                if lab[y0a, x0a] == 0:
                    continue
                ys, xs = np.where(lab == lab[y0a, x0a])
                if len(ys) < 12:
                    continue
                # crop around the SAMPLE PIXEL, not the component
                # centroid: the component is the whole arm
                r = int(max(math.sqrt(np.median(
                    [c_[3] for c_ in comps])), 6))
                keep = (np.abs(ys - y0a) < r) & (np.abs(xs - x0a) < r)
                ys, xs = ys[keep], xs[keep]
                if len(ys) < 12:
                    continue
                bb = (int(ys.min()), int(xs.min()),
                      int(ys.max()), int(xs.max()))
                fi = min(i * 3 + 1, len(F) - 1)
                c = stable_crop(F, bb, (ys, xs), fi, fi + 1)
                if c is not None:
                    agent_crops.append(c[0])
                    agent_masks.append(c[1])
        # background gallery: far from every event, a synthetic disc
        # mask of the corpus site scale so the crops live in the same
        # masked space as everything else
        if len(bg_crops) < 200:
            H, W = Fs.shape[1:3]
            r = max(S_med / 1.4, 4.0)
            for _ in range(6):
                x, y = rng.uniform(0.08, 0.92) * W, \
                    rng.uniform(0.08, 0.92) * H
                if any(np.hypot(x - px, y - py) < 3.0 * S_med
                       for px, py in ev_pos):
                    continue
                yy, xx = np.mgrid[0:H, 0:W]
                inside = (yy - y) ** 2 + (xx - x) ** 2 < r * r
                ys, xs = np.where(inside)
                if len(ys) < 12:
                    continue
                bb = (int(ys.min()), int(xs.min()),
                      int(ys.max()), int(xs.max()))
                i = int(rng.integers(len(F)))
                c = stable_crop(F, bb, (ys, xs), i, i + 1)
                if c is not None:
                    bg_crops.append(c[0])
                    bg_masks.append(c[1])
    print(f"events {len(all_events):,}  "
          f"({len(all_events)/max(len(views),1):.1f}/view)  "
          f"agent gallery {len(agent_crops)}  bg gallery {len(bg_crops)}",
          flush=True)
    return (all_events, crops, masks, agent_crops, agent_masks,
            bg_crops, bg_masks)


def classify_events(events):
    """Direction from the SURFACE MODEL, no learned appearance in the
    loop (the MAE/gallery route measured 24/75 at scale - hand-picked
    events lied): a side whose state sits at a dominant surface is
    empty, a side that doesn't is occupied.

        before occupied, after surface   -> departure (-1)
        before surface,  after occupied  -> arrival   (+1)
        both occupied                    -> push       (0)
        both surface                     -> junk (a moved shadow -
                                            shadows ARE surface under
                                            the angle metric)

    OCCUPIED = far from every surface in angle, OR near in angle but
    with NON-UNIFORM channel ratios (a table-hued block; a shadow is a
    uniform scaling). Both cuts fitted by 2-means on the pooled
    log-features of every event side."""
    ang = np.array([[e[8], e[9]] for e in events], np.float32)
    rst = np.array([[e[10], e[11]] for e in events], np.float32)
    la = np.log10(ang + 1.0)
    cut, m0, m1 = two_means(la.ravel())
    surf_cut = 10 ** cut - 1.0
    cutr, r0, r1 = two_means(np.log10(rst.ravel() + 1e-3))
    ratio_cut = 10 ** cutr - 1e-3
    b_obj = (ang[:, 0] > surf_cut) | (rst[:, 0] > ratio_cut)
    a_obj = (ang[:, 1] > surf_cut) | (rst[:, 1] > ratio_cut)
    if len(events) and len(events[0]) > 14:
        # peak-pixel second opinion (a small block dilutes the
        # quartile core; the strongest-change pixel is the purest
        # object sample)
        a2 = np.array([[e[13], e[14]] for e in events], np.float32)
        b_obj |= a2[:, 0] > surf_cut
        a_obj |= a2[:, 1] > surf_cut
    dirs = np.where(b_obj & a_obj, 0, np.where(a_obj, 1, -1))
    junk = (~b_obj) & (~a_obj)
    print(f"surface cut {surf_cut:.1f} (modes {10**m0-1:.1f}/"
          f"{10**m1-1:.1f})  ratio cut {ratio_cut:.3f} (modes "
          f"{10**r0-1e-3:.3f}/{10**r1-1e-3:.3f})  "
          f"junk {int(junk.sum())}/{len(events)}  "
          f"dirs arr {int(((dirs == 1) & ~junk).sum())} dep "
          f"{int(((dirs == -1) & ~junk).sum())} push "
          f"{int(((dirs == 0) & ~junk).sum())}", flush=True)
    return dirs, junk


def manipulations(events, dirs, junk, med_r=24.0):
    """OBJECT PERMANENCE AS SPOT BOOKKEEPING, no appearance identity:

    - events merge across views by time (the scene is one serial
      manipulation stream; both views share the clock by recording).
    - per (view, spot) a LIFO stack: an arrival pushes, a departure
      pops - physically exact for stacking, and a pop names the
      departing object as WHATEVER LAST ARRIVED THERE. A pop of an
      empty spot births a new cast member (a block resident since
      before recording).
    - a departure pairs with the NEXT arrival (at most one object is
      in flight in a serial stream) -> a manipulation; a push is a
      one-event manipulation.
    Slots come out of union-find over these links - the vision model
    decides only WHERE/WHEN states changed and which side is table;
    identity is pure bookkeeping."""
    by_ep = defaultdict(list)
    for ev, dr, jk in zip(events, dirs, junk):
        if jk:
            continue
        ep, sv, t, cx, cy = ev[:5]
        by_ep[ep].append((int(t), sv, cx, cy, int(dr)))
    out = {}
    for ep, evs in by_ep.items():
        evs.sort()
        # cross-view merge: same direction within 1s = one physical
        # event, keep every view's position
        merged = []
        for t, sv, cx, cy, dr in evs:
            hit = None
            for mm in merged:
                if mm["dir"] == dr and abs(t - mm["t"]) < int(1.0e9) \
                        and sv not in mm["pos"]:
                    hit = mm
                    break
            if hit is None:
                merged.append({"t": t, "dir": dr, "pos": {sv: (cx, cy)}})
            else:
                hit["pos"][sv] = (cx, cy)
        # spot registries per view; each spot holds a LIFO of object
        # tokens; union-find over tokens
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x, y):
            parent.setdefault(x, x)
            parent.setdefault(y, y)
            parent[find(x)] = find(y)

        spots = defaultdict(list)   # (view, spot_idx) -> LIFO of tokens
        reg = defaultdict(list)     # view -> [(x, y)]
        next_tok = [0]

        def spot_of(sv, x, y):
            for si, (rx, ry) in enumerate(reg[sv]):
                if np.hypot(x - rx, y - ry) < med_r:
                    return si
            reg[sv].append((x, y))
            return len(reg[sv]) - 1

        def new_tok():
            next_tok[0] += 1
            return next_tok[0] - 1

        open_dep = None             # (token, t, pos)
        mans = []                   # (t0, t1, token, travel, push)
        for i, mm in enumerate(merged):
            dr = mm["dir"]
            if dr == 0:
                # BOTH SIDES OCCUPIED = stack arrival, unstack
                # departure or push. Serial context decides: one arm
                # cannot push while carrying, so an open flight makes
                # this the flight's STACK ARRIVAL; if the next
                # decided event is an arrival, this must be the
                # departure that opened it (UNSTACK); else a push.
                if open_dep is not None:
                    dr = 1
                else:
                    nxt = next((m2["dir"] for m2 in merged[i + 1:]
                                if m2["dir"] != 0), None)
                    dr = -1 if nxt == 1 else 0
            if dr <= 0:             # departure (or push departs+lands)
                toks = []
                for sv, (x, y) in mm["pos"].items():
                    si = spot_of(sv, x, y)
                    st = spots[(sv, si)]
                    toks.append(st.pop() if st else new_tok())
                tok = toks[0]
                for t2 in toks[1:]:
                    union(tok, t2)
            if dr == 0:             # push: lands right back
                for sv, (x, y) in mm["pos"].items():
                    si = spot_of(sv, x, y)
                    spots[(sv, si)].append(tok)
                mans.append((mm["t"], mm["t"], tok, 0.0, True))
                continue
            if dr < 0:
                open_dep = (tok, mm["t"], mm["pos"])
                continue
            # arrival
            if open_dep is not None:
                tok, t0, pos0 = open_dep
                open_dep = None
            else:
                tok, t0, pos0 = new_tok(), mm["t"], {}
            for sv, (x, y) in mm["pos"].items():
                si = spot_of(sv, x, y)
                spots[(sv, si)].append(tok)
            trav = -1.0
            for sv, (x, y) in mm["pos"].items():
                if sv in pos0:
                    trav = max(trav, float(np.hypot(x - pos0[sv][0],
                                                    y - pos0[sv][1])))
            mans.append((t0, mm["t"], tok, trav, False))
        out[ep] = [(t0, t1, find(tok), trav, push)
                   for t0, t1, tok, trav, push in mans]
    return out


def tokenise(mans):
    trav = np.array([m[3] for v in mans.values() for m in v if m[3] > 0])
    tq = np.percentile(trav, [33, 66]) if len(trav) else [1, 2]
    out = {}
    for ep, ms in mans.items():
        slot_of = {}
        toks = []
        prev = None
        for t0, t1, c, d, push in ms:
            if c not in slot_of:
                slot_of[c] = len(slot_of)
            if prev is not None:
                gap = max((t0 - prev) / 1e9, 0.0)
                toks.append((("G", int(min(gap / 4.0, 2)), 0),
                             -1, 0.0, None))
            kind = ("P", 0, 0) if push else \
                (("M", int(np.searchsorted(tq, d)) if d >= 0 else 1, 0))
            toks.append((kind, slot_of[c], (t1 - t0) / 1e9, None))
            prev = t1
        out[ep] = toks
    return out


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv \
        else 0
    db = Store.open(str(store))
    views = episode_views(db)
    if limit:
        views = [v for v in views if v[0] < limit]
    print(f"{len(views)} episode-views", flush=True)

    cache = Path(os.environ.get(
        "ELIDEDB_DELTA_CACHE",
        "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
        "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
        "scratchpad/delta_events.npz"))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        events = [list(e) for e in z["events"]]
        crops = list(z["crops"])
        masks = list(z["masks"])
        agent_crops = list(z["agents"])
        agent_masks = list(z["agent_masks"])
        bg_crops = list(z["bg"])
        bg_masks = list(z["bg_masks"])
        print(f"events from cache: {len(events):,}")
    else:
        t0 = time.time()
        (events, crops, masks, agent_crops, agent_masks, bg_crops,
         bg_masks) = extract(db, views)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, events=obj_arr(events), crops=obj_arr(crops),
                 masks=obj_arr(masks), agents=obj_arr(agent_crops),
                 agent_masks=obj_arr(agent_masks),
                 bg=obj_arr(bg_crops), bg_masks=obj_arr(bg_masks))
        print(f"extraction {(time.time()-t0)/60:.1f} min", flush=True)

    dirs, junk = classify_events(events)
    med_r = 0.9 * DS * math.sqrt(np.median([e[5] for e in events])) \
        if events else 24.0
    mans = manipulations(events, dirs, junk, med_r)
    n = [len(v) for v in mans.values()]
    print(f"episodes {len(mans)}  mean manipulations "
          f"{np.mean(n) if n else 0:.1f}", flush=True)
    seqs = tokenise(mans)
    for e in sorted(seqs)[:6]:
        print(f"  ep{e}: " + " ".join(
            f"{k[0]}{s}" for k, s, _, _ in seqs[e]))

    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    # EVAL-SIDE diagnostic (truth is the grader, never a path input):
    # detected manipulation count + cast size vs the chain script's
    n_true = defaultdict(int)
    cast_true = defaultdict(set)
    for e, p, b in zip(t["episode"], t["prim"], t["block"]):
        if str(p) != "pick":            # a manipulation = pick+set-down
            n_true[int(e)] += 1
        cast_true[int(e)].add(str(b))
    got_n = [len(mans.get(e, [])) for e in sorted(mans)]
    tru_n = [n_true[e] for e in sorted(mans)]
    got_c = [len({m[2] for m in mans.get(e, [])}) for e in sorted(mans)]
    tru_c = [len(cast_true[e]) for e in sorted(mans)]
    print(f"EVAL count: detected {np.mean(got_n):.1f}/ep vs true "
          f"{np.mean(tru_n):.1f}  cast {np.mean(got_c):.1f} vs "
          f"{np.mean(tru_c):.1f}", flush=True)
    # slot-partition agreement: match manips to truth set-downs by
    # time, then pairwise same-block-vs-same-slot accuracy
    ep0_of = {int(e): int(t) for e, t in
              zip(db.table("episodes").scan().to_pydict()
                  ["episode_index"],
                  db.table("episodes").scan().to_pydict()["ts"])}
    tru_md = defaultdict(list)
    for e, p, b, a1 in zip(t["episode"], t["prim"], t["block"],
                           t["t1"]):
        if str(p) != "pick":
            tru_md[int(e)].append((float(a1), str(b)))
    agree = tot_pairs = 0
    for e, ms in mans.items():
        pairs = []
        used = set()
        for t0, t1, tok, trav, push in sorted(ms):
            te = (t1 - ep0_of[e]) / 1e9
            cands = [(abs(te - a1), j) for j, (a1, b)
                     in enumerate(tru_md[e])
                     if j not in used and abs(te - a1) < 3.5]
            if not cands:
                continue
            _, j = min(cands)
            used.add(j)
            pairs.append((tok, tru_md[e][j][1]))
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                tot_pairs += 1
                agree += (pairs[i][0] == pairs[j][0]) == \
                    (pairs[i][1] == pairs[j][1])
    print(f"EVAL slots: pairwise agreement "
          f"{agree}/{tot_pairs} = "
          f"{agree/max(tot_pairs,1):.2f}", flush=True)
    chain_qbe.W_KIND = 0.5
    dev = ("swap", "precarious", "push_then_build",
           "build_unstack_move")
    hold = ("relocate_build", "two_sites_merge")
    pools = defaultdict(int)
    for e in seqs:
        pools[tmpl.get(e)] += 1
    if all(pools[tm] > 6 for tm in dev + hold):
        bench(seqs, tmpl, "DEV delta", dev, w_slot=1.0)
        bench(seqs, tmpl, "HOLDOUT delta", hold, w_slot=1.0)
    else:
        print("(probe scale - bench skipped)")


if __name__ == "__main__":
    main()
