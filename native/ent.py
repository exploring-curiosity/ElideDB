"""L1-L3 of the general solution (native/PROBLEM.md): ENTITIES.

L1  persistence linking: appearance that travels together becomes a
    track. Converts fields into things. No detector of known objects,
    no classes - foreground is "differs from the recording's own
    persistent scene", components are linked frame to frame.
L2  roles by statistics: per-recording profile of each track
    (presence, involvement in change) -> scene / recurring agent /
    episodic entity. Figure-ground is computed, never asserted.
L3  entity records: state series stored RAW in the recording's
    private frame (compared relationally only, downstream); cheap
    appearance for linking; fingerprints attached later at moment
    build (they are a read-path concern, not an L1 gate concern).

Problem-check (PROBLEM.md #9): nothing here names arm/block/pen; the
same operators run on any fixed-camera recording. Known V1 limits,
stated: static camera per recording (moving cameras need the
scene-relative registration of PROBLEM.md #5); occlusion avoided at
capture (owner's V1 concession).

    python native/ent.py --build data/sim_probe            # cache tracks
    python native/ent.py --grade data/sim_probe            # vs sim truth
Truth (meta.json colors/spans) is EVAL-ONLY, used by --grade alone.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "ent"
W, H = 640, 480
FPS = 10.0
BG_STRIDE = 5            # frames sampled into the scene median
MIN_AREA_FRAC = 2e-4     # component noise floor, relative to frame area
GAP_BRIDGE = 6           # frames a track survives unmatched
MOVE_EXT = 0.35          # "moved" = displacement > this * own extent
OBJ_CADENCE = 5          # frames between object-configuration samples


def read_frames(mp4):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True)
    n = len(p.stdout) // (W * H * 3)
    return np.frombuffer(p.stdout, np.uint8).reshape(n, H, W, 3)


def otsu(x):
    h, edges = np.histogram(x, 128)
    mids = (edges[:-1] + edges[1:]) / 2
    w = h / max(h.sum(), 1)
    best, thr = -1.0, float(np.median(x))
    for i in range(1, 127):
        w0, w1 = w[:i].sum(), w[i:].sum()
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        m0 = (w[:i] * mids[:i]).sum() / w0
        m1 = (w[i:] * mids[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, float(mids[i])
    return thr


class Track:
    __slots__ = ("tid", "t", "cx", "cy", "w", "h", "rgb", "misses",
                 "contact")

    def __init__(self, tid, t, det):
        self.tid = tid
        self.t = [t]
        self.cx = [det[0]]
        self.cy = [det[1]]
        self.w = [det[2]]
        self.h = [det[3]]
        self.rgb = [det[4]]
        self.contact = [det[5]]
        self.misses = 0

    def last(self):
        return (self.cx[-1], self.cy[-1], self.w[-1], self.h[-1],
                self.rgb[-1])

    def add(self, t, det):
        self.t.append(t)
        self.cx.append(det[0]); self.cy.append(det[1])
        self.w.append(det[2]); self.h.append(det[3])
        self.rgb.append(det[4]); self.contact.append(det[5])
        self.misses = 0


def detections(fg, frame, lab, nlab):
    """Connected components -> (cx, cy, w, h, meanRGB, contact, bbox)."""
    out = []
    if nlab == 0:
        return out
    objs = ndimage.find_objects(lab)
    areas = ndimage.sum_labels(fg, lab, index=np.arange(1, nlab + 1))
    for i, sl in enumerate(objs):
        if sl is None or areas[i] < MIN_AREA_FRAC * W * H:
            continue
        ys, xs = sl
        m = lab[sl] == (i + 1)
        px = frame[sl][m]
        out.append((0.5 * (xs.start + xs.stop),
                    0.5 * (ys.start + ys.stop),
                    float(xs.stop - xs.start), float(ys.stop - ys.start),
                    px.mean(0).astype(np.float32), False,
                    (xs.start, ys.start, xs.stop, ys.stop)))
    return out


def _rect_dist(px, py, box):
    """Distance from a point to a bbox (0 inside). Center distance is
    meaningless for large blobs - a small entity inside a big merged
    component is AT the component, not far from its centroid."""
    x0, y0, x1, y1 = box
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return float(np.hypot(dx, dy))


def foreground(F, bg):
    """Chromaticity + intensity change vs the scene. A shadow darkens
    without changing hue; an entity changes chromaticity (or changes
    intensity massively). Generic: no colours are named, thresholds
    come from the recording's own distributions (Otsu)."""
    Ff = F.astype(np.float32)
    s_f = Ff.sum(-1, keepdims=True) + 3.0
    s_b = bg.sum(-1, keepdims=True) + 3.0
    chrom = np.abs(Ff / s_f - bg[None] / s_b).sum(-1)     # (T, H, W)
    inten = np.abs(Ff - bg[None]).sum(-1)
    tc = otsu(chrom[:: max(1, len(F) // 30)].ravel())
    ti = otsu(inten[:: max(1, len(F) // 30)].ravel())
    return (chrom > max(tc, 0.05)) | (inten > 2.5 * ti)


def _vel(tr):
    if len(tr.t) < 2:
        return 0.0, 0.0
    dt = max(tr.t[-1] - tr.t[-2], 1)
    return ((tr.cx[-1] - tr.cx[-2]) / dt, (tr.cy[-1] - tr.cy[-2]) / dt)


def settled_scenes(F, cadence=10, halfwin=15):
    """The scene as a PIECEWISE-CONSTANT series. A resting entity IS
    scene (persistence without co-variation); it enters the story via
    scene CHANGES. v2: the pieces of a piecewise-constant scene are
    the QUIET spans between motion, so snapshots anchor to local
    minima of global change energy - a fixed cadence straddles events
    and smears one physical change across several diffs (measured:
    every window saturated at 4+ sites and every fact flattened)."""
    T = len(F)
    Ff = F[::2].astype(np.float32)
    energy = np.abs(np.diff(Ff, axis=0)).sum((1, 2, 3))
    energy = np.interp(np.arange(T), np.arange(len(energy)) * 2, energy)
    k = np.ones(7) / 7
    e = np.convolve(energy, k, mode="same")
    thr = np.percentile(e, 40)
    quiet = e <= thr
    # quiet runs -> one snapshot at each run's midpoint, median taken
    # INSIDE the run so it never straddles adjacent motion
    runs, s = [], None
    for t in range(T):
        if quiet[t] and s is None:
            s = t
        elif not quiet[t] and s is not None:
            if t - s >= 5:
                runs.append((s, t))
            s = None
    if s is not None and T - s >= 5:
        runs.append((s, T))
    if not runs:
        runs = [(t, min(t + cadence, T))
                for t in range(0, T, cadence)]
    if runs[0][0] > 8:
        runs.insert(0, (0, min(8, T)))
    if T - runs[-1][1] > 8:
        runs.append((T - 8, T))
    ts, snaps = [], []
    for s0, s1 in runs:
        mid = (s0 + s1) // 2
        hw = min(halfwin, max((s1 - s0) // 2, 2))
        ts.append(mid)
        snaps.append(np.median(
            F[max(s0, mid - hw):min(s1, mid + hw + 1):2]
            .astype(np.float32), 0))
    return np.array(ts), snaps


def scene_changes(F, ts, snaps):
    """Regions where consecutive settled scenes differ: something
    appeared or vanished from the persistent world. Each site carries
    its time, bbox and the before/after appearance.

    The mask is CHROMATIC change only: shadows and illumination are
    same-chromaticity by construction, and the census showed they were
    most of the excess sites (tan->darker-tan, grey->grey). The known
    generic limit: an achromatic change on achromatic ground is
    indistinguishable from illumination in a two-snapshot diff - that
    ambiguity is real in any fixed-camera recording, not a corpus
    quirk."""
    sites = []
    for i in range(1, len(snaps)):
        a, b = snaps[i - 1], snaps[i]
        s_a = a.sum(-1, keepdims=True) + 3.0
        s_b = b.sum(-1, keepdims=True) + 3.0
        chrom = np.abs(a / s_a - b / s_b).sum(-1)
        m = chrom > 0.08
        # opening cuts thin penumbra bridges: shadow edges pass the
        # mask pixel-by-pixel and chain solid regions into one giant
        # component whose "change" is nothing (measured: a 174x112
        # site with core pre~post swallowing the real block sites)
        m = ndimage.binary_opening(m, structure=np.ones((3, 3), bool))
        lab, nlab = ndimage.label(m)
        if not nlab:
            continue
        for j, sl in enumerate(ndimage.find_objects(lab)):
            if sl is None:
                continue
            ys, xs = sl
            area = int((lab[sl] == j + 1).sum())
            if area < MIN_AREA_FRAC * W * H:
                continue
            mm = lab[sl] == j + 1
            # CORE colour, not region mean: the fringe of a change
            # region is shadow/blend and washes the colour until
            # everything matches everything (measured: colour
            # anchoring selected at chance). Median over the
            # top-quartile-changed pixels is the thing itself.
            mag = np.abs(a[sl] - b[sl]).sum(-1)
            core = mm & (mag >= np.percentile(mag[mm], 75))
            if core.sum() < 3:
                core = mm
            # DIRECTION (which side holds the thing): the thing
            # differs from its own surround, the ground matches it -
            # figure/ground from spatial coincidence alone. Compare
            # each side's core colour to the surrounding ring in the
            # SAME snapshot; dirg > 0 = thing on the pre side
            # (vanish-shaped), < 0 = thing on the post side
            # (appear-shaped). A whole-recording background cannot
            # answer this (a long-resting entity IS the background at
            # its own spot); the ring can.
            ys2 = slice(max(ys.start - 5, 0), min(ys.stop + 5, H))
            xs2 = slice(max(xs.start - 5, 0), min(xs.stop + 5, W))
            mm2 = np.zeros((ys2.stop - ys2.start,
                            xs2.stop - xs2.start), bool)
            mm2[ys.start - ys2.start:ys.stop - ys2.start,
                xs.start - xs2.start:xs.stop - xs2.start] = mm
            ring = ndimage.binary_dilation(mm2, iterations=4) & ~mm2
            pre_rgb = np.median(a[sl][core], 0).astype(np.float32)
            post_rgb = np.median(b[sl][core], 0).astype(np.float32)
            # core validity: the site's own medians must clear the
            # SAME chromatic bar the mask used - an aggregate of
            # penumbra edges has extreme pixels but near-equal cores,
            # i.e. it is not a change of the persistent world
            cd = np.abs(pre_rgb / (pre_rgb.sum() + 3.0)
                        - post_rgb / (post_rgb.sum() + 3.0)).sum()
            if cd < 0.08:
                continue
            if ring.sum() >= 8:
                ring_a = np.median(a[ys2, xs2][ring], 0)
                ring_b = np.median(b[ys2, xs2][ring], 0)
                d_pre = float(np.abs(pre_rgb - ring_a).sum())
                d_post = float(np.abs(post_rgb - ring_b).sum())
                dirg = (d_pre - d_post) / max(d_pre + d_post, 1e-6)
            else:
                dirg = 0.0
            # LOCALIZE the change inside the snapshot interval: when
            # events run back to back, one snapshot pair straddles
            # several physical changes and interval-overlap attachment
            # hands every window the whole bundle (measured: 4+ sites
            # per window, every fact saturated). The region itself
            # says when it changed: its distance-to-pre / distance-to-
            # post curves cross exactly once; a transient occlusion
            # (the agent passing over) bumps both curves without
            # faking a crossing.
            t0f, t1f = int(ts[i - 1]), int(ts[i])
            pre_c = a[sl][core]
            post_c = b[sl][core]
            samp = list(range(t0f, t1f + 1, 2))
            dpre = np.array(
                [np.abs(F[t][sl][core].astype(np.float32)
                        - pre_c).mean() for t in samp])
            dpost = np.array(
                [np.abs(F[t][sl][core].astype(np.float32)
                        - post_c).mean() for t in samp])
            sc = dpre - dpost
            tc = (t0f + t1f) // 2
            ki = len(sc) - 1
            for k in range(len(sc) - 1):
                if sc[k] > 0 and sc[k + 1] > 0:
                    tc, ki = samp[k], k
                    break
            # ONSET: last sample still at rest in the pre state before
            # the crossing. tc alone LAGS the physical change (the
            # crossing cannot complete until the agent clears the
            # spot), so a change is honestly located only as an
            # interval (ton, tc]; downstream, window membership is the
            # fractional overlap of that interval, not a point test.
            base = float(dpre[0])
            thr = base + 0.3 * (float(dpre[:ki + 1].max()) - base)
            ton = t0f
            for k in range(ki, -1, -1):
                if dpre[k] <= thr:
                    ton = samp[k]
                    break
            sites.append(dict(
                t0=t0f, t1=t1f, tc=int(tc), ton=int(min(ton, tc)),
                dirg=float(dirg),
                cx=0.5 * (xs.start + xs.stop),
                cy=0.5 * (ys.start + ys.stop),
                w=float(xs.stop - xs.start),
                h=float(ys.stop - ys.start),
                pre_rgb=pre_rgb, post_rgb=post_rgb))
    return sites


def _same_look(ci, cj):
    """Same THING, seen under different light. Chromaticity only, and
    a looser bar than the change detector uses: the two faces of one
    object differ almost entirely in intensity (measured: splitting
    on intensity too shattered blocks into lit/shaded halves, each
    below the area floor - entity presence fell 0.623 -> 0.477)."""
    si, sj = ci.sum() + 3.0, cj.sum() + 3.0
    return float(np.abs(ci / si - cj / sj).sum()) < 0.14


def _colour_regions(small, comp, floor, ithr):
    """One connected foreground blob may hold SEVERAL things that
    touch: two stacked entities are a single component against the
    surface, and the merge hides exactly the relation that separates
    'arrived on open ground' from 'arrived onto a particular thing'
    (measured: 38% of true entities absent, outsized regions at
    precisely the stacking snapshots).

    Things are split by colour coherence - spatial coincidence of
    LIKE measurements, the same atomic signal one level finer - then
    neighbours that look the same are re-merged so shading gradients
    do not shatter one thing into many."""
    # split on CHROMATICITY (which distinguishes things) plus a single
    # per-recording dark/light cut (which distinguishes achromatic
    # things - a black arm joint from a white one - without letting
    # shading cut one object in half)
    s = small.sum(-1) + 3.0
    q = (np.round(small[..., 0] / s / 0.05).astype(np.int32) * 1000
         + np.round(small[..., 1] / s / 0.05).astype(np.int32) * 10
         + (s > ithr).astype(np.int32))
    parts = []
    for val in np.unique(q[comp]):
        lab2, n2 = ndimage.label(comp & (q == val))
        for k in range(1, n2 + 1):
            mk = lab2 == k
            if mk.sum() >= floor:
                parts.append(mk)
    changed = True
    while changed and len(parts) > 1:
        changed = False
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                if not _same_look(np.median(small[parts[i]], 0),
                                  np.median(small[parts[j]], 0)):
                    continue
                if not (ndimage.binary_dilation(parts[i])
                        & parts[j]).any():
                    continue
                parts[i] = parts[i] | parts.pop(j)
                changed = True
                break
            if changed:
                break
    return parts


def scene_objects(ts, snaps):
    """L3: the THINGS a scene is made of - spatial coincidence WITHIN
    a still (PROBLEM.md atomic signal (a)), which the write path never
    used until now. It only ever saw change, so an entity that never
    moves did not exist; but the thing you stack ONTO is exactly such
    an entity, and it is the reference object of the relation that
    separates place from stack.

    A thing is a region that differs from the surface it sits on.
    'Surface' is estimated per-recording as the LOCAL colour mode
    (median over a neighbourhood many times an object's size), so
    nothing about tables, walls or blocks is asserted - on a street
    the same operator returns cars against road, on a bench parts
    against a workbench.

    THE SAMPLING IS REGULAR, NOT SETTLED-ONLY (measured 2026-08-10):
    local contrast needs no background model, so it works on ANY
    frame - only the change-SITE machinery needs quiet scenes. When
    configurations were read at quiet snapshots, a pick-carry-place
    sequence fell BETWEEN two of them and the resulting facts
    described several events at once: place events came out
    vanish-dominant (kind -0.31 where an arriving thing must be +1).
    A regularly sampled state series is also what PROBLEM.md #4
    means by a state series in the first place.

    Objects are linked across samples by position+colour, giving a
    resting particular its own state series exactly as a moving one
    gets from tracks."""
    D = 4                                # work at 1/4 scale
    regions = []                         # per sample: list of dicts
    diffs = []
    for sn in snaps:
        small = sn[::D, ::D]
        surf = np.stack([ndimage.median_filter(small[..., c], size=15)
                         for c in range(3)], -1)
        s_s = small.sum(-1, keepdims=True) + 3.0
        s_u = surf.sum(-1, keepdims=True) + 3.0
        diffs.append(np.abs(small / s_s - surf / s_u).sum(-1))
    thr = max(otsu(np.concatenate([d.ravel() for d in diffs])), 0.06)
    ithr = otsu(np.concatenate(
        [sn[::D, ::D].sum(-1).ravel() for sn in snaps]))
    for k, d in enumerate(diffs):
        small = snaps[k][::D, ::D]
        m = ndimage.binary_opening(d > thr, np.ones((2, 2), bool))
        lab, nlab = ndimage.label(m)
        objs = []
        for j in range(1, nlab + 1):
            comp = lab == j
            if comp.sum() < 12:          # 12 quarter-scale px
                continue
            for mk in _colour_regions(small, comp, 12, ithr):
                ys, xs = ndimage.find_objects(mk.astype(np.int8))[0]
                objs.append(dict(
                    t=int(ts[k]),
                    cx=D * 0.5 * (xs.start + xs.stop),
                    cy=D * 0.5 * (ys.start + ys.stop),
                    w=float(D * (xs.stop - xs.start)),
                    h=float(D * (ys.stop - ys.start)),
                    rgb=np.median(small[mk], 0).astype(np.float32)))
        regions.append(objs)
    # link across snapshots: a particular persists where position and
    # appearance both continue (identity by continuity, PROBLEM.md #4)
    objects, live = [], []
    for objs in regions:
        used = set()
        for tr in live:
            best, bj = None, -1
            for j, o in enumerate(objs):
                if j in used:
                    continue
                ext = max(tr["w"][-1], tr["h"][-1], 8.0)
                dist = np.hypot(o["cx"] - tr["cx"][-1],
                                o["cy"] - tr["cy"][-1])
                dcol = float(np.abs(o["rgb"] - tr["rgb"][-1]).sum())
                if dist > 0.8 * ext or dcol > 150:
                    continue
                cost = dist + 0.2 * dcol
                if best is None or cost < best:
                    best, bj = cost, j
            if bj >= 0:
                o = objs[bj]
                used.add(bj)
                for key in ("t", "cx", "cy", "w", "h", "rgb"):
                    tr[key].append(o[key])
        for j, o in enumerate(objs):
            if j not in used:
                live.append({k2: [v] for k2, v in o.items()})
    objects = [tr for tr in live if len(tr["t"]) >= 1]
    return objects


def link(mp4):
    """One recording (one view) -> (motion tracks, scene sites, T).
    Motion foreground is taken against the TIME-LOCAL settled scene,
    so only currently-moving things are tracked; resting entities live
    in the scene series and surface as change sites."""
    F = read_frames(mp4)
    T = len(F)
    ts, snaps = settled_scenes(F)
    sites = scene_changes(F, ts, snaps)
    # object configurations on a REGULAR cadence (see scene_objects):
    # sites need quiet scenes, objects do not
    otimes = np.arange(0, T, OBJ_CADENCE)
    objects = scene_objects(otimes, [F[t].astype(np.float32)
                                     for t in otimes])
    # per-frame background = nearest settled snapshot; thresholds are
    # per-RECORDING (one Otsu over all groups), not per-group
    idx = np.clip(np.searchsorted(ts, np.arange(T)), 0, len(ts) - 1)
    chrom = np.empty((T, H, W), np.float32)
    inten = np.empty((T, H, W), np.float32)
    for i, sn in enumerate(snaps):
        sel = idx == i
        if not sel.any():
            continue
        Ff = F[sel].astype(np.float32)
        s_f = Ff.sum(-1, keepdims=True) + 3.0
        s_b = sn.sum(-1, keepdims=True) + 3.0
        chrom[sel] = np.abs(Ff / s_f - sn[None] / s_b).sum(-1)
        inten[sel] = np.abs(Ff - sn[None]).sum(-1)
    tc = otsu(chrom[:: max(1, T // 30)].ravel())
    ti = otsu(inten[:: max(1, T // 30)].ravel())
    fg_all = (chrom > max(tc, 0.05)) | (inten > 2.5 * ti)
    del chrom, inten
    tracks, live, next_id = [], [], 0
    for t in range(T):
        lab, nlab = ndimage.label(fg_all[t])
        dets = detections(fg_all[t], F[t], lab, nlab)
        used = set()
        # match live tracks to detections (greedy by gated distance
        # from the velocity-extrapolated position; a detection may
        # serve SEVERAL tracks = contact/merged blob)
        claims = {}
        for tr in live:
            cx, cy, w, h, rgb = tr.last()
            vx, vy = _vel(tr)
            dt = t - tr.t[-1]
            px, py = cx + vx * dt, cy + vy * dt
            ext = max(w, h)
            best, bj = None, -1
            for j, d in enumerate(dets):
                # gate on the TRACK's own extent, distance to the
                # detection's bbox: a track may sit inside a large
                # merged blob, but may never leap by more than its
                # own scale
                dist = _rect_dist(px, py, d[6])
                if dist > ext * 1.2 + 12:
                    continue
                cost = dist + 0.15 * np.abs(d[4] - rgb).sum()
                if best is None or cost < best:
                    best, bj = cost, j
            if bj >= 0:
                claims.setdefault(bj, []).append(tr)
        for j, trs in claims.items():
            d = dets[j]
            multi = len(trs) > 1
            for tr in trs:
                if multi:
                    # merged blob: keep each track alive at the blob,
                    # flagged as contact - contact IS the interaction
                    # currency (L4), not an error to fix
                    cx, cy, w, h, rgb = tr.last()
                    tr.add(t, (d[0], d[1], w, h, rgb, True))
                else:
                    tr.add(t, d)
            used.add(j)
        for tr in live:
            if not tr.t or tr.t[-1] != t:
                tr.misses += 1
        live = [tr for tr in live if tr.misses <= GAP_BRIDGE
                or tracks.append(tr)]
        for j, d in enumerate(dets):
            if j not in used:
                live.append(Track(next_id, t, d))
                next_id += 1
    tracks += live
    tracks = [tr for tr in tracks if len(tr.t) >= 5]
    return stitch(tracks), sites, objects, T


def stitch(tracks, max_gap=25, pos_slack=2.5, col_tol=140.0):
    """Tracklet stitching: a track that ends and another that starts
    shortly after, nearby (relative to extent, allowing drift during
    the gap) and looking alike, are ONE entity whose link broke.
    Identity by continuity - resemblance only confirms, never merges
    across a spatial impossibility."""
    tracks.sort(key=lambda tr: tr.t[0])
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(tracks):
            best, bj = None, -1
            for j, b in enumerate(tracks):
                if i == j or b.t[0] <= a.t[-1]:
                    continue
                gap = b.t[0] - a.t[-1]
                if gap > max_gap:
                    continue
                ext = max(np.median(a.w), np.median(a.h), 8.0)
                dist = np.hypot(b.cx[0] - a.cx[-1], b.cy[0] - a.cy[-1])
                if dist > pos_slack * ext + 0.5 * gap:
                    continue
                dcol = np.abs(np.median(np.stack(a.rgb), 0)
                              - np.median(np.stack(b.rgb), 0)).sum()
                if dcol > col_tol:
                    continue
                cost = dist + 0.2 * dcol + gap
                if best is None or cost < best:
                    best, bj = cost, j
            if bj >= 0:
                b = tracks.pop(bj)
                a.t += b.t
                a.cx += b.cx; a.cy += b.cy
                a.w += b.w; a.h += b.h
                a.rgb += b.rgb; a.contact += b.contact
                merged = True
                break
    return tracks


def roles(tracks, T, n_change_windows=40):
    """L2: per-recording statistics -> role per track.
    agent = present through the recording AND involved in most change;
    episodic = the rest; (scene = the background, by construction)."""
    stats = []
    for tr in tracks:
        pres = len(tr.t) / max(T, 1)
        cx, cy = np.array(tr.cx), np.array(tr.cy)
        ext = max(np.median(tr.w), np.median(tr.h), 1.0)
        d = np.hypot(np.diff(cx), np.diff(cy))
        moving = d > 0.05 * ext
        stats.append((pres, float(moving.mean()), ext))
    agent_ids, residue_ids = set(), set()
    if stats:
        score = np.array([p * m for p, m, _ in stats])
        smax = score.max()
        for i, (p, m, ext) in enumerate(stats):
            if p > 0.5 and score[i] > 0.4 * smax and smax > 0:
                agent_ids.add(tracks[i].tid)
                continue
            # residue: never displaced beyond its own scale over its
            # whole life = illumination/shadow artifact, not a thing
            # that ever went anywhere. Entities displace; shadows
            # jitter in place.
            tr = tracks[i]
            span = np.hypot(max(tr.cx) - min(tr.cx),
                            max(tr.cy) - min(tr.cy))
            if span < 1.2 * ext:
                residue_ids.add(tr.tid)
    # CARRIED-ENTITY DEMOTION: a much-carried patient is statistically
    # agent-like (high presence, high moving, involved in every
    # change - measured: a blue block's carry track earned the agent
    # role, poisoned the body palette, and got its own vanish site
    # flagged as agent-body). What separates it is ASYMMETRIC
    # co-movement: the carried thing moves only while its carrier
    # moves, and it is born later - it did not exist as a mover
    # before the world started changing.
    cands = sorted((tr for tr in tracks if tr.tid in agent_ids),
                   key=lambda tr: tr.t[0])
    demoted = set()
    for i, tri in enumerate(cands):
        ti = np.array(tri.t)
        di = np.hypot(np.diff(tri.cx), np.diff(tri.cy))
        exti = max(np.median(tri.w), np.median(tri.h), 1.0)
        mov_i = set(int(t) for t in ti[1:][di > 0.05 * exti])
        if not mov_i:
            continue
        for trj in cands[:i]:
            if trj.tid in demoted \
                    or tri.t[0] - trj.t[0] < 0.1 * T:
                continue
            tj = np.array(trj.t)
            dj = np.hypot(np.diff(trj.cx), np.diff(trj.cy))
            extj = max(np.median(trj.w), np.median(trj.h), 1.0)
            mov_j = set()
            for t in tj[1:][dj > 0.05 * extj]:
                mov_j.update(range(int(t) - 3, int(t) + 4))
            if len(mov_i & mov_j) / len(mov_i) > 0.9:
                demoted.add(tri.tid)
                break
    agent_ids -= demoted
    # INDEPENDENT (self-moving) entities: displaced substantially with
    # almost no contact coupling - people in car footage, other
    # vehicles, animals. Nothing requires an agent to act on them
    # (PROBLEM.md L2); the empty coupling profile is itself structure.
    self_ids = set()
    for i, tr in enumerate(tracks):
        if tr.tid in agent_ids or tr.tid in residue_ids:
            continue
        if len(tr.contact) and float(np.mean(tr.contact)) < 0.15:
            self_ids.add(tr.tid)
    return agent_ids, residue_ids, self_ids, stats


def _agent_explains(agents, t_ref, s, side_rgb, pal):
    """Is this site end the agent's own body? Two conditions: some
    fragment was parked at the site around t_ref, AND the site's
    appearance at that end looks like the agent's body (palette) - a
    ghost's changed side IS the body by definition, so both must
    hold; position alone flags real sites the agent parks over.
    At snapshot times the agent is invisible to the motion layer, so
    the nearest sample IS the parked pose - a track ends where the
    body stopped. The agent is FRAGMENTARY by construction (foreground
    is diffed against a scene that contains the parked body, so only
    the parts clear of the old silhouette show); per-fragment
    centre-inside-site or half-coverage, any fragment sufficing."""
    x0, x1 = s["cx"] - s["w"] / 2, s["cx"] + s["w"] / 2
    y0, y1 = s["cy"] - s["h"] / 2, s["cy"] + s["h"] / 2
    area = max((x1 - x0) * (y1 - y0), 1.0)
    hit = False
    for tr in agents:
        t = np.asarray(tr.t)
        i = int(np.clip(np.searchsorted(t, t_ref), 0, len(t) - 1))
        if i > 0 and abs(t[i - 1] - t_ref) < abs(t[i] - t_ref):
            i -= 1
        mx = 0.15 * max(x1 - x0, y1 - y0)
        if x0 - mx <= tr.cx[i] <= x1 + mx \
                and y0 - mx <= tr.cy[i] <= y1 + mx:
            hit = True
            break
        w, h = tr.w[i] * 1.15, tr.h[i] * 1.15
        ix = max(0.0, min(x1, tr.cx[i] + w / 2)
                 - max(x0, tr.cx[i] - w / 2))
        iy = max(0.0, min(y1, tr.cy[i] + h / 2)
                 - max(y0, tr.cy[i] - h / 2))
        if ix * iy / area >= 0.5:
            hit = True
            break
    if not hit or pal is None or not len(pal):
        return False
    # the site side must LOOK like the agent's own body (measured:
    # position alone flagged the carried purple cylinder's real
    # appear site - the arm parks over what it just released).
    # Chromaticity-first comparison: raw RGB L1 calls a mid-grey and
    # a dark purple "close" (same luminance), exactly the confusion
    # that let the false flag through.
    sr = side_rgb / (side_rgb.sum() + 3.0)
    pr = pal / (pal.sum(1, keepdims=True) + 3.0)
    chrom = np.abs(pr - sr).sum(1)
    inten = np.abs(pal.sum(1) - side_rgb.sum())
    return bool(((chrom < 0.12) & (inten < 300)).any())


CACHE_VER = 3            # 3: object configs on a regular cadence


def views_of(rows):
    """View names in a cache, excluding the version marker row.
    Every reader must go through this: the marker's empty view sorts
    FIRST, so a naive set-of-views would hand a caller '' as view 0."""
    return sorted({r["view"] for r in rows if r["role"] != "_ver"})


def build_one(ep, out):
    rows = [dict(role="_ver", ver=CACHE_VER, view="")]
    for cam in sorted(ep.glob("cam*.mp4")):
        tracks, sites, objects, T = link(cam)
        agent_ids, residue_ids, self_ids, stats = roles(tracks, T)
        for o in objects:
            rows.append(dict(
                view=cam.stem, role="object",
                t=np.array(o["t"], np.int32),
                cx=np.array(o["cx"], np.float32),
                cy=np.array(o["cy"], np.float32),
                w=np.array(o["w"], np.float32),
                h=np.array(o["h"], np.float32),
                rgb=np.stack(o["rgb"]).astype(np.float32)))
        # AGENT-BODY sites: the agent parked in different poses across
        # quiet spans shows up as vanish/appear-shaped scene change
        # (measured: arm links against the backdrop, red gripper pad -
        # chromatic, so no illumination filter can catch it). Role-
        # consistent, not colour-based: the agent is not scene, so a
        # site end the agent's body occupied is the agent, not an
        # entity event. The flags are PER END; the read side must
        # apply them direction-aware (a real vanish site's POST end is
        # legitimately agent-visited - it picked the thing up).
        # Flagged, kept in the store, excluded only at moment build.
        ag_trs = [tr for tr in tracks if tr.tid in agent_ids]
        # body palette = each agent fragment's WHOLE-LIFE median
        # colour. Not per-sample: a carried entity is merged into the
        # agent's blob (it has no track of its own under the piecewise
        # scene), so samples during a carry wear the entity's colour -
        # a life median is immune to the brief carry, and different
        # fragments supply the body's different colour modes.
        pal = (np.stack([np.median(np.stack(tr.rgb), 0)
                         for tr in ag_trs]).astype(np.float32)
               if ag_trs else None)
        for s in sites:
            s["ab_pre"] = _agent_explains(
                ag_trs, s["t0"], s, s["pre_rgb"], pal)
            s["ab_post"] = _agent_explains(
                ag_trs, s["t1"], s, s["post_rgb"], pal)
            rows.append(dict(view=cam.stem, role="site", **s))
        for tr, (pres, mov, ext) in zip(tracks, stats):
            rows.append(dict(
                view=cam.stem, tid=tr.tid,
                selfmove=tr.tid in self_ids,
                role=("agent" if tr.tid in agent_ids else
                      "residue" if tr.tid in residue_ids else "entity"),
                presence=pres, moving=mov, extent=ext,
                t=np.array(tr.t, np.int32),
                cx=np.array(tr.cx, np.float32),
                cy=np.array(tr.cy, np.float32),
                w=np.array(tr.w, np.float32),
                h=np.array(tr.h, np.float32),
                contact=np.array(tr.contact, bool),
                rgb=np.stack(tr.rgb).astype(np.float32)))
    np.save(out, np.array(rows, dtype=object), allow_pickle=True)
    return rows


def _current(out):
    """A cache is reusable only if it carries THIS build's version.
    Existence is not freshness: an interrupted build from an earlier
    format silently survived a rebuild and crashed the bench 45 min
    later (2026-08-10). An explicit marker row beats inferring the
    format from fields, which cannot distinguish 'new layer missing'
    from 'this recording had none of that layer'."""
    if not out.exists():
        return False
    try:
        rows = np.load(out, allow_pickle=True)
    except Exception:
        return False
    return bool(len(rows)) and rows[0].get("role") == "_ver" \
        and rows[0].get("ver") == CACHE_VER


def build(corpus):
    CACHE.mkdir(parents=True, exist_ok=True)
    eps = sorted((ROOT / corpus).glob("ep*"))
    from tqdm import tqdm
    name = Path(corpus).name
    for ep in tqdm(eps, unit="ep", desc=f"ent/{name}"):
        out = CACHE / f"{name}_{ep.name}.npy"
        if not _current(out):
            build_one(ep, out)


def grade(corpus):
    """EVAL-ONLY truth: per ok-event, did exactly one episodic track
    move in the span, and does its colour match the true block's?
    Plus fragmentation (tracks per block) and agent coverage."""
    name = Path(corpus).name
    eps = sorted((ROOT / corpus).glob("ep*"))
    ev_tot = ev_moved = ev_color = 0
    frag, agent_cov, ntracks = [], [], []
    from tqdm import tqdm
    for ep in tqdm(eps, unit="ep", desc=f"grade/{name}"):
        f = CACHE / f"{name}_{ep.name}.npy"
        if not f.exists():
            continue
        rows = np.load(f, allow_pickle=True)
        meta = json.loads((ep / "meta.json").read_text())
        rgba = {b["name"]: 255 * np.array(b["rgba"][:3])
                for b in meta["blocks"]}
        for view in views_of(rows):
            vr = [r for r in rows if r["view"] == view]
            ents = [r for r in vr if r["role"] == "entity"]
            agents = [r for r in vr if r["role"] == "agent"]
            vsites = [r for r in vr if r["role"] == "site"]
            ntracks.append(len(ents))
            claimed = {b: set() for b in rgba}
            ag_hit = ag_tot = 0
            for e in meta["events"]:
                if not e["ok"]:
                    continue
                a, b = int(e["t0"] * FPS), int(e["t1"] * FPS)
                ev_tot += 1
                ag_tot += 1
                movers = []
                for r in ents:
                    m = (r["t"] >= a) & (r["t"] <= b)
                    if m.sum() < 3:
                        continue
                    ext = max(r["extent"], 1.0)
                    dx = r["cx"][m].max() - r["cx"][m].min()
                    dy = r["cy"][m].max() - r["cy"][m].min()
                    if np.hypot(dx, dy) > MOVE_EXT * ext:
                        movers.append(r)
                # scene-change sites overlapping the span (slack one
                # snapshot either side): the settled world changed here
                esites = [s for s in vsites
                          if s["t0"] <= b + 10 and s["t1"] >= a - 10]
                if movers or esites:
                    ev_moved += 1
                    tgt = rgba[e["block"]]
                    ok = False
                    for r in movers:
                        m = (r["t"] >= a) & (r["t"] <= b) \
                            & ~r["contact"]
                        mm = m if m.sum() >= 2 else \
                            ((r["t"] >= a) & (r["t"] <= b))
                        col = r["rgb"][mm].mean(0)
                        if np.abs(col - tgt).sum() < 180:
                            ok = True
                            claimed[e["block"]].add(r["tid"])
                    for s in esites:
                        if min(np.abs(s["pre_rgb"] - tgt).sum(),
                               np.abs(s["post_rgb"] - tgt).sum()) < 180:
                            ok = True
                    ev_color += ok
                for r in agents:
                    m = (r["t"] >= a) & (r["t"] <= b)
                    if m.sum() >= 3:
                        d = np.hypot(np.diff(r["cx"][m]),
                                     np.diff(r["cy"][m])).sum()
                        if d > 0.3 * r["extent"]:
                            ag_hit += 1
                            break
            agent_cov.append(ag_hit / max(ag_tot, 1))
            for b, s in claimed.items():
                if s:
                    frag.append(len(s))
    print(f"\n{name}: {ev_tot} graded events")
    print(f"  event recall (a mover exists)   {ev_moved / max(ev_tot,1):.3f}")
    print(f"  colour-correct recall           {ev_color / max(ev_tot,1):.3f}")
    print(f"  tracks per moved block (frag)   {np.mean(frag):.2f}")
    print(f"  agent coverage of events        {np.mean(agent_cov):.3f}")
    print(f"  episodic tracks per view        {np.mean(ntracks):.1f} "
          f"(true blocks ~{np.mean([len(json.loads((e/'meta.json').read_text())['blocks']) for e in eps[:20]]):.1f})")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build(sys.argv[sys.argv.index("--build") + 1])
    if "--grade" in sys.argv:
        grade(sys.argv[sys.argv.index("--grade") + 1])
