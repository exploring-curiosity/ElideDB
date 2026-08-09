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
    """The scene as a PIECEWISE-CONSTANT series: a settled snapshot
    every `cadence` frames, each the pixel median of its neighbourhood.
    A resting entity IS scene (persistence without co-variation - the
    atomic definition); it enters the story via scene CHANGES."""
    T = len(F)
    ts = list(range(0, T, cadence))
    snaps = [np.median(F[max(0, t - halfwin):min(T, t + halfwin):3]
                       .astype(np.float32), 0) for t in ts]
    return np.array(ts), snaps


def scene_changes(ts, snaps):
    """Regions where consecutive settled scenes differ: something
    appeared or vanished from the persistent world. Each site carries
    its time, bbox and the before/after appearance."""
    sites = []
    for i in range(1, len(snaps)):
        a, b = snaps[i - 1], snaps[i]
        s_a = a.sum(-1, keepdims=True) + 3.0
        s_b = b.sum(-1, keepdims=True) + 3.0
        chrom = np.abs(a / s_a - b / s_b).sum(-1)
        inten = np.abs(a - b).sum(-1)
        m = (chrom > 0.08) | (inten > 90)
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
            sites.append(dict(
                t0=int(ts[i - 1]), t1=int(ts[i]),
                cx=0.5 * (xs.start + xs.stop),
                cy=0.5 * (ys.start + ys.stop),
                w=float(xs.stop - xs.start),
                h=float(ys.stop - ys.start),
                pre_rgb=a[sl][mm].mean(0).astype(np.float32),
                post_rgb=b[sl][mm].mean(0).astype(np.float32)))
    return sites


def link(mp4):
    """One recording (one view) -> (motion tracks, scene sites, T).
    Motion foreground is taken against the TIME-LOCAL settled scene,
    so only currently-moving things are tracked; resting entities live
    in the scene series and surface as change sites."""
    F = read_frames(mp4)
    T = len(F)
    ts, snaps = settled_scenes(F)
    sites = scene_changes(ts, snaps)
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
    return stitch(tracks), sites, T


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
    return agent_ids, residue_ids, stats


def build_one(ep, out):
    rows = []
    for cam in sorted(ep.glob("cam*.mp4")):
        tracks, sites, T = link(cam)
        agent_ids, residue_ids, stats = roles(tracks, T)
        for s in sites:
            rows.append(dict(view=cam.stem, role="site", **s))
        for tr, (pres, mov, ext) in zip(tracks, stats):
            rows.append(dict(
                view=cam.stem, tid=tr.tid,
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


def build(corpus):
    CACHE.mkdir(parents=True, exist_ok=True)
    eps = sorted((ROOT / corpus).glob("ep*"))
    from tqdm import tqdm
    name = Path(corpus).name
    for ep in tqdm(eps, unit="ep", desc=f"ent/{name}"):
        out = CACHE / f"{name}_{ep.name}.npy"
        if not out.exists():
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
        for view in sorted({r["view"] for r in rows}):
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
