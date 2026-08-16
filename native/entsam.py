"""Perception done properly (SAM-class, owner-approved minimum) +
the happened-facts readout, gated on YIELD.

WHY THIS FILE (the measurements that force it):
  - The primitive's happened-facts had the right DIRECTIONS all along
    (pick vanish-dominant, place appear-dominant) but were computed
    over an entity set that was wrong ~40% of the time (flicker,
    achromatic blindness, stacked-pair merges); noise ~2x signal.
  - Cosine over pooled descriptors is the BUG, per the owner, said
    repeatedly: it compares what a moment LOOKS like. This file
    compares WHAT HAPPENED: did a thing leave its rest, did a thing
    arrive, next to another thing or onto open ground, did one thing
    do both (a push). For simple primitives in clean sim these facts
    separate with no learning and no labels.
  - Single view ONLY (cross-view is barred). No labels anywhere on
    the write or read path; truth grades output only.

Minimum footprint: FastSAM-s (smallest), everything-mode masks at a
regular cadence, linked by continuity. No prompts, no text, no
classes - masks are anonymous regions; roles come from statistics.

    python native/entsam.py --build data/sim_probe     # cache objects
    python native/entsam.py --gate  data/sim_probe     # entity gate
    python native/entsam.py --yield data/sim_probe     # THE gate
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
import ent  # noqa: E402  (read_frames)

CACHE = ROOT / "data" / "cache" / "entsam"
FPS = 10.0
CAD = 5                  # sample cadence (frames)
PRIMS = ("pick", "place", "stack", "unstack", "push")
VER = 4                  # 4: workspace-crop tracking (2-3x effective res)
IMGSZ = 640              # SAM2/FastSAM inference size (A/B via override)
ELEV_W = 0.0             # elevation-fact weight (see agree(): support prior)
ORACLE_BIND = False      # EVAL-ONLY: true-patient binding diagnostic

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import FastSAM
        _MODEL = FastSAM("FastSAM-s.pt")
    return _MODEL


def detect(frame):
    """Anonymous object regions in one frame. Filters are generic:
    scale (not specks, not scenery), border coverage (backdrop and
    table span the frame edge), NMS by containment."""
    H, W = frame.shape[:2]
    res = model()(frame, device="mps", retina_masks=True, imgsz=640,
                  conf=0.3, iou=0.8, verbose=False)[0]
    if res.masks is None:
        return []
    out = []
    for box, mk in zip(res.boxes.xyxy.cpu().numpy(),
                       res.masks.data.cpu().numpy()):
        x0, y0, x1, y1 = box
        area = float(mk.sum())
        if area < 64 or area > 0.12 * W * H:
            continue
        if (x1 - x0) > 0.6 * W or (y1 - y0) > 0.75 * H:
            continue
        ys, xs = np.where(mk > 0)
        rgb = np.median(frame[ys, xs], 0).astype(np.float32)
        out.append(dict(cx=0.5 * (x0 + x1), cy=0.5 * (y0 + y1),
                        w=float(x1 - x0), h=float(y1 - y0),
                        area=area, rgb=rgb))
    # containment NMS: FastSAM emits part and whole; keep the whole
    out.sort(key=lambda d: -d["area"])
    keep = []
    for d in out:
        inside = False
        for kd in keep:
            ix = max(0.0, min(d["cx"] + d["w"] / 2,
                              kd["cx"] + kd["w"] / 2)
                     - max(d["cx"] - d["w"] / 2, kd["cx"] - kd["w"] / 2))
            iy = max(0.0, min(d["cy"] + d["h"] / 2,
                              kd["cy"] + kd["h"] / 2)
                     - max(d["cy"] - d["h"] / 2, kd["cy"] - kd["h"] / 2))
            if ix * iy > 0.7 * d["w"] * d["h"]:
                inside = True
                break
        if not inside:
            keep.append(d)
    return keep


def prompt_boxes(F0):
    """Frame-0 prompts for SAM2 video propagation: every anonymous
    region that is neither scenery nor a speck. Unlike detect(), the
    AGENT BODY must be kept - roles() needs it tracked - so the caps
    exclude only frame-spanning surfaces (table, backdrop, floor
    bands), not merely large things."""
    H, W = F0.shape[:2]
    res = model()(F0, device="mps", retina_masks=True, imgsz=IMGSZ,
                  conf=0.25, iou=0.8, verbose=False)[0]
    if res.masks is None:
        return []
    cands = []
    for box, mk in zip(res.boxes.xyxy.cpu().numpy(),
                      res.masks.data.cpu().numpy()):
        x0, y0, x1, y1 = [float(v) for v in box]
        area = float(mk.sum())
        if area < 64 or area > 0.30 * W * H:
            continue
        if (x1 - x0) > 0.80 * W or (y1 - y0) > 0.90 * H:
            continue
        ys, xs = np.where(mk > 0)
        rgb = np.median(F0[ys, xs], 0).astype(np.float32)
        chrom = rgb / max(float(rgb.sum()), 1.0)
        cands.append((area, [x0, y0, x1, y1], mk > 0, chrom))
    # part-of-whole NMS: pixel containment AND appearance agreement.
    # Pixel not bbox (bbox containment deleted every block sitting in
    # a shadow's box); and appearance too, because FastSAM's coarse
    # region masks BLEED over small things inside them - a block is
    # 70% "inside" the shadow band's mask by pixels, but a true part
    # shares its whole's material look and the block does not.
    cands.sort(key=lambda c: -c[0])
    keep = []
    for area, b, mk, ch in cands:
        inside = False
        for _, _, km, kch in keep:
            if (mk & km).sum() > 0.7 * area \
                    and float(np.abs(ch - kch).sum()) < 0.10:
                inside = True
                break
        if not inside:
            keep.append((area, b, mk, ch))
    return [b for _, b, _, _ in keep]


def track(dets_per_t, times):
    """Identity by continuity across cadence samples: nearest match
    within own extent, confirmed by colour. Anonymous throughout."""
    objs, live = [], []
    for t, dets in zip(times, dets_per_t):
        used = set()
        for tr in live:
            best, bj = None, -1
            for j, d in enumerate(dets):
                if j in used:
                    continue
                ext = max(tr["w"][-1], tr["h"][-1], 8.0)
                dist = np.hypot(d["cx"] - tr["cx"][-1],
                                d["cy"] - tr["cy"][-1])
                dcol = float(np.abs(d["rgb"] - tr["rgb"][-1]).sum())
                if dist > 1.6 * ext or dcol > 140:
                    continue
                cost = dist + 0.2 * dcol
                if best is None or cost < best:
                    best, bj = cost, j
            if bj >= 0:
                d = dets[bj]
                used.add(bj)
                tr["t"].append(t)
                for k2 in ("cx", "cy", "w", "h", "rgb"):
                    tr[k2].append(d[k2])
            else:
                tr["miss"] += 1
        live = [tr for tr in live
                if tr["miss"] <= 3 or objs.append(tr)]
        for j, d in enumerate(dets):
            if j not in used:
                live.append(dict(t=[t], cx=[d["cx"]], cy=[d["cy"]],
                                 w=[d["w"]], h=[d["h"]],
                                 rgb=[d["rgb"]], miss=0))
    objs += live
    return [o for o in objs if len(o["t"]) >= 2]


def stitch_rest(objs):
    """A track that dies and a same-looking track that starts at the
    SAME SPOT later are one particular whose detection flickered -
    the thing never moved, so continuity holds regardless of the gap.
    (Same-spot only: a moved thing changes position, and that join
    must pass physical verification instead.)"""
    objs.sort(key=lambda o: o["t"][0])
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(objs):
            for j, b in enumerate(objs):
                if i == j or b["t"][0] <= a["t"][-1]:
                    continue
                ext = max(np.median(a["w"]), np.median(a["h"]), 8.0)
                dist = np.hypot(b["cx"][0] - a["cx"][-1],
                                b["cy"][0] - a["cy"][-1])
                dcol = float(np.abs(
                    np.median(np.asarray(a["rgb"]), 0)
                    - np.median(np.asarray(b["rgb"]), 0)).sum())
                if dist > 1.0 * ext or dcol > 110:
                    continue
                for k2 in ("t", "cx", "cy", "w", "h", "rgb"):
                    a[k2] = list(a[k2]) + list(b[k2])
                objs.pop(j)
                merged = True
                break
            if merged:
                break
    return objs


def _colour_at(F, t, cx, cy, ext, c):
    """Is appearance c physically present near (cx,cy) at time t?
    Direct pixel check - the arbiter between tracking artifacts and
    physics. Median over a small grid of patches so one occluding
    pixel or shadow edge cannot lie."""
    T = len(F)
    t = int(np.clip(t, 0, T - 1))
    H, W = F[t].shape[:2]
    r = max(int(ext * 0.7), 6)
    y0, y1 = max(int(cy - r), 0), min(int(cy + r), H)
    x0, x1 = max(int(cx - r), 0), min(int(cx + r), W)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return False
    # F[t][...] - indexing F directly sliced the FRAME AXIS by image
    # rows, so every pixel verification compared garbage (latent bug,
    # found via empty-slice warnings from the theft correction)
    patch = F[t][y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if patch.size == 0:
        return False
    d = np.abs(patch - c).sum(1)
    # enough pixels of that colour to be the thing, not a fringe
    return float((d < 110).mean()) > 0.08


def verify(objs, F):
    """Births and deaths must be PHYSICAL to count as events later:
    a death where the colour is still at the spot afterwards was
    detection flicker (died_real=False); a birth where the colour was
    already there before was flicker too. Track ends at the recording
    boundary are neither."""
    T = len(F)
    for o in objs:
        c = np.median(np.asarray(o["rgb"]), 0)
        ext = max(np.median(o["w"]), np.median(o["h"]), 8.0)
        tb, td = int(o["t"][0]), int(o["t"][-1])
        if tb <= CAD:
            o["born_real"] = False       # existed from the start
        else:
            o["born_real"] = not any(
                _colour_at(F, tb - dt, o["cx"][0], o["cy"][0], ext, c)
                for dt in (CAD, 2 * CAD, 3 * CAD))
        if td >= T - CAD - 1:
            o["died_real"] = False       # survived to the end
        else:
            o["died_real"] = not any(
                _colour_at(F, td + dt, o["cx"][-1], o["cy"][-1],
                           ext, c)
                for dt in (CAD, 2 * CAD, 3 * CAD))
    return objs


def roles(objs, T):
    """Agent for masklet world. The anchor is the top presence x
    moving masklet (the body). Other masklets join the agent role
    only if they BOTH move exclusively when the anchor moves AND stay
    near it over their whole life (gripper parts, cast shadow). A
    carried entity co-moves but rests FAR from the anchor most of its
    life, so the distance term keeps it an object - the born-later
    demotion cannot work here because every masklet starts at frame 0
    (measured: the carried purple block took the agent role)."""
    if not objs:
        return set()
    stats = []
    for o in objs:
        pres = len(o["t"]) / max(T, 1)
        cx, cy = np.array(o["cx"]), np.array(o["cy"])
        ext = max(np.median(o["w"]), np.median(o["h"]), 1.0)
        d = np.hypot(np.diff(cx), np.diff(cy))
        stats.append((pres, float((d > 0.1 * ext).mean())
                      if len(d) else 0.0, ext))
    score = np.array([p * m for p, m, _ in stats])
    top = int(np.argmax(score))
    o1 = objs[top]
    t1 = np.asarray(o1["t"])
    ext1 = max(np.median(o1["w"]), np.median(o1["h"]), 8.0)
    d1 = np.hypot(np.diff(o1["cx"]), np.diff(o1["cy"]))
    mov1 = set(int(t) for t in t1[1:][d1 > 0.1 * ext1])
    movpad = set()
    for t in mov1:
        movpad.update(range(t - 3, t + 4))
    agent = {top}
    for i, o in enumerate(objs):
        if i == top:
            continue
        ti = np.asarray(o["t"])
        exti = max(np.median(o["w"]), np.median(o["h"]), 1.0)
        di = np.hypot(np.diff(o["cx"]), np.diff(o["cy"]))
        mov_i = [int(t) for t in ti[1:][di > 0.1 * exti]]
        co = (np.mean([t in movpad for t in mov_i])
              if mov_i else 0.0)
        ds = []
        for k2 in range(0, len(ti), 3):
            j = int(np.clip(np.searchsorted(t1, ti[k2]), 0,
                            len(t1) - 1))
            ds.append(np.hypot(o["cx"][k2] - o1["cx"][j],
                               o["cy"][k2] - o1["cy"][j]))
        neardist = float(np.median(ds)) if ds else 9e9
        if co > 0.9 and neardist < 1.2 * ext1:
            agent.add(i)
    return agent


def _take(slots, last, si, k, feats, fi):
    ft = feats[k]
    last[si] = (ft[0], ft[1])
    sl = slots[si]
    sl["t"].append(fi)
    sl["cx"].append(ft[0])
    sl["cy"].append(ft[1])
    sl["w"].append(ft[2])
    sl["h"].append(ft[3])
    sl["rgb"].append(ft[4])


def workspace_crop(F):
    """The recording's own action region: where pixels ever change.
    Frame-difference energy -> bounding rect + margin. Nothing about
    content - motion defines the workspace. Tracking inside the crop
    gives 2-3x effective resolution on the manipulated things."""
    step = max(1, len(F) // 40)
    S = F[::step].astype(np.int16)
    if len(S) < 3:
        return 0, 0, F.shape[2], F.shape[1]
    E = (np.abs(np.diff(S, axis=0)).sum(-1) > 30).mean(0)
    ys, xs = np.where(E > 0.02)
    H, W = F.shape[1:3]
    if len(ys) < 50:
        return 0, 0, W, H
    x0 = max(int(xs.min()) - 24, 0)
    x1 = min(int(xs.max()) + 24, W)
    y0 = max(int(ys.min()) - 24, 0)
    y1 = min(int(ys.max()) + 24, H)
    # enforce a workable minimum and even dims for the codec
    if x1 - x0 < 288:
        c = (x0 + x1) // 2
        x0, x1 = max(c - 144, 0), min(c + 144, W)
    if y1 - y0 < 288:
        c = (y0 + y1) // 2
        y0, y1 = max(c - 144, 0), min(c + 144, H)
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2
    x1, y1 = x1 - (x1 - x0) % 2, y1 - (y1 - y0) % 2
    return x0, y0, x1 - x0, y1 - y0


def build_one(ep, out, view=None):
    """SAM2 video propagation (VER 3). The tracker's temporal
    identity replaces the whole compensation stack the per-frame
    segmenter needed (stitch, pixel verification, appendage rules -
    all measured insufficient): masklets exist only for frame-0
    prompts, so nothing respawns per frame, and a slot with no mask
    is OCCLUDED OR HELD - absence is a fact, not flicker."""
    from ultralytics.models.sam import SAM2VideoPredictor
    cams = sorted(ep.glob("cam*.mp4"))
    cam = cams[0] if view is None else ep / f"{view}.mp4"
    Fall = ent.read_frames(cam)
    cx0, cy0, cw, chh = workspace_crop(Fall)
    src = cam
    if (cw, chh) != (Fall.shape[2], Fall.shape[1]):
        import subprocess as _sp
        import tempfile
        src = Path(tempfile.gettempdir()) / f"crop_{ep.name}.mp4"
        _sp.run(["ffmpeg", "-y", "-v", "error", "-i", str(cam),
                 "-vf", f"crop={cw}:{chh}:{cx0}:{cy0}",
                 "-c:v", "libx264", "-crf", "15", str(src)],
                check=True)
    F0 = Fall[0][cy0:cy0 + chh, cx0:cx0 + cw]
    boxes = prompt_boxes(F0)
    overrides = dict(conf=0.25, task="segment", mode="predict",
                     imgsz=IMGSZ, model="sam2.1_t.pt", device="mps",
                     verbose=False, save=False)
    pred = SAM2VideoPredictor(overrides=overrides)
    res = pred(source=str(src), bboxes=boxes, stream=True)
    slots = [dict(t=[], cx=[], cy=[], w=[], h=[], rgb=[])
             for _ in boxes]
    last = [(0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])) for b in boxes]
    T = 0
    for fi, r in enumerate(res):
        T = fi + 1
        if r.masks is None:
            continue
        M = r.masks.data.cpu().numpy()
        img = r.orig_img[..., ::-1]          # BGR -> RGB
        feats = []
        for k in range(M.shape[0]):
            ys, xs = np.where(M[k] > 0)
            if len(ys) < 20:
                feats.append(None)
                continue
            feats.append((float(xs.mean()), float(ys.mean()),
                          float(xs.max() - xs.min() + 1),
                          float(ys.max() - ys.min() + 1),
                          np.median(img[ys, xs], 0).astype(np.float32)))
        # greedy match masks -> slots by distance to the slot's last
        # position (masks barely move frame to frame)
        pairs = []
        for k, ft in enumerate(feats):
            if ft is None:
                continue
            for si, (lx, ly) in enumerate(last):
                d = float(np.hypot(ft[0] - lx, ft[1] - ly))
                pairs.append((d, k, si))
        pairs.sort()
        um, us = set(), set()
        for d, k, si in pairs:
            if k in um or si in us:
                continue
            if d > max(60.0, 1.5 * max(feats[k][2], feats[k][3])):
                continue
            um.add(k)
            us.add(si)
            _take(slots, last, si, k, feats, fi)
        # RE-ACQUISITION: a mask far from every slot is a slot coming
        # back after a carry - graded instance re-id by look and size
        # against slots absent right now (identity by continuity
        # first; resemblance only to CONFIRM a reappearance)
        for k, ft in enumerate(feats):
            if ft is None or k in um:
                continue
            best, bs = None, -1
            for si, sl in enumerate(slots):
                if si in us or not sl["t"]:
                    continue
                if fi - sl["t"][-1] < 3:
                    continue
                dcol = float(np.abs(ft[4] - sl["rgb"][-1]).sum())
                sz = max(ft[2], ft[3]) / max(sl["w"][-1],
                                             sl["h"][-1], 8.0)
                if dcol > 110 or not 0.5 < sz < 2.0:
                    continue
                if best is None or dcol < best:
                    best, bs = dcol, si
            if bs >= 0:
                um.add(k)
                us.add(bs)
                _take(slots, last, bs, k, feats, fi)
    objs = [sl for sl in slots if len(sl["t"]) >= 3]
    for o in objs:                       # back to full-frame coords
        o["cx"] = [v + cx0 for v in o["cx"]]
        o["cy"] = [v + cy0 for v in o["cy"]]
    agent = roles(objs, T)
    rows = [dict(role="_ver", ver=VER, T=T, view=cam.name,
                 crop=(cx0, cy0, cw, chh))]
    for i, o in enumerate(objs):
        rows.append(dict(
            role="agent" if i in agent else "object",
            t=np.array(o["t"], np.int32),
            cx=np.array(o["cx"], np.float32),
            cy=np.array(o["cy"], np.float32),
            w=np.array(o["w"], np.float32),
            h=np.array(o["h"], np.float32),
            rgb=np.stack(o["rgb"]).astype(np.float32)))
    np.save(out, np.array(rows, dtype=object), allow_pickle=True)
    return rows


def small_region(rows, W=640, H=480):
    """Where the SMALL tracks live, from a first full-frame pass:
    union of sub-80px tracks' positions + margin. The refinement
    crop - self-calibrated from the recording's own first-pass
    output, no content priors. (A motion-energy crop was tried and
    measured useless: the agent and its shadows sweep the frame.)"""
    xs, ys = [], []
    for o in rows:
        if o["role"] == "_ver":
            continue
        ext = max(float(np.median(o["w"])), float(np.median(o["h"])))
        if ext >= 80:
            continue
        xs += [float(np.min(o["cx"])), float(np.max(o["cx"]))]
        ys += [float(np.min(o["cy"])), float(np.max(o["cy"]))]
    if not xs:
        return 0, 0, W, H
    x0 = max(int(min(xs)) - 40, 0)
    x1 = min(int(max(xs)) + 40, W)
    y0 = max(int(min(ys)) - 40, 0)
    y1 = min(int(max(ys)) + 40, H)
    if x1 - x0 < 200:
        c = (x0 + x1) // 2
        x0, x1 = max(c - 100, 0), min(c + 100, W)
    if y1 - y0 < 200:
        c = (y0 + y1) // 2
        y0, y1 = max(c - 100, 0), min(c + 100, H)
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2
    x1, y1 = x1 - (x1 - x0) % 2, y1 - (y1 - y0) % 2
    return x0, y0, x1 - x0, y1 - y0


def build_refine(ep, base_rows, out, view=None):
    """Pass 2: re-track the small-entity region at 2-3x effective
    resolution; keep pass-1's large tracks (they were never the
    problem), replace the small ones."""
    import subprocess as _sp
    import tempfile
    cams = sorted(ep.glob("cam*.mp4"))
    cam = cams[0] if view is None else ep / f"{view}.mp4"
    cx0, cy0, cw, chh = small_region(base_rows)
    src = Path(tempfile.gettempdir()) / f"ref_{ep.name}.mp4"
    _sp.run(["ffmpeg", "-y", "-v", "error", "-i", str(cam),
             "-vf", f"crop={cw}:{chh}:{cx0}:{cy0}",
             "-c:v", "libx264", "-crf", "15", str(src)], check=True)
    Fall = ent.read_frames(cam)
    F0 = Fall[0][cy0:cy0 + chh, cx0:cx0 + cw]
    boxes = prompt_boxes(F0)
    from ultralytics.models.sam import SAM2VideoPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict",
                     imgsz=IMGSZ, model="sam2.1_t.pt", device="mps",
                     verbose=False, save=False)
    pred = SAM2VideoPredictor(overrides=overrides)
    res = pred(source=str(src), bboxes=boxes, stream=True)
    slots = [dict(t=[], cx=[], cy=[], w=[], h=[], rgb=[])
             for _ in boxes]
    last = [(0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])) for b in boxes]
    T = 0
    for fi, r in enumerate(res):
        T = fi + 1
        if r.masks is None:
            continue
        M = r.masks.data.cpu().numpy()
        img = r.orig_img[..., ::-1]
        feats = []
        for k in range(M.shape[0]):
            ys, xs = np.where(M[k] > 0)
            if len(ys) < 20:
                feats.append(None)
                continue
            feats.append((float(xs.mean()), float(ys.mean()),
                          float(xs.max() - xs.min() + 1),
                          float(ys.max() - ys.min() + 1),
                          np.median(img[ys, xs], 0).astype(np.float32)))
        pairs = []
        for k, ft in enumerate(feats):
            if ft is None:
                continue
            for si, (lx, ly) in enumerate(last):
                d = float(np.hypot(ft[0] - lx, ft[1] - ly))
                pairs.append((d, k, si))
        pairs.sort()
        um, us = set(), set()
        for d, k, si in pairs:
            if k in um or si in us:
                continue
            if d > max(60.0, 1.5 * max(feats[k][2], feats[k][3])):
                continue
            um.add(k)
            us.add(si)
            _take(slots, last, si, k, feats, fi)
    objs = [sl for sl in slots if len(sl["t"]) >= 3]
    for o in objs:
        o["cx"] = [v + cx0 for v in o["cx"]]
        o["cy"] = [v + cy0 for v in o["cy"]]
    # merge: pass-1 large + pass-2 everything inside the region
    keep_large = [dict(r) for r in base_rows
                  if r["role"] != "_ver"
                  and max(float(np.median(r["w"])),
                          float(np.median(r["h"]))) >= 80]
    ver = next(r for r in base_rows if r["role"] == "_ver")
    rows = [dict(role="_ver", ver=VER, T=ver["T"],
                 view=ver.get("view", cam.name),
                 crop=(cx0, cy0, cw, chh))]
    for o in keep_large:
        rows.append(o)
    for o in objs:
        rows.append(dict(
            role="object",
            t=np.array(o["t"], np.int32),
            cx=np.array(o["cx"], np.float32),
            cy=np.array(o["cy"], np.float32),
            w=np.array(o["w"], np.float32),
            h=np.array(o["h"], np.float32),
            rgb=np.stack(o["rgb"]).astype(np.float32)))
    np.save(out, np.array(rows, dtype=object), allow_pickle=True)
    return rows


def build(corpus):
    CACHE.mkdir(parents=True, exist_ok=True)
    eps = sorted((ROOT / corpus).glob("ep*"))
    from tqdm import tqdm
    name = Path(corpus).name
    model()                      # load BEFORE the bar
    for ep in tqdm(eps, unit="ep", desc=f"entsam/{name}"):
        out = CACHE / f"{name}_{ep.name}.npy"
        ok = False
        if out.exists():
            try:
                r = np.load(out, allow_pickle=True)
                ok = len(r) and r[0].get("ver") == VER
            except Exception:
                ok = False
        if not ok:
            build_one(ep, out)


# ---------------- happened-facts + yield ----------------
#
# Everything below runs on CACHED tracks - the expensive layer is the
# tracker; roles and facts are re-derived at read time so they can be
# iterated without re-tracking the corpus. The cache's stored "role"
# field is ignored (superseded by roles2's change-coverage statistic).


def residue_flags(rows, cam):
    """Scene residue, per recording: a TRAVELLING region whose
    appearance matches the scene wherever it goes - a cast shadow, a
    swept highlight - is not an entity; it is the persistent
    remainder's modulation (an entity is appearance that TRAVELS,
    which requires appearance of its own). Two hard-won constraints:
      - test only EXCURSION samples (far from the track's own median
        position): the temporal median contains every near-static
        thing, so testing a thing at its home compares it to ITSELF
        (measured: the green block flagged 5/5, the dark arm eaten);
      - things that never travel are never residue here - they cannot
        win the patient race, and adjacency guards itself by scale.
    Self-calibrated; no colours named; genuine camouflage is eaten,
    which is the honest limit of vision, not a bug."""
    import ent as _ent
    F = _ent.read_frames(cam)
    step = max(1, len(F) // 15)
    med = np.median(F[::step].astype(np.float32), 0)
    H, W = med.shape[:2]

    def _chrom(v):
        return v / max(float(v.sum()), 1.0)

    flags = []
    for r in rows:
        if r["role"] == "_ver":
            continue
        o = r
        ext = max(float(np.median(o["w"])),
                  float(np.median(o["h"])), 8.0)
        # valid test samples: locations this track occupies only
        # TRANSIENTLY (the median frame there is true background).
        # Centre-excursion fails for elongated regions; per-sample
        # occupancy encodes "the median does not contain me here"
        # directly, whatever the shape
        cxs = np.asarray(o["cx"])
        cys = np.asarray(o["cy"])
        nall = len(cxs)
        idx = []
        for i in range(nall):
            occ = float(np.mean(
                np.hypot(cxs - cxs[i], cys - cys[i]) < 0.7 * ext))
            if occ < 0.25:
                idx.append(i)
        if len(idx) < 6:
            flags.append(False)
            continue
        hits = n = 0
        for i in idx[:: max(1, len(idx) // 30)]:
            cy = int(np.clip(o["cy"][i], 2, H - 3))
            cx = int(np.clip(o["cx"][i], 2, W - 3))
            scm = np.median(
                med[cy - 2:cy + 3, cx - 2:cx + 3].reshape(-1, 3), 0)
            d = float(np.abs(_chrom(o["rgb"][i])
                             - _chrom(scm)).sum())
            lr = float(o["rgb"][i].sum()) / max(float(scm.sum()), 1.0)
            n += 1
            if d < 0.10 and lr < 1.15:
                hits += 1
        flags.append(bool(n and hits / n > 0.6))
    return np.asarray(flags, bool)


_DINO = None


def _dino():
    global _DINO
    if _DINO is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        # ConvNeXt-Tiny in fp32: the ViT fp16 path NaNs on MPS (the
        # trap this repo already hit once - the identity cut shipped
        # on ConvNeXt for the same reason)
        proc = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
        mod = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
            ).to("mps").eval()
        _DINO = (proc, mod)
    return _DINO


def thingness(frame, cx, cy, w, h):
    """Is this box a THING or a continuation of its surroundings?
    A 3x-box context crop through the encoder (the full frame
    collapses to a 7x7 grid - one cell is 91px, blind at block
    scale); box cells vs ring cells similarity. A real object is
    feature-distinct from its ring; a shadow, lit patch, or arm-
    reflection is the same surface continuing. Returns box-ring
    similarity in [-1,1]: LOW = thing."""
    import torch
    proc, mod = _dino()
    H, W = frame.shape[:2]
    R = 1.5 * max(w, h, 24.0)
    x0, x1 = int(max(cx - R, 0)), int(min(cx + R, W))
    y0, y1 = int(max(cy - R, 0)), int(min(cy + R, H))
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    if crop.size == 0:
        return 1.0
    inputs = proc(images=crop, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = mod(**inputs)
    hs = out.last_hidden_state[0].float().cpu().numpy()
    g = int(round((hs.shape[0] - 1) ** 0.5))
    G = hs[1:1 + g * g].reshape(g, g, -1)
    # the box occupies the central band of the crop
    fx0 = (cx - w / 2 - x0) / max(x1 - x0, 1) * g
    fx1 = (cx + w / 2 - x0) / max(x1 - x0, 1) * g
    fy0 = (cy - h / 2 - y0) / max(y1 - y0, 1) * g
    fy1 = (cy + h / 2 - y0) / max(y1 - y0, 1) * g
    bx, rx = [], []
    for yy in range(g):
        for xx in range(g):
            v = G[yy, xx]
            if fy0 - 0.3 <= yy + 0.5 <= fy1 + 0.3 \
                    and fx0 - 0.3 <= xx + 0.5 <= fx1 + 0.3:
                bx.append(v)
            else:
                rx.append(v)
    if not bx or not rx:
        return 1.0
    vb = np.mean(bx, 0)
    vr = np.mean(rx, 0)
    vb = vb / max(float(np.linalg.norm(vb)), 1e-6)
    vr = vr / max(float(np.linalg.norm(vr)), 1e-6)
    return float(vb @ vr)


def dino_residue_flags(rows, cam):
    """The residue test with the sanctioned L0's eyes: a track whose
    DINO features MATCH the scene's features at its transiently-
    occupied places is scene modulation (shadow, highlight, arm
    reflection), not a thing. Same occupancy-sampled logic as the
    chroma test - which the refined track pool defeated (measured:
    104 wrong patients, 57 shadow/arm-class) - but a general encoder
    replaces the hand rule. No labels, no names; the threshold is a
    fixed high similarity bar."""
    import ent as _ent
    F = _ent.read_frames(cam)
    step = max(1, len(F) // 15)
    med = np.median(F[::step].astype(np.float32), 0).astype(np.uint8)
    Gm, sy, sx = _dino_grid(med)
    grids = {}
    flags = []
    for r in rows:
        if r["role"] == "_ver":
            continue
        o = r
        ext = max(float(np.median(o["w"])),
                  float(np.median(o["h"])), 8.0)
        cxs = np.asarray(o["cx"])
        cys = np.asarray(o["cy"])
        idx = []
        for i in range(len(cxs)):
            occ = float(np.mean(
                np.hypot(cxs - cxs[i], cys - cys[i]) < 0.7 * ext))
            if occ < 0.25:
                idx.append(i)
        if len(idx) < 4:
            flags.append(False)
            continue
        sims = []
        for i in idx[:: max(1, len(idx) // 6)][:6]:
            fi = int(np.clip(int(o["t"][i]), 0, len(F) - 1))
            if fi not in grids:
                if len(grids) > 12:
                    grids.pop(next(iter(grids)))
                grids[fi] = _dino_grid(F[fi])[0]
            Gf = grids[fi]
            ft = _box_feat(Gf, sy, sx, o["cx"][i], o["cy"][i],
                           o["w"][i], o["h"][i])
            fb = _box_feat(Gm, sy, sx, o["cx"][i], o["cy"][i],
                           o["w"][i], o["h"][i])
            sims.append(float(ft @ fb))
        flags.append(bool(np.median(sims) > 0.80))
    return np.asarray(flags, bool)


def residue_cached(rows, ep):
    name = ep.parent.name
    view = next((x.get("view") for x in rows
                 if x["role"] == "_ver"), None)
    vk = f"_{Path(view).stem}" if view else ""
    f = CACHE / f"{name}_{ep.name}{vk}_res.npy"
    if f.exists():
        r = np.load(f)
        if len(r) == sum(1 for x in rows if x["role"] != "_ver"):
            return r
    view = next((x.get("view") for x in rows
                 if x["role"] == "_ver"), None)
    cam = ep / view if view else sorted(ep.glob("cam*.mp4"))[0]
    r = residue_flags(rows, cam)
    np.save(f, r)
    return r


def entities(rows, res=None):
    """Dedup masklets of one particular: frame-0 prompting gives a
    cylinder's lit face and its whole separate masklets; two tracks
    whose boxes coincide most of their common life are one thing.
    Keep the larger (the whole). Identity stays continuity-only -
    the merge criterion is spatial coincidence, not resemblance."""
    objs = [dict(r) for r in rows if r["role"] != "_ver"]
    if res is not None:
        objs = [o for o, f in zip(objs, res) if not f]
    # a "thing" at scene scale is scene: masklets that leak over the
    # frame during tracking (measured at w 569-591 of 640 with the
    # patient inside, its rest unreadable) are the persistent
    # remainder, not particulars. Frame size is estimated from the
    # data itself, not assumed
    est_w = max((float(np.max(o["cx"])) for o in objs), default=640.0)
    objs = [o for o in objs
            if max(float(np.median(o["w"])),
                   float(np.median(o["h"]))) < 0.45 * est_w]
    for o in objs:
        o["_tidx"] = {int(t): i for i, t in enumerate(o["t"])}
        o["_area"] = float(np.median(o["w"]) * np.median(o["h"]))

    def iou_med(a, b):
        vals = []
        for tb, j in b["_tidx"].items():
            i = a["_tidx"].get(tb)
            if i is None:
                continue
            ax0 = a["cx"][i] - a["w"][i] / 2
            ax1 = a["cx"][i] + a["w"][i] / 2
            ay0 = a["cy"][i] - a["h"][i] / 2
            ay1 = a["cy"][i] + a["h"][i] / 2
            bx0 = b["cx"][j] - b["w"][j] / 2
            bx1 = b["cx"][j] + b["w"][j] / 2
            by0 = b["cy"][j] - b["h"][j] / 2
            by1 = b["cy"][j] + b["h"][j] / 2
            ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            iy = max(0.0, min(ay1, by1) - max(ay0, by0))
            inter = ix * iy
            u = (ax1 - ax0) * (ay1 - ay0) \
                + (bx1 - bx0) * (by1 - by0) - inter
            vals.append(inter / max(u, 1.0))
        return float(np.median(vals)) if len(vals) >= 3 else 0.0

    objs.sort(key=lambda o: -o["_area"])
    kept = []
    for o in objs:
        if any(iou_med(k, o) > 0.4 for k in kept):
            continue
        kept.append(o)

    # a track whose box lives INSIDE a much larger track's box is a
    # PART of it (a specular highlight riding on the arm escaped the
    # residue test - brighter than the scene - and won the patient
    # race with range 6). Scale-gated x3 so a block under a merged
    # stacked-pair track is never swallowed
    def inside_frac(part, whole):
        vals = []
        for tb, j in part["_tidx"].items():
            i = whole["_tidx"].get(tb)
            if i is None:
                continue
            px0 = part["cx"][j] - part["w"][j] / 2
            px1 = part["cx"][j] + part["w"][j] / 2
            py0 = part["cy"][j] - part["h"][j] / 2
            py1 = part["cy"][j] + part["h"][j] / 2
            wx0 = whole["cx"][i] - whole["w"][i] / 2
            wx1 = whole["cx"][i] + whole["w"][i] / 2
            wy0 = whole["cy"][i] - whole["h"][i] / 2
            wy1 = whole["cy"][i] + whole["h"][i] / 2
            ix = max(0.0, min(px1, wx1) - max(px0, wx0))
            iy = max(0.0, min(py1, wy1) - max(py0, wy0))
            pa = max((px1 - px0) * (py1 - py0), 1.0)
            vals.append(ix * iy / pa)
        return float(np.mean(vals)) if len(vals) >= 5 else 0.0

    def _e(o):
        return max(float(np.median(o["w"])),
                   float(np.median(o["h"])), 8.0)

    def _ch(o):
        v = np.median(np.asarray(o["rgb"], np.float32), 0)
        return v / max(float(v.sum()), 1.0)

    # part requires appearance agreement too - the same lesson as the
    # prompt NMS: a sprawling shadow band's BOX covers half the table,
    # and by box alone it swallowed every block resting inside it
    final = []
    for o in kept:
        part_of = any(
            w is not o and _e(w) >= 3.0 * _e(o)
            and inside_frac(o, w) > 0.75
            and float(np.abs(_ch(o) - _ch(w)).sum()) < 0.10
            for w in kept)
        if not part_of:
            final.append(o)
    return final


def _mov_frames(o):
    """Frames at which this track is moving: smoothed step above a
    floor that is absolute (SAM2 centre jitter on a static thing is
    sub-pixel) plus a small own-extent term. NOT 0.1*ext - that made
    big slow things immobile and small flicker mobile (measured: the
    arm body scored below its own shadow fragments)."""
    ts = np.asarray(o["t"])
    if len(ts) < 3:
        return set()
    d = np.hypot(np.diff(o["cx"]), np.diff(o["cy"]))
    if len(d) >= 3:                # "same" pads to kernel size below
        d = np.convolve(d, np.ones(3) / 3.0, mode="same")
    ext = max(np.median(o["w"]), np.median(o["h"]), 8.0)
    thr = max(1.2, 0.02 * ext)
    return set(int(t) for t in ts[1:][d > thr])


def roles2(ents):
    """Roles purely by the L2 statistic, two axes:
      1. co-variation coverage - the recurring agent moves in nearly
         every interval in which anything moves;
      2. lifelong proximity - an APPENDAGE (gripper part, cast
         shadow) lives near the agent its entire life, while a
         carried entity is far from it before its pick and after its
         release (measured: coverage alone cannot split a shadow at
         0.55 from a long-carried block at 0.56 - the proximity axis
         can, because only the shadow is near the arm always).
    Anchor = highest coverage; members join with cov > 0.45 AND
    90th-pct distance-to-anchor < 2.5 anchor extents."""
    mov = [_mov_frames(o) for o in ents]
    change = set()
    for m in mov:
        change |= m
    if not change or not ents:
        return set()
    cov = []
    for i, o in enumerate(ents):
        pres = set(int(t) for t in o["t"])
        rel = change & pres
        cov.append(len(mov[i] & rel) / len(rel) if rel else 0.0)
    anchor = int(np.argmax(cov))
    o1 = ents[anchor]
    mov1 = mov[anchor]
    agent = {anchor}
    for i, o in enumerate(ents):
        if i == anchor:
            continue
        # an appendage (shadow, gripper part) moves in nearly ALL of
        # the anchor's moving frames. Coverage alone is jitter-
        # inflatable (an approaching gripper flickers the patient's
        # mask edge, measured absorbing carried blocks in 3 of 4
        # audited failures), so the decisive test is a VETO grounded
        # in physics: every manipulation begins with an approach, so
        # a patient has a long stationary run while the anchor works;
        # an appendage never sits out while the body moves.
        pres = set(int(t) for t in o["t"])
        rel1 = mov1 & pres
        acov = len(mov[i] & rel1) / len(rel1) if rel1 else 0.0
        if acov <= 0.6:
            continue
        seq = sorted(mov1)
        sits_out = False
        for s0 in range(0, max(len(seq) - 11, 0)):
            win = seq[s0:s0 + 12]
            present = sum(1 for t in win if t in pres)
            moving = sum(1 for t in win if t in mov[i])
            if present >= 10 and moving <= 3:
                sits_out = True
                break
        if not sits_out:
            agent.add(i)
    return agent


def depth_cached(ep):
    """Endpoint depth maps (Depth Pro, downsampled x4) per episode,
    or None when absent - every 3D fact degrades to its image-plane
    form gracefully."""
    f = CACHE / f"depth_{ep.name}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    frames = sorted(int(k[1:]) for k in z.files if k.startswith("d"))
    return dict(z=z, frames=frames)


def _depth_at(dc, t):
    if dc is None or not dc["frames"]:
        return None
    fi = min(dc["frames"], key=lambda x: abs(x - t))
    if abs(fi - t) > 8:
        return None
    return (np.asarray(dc["z"][f"d{fi}"], np.float32),
            float(dc["z"][f"f{fi}"]))


def _unproj(s, D, f):
    """Entity state (cx,cy,w,h px) -> 3D point + metric extent. The
    depth median is over the inner box; the downsampled map indexes
    by //4."""
    cx, cy, ext = s
    r = max(int(ext / 3) // 4, 1)
    y0, x0 = int(cy) // 4, int(cx) // 4
    H4, W4 = D.shape
    patch = D[max(y0 - r, 0):min(y0 + r + 1, H4),
              max(x0 - r, 0):min(x0 + r + 1, W4)]
    if patch.size == 0:
        return None
    z = float(np.median(patch))
    return (np.array([(cx - 320.0) * z / f, (cy - 240.0) * z / f, z]),
            ext * z / f)


def support_plane(ents, ag, dc):
    """The recording's own support structure: fit a plane to the 3D
    rest positions of its non-agent entities across the cached
    frames (RANSAC-lite: SVD fit, trim, refit). Normal oriented
    toward the agent's side - the mover works on the free side of
    the support. This is section 5's scene frame made literal, with
    zero world priors: the plane is wherever THIS recording's things
    rest."""
    if dc is None:
        return None
    pts = []
    for fi in dc["frames"]:
        Df = _depth_at(dc, fi)
        if Df is None:
            continue
        D, f = Df
        for i, o in enumerate(ents):
            if i in ag or not _resting(o, fi, -1):
                continue
            s = _state_at(o, fi, -1)
            if s is None:
                continue
            u = _unproj(s, D, f)
            if u is not None:
                pts.append(u[0])
    if len(pts) < 5:
        return None
    P = np.asarray(pts)
    for _ in range(2):
        c = P.mean(0)
        _, _, V = np.linalg.svd(P - c)
        n = V[-1]
        r = np.abs((P - c) @ n)
        keep = r <= max(2.0 * np.median(r), 1e-4)
        if keep.sum() < 5:
            break
        P = P[keep]
    c = P.mean(0)
    _, _, V = np.linalg.svd(P - c)
    n = V[-1]
    # orient toward the agent side
    aref = None
    for i in ag:
        o = ents[i]
        s = (float(np.median(o["cx"])), float(np.median(o["cy"])),
             max(float(np.median(o["w"])), float(np.median(o["h"])),
                 8.0))
        Df = _depth_at(dc, int(np.median(o["t"])))
        if Df is not None:
            u = _unproj(s, *Df)
            if u is not None:
                aref = u[0]
                break
    if aref is None:
        aref = np.zeros(3)          # camera side
    if float((aref - c) @ n) < 0:
        n = -n
    return c, n


def _state_at(o, t, side):
    """Nearest sample state; side<0 = at/before t, side>0 = at/after.
    Returns (cx, cy, ext) or None if the object has no sample there."""
    ts = o["t"]
    if side < 0:
        idx = np.where(ts <= t + 3)[0]
    else:
        idx = np.where(ts >= t - 3)[0]
    if not len(idx):
        return None
    i = idx[-1] if side < 0 else idx[0]
    if abs(int(ts[i]) - t) > 12:
        return None
    return (float(o["cx"][i]), float(o["cy"][i]),
            max(float(o["w"][i]), float(o["h"][i]), 8.0))


def _resting(o, t, side, win=8):
    """At rest = stationary across a STRICTLY ONE-SIDED window:
    before t for side<0, after t for side>0. A held block is
    stationary at a window edge (the arm pauses between events), but
    it moves again immediately - a one-sided look catches that; a
    symmetric window called a grasp-pause "rest" (measured: pick at
    the episode start read as ARRIVED). Short window: in a chained
    recording the next event begins ~1s after this one settles, and
    a 15-frame look caught the re-grasp as unrest (measured on every
    mid-episode stack)."""
    ts = np.asarray(o["t"])
    if side < 0:
        near = np.where((ts >= t - win) & (ts <= t + 2))[0]
    else:
        near = np.where((ts >= t - 2) & (ts <= t + win))[0]
    if len(near) < 3:
        return False
    cx, cy = np.array(o["cx"])[near], np.array(o["cy"])[near]
    ext = max(np.median(np.array(o["w"])[near]),
              np.median(np.array(o["h"])[near]), 8.0)
    return float(np.hypot(cx.max() - cx.min(),
                          cy.max() - cy.min())) < 0.4 * ext


def moment(rows, a, b, res=None, F=None, cap=None, dc=None,
           want_rgb=None):
    """WHAT HAPPENED in [a,b], as a relational story about the
    most-changed non-agent particular (the patient), per PROBLEM.md
    section 5's three generic frames - its own past, co-present
    entities pairwise, the scene - with every distance in units of
    the participants' own extents. No height axis, no names:
      rest_b/rest_a  at rest when the window opens / closes
      disp           net displacement, own extents
      arc            max deviation from the endpoint chord during
                     motion: a carried thing leaves its support and
                     arcs; a pushed thing slides along it (chord ~ 0)
      adjB/adjA      at-rest adjacency to ANOTHER resting entity
                     (stacked-on vs alone), before / after
      agB/agA        nearest agent-role proximity at open / close:
                     a released thing is left alone (agent departs);
                     a held thing keeps its agent (this is how "held
                     at end" is told from "resting at end" without
                     any world axis)
      absent         fraction of the window with no mask - in
                     tracker-world absence IS the held/occluded fact
    Nothing pooled, no cosine anywhere."""
    ents = entities(rows, res)
    ag = roles2(ents)
    T = next(r["T"] for r in rows if r["role"] == "_ver")
    objs = [(i, o) for i, o in enumerate(ents) if i not in ag]
    if not objs:
        return None

    def _ext(o):
        return max(float(np.median(o["w"])),
                   float(np.median(o["h"])), 8.0)

    # the patient: the non-agent particular that CHANGED most.
    # Change is position RANGE in own extents (never path-sum: a
    # jittering fragment accumulates path while going nowhere), OR -
    # equally - a rest-anchored DISAPPEARANCE: a fully-enclosed grip
    # leaves no visible displacement at all, and in tracker-world
    # that absence IS the taking (measured: 12 picks with the block
    # occluded at the grip read range 0.1-0.3 and were refused)
    best, patient = 0.0, None
    for i, o in objs:
        ts = np.asarray(o["t"])
        In = (ts >= a - 3) & (ts <= b + 3)
        if In.sum() >= 3:
            cx = np.asarray(o["cx"])[In]
            cy = np.asarray(o["cy"])[In]
            rng = float(np.hypot(cx.max() - cx.min(),
                                 cy.max() - cy.min())) / _ext(o)
        else:
            rng = 0.0
        span = max(b - a, 1)
        n_in = int(((ts >= a) & (ts <= b)).sum())
        absent_frac = max(0.0, 1.0 - n_in / (span + 1.0))
        edge_rest = _resting(o, a, -1) or _resting(o, b, +1)
        qual = rng
        if absent_frac > 0.25 and edge_rest:
            qual = max(qual, 2.0)
        if qual > best:
            best, patient = qual, (i, o)
    if want_rgb is not None:            # EVAL-ONLY oracle binding
        for i, o in objs:
            if np.abs(np.median(o["rgb"], 0)
                      - want_rgb).sum() < 150:
                patient, best = (i, o), max(best, 1.0)
                break
    if patient is None or best < 0.6:
        return None
    pi, p = patient
    ext = _ext(p)
    ts = np.asarray(p["t"])

    rest_b = _resting(p, a, -1)
    rest_a = _resting(p, b, +1) \
        or (cap is not None and _resting(p, min(b + 6, cap - 8), +1))
    sb = _state_at(p, a, -1)
    sa = _state_at(p, b, +1)
    disp = 0.0
    if sb and sa:
        disp = float(np.hypot(sa[0] - sb[0], sa[1] - sb[1])) / ext

    # arc: own-past frame. Deviation from the endpoint chord over the
    # in-window samples; a slide stays on its chord, a carry leaves it
    arc = 0.0
    if sb and sa:
        In = np.where((ts >= a - 3) & (ts <= b + 3))[0]
        if len(In) >= 3:
            p0 = np.array([sb[0], sb[1]])
            p1 = np.array([sa[0], sa[1]])
            ch = p1 - p0
            L = float(np.hypot(*ch))
            pts = np.stack([np.asarray(p["cx"])[In],
                            np.asarray(p["cy"])[In]], 1) - p0
            if L > 1e-6:
                perp = np.abs(pts[:, 0] * ch[1] - pts[:, 1] * ch[0]) \
                    / L
            else:
                perp = np.hypot(pts[:, 0], pts[:, 1])
            arc = float(perp.max()) / ext

    # window absence: in tracker-world a missing mask is the
    # held/occluded fact itself, never flicker
    span = max(b - a, 1)
    n_in = int(((ts >= a) & (ts <= b)).sum())
    absent = max(0.0, 1.0 - n_in / (span + 1.0))

    plane = support_plane(ents, ag, dc)

    def _p3(s, t):
        Df = _depth_at(dc, t)
        if Df is None or s is None:
            return None
        return _unproj((s[0], s[1], s[2]), *Df)

    def elev(t, side, pos=None):
        # height above the recording's own support plane, in the
        # patient's METRIC extent: the fact no image row can carry -
        # a held-still block at recording end is ELEVATED; a resting
        # one is not, whatever the view angle
        if plane is None:
            return 0.0
        s = pos if pos is not None else _state_at(p, t, side)
        u = _p3(s, t)
        if u is None:
            return 0.0
        c, n = plane
        h = float((u[0] - c) @ n) / max(u[1], 1e-3)
        # /1.5 not /4: monocular depth compresses metric scale on
        # these frames (measured 10cm true -> 2.9cm predicted), so a
        # physical 3-extent lift reads ~0.35 extents; the tighter
        # normalizer spreads held-vs-landed over the fact's range
        return float(np.clip(h / 1.5, -0.3, 1.0))

    def adj(t, side, pos=None):
        # at-rest CONTACT with another resting entity of comparable
        # scale: box gap (not centre distance - a huge region's
        # centre is meaningless), in units of the patient. Scenery
        # bands dwarf the patient; "next to" is only defined between
        # things of the same order (units-of-the-things-themselves)
        s = pos if pos is not None else _state_at(p, t, side)
        if not s:
            return 0.0
        val = 0.0
        for j, o2 in objs:
            if j == pi or _ext(o2) > 3.5 * ext:
                continue
            s2 = None
            if _resting(o2, t, side):
                s2 = _state_at(o2, t, side)
            if s2 is None and side > 0:
                # a partner whose track ENDED (masklet merged into
                # the arriving patient) does not stop having existed:
                # its last known rest is still the other end of the
                # relation (the L3 biography). Only for the after-
                # side - before the window, absence is just absence
                ts2 = np.asarray(o2["t"])
                if len(ts2) >= 5 and ts2[-1] < t - 3 \
                        and _resting(o2, int(ts2[-1]), -1):
                    s2 = _state_at(o2, int(ts2[-1]), -1)
            if s2 is None:
                continue
            u1, u2 = _p3(s, t), _p3(s2, t)
            if u1 is not None and u2 is not None:
                # 3D contact in the entities' own metric units - the
                # image-plane gap was measured BLIND (median 0.00 for
                # both true-contact and true-separate pairs); depth-
                # lifted distance separates them 5.1cm vs 15.9cm
                d3 = float(np.linalg.norm(u1[0] - u2[0]))
                dn = d3 / max(0.5 * (u1[1] + u2[1]), 1e-3)
                val = max(val, float(np.exp(-max(dn - 1.2, 0.0))))
            else:
                gx = max(0.0,
                         abs(s[0] - s2[0]) - 0.5 * (s[2] + s2[2]))
                gy = max(0.0,
                         abs(s[1] - s2[1]) - 0.5 * (s[2] + s2[2]))
                gap = float(np.hypot(gx, gy)) / ext
                val = max(val, float(np.exp(-gap / 0.5)))
        return val

    def agent_near(t, side):
        s = _state_at(p, t, side)
        if not s:
            # absent = enclosed by the agent: maximal proximity
            return 1.0
        val = 0.0
        for j in ag:
            o2 = ents[j]
            s2 = _state_at(o2, t, side)
            if not s2:
                continue
            d = np.hypot(s[0] - s2[0], s[1] - s2[1]) \
                / max(0.5 * (s[2] + s2[2]), 1.0)
            val = max(val, float(np.exp(-max(d - 1.2, 0.0))))
        return val

    def _agent_dist(t):
        s = _state_at(p, t, +1) or _state_at(p, t, -1)
        if not s:
            return None
        best = None
        for j in ag:
            o2 = ents[j]
            s2 = _state_at(o2, t, +1) or _state_at(o2, t, -1)
            if not s2:
                continue
            d = float(np.hypot(s[0] - s2[0], s[1] - s2[1])) \
                / max(0.5 * (s[2] + s2[2]), 1.0)
            best = d if best is None else min(best, d)
        return best

    # HELD vs RELEASED at the end, with no world axis and no need
    # for a gripper track: after the patient's LAST MOTION, a
    # released thing is left (agent separation GROWS as it retreats);
    # a held thing keeps its agent (separation stays). Instantaneous
    # proximity fails here - the grip point need not be a track of
    # its own (measured on the composed gripper: nearest agent
    # member sat 3.4 extents from a genuinely held block).
    mv = sorted(_mov_frames(p))
    mv_in = [t for t in mv if a - 3 <= t <= b + 3]
    t_last = mv_in[-1] if mv_in else b
    d0 = _agent_dist(t_last)
    d1 = _agent_dist(min(t_last + 25, T - 1))
    if d0 is None or d1 is None:
        ag_stay = 0.5              # unknown, neutral
    else:
        ag_stay = float(np.exp(-max(d1 - d0 - 0.3, 0.0) / 0.8))
    if not _state_at(p, min(b + 12, T - 1), +1):
        ag_stay = 1.0              # ends enclosed by the agent

    # the end look must not cross into the NEXT event: a fixed b+12
    # landed inside the following re-grasp in chained recordings and
    # read its wobble as unrest (measured on every stack inside an
    # unstack episode). The event list itself provides the cap - the
    # spans are the given retrieval units, nothing is leaked. (A
    # settle-run search here was tried and REVERTED: it forced
    # rest_a=1 for held-still picks, the exact trap the one-sided
    # window exists to avoid.)
    b2 = min(b + 12, T - 1)
    if cap is not None:
        b2 = max(min(b2, cap - 10), b + 1)

    # MASKLET THEFT correction, pixel-arbitrated: at release the
    # patient's mask can jump onto the retreating gripper (measured:
    # a stacked block "flying" up-right after its event, reading
    # rest_a=0 / adjA=0). If the track has left its last in-window
    # rest but the patient's COLOUR is still at that rest, the track
    # was stolen - the thing never moved. A genuine re-grasp leaves
    # no colour behind, so the two cases separate physically.
    if F is not None:
        In2 = np.where((ts >= a - 3) & (ts <= b2 + 3))[0]
        p_rest = None
        run = []
        for i2 in In2:
            if run and np.hypot(
                    p["cx"][i2] - np.median([p["cx"][r] for r in run]),
                    p["cy"][i2] - np.median([p["cy"][r] for r in run])
                    ) > 0.35 * ext:
                if len(run) >= 4:
                    p_rest = (float(np.median(
                        [p["cx"][r] for r in run])),
                        float(np.median([p["cy"][r] for r in run])))
                run = []
            run.append(i2)
        if len(run) >= 4:
            p_rest = (float(np.median([p["cx"][r] for r in run])),
                      float(np.median([p["cy"][r] for r in run])))
        s_now = _state_at(p, b2, +1) or _state_at(p, b2, -1)
        prgb_ = np.median(p["rgb"], 0)
        if p_rest is not None and s_now is not None \
                and np.hypot(s_now[0] - p_rest[0],
                             s_now[1] - p_rest[1]) > 1.5 * ext \
                and _colour_at(F, b2, p_rest[0], p_rest[1],
                               ext, prgb_):
            sa2 = (p_rest[0], p_rest[1], ext)
            disp2 = disp
            if sb:
                disp2 = float(np.hypot(sa2[0] - sb[0],
                                       sa2[1] - sb[1])) / ext
            facts = dict(rest_b=float(rest_b), rest_a=1.0,
                         disp=min(disp2, 8.0) / 8.0,
                         arc=min(arc, 4.0) / 4.0,
                         adjB=adj(a, -1),
                         adjA=adj(b2, +1, pos=sa2),
                         agB=agent_near(a, -1), agA=0.5,
                         absent=absent,
                         elevB=elev(a, -1),
                         elevA=elev(b2, +1, pos=sa2),
                         tail=min((T - 1 - b) / 25.0, 1.0))
            return dict(facts=facts, prgb=prgb_)

    facts = dict(rest_b=float(rest_b), rest_a=float(rest_a),
                 disp=min(disp, 8.0) / 8.0,
                 arc=min(arc, 4.0) / 4.0,
                 adjB=adj(a, -1), adjA=adj(b2, +1),
                 agB=agent_near(a, -1), agA=ag_stay,
                 absent=absent,
                 elevB=elev(a, -1), elevA=elev(b2, +1),
                 tail=min((T - 1 - b) / 25.0, 1.0))
    return dict(facts=facts, prgb=np.median(p["rgb"], 0))


def rel_series(rows, a, b, res=None, n=16, want_rgb=None):
    """The trajectory ITSELF, relationally: pairwise distance time
    series over the window - patient to agent, patient to nearest
    other entity, patient to its own start - plus speed. Scalars over
    time in the participants' own extents: no axis of any kind is
    defined, nothing is pooled, and the motion the endpoint facts
    discarded is what gets compared. Absence reads as distance zero
    to the agent (enclosed)."""
    ents = entities(rows, res)
    ag = roles2(ents)
    objs = [(i, o) for i, o in enumerate(ents) if i not in ag]
    if not objs or not ag:
        return None
    best, patient = 0.0, None
    for i, o in objs:
        ts = np.asarray(o["t"])
        In = (ts >= a - 3) & (ts <= b + 3)
        if In.sum() < 3:
            continue
        cx, cy = np.asarray(o["cx"])[In], np.asarray(o["cy"])[In]
        ext_ = max(float(np.median(o["w"])),
                   float(np.median(o["h"])), 8.0)
        rng = float(np.hypot(cx.max() - cx.min(),
                             cy.max() - cy.min())) / ext_
        span = max(b - a, 1)
        n_in = int(((ts >= a) & (ts <= b)).sum())
        af = max(0.0, 1.0 - n_in / (span + 1.0))
        q = max(rng, 2.0 if af > 0.25 else rng)
        if q > best:
            best, patient = q, (i, o)
    if want_rgb is not None:            # EVAL-ONLY oracle binding
        for i, o in objs:
            if np.abs(np.median(o["rgb"], 0)
                      - want_rgb).sum() < 150:
                patient, best = (i, o), max(best, 1.0)
                break
    if patient is None or best < 0.6:
        return None
    pi, p_ = patient
    ext = max(float(np.median(p_["w"])), float(np.median(p_["h"])),
              8.0)
    ts = np.asarray(p_["t"], float)

    def interp(o, tq):
        ot = np.asarray(o["t"], float)
        return (np.interp(tq, ot, np.asarray(o["cx"], float)),
                np.interp(tq, ot, np.asarray(o["cy"], float)))

    tq = np.linspace(a - 2, b + 4, n)
    px, py = interp(p_, tq)
    present = np.array([bool((np.abs(ts - t) <= 6).any())
                        for t in tq])
    # nearest agent member per step
    s_ag = np.zeros(n)
    for k in range(n):
        if not present[k]:
            s_ag[k] = 0.0            # enclosed by the agent
            continue
        dbest = 9.0
        for j in ag:
            axk, ayk = interp(ents[j], tq[k:k + 1])
            e2 = max(float(np.median(ents[j]["w"])),
                     float(np.median(ents[j]["h"])), 8.0)
            d = float(np.hypot(px[k] - axk[0], py[k] - ayk[0])) \
                / max(0.5 * (ext + e2), 1.0)
            dbest = min(dbest, d)
        s_ag[k] = min(dbest, 6.0) / 6.0
    # nearest other non-agent entity (comparable scale)
    s_ot = np.full(n, 1.0)
    for j, o2 in objs:
        if j == pi:
            continue
        e2 = max(float(np.median(o2["w"])),
                 float(np.median(o2["h"])), 8.0)
        if e2 > 3.5 * ext:
            continue
        ox, oy = interp(o2, tq)
        d = np.hypot(px - ox, py - oy) / max(0.5 * (ext + e2), 1.0)
        s_ot = np.minimum(s_ot, np.minimum(d, 6.0) / 6.0)
    # displacement from own start + speed
    s_disp = np.minimum(np.hypot(px - px[0], py - py[0]) / ext,
                        8.0) / 8.0
    dt = max(float(tq[1] - tq[0]), 1e-6)
    sp = np.hypot(np.diff(px), np.diff(py)) / ext / dt
    s_sp = np.concatenate([[0.0], np.minimum(sp, 1.5) / 1.5])
    return np.stack([s_ag, s_ot, s_disp, s_sp])


def traj_agree(S1, S2):
    if S1 is None or S2 is None:
        return 0.25
    w = np.array([1.0, 1.0, 0.7, 0.7])
    d = np.abs(S1 - S2).mean(1)
    return float(np.exp(-2.5 * float((w * d).sum() / w.sum())))


FKEYS = ("rest_b", "rest_a", "disp", "arc", "adjB", "adjA",
         "agB", "agA", "absent", "elevB", "elevA")


def agree(f1, f2):
    """Fact agreement: the rest-story bits must match (soft floor);
    graded facts compare by closeness. NOT cosine over a pooled blob
    - each dimension is a claim about what happened, and the product
    form means one flatly contradicted claim sinks the pair however
    well the rest align.

    CODA facts (what held after the window: rest_a, adjA, agA) are
    only as reliable as the tail evidence behind them - a recording
    that ends 0.5s after the event asserts its after-state on ~5
    frames. Their penalties scale with the JOINT tail (evidence
    quantity, never a class): a terminal pick and a mid-episode pick
    honestly disagree on rest_a, but the terminal one barely has a
    coda to disagree with (measured: rest_a split 36/108 inside the
    pick class along exactly the terminal/mid line)."""
    # tail-weighted coda penalties were TRIED AND REVERTED here:
    # softening rest_a/adjA for short-tail pairs recovered terminal
    # pick-pick agreement but let terminal picks collide with
    # terminal places/stacks, a net loss (0.499 -> 0.469 measured)
    s = 1.0
    for k in ("rest_b", "rest_a"):
        s *= 1.0 if f1[k] == f2[k] else 0.25
    for k in ("arc", "adjB", "adjA", "absent"):
        s *= 1.0 - 0.7 * abs(f1[k] - f2[k])
    # DIFFERENTIAL elevation, not absolutes: per-episode plane bias
    # and per-frame depth scale cancel in (elevA - elevB); the
    # absolutes carried std ~0.45 against mean gaps of ~0.2 and
    # dragged every class down (measured 0.513 -> 0.458)
    # ELEV_W=0 pending a depth read that survives the support prior:
    # monocular depth erases the LIFTED state at this object scale
    # (measured: true falls read -0.26, true rises read +0.00 - the
    # prior assumes things rest on surfaces, which is exactly the
    # fact being asked). Falls-only is not worth two 0.5-weight slots
    d1 = f1.get("elevA", 0.0) - f1.get("elevB", 0.0)
    d2_ = f2.get("elevA", 0.0) - f2.get("elevB", 0.0)
    s *= 1.0 - ELEV_W * abs(d1 - d2_)
    for k in ("disp", "agB", "agA"):
        s *= 1.0 - 0.3 * abs(f1[k] - f2[k])
    return s


USE_R2 = False           # pass-2 refined caches: better TRACKING
                         # (gate 0.870->0.904, verified on film) but
                         # the junk they add steals patient selection
                         # (0.720->0.641) and three filter designs
                         # measured dead (chroma-transient: blind to
                         # semi-static; DINO-transient: same; DINO
                         # ring-contrast: no separation on flat-shaded
                         # surfaces). Opt in once a junk-robust entity
                         # layer exists.


def cache_for(name, epname):
    r2 = CACHE / f"{name}_{epname}_r2.npy"
    if USE_R2 and r2.exists():
        return r2
    return CACHE / f"{name}_{epname}.npy"


def load_events(corpus):
    name = Path(corpus).name
    evs = []
    from tqdm import tqdm
    eps = sorted((ROOT / corpus).glob("ep*"))
    for ep in tqdm(eps, unit="ep", desc="facts"):
        f = cache_for(name, ep.name)
        if not f.exists():
            continue
        rows = np.load(f, allow_pickle=True)
        res = residue_cached(rows, ep)
        dc = depth_cached(ep)
        view = next((x.get("view") for x in rows
                     if x["role"] == "_ver"), None)
        F = ent.read_frames(ep / view if view
                            else sorted(ep.glob("cam*.mp4"))[0])
        meta = json.loads((ep / "meta.json").read_text())
        arm = meta.get("arm", "?")          # EVAL-ONLY: slices the
        evts = [e for e in meta["events"] if e["ok"]]
        for ei, e in enumerate(evts):       # benchmark, never scores
            a, b = int(e["t0"] * FPS), int(e["t1"] * FPS)
            if b - a < 4:
                continue
            cap = int(evts[ei + 1]["t0"] * FPS) \
                if ei + 1 < len(evts) else None
            wr = None
            if ORACLE_BIND:              # EVAL-ONLY diagnostic mode
                bl = {x["name"]: 255 * np.array(x["rgba"][:3])
                      for x in meta["blocks"]}
                wr = bl.get(e["block"])
            m = moment(rows, a, b, res, F=F, cap=cap, dc=dc,
                       want_rgb=wr)
            if m is not None:
                m["traj"] = rel_series(rows, a, b, res, want_rgb=wr)
            evs.append((e["prim"], f"{name}/{ep.name}", m, arm))
    return evs


def agree_q(fq, fc, w):
    """QUERY-IMPLIED weighting (the L7 spec, finally): the query
    decides which facts matter. w[k] in [0,1] is the query's
    assertiveness on fact k - its distance from the corpus marginal
    in corpus stds. A fact the query does not assert is not allowed
    to punish candidates (the global tail-weighting attempt failed
    because it softened facts for BOTH sides; per-query, only the
    uninformative QUERY drops its own claim). Label-free: the
    marginals are the corpus's own."""
    s = 1.0
    for k in ("rest_b", "rest_a"):
        if fq[k] != fc[k]:
            s *= 0.25 ** w[k]
    for k in ("arc", "adjB", "adjA", "absent"):
        s *= 1.0 - 0.7 * w[k] * abs(fq.get(k, 0.0) - fc.get(k, 0.0))
    dq = fq.get("elevA", 0.0) - fq.get("elevB", 0.0)
    dcv = fc.get("elevA", 0.0) - fc.get("elevB", 0.0)
    wd = 0.5 * (w.get("elevA", 0.0) + w.get("elevB", 0.0))
    s *= 1.0 - ELEV_W * wd * abs(dq - dcv)
    for k in ("disp", "agB", "agA"):
        s *= 1.0 - 0.3 * w[k] * abs(fq[k] - fc[k])
    return s


_QKEYS = ("rest_b", "rest_a", "disp", "arc",
          "adjB", "adjA", "agB", "agA", "absent", "elevB", "elevA")


def query_weights(evs):
    """Per-query fact assertiveness vs the corpus marginal."""
    ok = [i for i in range(len(evs)) if evs[i][2] is not None]
    mu, sd = {}, {}
    for k in _QKEYS:
        v = np.array([evs[i][2]["facts"].get(k, 0.0) for i in ok])
        mu[k], sd[k] = float(v.mean()), float(v.std()) + 1e-6
    W = {}
    for i in ok:
        f = evs[i][2]["facts"]
        W[i] = {k: min(abs(f.get(k, 0.0) - mu[k]) / sd[k], 1.5)
                / 1.5 for k in _QKEYS}
    return W


def _agree_matrix(evs):
    n = len(evs)
    S = np.zeros((n, n), np.float32)
    ok = [i for i in range(n) if evs[i][2] is not None]
    for ii, i in enumerate(ok):
        fi = evs[i][2]["facts"]
        for j in ok[ii + 1:]:
            s = agree(fi, evs[j][2]["facts"])
            S[i, j] = S[j, i] = s
    return S


def _l8_scores(S, i, forbid, alpha=0.55, iters=8, topk=12):
    """L8 corpus geometry on the fact-agreement graph: CSLS hubness
    correction + short diffusion. Axis-agnostic - it never looks at
    the facts, only at the similarity structure the corpus itself
    induces (the validated +50%-AP mechanism, re-applied). Columns of
    the query's own episode are cut from the walk so leave-episode-
    out stays honest."""
    n = len(S)
    r = np.zeros(n, np.float32)
    for j in range(n):
        row = np.sort(S[j])[::-1]
        r[j] = row[1:topk + 1].mean()
    C = 2.0 * S - r[None, :] - r[:, None]
    W = np.exp((C - C.max()) / 0.35)
    W[:, forbid] = 0.0
    W[forbid, :] = 0.0
    np.fill_diagonal(W, 0.0)
    # sparsify to top-k neighbourhoods, then row-normalize
    for j in range(n):
        row = W[j]
        if row.max() <= 0:
            continue
        thr = np.sort(row)[::-1][min(topk, n - 1)]
        row[row < thr] = 0.0
    Z = W.sum(1, keepdims=True)
    Z[Z == 0] = 1.0
    W = W / Z
    s = S[i].copy()
    s[forbid] = 0.0
    f = s.copy()
    for _ in range(iters):
        f = (1 - alpha) * s + alpha * (W @ f)
    return f


def _rank_yield(evs, qidx, cidx, l8=False, S=None, qw=None):
    """yield at k = support for query set qidx over corpus cidx,
    leave-episode-out. Returns per-prim lists."""
    prims = np.array([e[0] for e in evs])
    eps_ = np.array([e[1] for e in evs])
    per = defaultdict(list)
    cset = np.zeros(len(evs), bool)
    cset[cidx] = True
    if S is None:
        S = _agree_matrix(evs)
    none = np.array([e[2] is None for e in evs])
    for i in qidx:
        if evs[i][2] is None:
            continue
        mask = cset & (eps_ != eps_[i]) & ~none
        sup = int((prims[mask] == prims[i]).sum())
        if not sup:
            continue
        if qw is not None:
            scores = np.full(len(evs), -1.0)
            for j in np.where(mask)[0]:
                scores[j] = agree_q(evs[i][2]["facts"],
                                    evs[j][2]["facts"], qw[i])
        elif l8:
            forbid = np.where(~mask)[0]
            scores = _l8_scores(S, i, forbid)
        else:
            scores = S[i].copy()
        order = np.argsort(-scores[mask])
        lab = (prims[mask] == prims[i])[order[:sup]]
        per[prims[i]].append(int(lab.sum()) / sup)
    return per


def _print_yield(title, per):
    ally = [y for p in per for y in per[p]]
    print(f"\n--- {title} ---")
    for p in PRIMS:
        if per[p]:
            print(f"  {p:9s} n {len(per[p]):4d}  "
                  f"yield {np.mean(per[p]):.3f}")
    print(f"  {'ALL':9s} n {len(ally):4d}  "
          f"yield {np.mean(ally):.3f}" if ally else "  (empty)")


def yield_bench(corpus):
    evs = load_events(corpus)
    prims = np.array([e[0] for e in evs])
    n = len(evs)
    none = sum(1 for e in evs if e[2] is None)
    print(f"{n} events ({none} with no moment) | "
          + "  ".join(f"{p} {c}" for p, c in Counter(prims).items()))
    # per-class fact table (EVAL-ONLY diagnosis: means +- std)
    print(f"\n{'':9s}" + "".join(f"{k:>12s}" for k in FKEYS))
    for pr in PRIMS:
        rows_ = [evs[i][2]["facts"] for i in range(n)
                 if prims[i] == pr and evs[i][2] is not None]
        if not rows_:
            continue
        line = f"{pr:9s}"
        for k in FKEYS:
            v = [r.get(k, 0.0) for r in rows_]
            line += f" {np.mean(v):5.2f}~{np.std(v):4.2f}"
        print(line)
    allidx = np.arange(n)
    S = _agree_matrix(evs)
    per = _rank_yield(evs, allidx, allidx, S=S)
    _print_yield("YIELD at k=support (happened-facts, single view)",
                 per)
    per8 = _rank_yield(evs, allidx, allidx, l8=True, S=S)
    _print_yield("YIELD + L8 (CSLS + diffusion on the fact graph)",
                 per8)
    qw = query_weights(evs)
    perq = _rank_yield(evs, allidx, allidx, S=S, qw=qw)
    _print_yield("YIELD + QUERY-IMPLIED WEIGHTING (L7)", perq)
    n_ = len(evs)
    St = np.zeros((n_, n_), np.float32)
    ok = [i for i in range(n_) if evs[i][2] is not None]
    for ii, i in enumerate(ok):
        for j in ok[ii + 1:]:
            t = traj_agree(evs[i][2].get("traj"),
                           evs[j][2].get("traj"))
            St[i, j] = St[j, i] = t
    perT = _rank_yield(evs, allidx, allidx, S=St)
    _print_yield("YIELD trajectory-series ONLY", perT)
    Sc = np.sqrt(np.maximum(S, 1e-6) * np.maximum(St, 1e-6))
    perC = _rank_yield(evs, allidx, allidx, S=Sc)
    _print_yield("YIELD facts x trajectory (geometric mean)", perC)
    # cross-embodiment slice (the pen-test proxy): each arm's events
    # query a store holding only the OTHER arms' episodes
    arms = np.array([e[3] for e in evs])
    per_x = defaultdict(list)
    for aname in sorted(set(arms)):
        qidx = np.where(arms == aname)[0]
        cidx = np.where(arms != aname)[0]
        px = _rank_yield(evs, qidx, cidx, l8=True, S=S)
        for k, v in px.items():
            per_x[k] += v
    _print_yield("CROSS-EMBODIMENT yield (query arm vs other arms)",
                 per_x)


def gate(corpus):
    """Entity gate (EVAL-ONLY truth): true block visible as a tracked
    non-agent entity with the right colour, per event."""
    name = Path(corpus).name
    tot = hit = 0
    for ep in sorted((ROOT / corpus).glob("ep*")):
        f = cache_for(name, ep.name)
        if not f.exists():
            continue
        rows = np.load(f, allow_pickle=True)
        ents = entities(rows, residue_cached(rows, ep))
        ag = roles2(ents)
        objs = [o for i, o in enumerate(ents) if i not in ag]
        meta = json.loads((ep / "meta.json").read_text())
        rgba = {b["name"]: 255 * np.array(b["rgba"][:3])
                for b in meta["blocks"]}
        for e in meta["events"]:
            if not e["ok"]:
                continue
            tgt = rgba[e["block"]]
            a, b = int(e["t0"] * FPS), int(e["t1"] * FPS)
            tot += 1
            for o in objs:
                near = np.abs(np.asarray(o["t"])
                              - 0.5 * (a + b)) < 5 * CAD
                if not near.any():
                    continue
                med = np.median(np.asarray(o["rgb"])[near], 0)
                if np.abs(med - tgt).sum() < 180:
                    hit += 1
                    break
    print(f"{name}: true block tracked near its event: "
          f"{hit}/{tot} = {hit / max(tot, 1):.3f}")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build(sys.argv[sys.argv.index("--build") + 1])
    if "--gate" in sys.argv:
        gate(sys.argv[sys.argv.index("--gate") + 1])
    if "--yield" in sys.argv:
        yield_bench(sys.argv[sys.argv.index("--yield") + 1])
