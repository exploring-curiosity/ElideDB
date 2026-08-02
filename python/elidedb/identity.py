"""OBJECT IDENTITY on the write path, text-free, riding on tracks.

Identity is a property of a physical object, not of a name and not of a
category. The same spoon across every demo gets one id. A spatula gets
a different id. A red pepper and a green pepper get different ids, even
though every text-aligned encoder calls them both "pepper".

Nothing here turns pixels into language. Regions come from a
class-agnostic segmenter, identity from a ReID encoder; both vision
only, both real time. This is separate from objects.py, which proposes
regions for SEMANTIC retrieval - that channel answers "does this look
like a cloth", this one answers "is this the same object again".

CONTINUITY IS THE ARCHITECTURE
------------------------------
The earlier version sampled 3 frames per episode and asked the gallery
about every crop independently. That is the wrong question asked ~500
times: it made identity a threshold problem, and the threshold had no
good setting - 0.55 built attractor ids that swallowed a green box, a
broccoli, a toy mouse and a carrot; 0.85 fragmented to 1.15 sightings
per object. Neither end works because a single crop of an object from
one angle simply is not enough evidence.

A tracker answers it for free instead. While an object stays visible,
IoU + Kalman association says "same object" from GEOMETRY, at 0.25
ms/frame, with no appearance model and no threshold. The object store
is consulted only when continuity BREAKS - a new track appears - and
then it is asked once, about a descriptor pooled over the whole track,
not once per frame about one view.

Measured over 24 episodes / 883 frames of Bridge:

    median track length      24-48 frames (episodes are ~37)
    singleton tracks         5%
    object-store uploads     19, against 415 tracked detections
                             = 95% fewer questions asked

That 95% is the design. Fewer questions, each with far more evidence
behind it.

COST, and where it goes
-----------------------
Detection is STATELESS, so it batches across episode boundaries;
association is STATEFUL, so it runs per episode, sequentially. Splitting
them is what makes every-frame tracking affordable - ultralytics' own
track() ties them together and pays batch-size-1 prices for both.

    per-frame track() calls          15.8 ms/frame
    batched track() (list per ep)     9.3
    detect batched + associate        4.8   <- this module
      of which detect  (fp16, 448)    4.16
      of which associate              0.13
      of which reid                   0.50  (was 4.1 per FRAME)

At 18,000 frames per hour of video that is 1.43 min/hour marginal,
against a 1 min/hour write budget of which 0.23 is already spent by the
existing pass. STATED STRAIGHT: identity does not fit the budget. What
riding on tracks bought is the ReID half, which used to dominate and is
now 0.1 min/hour; the detector is the entire remaining cost, so closing
the gap is a detector question, not a design one.

fp16 is worth 14% of detect (4.94 -> 4.16 ms/frame) with the fitted cut
and track count unchanged. That took interleaved runs to establish - a
first attempt compared an fp16 run against fp32 runs taken earlier in a
long session and read the thermal drift as fp16 being SLOWER.

DEVICE. Defaults to the GPU, and this is now measured rather than
assumed. The old 3-frames-per-episode path put CPU and MPS within 5%
(19.3 vs 20.3 ms/frame) because per-call overhead dominated a batch of
three. On real batches the gap is 3.7x - mps 6.7 ms/frame vs cpu 24.8
at imgsz 640 - and the CPU gets WORSE with batching (93.9 ms/frame at
batch 16) as threads contend. Frames are already resident on the GPU
for the FDNN-V pass, and on CUDA a host round trip per frame costs more
than the inference. CPU is the fallback, never the default.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

_M: dict = {}

# WEIGHTS LIVE IN models/weights, not the repo root. ultralytics
# downloads into the CWD by default, which is how 86 MB of checkpoints
# ended up beside the README with 62 MB of it committed to git. _weights
# resolves a bare name against models/weights and falls back to the bare
# name so a missing file still auto-downloads - into models/weights,
# because that is where we then look.
def _weights(name):
    from pathlib import Path as _P
    d = _P(__file__).resolve().parents[2] / "models" / "weights"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    return str(p) if p.exists() else str(d / name)


SEG_MODEL = os.environ.get("ELIDEDB_SEG") or _weights("yolo11n-seg.pt")
REID_MODEL = os.environ.get("ELIDEDB_REID") or _weights("yolo26n-reid.onnx")
# imgsz 448 rather than 384, deliberately paying 0.5 ms/frame for it.
# 384 is cheaper (3.76 vs 4.23 ms/frame with fp16) and finds MORE boxes
# (2.63 vs 2.44 per frame), but the extra boxes are marginal ones: its
# proven-different pairs reach 0.757 against 0.555 at 448, i.e. noisier
# negatives, and it recovered 72 multi-episode objects against 81. A
# cheaper size that degrades the identity is not cheaper.
DET_SZ = int(os.environ.get("ELIDEDB_DET_SZ", "448"))
# free: same 2.44 boxes/frame, 4.74 -> 4.23 ms/frame
DET_FP16 = os.environ.get("ELIDEDB_DET_FP16", "1") not in ("0", "")
# detection is stateless, so the batch may cross episode boundaries;
# 3.87 ms/frame at ~37 (one episode) against 3.32 at 64
DET_BATCH = int(os.environ.get("ELIDEDB_DET_BATCH", "64"))
# Global motion compensation estimates camera movement before
# association. Default OFF from MEASUREMENT, not from an assumption
# about this corpus: with sparseOptFlow, median track length was 47 at
# imgsz 448 and 38 at 640; with none, 48 and 38 - identical - while the
# optical flow cost 1.8 ms/frame, 7x the entire association step. A
# corpus with a moving camera must re-measure and will likely want it
# back; that is what the knob is for.
GMC = os.environ.get("ELIDEDB_GMC", "none")
# how many views of a track to embed. The pooled descriptor is what the
# object store sees, so this trades evidence against ReID calls.
EXEMPLARS = int(os.environ.get("ELIDEDB_OBJ_VIEWS", "4"))

# "same physical object" cut. Not a constant to be tuned by hand - see
# calibrate(), which fits it from the corpus being ingested. This value
# is only the fallback for a corpus too small to fit on.
MATCH = float(os.environ.get("ELIDEDB_OBJ_MATCH", "0.85"))
# quantile of the free-negative distribution used as the cut
CAL_Q = float(os.environ.get("ELIDEDB_OBJ_CAL_Q", "99.5"))
# Adding an exemplar WIDENS the id's acceptance region, because matching
# takes a max over the gallery. Unchecked that is a runaway: absorb ->
# widen -> absorb more. A new view is admitted only if it is close to
# EVERY exemplar already held, which keeps an id tight instead of
# letting it sprawl across a chain of intermediate appearances.
MAX_EXEMPLARS = 5
ADMIT = float(os.environ.get("ELIDEDB_OBJ_ADMIT", "0.70"))
MIN_SIDE = 16
MAX_AREA = 0.5


def device():
    d = os.environ.get("ELIDEDB_DEVICE")
    if d:
        return d
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load():
    # check for THIS function's own keys, not for _M being non-empty:
    # propose() also caches into _M, so `if _M` short-circuited here and
    # left reid unloaded the moment a proposer had run first.
    if "reid" in _M:
        return _M
    from ultralytics import YOLO
    from ultralytics.trackers.utils.reid import ReID
    dev = device()
    _M["seg"] = YOLO(SEG_MODEL)
    # the ReID encoder runs under onnxruntime and picks its own
    # execution provider; it does not share the torch device
    _M["reid"] = ReID(REID_MODEL, device="cpu" if dev == "mps" else dev)
    _M["dev"] = dev
    return _M


PROPOSER = os.environ.get("ELIDEDB_PROPOSER") or _weights("FastSAM-s.pt")


def propose(frames, imgsz=DET_SZ, batch=DET_BATCH, conf=0.25):
    """CLASS-AGNOSTIC region proposal. Not YOLO-with-single_cls.

    MEASURED, same frames, same model: single_cls=True and
    single_cls=False give byte-identical output - 3.10 det/frame, all of
    it {oven, sink, bowl, person, spoon, wine glass}. single_cls is a
    TRAINING flag; at inference the detector still fires only on
    COCO-shaped things. Everything built on it inherited an 80-class
    prior that was never intended, which is why looking for a container
    found 0.7 regions per episode and they were ovens.

    FastSAM is trained on SA-1B with no class list at all, and on this
    corpus returns 45.3 regions/frame against COCO YOLO's 2.6, with 3.5
    large structures per frame against 1.1.

    It costs 24.93 ms/frame batched fp16 = 7.48 min per hour of video,
    which is 7x the write budget - so this is a TEACHER. It is paid once
    to make training pairs and then distilled, exactly like every other
    teacher here. Never put it on the write path directly.
    """
    from ultralytics import FastSAM
    if "prop" not in _M:
        _M["prop"] = FastSAM(PROPOSER)
    out = []
    kw = {"quantize": "fp16"} if DET_FP16 else {}
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        res = _M["prop"].predict(chunk, device=device(), verbose=False,
                                 imgsz=imgsz, conf=conf, **kw)
        for im, r in zip(chunk, res):
            H, W = im.shape[:2]
            if r.boxes is None or not len(r.boxes):
                out.append(np.zeros((0, 4), np.int32))
                continue
            b = r.boxes.xyxy.cpu().numpy()
            b = np.stack([b[:, 0].clip(0, W), b[:, 1].clip(0, H),
                          b[:, 2].clip(0, W), b[:, 3].clip(0, H)],
                         1).astype(np.int32)
            wh = (b[:, 2] - b[:, 0]), (b[:, 3] - b[:, 1])
            out.append(b[(wh[0] >= MIN_SIDE) & (wh[1] >= MIN_SIDE)])
    return out


def detect(frames, imgsz=DET_SZ, batch=DET_BATCH, conf=0.25):
    """Class-agnostic regions per frame: [(boxes Nx4, conf N, area N)].

    Stateless and therefore batched - the caller may hand in frames from
    several episodes at once. single_cls collapses the 80 COCO classes
    into one anonymous class, so the model reports "a thing is here" and
    never a label.
    """
    m = _load()
    out = []
    kw = {"quantize": "fp16"} if DET_FP16 else {}
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        res = m["seg"].predict(chunk, device=m["dev"], single_cls=True,
                               verbose=False, imgsz=imgsz, conf=conf, **kw)
        for im, r in zip(chunk, res):
            H, W = im.shape[:2]
            if r.boxes is None or not len(r.boxes):
                out.append((np.zeros((0, 4), np.int32), np.zeros(0, np.float32),
                            np.zeros(0, np.float32)))
                continue
            xy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            ar = (r.masks.data.sum((1, 2)).cpu().numpy()
                  if r.masks is not None else np.zeros(len(xy)))
            b = np.stack([xy[:, 0].clip(0, W), xy[:, 1].clip(0, H),
                          xy[:, 2].clip(0, W), xy[:, 3].clip(0, H)],
                         1).astype(np.int32)
            wh = (b[:, 2] - b[:, 0]), (b[:, 3] - b[:, 1])
            ok = ((wh[0] >= MIN_SIDE) & (wh[1] >= MIN_SIDE)
                  & (wh[0] * wh[1] <= MAX_AREA * W * H))
            out.append((b[ok], cf[ok].astype(np.float32),
                        ar[ok].astype(np.float32)))
    return out


class _Dets:
    """The minimal shape ultralytics' trackers consume: xywh/conf/cls,
    a length, and boolean-mask indexing (they split detections into
    high- and low-confidence subsets). Handing them this instead of a
    Results object is what lets detection run batched somewhere else."""

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh, conf, cls):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, m):
        return _Dets(self.xywh[m], self.conf[m], self.cls[m])

    @property
    def xyxy(self):
        """Corner boxes, which global motion compensation asks for.

        byte_tracker.py calls `self.gmc.apply(img, results_high.xyxy)` to
        mask moving objects out before estimating camera motion, inside a
        try/except that WARNS and falls back to an identity warp. Without
        this property every frame took that fallback: 1,434 warnings in
        the first two minutes of a write, and camera motion silently not
        compensated.

        It was invisible on this corpus because Bridge's camera is fixed,
        where identity is the correct warp anyway - so the bug cost
        nothing here and everything on a corpus that moves. A robot that
        drives or a vehicle camera would have had its tracks fragmented
        by exactly the motion GMC exists to remove.

        xywh is centre-based (built that way in Stream.push, and what
        ultralytics' xywh2ltwh assumes), so the corners are centre +/-
        half-extent.
        """
        cx, cy, w, h = (self.xywh[:, 0], self.xywh[:, 1],
                        self.xywh[:, 2], self.xywh[:, 3])
        return np.stack([cx - w / 2, cy - h / 2,
                         cx + w / 2, cy + h / 2], 1)


def _tracker(gmc=None):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import YAML
    import ultralytics
    from pathlib import Path
    cfg = YAML.load(Path(ultralytics.__file__).parent
                    / "cfg/trackers/botsort.yaml")
    cfg["gmc_method"] = gmc or GMC
    # BoT-SORT's own ReID would embed every detection on every frame,
    # which is the cost this design exists to avoid. Association here is
    # geometry only; appearance is consulted once per track, later.
    cfg["with_reid"] = False
    return BOTSORT(args=SimpleNamespace(**cfg))


class Stream:
    """CONTINUOUS tracking over a stream. No episode boundaries.

    link() below takes one episode's frames and builds a fresh tracker
    for each - correct when a corpus ships discrete demos, and an
    assumption the engine has no right to make. Raw capture is a
    continuous recording; episodes are something the engine must
    PRODUCE. A driving log, a surveillance feed and a surgical recording
    have no cuts to reset on.

    So the tracker runs for the life of the stream and a track ends when
    the OBJECT does - it leaves frame, is occluded past the buffer, or
    is carried away. That track's span is exactly a PRESENCE INTERVAL:
    "this object was here, from t0 to t1", one row however long it
    lasted. An object sitting still for four hours is one row, not
    72,000, and it is finally recorded at all - the motion-triggered
    element path never produced a row for anything that did not move.

    Online by construction: detections go in frame by frame, closed
    tracks come out as they close, and only the open tracks are held.
    A stream that does not fit in memory is the normal case, not an
    edge case.

    Exemplar crops are retained per OPEN track and embedded once, at
    close - the same "ask the object store one question per track, with
    a whole sighting behind it" that made identity work, now without
    needing the episode to know when to ask.
    """

    def __init__(self, gmc=None, views=EXEMPLARS, buffer_s=6.0):
        self.tr = _tracker(gmc)
        self.views = views
        # how long a track survives with no detection before it is
        # declared ended, in SECONDS not frames: a frame count means
        # different things at 5 fps and 30 fps, and the corpus chooses
        # the frame rate.
        self.buffer_s = buffer_s
        self.open: dict[int, dict] = {}
        self.last_ts = None

    def update(self, ts, det, frame=None):
        """One frame in; the tracks that CLOSED at this frame out.

        `det` is (boxes Nx4, conf N, area N) from detect()/propose().
        """
        b, c, a = det
        self.last_ts = int(ts)
        seen = set()
        if len(b):
            xywh = np.stack([(b[:, 0] + b[:, 2]) / 2,
                             (b[:, 1] + b[:, 3]) / 2,
                             b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            rows = self.tr.update(
                _Dets(xywh.astype(np.float32), c,
                      np.zeros(len(b), np.float32)), frame)
            for row in rows:
                tid, di = int(row[-4]), int(row[-1])
                if di >= len(b):
                    continue
                seen.add(tid)
                t = self.open.setdefault(tid, {
                    "ts": int(ts), "t1": int(ts), "n": 0, "t": [],
                    "box": [], "conf": [], "area": [], "crops": []})
                t["t1"] = int(ts)
                t["n"] += 1
                # the box list without its timestamps is a SHAPE, not a
                # trajectory; every consumer that wanted motion had to
                # guess the time axis back from (ts, t1, n)
                t["t"].append(int(ts))
                t["box"].append(b[di])
                t["conf"].append(float(c[di]))
                t["area"].append(float(a[di]))
                # retain a bounded, spread set of views for the one
                # ReID call this track will ever cost
                if frame is not None and len(t["crops"]) < self.views:
                    x0, y0, x1, y1 = (int(v) for v in b[di])
                    if x1 > x0 and y1 > y0:
                        t["crops"].append((b[di], frame[y0:y1, x0:x1]))
        return self._reap(seen)

    def _reap(self, seen):
        gap = self.buffer_s * 1e9
        done = []
        for tid in list(self.open):
            if tid in seen:
                continue
            if self.last_ts - self.open[tid]["t1"] > gap:
                done.append((tid, self.open.pop(tid)))
        return done

    def flush(self):
        """End of stream: everything still open is still real."""
        out = list(self.open.items())
        self.open = {}
        return out


def link(dets, frames=None, gmc=None):
    """Associate one episode's consecutive detections into tracks.

    Returns {track_id: {"f": [frame idx], "box": [...], "conf": [...],
    "area": [...]}}. A fresh tracker per call, which is right only when
    the caller really does have a scene cut. Prefer Stream for
    continuous capture; this remains for corpora that ship discrete
    clips, where resetting at a genuine cut avoids carrying Kalman state
    across it.
    """
    tr = _tracker(gmc)
    out: dict[int, dict] = {}
    for j, (b, c, a) in enumerate(dets):
        if not len(b):
            continue
        xywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                         b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        det = _Dets(xywh.astype(np.float32), c, np.zeros(len(b), np.float32))
        rows = tr.update(det, None if frames is None else frames[j])
        for row in rows:
            # ultralytics packs [*xyxy, track_id, conf, cls, det_idx]
            tid, di = int(row[-4]), int(row[-1])
            if di >= len(b):
                continue
            t = out.setdefault(tid, {"f": [], "box": [], "conf": [],
                                     "area": []})
            t["f"].append(j)
            t["box"].append(b[di])
            t["conf"].append(float(c[di]))
            t["area"].append(float(a[di]))
    return out


def _views(track, k=EXEMPLARS):
    """Pick up to k frames of a track: the best-scoring view in each of
    k equal slices of the track's life. Spread matters more than score -
    k views of the same instant carry one view's worth of evidence."""
    n = len(track["f"])
    if n <= k:
        return list(range(n))
    edges = np.linspace(0, n, k + 1).round().astype(int)
    conf = np.asarray(track["conf"])
    return [int(a + conf[a:b].argmax())
            for a, b in zip(edges[:-1], edges[1:]) if b > a]


def features(frame, boxes):
    """Appearance embedding per region. Vision only, no text."""
    if not len(boxes):
        return np.zeros((0, 512), np.float32)
    m = _load()
    b = np.asarray(boxes, np.float32)
    xywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                     b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
    f = np.asarray(m["reid"](frame, xywh))
    f = f.reshape(len(boxes), -1).astype(np.float32)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


def descriptors(frames, tracks, k=EXEMPLARS):
    """One pooled appearance vector per track - the unit of upload.

    This is the "break in continuity" moment: while a track holds, the
    object store hears nothing about it. When the track is done, it is
    described ONCE, by a mean over k views spread across its life, and
    that single descriptor is what the store is asked about.
    """
    ids, out = [], []
    for tid, t in tracks.items():
        V = []
        for i in _views(t, k):
            V.append(features(frames[t["f"][i]], [t["box"][i]])[0])
        v = np.mean(V, 0)
        ids.append(tid)
        out.append(v / (np.linalg.norm(v) + 1e-8))
    return ids, (np.stack(out) if out else np.zeros((0, 512), np.float32))


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0) * max(y1 - y0, 0)
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def free_negatives(tracks, iou_max=0.1):
    """Track-id pairs that are CERTAINLY different physical objects.

    Two tracks occupying DISJOINT REGIONS OF THE SAME FRAME are two
    different objects: one object cannot be in two places at one
    instant, and the tracker has already linked each object to itself,
    so a second track is a second thing. No annotation, no dataset
    knowledge, no prior - geometry alone.

    The disjointness test is not decoration. Co-existence ALONE is not
    enough, because a detector will sometimes put two boxes on one
    object; those two tracks co-exist and look identical, and they land
    exactly in the high tail that a calibration quantile reads. Fitting
    on co-existence alone drove the cut to its 0.95 ceiling on a 300-
    episode run and fragmented the store to 1.21 sightings per object.
    Requiring the boxes to be apart removes the double detections, which
    are the only way two tracks of the same object can co-exist.
    """
    out, ids = [], list(tracks)
    for x in range(len(ids)):
        a = tracks[ids[x]]
        fa = {f: i for i, f in enumerate(a["f"])}
        for y in range(x + 1, len(ids)):
            b = tracks[ids[y]]
            shared = [f for f in b["f"] if f in fa]
            if not shared:
                continue
            bf = {f: i for i, f in enumerate(b["f"])}
            if all(_iou(a["box"][fa[f]], b["box"][bf[f]]) <= iou_max
                   for f in shared):
                out.append((ids[x], ids[y]))
    return out


def free_positives(tracks, iou_min=0.8):
    """Track-id pairs that are CERTAINLY the SAME physical object.

    The mirror of free_negatives, from the mirror of its argument. That
    one says two tracks in DISJOINT regions of one frame are two objects,
    because one object cannot be in two places at once. This one says two
    tracks in the SAME region of one frame are ONE object, because two
    objects cannot occupy one place at once. Both are geometry; neither
    needs an annotation, a label or a dataset prior.

    These pairs are DOUBLE DETECTIONS - the detector put two boxes on one
    thing - which is why free_negatives works to exclude them. They were
    treated as waste. They are not waste: they are the only proven-same
    evidence this corpus can produce for free, and without them the cut
    was fitted from one side of a two-sided decision.

    iou_min is high on purpose. Overlap alone does not prove identity:
    nested and contacting things - a lid on a jar, a hand on a pot -
    overlap heavily and are two objects. Measured on fresh_bench, the
    positive sample gets monotonically cleaner as the bar rises, and the
    AUC it implies rises with it (0.860 at IoU>0.5 to 0.902 at IoU>0.95),
    i.e. the loose bands are contaminated by exactly those nested pairs.
    0.8 keeps thousands of pairs while paying most of that gap.

    BIAS, stated, because it bounds what this can conclude: a same-frame
    positive is the EASIEST positive there is - one instant, one
    viewpoint, one exposure - while a same-frame negative is the HARDEST
    negative. So the fitted cut is bracketed by two optimistic samples
    pulling in opposite directions, and a genuine cross-episode
    re-identification is harder than anything measured here.
    """
    out, ids = [], list(tracks)
    for x in range(len(ids)):
        a = tracks[ids[x]]
        fa = {f: i for i, f in enumerate(a["f"])}
        for y in range(x + 1, len(ids)):
            b = tracks[ids[y]]
            shared = [f for f in b["f"] if f in fa]
            if not shared:
                continue
            bf = {f: i for i, f in enumerate(b["f"])}
            if all(_iou(a["box"][fa[f]], b["box"][bf[f]]) >= iou_min
                   for f in shared):
                out.append((ids[x], ids[y]))
    return out


def interval_pairs(rows, iou_diff=0.1, iou_same=0.8, start_frac=0.25):
    """free_negatives and free_positives over CLOSED intervals.

    The write path does not hold tracks - it holds closed presence
    intervals, each reduced to one box. Same two geometric arguments,
    applied to what the writer actually has.

    APPROXIMATION, stated: the retained box is the track's FIRST box, so
    two boxes are only comparable when the tracks START together. That is
    always true of the double detections the positive test is looking
    for, and start_frac enforces it rather than assuming it. The negative
    test does not need the guard - a pair wrongly called disjoint is a
    pair dropped, not a pair mislabelled.

    Args:
        rows: (stream, meta) or (stream, meta, vec) as close_track emits;
              meta carries ts, t1 and box.
    Returns:
        (neg, pos) index-pair lists into `rows`.
    """
    n = len(rows)
    key = sorted(range(n), key=lambda i: (rows[i][0], rows[i][1]["ts"]))
    neg, pos = [], []
    for a in range(len(key)):
        i = key[a]
        si, mi = rows[i][0], rows[i][1]
        for b in range(a + 1, len(key)):
            j = key[b]
            sj, mj = rows[j][0], rows[j][1]
            if sj != si or mj["ts"] > mi["t1"]:
                break                       # sorted: no later one overlaps
            ov = _iou(mi["box"], mj["box"])
            # the sweep runs in time order, so emit (min, max) rather
            # than (earlier, later) - a pair is unordered and callers
            # should not have to know which way round it came out.
            e = (i, j) if i < j else (j, i)
            if ov <= iou_diff:
                neg.append(e)
            elif ov >= iou_same:
                span = min(mi["t1"] - mi["ts"], mj["t1"] - mj["ts"])
                if abs(mj["ts"] - mi["ts"]) <= start_frac * max(span, 1):
                    pos.append(e)
    return neg, pos


def calibrate(V, pairs, pos=None, q=CAL_Q, floor=0.5, ceil=0.95):
    """Fit the match cut from proven pairs, on this corpus.

    TWO-SIDED when proven-same pairs are supplied, and that is the
    correction. The one-sided form below reads the q-th percentile of the
    proven-DIFFERENT similarities - "tighter than all but q% of what I
    can show is not the same object" - which is sound only if that sample
    is clean. On fresh_bench it was not: double detections are 0.5% of
    co-existing pairs and q=99.5 reads the top 0.5%, so the cut was very
    nearly a readout of the contamination. It sat at 0.739, where only
    26% of proven-same pairs are accepted, and the store fragmented to
    78% singletons.

    Fixing the contamination alone moves it to 0.692 - still 34% recall.
    A one-sided fit cannot do better, because the tail of the negatives
    says nothing about where the positives are. Hence Youden's J over
    both: the cut that maximises (true accept rate - false merge rate),
    the standard criterion when neither error has a stated price. It is
    fitted, not chosen; no number here is hand-picked.

    Fails safe in stages: no positives, use the one-sided percentile; too
    few negatives to estimate at all, keep MATCH.

    Args:
        V: (N, D) unit-norm track descriptors.
        pairs: proven-DIFFERENT index pairs, from free_negatives.
        pos: proven-SAME index pairs, from free_positives. Optional.
    """
    if len(pairs) < 20:
        return MATCH, len(pairs)
    neg = np.array([float(V[i] @ V[j]) for i, j in pairs])
    if pos is None or len(pos) < 20:
        return float(np.clip(np.percentile(neg, q), floor, ceil)), len(neg)
    p = np.array([float(V[i] @ V[j]) for i, j in pos])
    grid = np.linspace(0.0, 1.0, 201)
    j = ((p[None, :] >= grid[:, None]).mean(1)
         - (neg[None, :] >= grid[:, None]).mean(1))
    return float(np.clip(grid[int(j.argmax())], floor, ceil)), len(neg)


def fit_cut(V, groups, neg, pos=None, grid=None):
    """Fit the cut by MAXIMISING what identity is actually for.

    Both free samples - proven-different and proven-same - are SAME-FRAME
    pairs. The decision the gallery actually makes is not that one. It is
    "is this new track the object I saw in a different episode?", and no
    same-frame pair is evidence about it. So a statistic of those pairs
    can bound the descriptor's quality but cannot choose the operating
    point, and choosing it from one anyway is how the cut ended up where
    it did.

    The target quantity is measurable directly and needs no annotation:
    HOW MANY OBJECTS RECUR ACROSS GROUPS. That is the whole point of a
    persistent id, and it is self-limiting as an objective - too tight
    and every sighting is a new id that recurs nowhere; too loose and
    distinct objects collapse into one attractor, which also recurs
    nowhere because there are no longer distinct objects to recur. It
    therefore has an interior maximum, and that maximum is the fit.

    Measured on fresh_bench (2,097 episodes, 73,521 intervals):

        cut    objects  singletons  RECUR   worst false-merge
        0.75    34,063     80.2%    6,447        0.18%
        0.70    26,159     74.0%    6,475        0.22%   <- peak
        0.60    14,011     57.3%    5,732        0.54%
        0.50     6,320     36.8%    3,876        1.05%
        0.42     2,618     19.4%    2,075        1.72%

    which also disposes of the singleton rate as a target: driving it
    from 80% to 19% costs two thirds of the recurrence and multiplies
    proven-wrong merges by eight. Fewer, bigger, wronger objects.

    Sweeping a cut used to be impossible - Gallery.assign was ~20 minutes
    per pass - so the fit had to be a closed-form statistic. It is now
    seconds, and an objective beats a proxy.

    Args:
        V: (N, D) unit-norm descriptors.
        groups: (N,) group label per descriptor - the episode it sits in.
        neg: proven-different pairs, for the reported false-merge rate.
        pos: proven-same pairs, for the reported recovery rate.
        grid: candidate cuts. Defaults to percentiles of the proven-
              different similarities, so the range adapts to the corpus
              rather than being a hand-written span.
    Returns:
        (cut, report) - report lists every candidate that was tried.
    """
    g = np.asarray(groups)
    g = g - g.min() + 1
    span = int(g.max()) + 2
    N = np.asarray(neg, np.int64).reshape(-1, 2)
    Pp = np.asarray(pos if pos is not None else [], np.int64).reshape(-1, 2)
    if grid is None:
        s = (np.einsum("ij,ij->i", V[N[:, 0]], V[N[:, 1]]) if len(N)
             else np.array([MATCH]))
        grid = np.unique(np.round(np.percentile(
            s, [90, 95, 97.5, 99, 99.3, 99.5, 99.7, 99.9]), 3))
    report = []
    for cut in grid:
        ids = Gallery(match=float(cut)).assign(V)
        n = int(ids.max()) + 1
        # objects present in more than one group
        u = np.unique(ids.astype(np.int64) * span + g)
        per = np.bincount((u // span).astype(np.int64), minlength=n)
        row = {"cut": float(cut), "objects": n,
               "recur": int((per > 1).sum()),
               "singleton_pct": round(100.0 * float(
                   (np.bincount(ids) == 1).mean()), 1)}
        if len(N):
            row["false_merge_pct"] = round(100.0 * float(
                (ids[N[:, 0]] == ids[N[:, 1]]).mean()), 3)
        if len(Pp):
            row["recovered_pct"] = round(100.0 * float(
                (ids[Pp[:, 0]] == ids[Pp[:, 1]]).mean()), 1)
        report.append(row)
    best = max(report, key=lambda r: r["recur"])
    # The grid is percentiles of the proven-DIFFERENT similarities, and
    # that anchor breaks when the descriptor is much better than the
    # negatives are hard: recurrence keeps rising past the negatives'
    # entire tail, the argmax lands on the grid's top edge, and the
    # "interior maximum" was never actually bracketed. Measured on the
    # DINOv3 rebuild: recur still climbing 7,590 -> 10,541 at the last
    # point. So while the best cut IS the top edge, keep extending
    # upward (midpoint steps toward 0.99) until the maximum is interior
    # or the ceiling is reached - the fit must end bracketed, not
    # truncated by an artifact of where the negatives happened to end.
    while best["cut"] == report[-1]["cut"] and best["cut"] < 0.985:
        cut = round(best["cut"] + (0.99 - best["cut"]) / 2, 3)
        ids = Gallery(match=float(cut)).assign(V)
        n = int(ids.max()) + 1
        u = np.unique(ids.astype(np.int64) * span + g)
        per = np.bincount((u // span).astype(np.int64), minlength=n)
        row = {"cut": float(cut), "objects": n,
               "recur": int((per > 1).sum()),
               "singleton_pct": round(100.0 * float(
                   (np.bincount(ids) == 1).mean()), 1)}
        if len(N):
            row["false_merge_pct"] = round(100.0 * float(
                (ids[N[:, 0]] == ids[N[:, 1]]).mean()), 3)
        if len(Pp):
            row["recovered_pct"] = round(100.0 * float(
                (ids[Pp[:, 0]] == ids[Pp[:, 1]]).mean()), 1)
        report.append(row)
        if row["recur"] > best["recur"]:
            best = row
        else:
            break
    return best["cut"], report


class Gallery:
    """Persistent identities, matched by appearance.

    A GALLERY of exemplars per object, not a running mean. The mean is
    what broke the earlier attempt: a centroid that updates as it
    absorbs members lets a group CHAIN - green, to a slightly different
    green, to red - so two objects merge through a path of intermediate
    views. Several fixed exemplars, matched against the best of them,
    has no such path. This is ordinary ReID gallery practice.

    What it is asked about changed: entries are now TRACK descriptors
    pooled over many frames, not single crops, so each question carries
    a whole sighting's worth of evidence.
    """

    def __init__(self, match=MATCH, max_ex=MAX_EXEMPLARS, admit=ADMIT):
        self.match, self.max_ex, self.admit = match, max_ex, admit
        self.ex: list[np.ndarray] = []
        self.n: list[int] = []
        # every exemplar of every object in ONE matrix, plus which object
        # owns each row. self.ex stays the public view of the same thing.
        self._M = np.zeros((0, 0), np.float32)
        self._own = np.zeros(0, np.int32)
        self._m = 0

    def _grow(self, dim):
        if self._M.shape[0] > self._m:
            return
        cap = max(1024, self._M.shape[0] * 2)
        M = np.zeros((cap, dim), np.float32)
        if self._m:
            M[:self._m] = self._M[:self._m]
        own = np.zeros(cap, np.int32)
        own[:self._m] = self._own[:self._m]
        self._M, self._own = M, own

    def assign(self, feats):
        """Which object is each descriptor, in arrival order.

        ONE matrix-vector product per descriptor, not one per object.
        The old form looped over self.ex in Python, so a 73,521-track
        corpus that had grown 32,756 objects did 2.4 billion iterations
        of a two-line body - about 20 of the write's 54 minutes, and slow
        enough that the cut could not be swept to find out it was wrong.

        The reduction is exact, not an approximation: the best object is
        the one owning the best EXEMPLAR, because max over objects of
        (max over that object's exemplars) IS the max over all exemplars.
        So a single argmax replaces the per-object max-then-compare.
        """
        ids = []
        for v in feats:
            v = np.ascontiguousarray(v, np.float32)
            best, bs = -1, -1.0
            if self._m:
                s = self._M[:self._m] @ v
                k = int(s.argmax())
                best, bs = int(self._own[k]), float(s[k])
            if best >= 0 and bs >= self.match:
                ids.append(best)
                self.n[best] += 1
                E = self.ex[best]
                # keep a view only if it ADDS one; near-duplicates of a
                # stored exemplar teach the gallery nothing. Admit it
                # only if it agrees with EVERY held exemplar - min(),
                # not max() - so the acceptance region cannot sprawl.
                if (len(E) < self.max_ex and bs < 0.95
                        and float((E @ v).min()) >= self.admit):
                    self.ex[best] = np.vstack([E, v])
                    self._grow(len(v))
                    self._M[self._m] = v
                    self._own[self._m] = best
                    self._m += 1
            else:
                self.ex.append(v[None, :])
                self.n.append(1)
                best = len(self.ex) - 1
                ids.append(best)
                self._grow(len(v))
                self._M[self._m] = v
                self._own[self._m] = best
                self._m += 1
        return np.asarray(ids, np.int32)

    def centroids(self):
        return np.stack([E.mean(0) / (np.linalg.norm(E.mean(0)) + 1e-8)
                         for E in self.ex]) if self.ex else np.zeros(
                             (0, 512), np.float32)

    def __len__(self):
        return len(self.ex)
