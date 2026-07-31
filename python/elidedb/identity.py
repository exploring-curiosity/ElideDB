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
    detect batched + associate        4.1   <- this module
      of which detect                 3.9
      of which associate              0.25

At 18,000 frames per hour of video that last number is 1.23 min/hour,
against a 1 min/hour write budget of which 0.23 is already spent by the
existing pass. Identity therefore does NOT fit the budget for free; it
fits only because the ReID half - the part that used to dominate - has
collapsed to ~0.1 min/hour by riding on tracks. The detector is now the
whole cost, and shrinking it is a detector question, not a design one.

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

SEG_MODEL = os.environ.get("ELIDEDB_SEG", "yolo11n-seg.pt")
REID_MODEL = os.environ.get("ELIDEDB_REID", "yolo26n-reid.onnx")
DET_SZ = int(os.environ.get("ELIDEDB_DET_SZ", "448"))
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
    if _M:
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


def detect(frames, imgsz=DET_SZ, batch=DET_BATCH, conf=0.25):
    """Class-agnostic regions per frame: [(boxes Nx4, conf N, area N)].

    Stateless and therefore batched - the caller may hand in frames from
    several episodes at once. single_cls collapses the 80 COCO classes
    into one anonymous class, so the model reports "a thing is here" and
    never a label.
    """
    m = _load()
    out = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        res = m["seg"].predict(chunk, device=m["dev"], single_cls=True,
                               verbose=False, imgsz=imgsz, conf=conf)
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


def link(dets, frames=None, gmc=None):
    """Associate one episode's consecutive detections into tracks.

    Returns {track_id: {"f": [frame idx], "box": [...], "conf": [...],
    "area": [...]}}. A fresh tracker per episode: a new episode is a new
    scene, and carrying Kalman state across a cut invents motion that
    never happened.
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


def calibrate(V, pairs, q=CAL_Q, floor=0.5, ceil=0.95):
    """Fit the match cut from the free negatives, on this corpus.

    The cut is the q-th percentile of similarity over pairs proven
    different, i.e. "tighter than all but q% of what I can show is not
    the same object". Hand-picking a number here would be exactly the
    dataset prior this codebase forbids, and it would also be wrong: the
    right cut moved from 0.85 to 0.65 the moment descriptors became
    track-pooled instead of single-crop.

    Fails safe: too few negatives to estimate from, keep MATCH.

    Args:
        V: (N, D) unit-norm track descriptors.
        pairs: index pairs into V, from free_negatives().
    """
    if len(pairs) < 20:
        return MATCH, len(pairs)
    neg = np.array([float(V[i] @ V[j]) for i, j in pairs])
    return float(np.clip(np.percentile(neg, q), floor, ceil)), len(neg)


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

    def assign(self, feats):
        ids = []
        for v in feats:
            best, bs = -1, -1.0
            for i, E in enumerate(self.ex):
                s = float((E @ v).max())
                if s > bs:
                    best, bs = i, s
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
            else:
                self.ex.append(v[None, :])
                self.n.append(1)
                ids.append(len(self.ex) - 1)
        return np.asarray(ids, np.int32)

    def centroids(self):
        return np.stack([E.mean(0) / (np.linalg.norm(E.mean(0)) + 1e-8)
                         for E in self.ex]) if self.ex else np.zeros(
                             (0, 512), np.float32)

    def __len__(self):
        return len(self.ex)
