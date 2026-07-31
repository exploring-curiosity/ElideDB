"""OBJECT IDENTITY, on the write path, text-free.

Identity is a property of a physical object, not of a name and not of a
category. The same spoon across every demo gets one id. A spatula gets
a different id. A red pepper and a green pepper get different ids, even
though every text-aligned encoder calls them both "pepper".

Nothing here turns pixels into language. Regions come from a
class-agnostic segmenter, identity from a ReID encoder; both vision
only, both real time. This is separate from objects.py, which proposes
regions for SEMANTIC retrieval - that channel answers "does this look
like a cloth", this one answers "is this the same object again".

WHY THIS SHAPE, measured rather than assumed
--------------------------------------------
Regions:
    motion blobs      the old crop - bbox of a connected component of
                      thresholded flow, padded 60%. Not an object, just
                      whatever MOVED plus background, which is why the
                      name vocabulary filled with "black object" and
                      "white paper".
    Grounding-DINO    real objects but 624 ms/frame = 9 to 104 min per
                      hour of video. Batching makes it worse: 14,296
                      ms/frame at batch 32 on MPS.
    YOLO11n-seg       19.3 ms/frame, 2.9 objects/frame, masks not
      single_cls      boxes. single_cls collapses the 80 COCO classes
                      into one anonymous class, so the model says "a
                      thing is here" and never a label.

Identity:
    SigLIP            merges a green pepper with a red one. Semantic by
                      construction - built to collapse instances into
                      categories.
    DINOv2            separates them (0.526 same vs 0.209 cross) but is
                      a heavyweight general-purpose encoder.
    yolo26n-reid      0.941 same instance vs 0.160 different, 1-NN
                      colour purity 1.000, 3.3 ms/object. Trained for
                      "is this the same object", which is the question.

Budget: 19.3 + 8.4 = ~28 ms/frame. At 3 frames/episode that is 0.40 min
per hour of video on top of the existing 0.23 - inside the 1 min/hour
write budget. There is no offline tier: if it did not fit, the model
would have to change, not the tier.

DEVICE. Defaults to the GPU. On this machine CPU and MPS are within 5%
(19.3 vs 20.3 ms/frame), which makes the GPU look optional - it is not.
Frames are already resident on the GPU for the FDNN-V pass, and on CUDA
a host round trip per frame costs more than the inference. Keeping
tensors where they already are is the portable choice; CPU is the
fallback, never the default.
"""
from __future__ import annotations

import os

import numpy as np

_M: dict = {}

SEG_MODEL = os.environ.get("ELIDEDB_SEG", "yolo11n-seg.pt")
REID_MODEL = os.environ.get("ELIDEDB_REID", "yolo26n-reid.onnx")
# "same physical object" cut, CALIBRATED from the observed distribution
# rather than guessed. Over 241 regions from 60 episodes:
#     same-episode pairs   mean 0.451, p90 0.830
#     cross-episode pairs  mean 0.175, p99 0.707, p99.9 0.813
#     thr   same accepted   cross accepted
#     0.55      35.0%           3.45%      <- first guess, WRONG
#     0.75      17.2%           0.56%
#     0.85       7.7%           0.02%      <- here
# 3.45% of 28,375 cross-episode pairs is ~980 false merges, and one of
# them becomes an attractor: id 15 swallowed a green box, broccoli, a
# toy mouse and a carrot. The asymmetry is the whole argument for being
# strict - splitting one object into two ids costs a duplicate row,
# merging two objects into one id corrupts every query touching either.
MATCH = float(os.environ.get("ELIDEDB_OBJ_MATCH", "0.85"))
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


def regions(frames, imgsz=640, conf=0.25):
    """Class-agnostic instances per frame: [(box, mask_px, conf)]."""
    m = _load()
    out = m["seg"].predict(frames, device=m["dev"], single_cls=True,
                           verbose=False, imgsz=imgsz, conf=conf)
    per = []
    for im, r in zip(frames, out):
        H, W = im.shape[:2]
        got = []
        if r.boxes is not None and len(r.boxes):
            xy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            ar = (r.masks.data.sum((1, 2)).cpu().numpy()
                  if r.masks is not None else np.zeros(len(xy)))
            for b, c, a in zip(xy, cf, ar):
                x0, y0, x1, y1 = [int(v) for v in b]
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, W), min(y1, H)
                if (x1 - x0) < MIN_SIDE or (y1 - y0) < MIN_SIDE:
                    continue
                if (x1 - x0) * (y1 - y0) > MAX_AREA * W * H:
                    continue
                got.append(((x0, y0, x1, y1), float(a), float(c)))
        per.append(got)
    return per


def features(frame, boxes):
    """Appearance embedding per region. Vision only, no text."""
    if not len(boxes):
        return np.zeros((0, 512), np.float32)
    m = _load()
    f = np.asarray(m["reid"](frame, np.asarray(boxes, np.float32)))
    f = f.reshape(len(boxes), -1).astype(np.float32)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


class Gallery:
    """Persistent identities, matched by appearance.

    A GALLERY of exemplars per object, not a running mean. The mean is
    what broke the earlier attempt: a centroid that updates as it
    absorbs members lets a group CHAIN - green, to a slightly different
    green, to red - so two objects merge through a path of intermediate
    views. Several fixed exemplars, matched against the best of them,
    has no such path. This is ordinary ReID gallery practice.
    """

    def __init__(self, match=MATCH, max_ex=MAX_EXEMPLARS):
        self.match, self.max_ex = match, max_ex
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
                # stored exemplar teach the gallery nothing
                # admit a new view only if it agrees with EVERY held
                # exemplar - see ADMIT. min(), not max().
                if (len(E) < self.max_ex and bs < 0.95
                        and float((E @ v).min()) >= ADMIT):
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
