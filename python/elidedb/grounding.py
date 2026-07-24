"""GEOMETRIC RELATIONAL VERIFIER — boxes over time instead of a VLM.

Calibration verdict that forced this: VLM judges see the right scene
and say yes (7B est 1.0 on visually ~10%-pure sets); they cannot bind
"X ends up inside/on Y". A detector can: ground the query's noun
phrases to boxes on sampled frames, and the relation IS the geometry —
containment (intersection-over-area of X in Y) low at the start and
high at the end verifies "put X in/on Y"; the reverse verifies "take X
out of Y". Deterministic, explainable, video-evidence-only.

Interim detector: Grounding DINO (open-vocabulary, ungated). Measured
on our frames (2026-07-24): green-object→drawer and lid→pot both
verified by end-state containment; one mid-flight false positive under
occlusion (sink read as lid) — which start/end predicates never see.
SAM 3 (pending license approval) swaps in for masks + tracking later.

Latency: ~1.2 s/frame on MPS — a VERIFICATION-tier cost (top-K
candidates, 6 frames each), never a scan cost.
"""
from __future__ import annotations

import numpy as np

_G = {}

MODEL_ID = "IDEA-Research/grounding-dino-base"

# "put X in Y" family vs "take X out of Y" family: sign of the expected
# containment CHANGE. Generic English, nothing per-dataset.
_INWARD = ("in", "into", "inside", "on", "onto", "on top of", "over")
_OUTWARD = ("out", "out of", "from")


def _load():
    if "model" in _G:
        return _G
    import torch
    from transformers import (AutoModelForZeroShotObjectDetection,
                              AutoProcessor)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    _G["proc"] = AutoProcessor.from_pretrained(MODEL_ID)
    _G["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID).to(dev).eval()
    _G["dev"] = dev
    return _G


def detect_phrases(images, phrases, threshold=0.3):
    """PIL images x phrases -> per image {phrase: (box, score)|None},
    best box per phrase."""
    import torch
    g = _load()
    prompt = " . ".join(p.lower().strip(". ") for p in phrases) + " ."
    inputs = g["proc"](images=images, text=[prompt] * len(images),
                       return_tensors="pt").to(g["dev"])
    with torch.no_grad():
        out = g["model"](**inputs)
    res = g["proc"].post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=threshold, text_threshold=0.25,
        target_sizes=[im.size[::-1] for im in images])
    per = []
    for r in res:
        best = {}
        for box, sc, lb in zip(r["boxes"], r["scores"],
                               r["text_labels"]):
            for p in phrases:
                if p.lower().strip(". ") in lb or lb in p.lower():
                    if p not in best or sc > best[p][1]:
                        best[p] = (np.array([float(v) for v in box]),
                                   float(sc))
        per.append({p: best.get(p) for p in phrases})
    return per


def _detector():
    """SAM 3 when available (masks + presence-token discrimination),
    Grounding DINO otherwise — the geometry consumes boxes either way."""
    try:
        from .sam3x import detect_phrases_sam3
        return detect_phrases_sam3
    except Exception:
        return detect_phrases


def _ioa(a, b):
    """Intersection over area of A — how much of X sits inside Y."""
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def parse_relation(text):
    """'put the green object in the drawer' ->
    ('a green object', 'in', 'a drawer') | None. Mechanical determiner-
    phrase split around a spatial preposition; generic English only."""
    import re
    t = text.lower()
    prep = None
    for p in sorted(_INWARD + _OUTWARD, key=len, reverse=True):
        m = re.search(rf"\b{p}\b", t)
        if m:
            prep = (p, m.start(), m.end())
            break
    if not prep:
        return None
    left, right = t[:prep[1]], t[prep[2]:]
    # the landmark ends at the first conjunction — "out of the drawer
    # and put it..." must not leak "and put" into the phrase
    right = right.split(" and ")[0].split(" then ")[0]

    def np_of(seg, last):
        # GREEDY middle words: non-greedy truncated "the stuffed toy"
        # to "a stuffed" (bench-caught); the preposition split already
        # bounds the segment so greed cannot swallow the landmark
        ms = list(re.finditer(
            r"\b(?:a|an|the)\s+(?:\w+\s+){0,2}\w+(?=\s|$|\.)", seg))
        if not ms:
            return None
        m = ms[-1] if last else ms[0]
        w = m.group(0).split()
        while len(w) > 1 and w[-1] in ("and", "then", "it", "of", "to"):
            w.pop()
        head = " ".join(w[1:])
        art = "an" if head[:1] in "aeiou" else "a"
        return f"{art} {head}"
    x, y = np_of(left, last=True), np_of(right, last=False)
    if not x or not y:
        return None
    return x, prep[0], y


def relation_margin(store, stream, t0, t1, x_phrase, y_phrase,
                    inward, n_frames=6):
    """Containment-change margin for one episode. Positive = geometry
    agrees with the query direction; NaN = abstain.

    Two complementary signals, because containers OCCLUDE (bench-
    caught: every put-in-drawer delta was 0 — the object disappears
    inside, so box overlap cannot see the end state):
      IoA change   — X's box overlap with Y rises (put ON / open
                     container where X stays visible)
      presence     — X stops being detected while Y persists (put IN),
        transition   or starts being detected (take OUT)
    The margin is the mean of whichever signals are defined; NaN only
    when neither is."""
    import pyarrow.compute as pc
    from PIL import Image

    from .video import FrameSet
    frames_tbl = store.table("frames").scan()
    sel = frames_tbl.filter(pc.and_(
        pc.equal(frames_tbl.column("stream"), stream),
        pc.and_(pc.greater_equal(frames_tbl.column("ts"), t0),
                pc.less_equal(frames_tbl.column("ts"), t1))))
    if len(sel) < n_frames:
        return float("nan")
    pick = np.linspace(0, len(sel) - 1, n_frames).round().astype(int)
    dec = FrameSet(store, "frames", sel.take(pick)).decode(width=448)
    if len(dec) < n_frames:
        return float("nan")
    imgs = [Image.fromarray(d[1]) for d in sorted(dec)]
    per = _detector()(imgs, [x_phrase, y_phrase])
    k = max(1, len(per) // 3)

    sigs = []
    # signal 1: IoA change where both visible
    traj = [(i, _ioa(d[x_phrase][0], d[y_phrase][0]))
            for i, d in enumerate(per)
            if d[x_phrase] and d[y_phrase]]
    if len(traj) >= 2:
        kk = max(1, len(traj) // 3)
        sigs.append(float(np.mean([v for _, v in traj[-kk:]])
                          - np.mean([v for _, v in traj[:kk]])))
    # signal 2: presence transition of X (Y must be seen at all —
    # otherwise the scene itself is off and we abstain)
    if any(d[y_phrase] for d in per):
        pres = [1.0 if d[x_phrase] else 0.0 for d in per]
        early, late = float(np.mean(pres[:k])), float(np.mean(pres[-k:]))
        if early != late:
            # disappearing INTO the container is inward-positive
            sigs.append(early - late)
    if not sigs:
        return float("nan")
    delta = float(np.mean(sigs))
    return delta if inward else -delta


def verify_relation(store, text, episodes, n_frames=6):
    """Batch: query text + [(stream,t0,t1)] -> margins array (NaN =
    abstain). Only called on verification-tier candidates."""
    rel = parse_relation(text)
    if rel is None:
        return None
    x, prep, y = rel
    inward = prep in _INWARD
    return np.array([relation_margin(store, s, a, b, x, y, inward,
                                     n_frames=n_frames)
                     for s, a, b in episodes])
