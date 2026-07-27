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
        # GREEDY middle words bounded at closed-class function words
        # (same _STOP mechanism as sig2.atoms_of): without the
        # boundary this produced "a vessel and put" as the audit's X
        # phrase — it grounded nowhere and the audit executed a 9/10
        # true set (audit-bench-caught)
        from .sig2 import _STOP
        ms = list(re.finditer(
            r"\b(?:a|an|the)\s+(?:(?!(?:%s)\b)\w+\s+){0,2}\w+"
            r"(?=\s|$|\.)" % "|".join(_STOP), seg))
        if not ms:
            return None
        m = ms[-1] if last else ms[0]
        w = m.group(0).split()
        while len(w) > 1 and w[-1] in _STOP:
            w.pop()
        head = " ".join(w[1:])
        art = "an" if head[:1] in "aeiou" else "a"
        return f"{art} {head}"
    x, y = np_of(left, last=True), np_of(right, last=False)
    if not x or not y:
        return None
    return x, prep[0], y


def relation_margin(store, stream, t0, t1, x_phrase, y_phrase,
                    inward, n_frames=8):
    """Containment-change margin for one episode, computed on the SAM
    3.1 VIDEO tracker's masklets (user rule: never per-frame images
    when a video API exists — tracked identity through time is the
    point). Positive = geometry agrees with the query direction; NaN =
    abstain (landmark never seen, or no signal defined).

    Two complementary signals, because containers OCCLUDE (bench-
    caught: every put-in-drawer IoA delta was 0 — the object disappears
    inside):
      IoA change   — X's mask/box overlap with Y rises (put ON, or
                     open container where X stays visible)
      presence     — the tracker loses X while Y persists (put IN), or
        transition   acquires X late (take OUT); tracker probabilities,
                     not thresholded detections
    """
    from .sam3x import track_concepts
    tr = track_concepts(store, stream, t0, t1, [x_phrase, y_phrase],
                        n_frames=n_frames)
    if tr is None:
        return float("nan")
    X, Y = tr[x_phrase], tr[y_phrase]
    if max(Y["presence"]) <= 0:
        return float("nan")          # scene lacks the landmark: abstain
    n = len(X["presence"])
    k = max(1, n // 3)

    sigs = []
    traj = []
    for i in range(n):
        if X["masks"][i] is not None and Y["masks"][i] is not None:
            inter = float((X["masks"][i] & Y["masks"][i]).sum())
            traj.append((i, inter / max(1.0,
                                        float(X["masks"][i].sum()))))
        elif X["boxes"][i] is not None and Y["boxes"][i] is not None:
            traj.append((i, _ioa(X["boxes"][i], Y["boxes"][i])))
    if len(traj) >= 2:
        kk = max(1, len(traj) // 3)
        sigs.append(float(np.mean([v for _, v in traj[-kk:]])
                          - np.mean([v for _, v in traj[:kk]])))
    pres = X["presence"]
    early, late = float(np.mean(pres[:k])), float(np.mean(pres[-k:]))
    if abs(early - late) > 0.2:
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
