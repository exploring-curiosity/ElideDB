"""Object identity: the free-negative label and the cut fitted from it.

Two pieces carry the design and both are pure geometry, so both are
testable without a model.

free_negatives is the labelling rule: two tracks holding DISJOINT
regions of the SAME frame are provably different physical objects. The
disjointness half is not decoration - without it a detector that puts
two boxes on one object contributes a pair of identical-looking
"negatives", and those land in exactly the high tail a calibration
quantile reads. That contamination drove the fitted cut to its ceiling
on a 300-episode run and fragmented the store to 1.21 sightings per
object, so it gets a test.

calibrate turns those pairs into the match threshold. Its contract is
that it fails SAFE: too few negatives to estimate from and it returns
the fallback rather than a number fitted on noise.

Run: python tests/test_identity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.identity import (MATCH, Gallery, _Dets,  # noqa: E402
                              calibrate,
                              free_negatives, _views)


def track(frames, boxes):
    return {"f": list(frames), "box": [np.array(b) for b in boxes],
            "conf": [1.0] * len(frames), "area": [1.0] * len(frames)}


def test_disjoint_pair_is_a_negative():
    t = {1: track([0, 1, 2], [(0, 0, 10, 10)] * 3),
         2: track([0, 1, 2], [(50, 50, 60, 60)] * 3)}
    assert free_negatives(t) == [(1, 2)]


def test_overlapping_boxes_are_not_a_negative():
    """A double detection on one object: co-existing but not disjoint."""
    t = {1: track([0, 1], [(0, 0, 10, 10)] * 2),
         2: track([0, 1], [(1, 1, 11, 11)] * 2)}
    assert free_negatives(t) == []


def test_no_shared_frame_is_not_a_negative():
    """Sequential tracks prove nothing - they could be one object that
    the tracker lost and re-acquired, which is the case the object
    store exists to adjudicate."""
    t = {1: track([0, 1], [(0, 0, 10, 10)] * 2),
         2: track([5, 6], [(50, 50, 60, 60)] * 2)}
    assert free_negatives(t) == []


def test_overlap_on_any_shared_frame_disqualifies():
    """Disjoint on one frame, overlapping on another - not provable."""
    t = {1: track([0, 1], [(0, 0, 10, 10), (0, 0, 10, 10)]),
         2: track([0, 1], [(50, 50, 60, 60), (1, 1, 11, 11)])}
    assert free_negatives(t) == []


def test_calibrate_fails_safe_on_too_few_negatives():
    V = np.eye(4, 8, dtype=np.float32)
    cut, n = calibrate(V, [(0, 1), (2, 3)])
    assert cut == MATCH and n == 2


def test_calibrate_sits_above_the_negative_mass():
    """The cut must exceed nearly every proven-different similarity."""
    rng = np.random.default_rng(0)
    V = rng.normal(size=(80, 16)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    pairs = [(i, j) for i in range(0, 80, 2) for j in range(i + 1, 80, 2)]
    cut, n = calibrate(V, pairs)
    sims = np.array([V[i] @ V[j] for i, j in pairs])
    assert n == len(pairs)
    assert (sims >= cut).mean() <= 0.01


def test_gallery_admit_blocks_the_chain():
    """A new view is admitted only if it agrees with EVERY exemplar.
    Without that, absorbing a bridge view widens the acceptance region
    and lets green reach red through the middle."""
    a = np.array([1.0, 0.0, 0.0], np.float32)
    mid = np.array([0.75, 0.66, 0.0], np.float32)
    mid /= np.linalg.norm(mid)
    g = Gallery(match=0.6, admit=0.9)
    g.assign(np.stack([a, mid]))
    assert len(g.ex[0]) == 1, "bridge view must not join the gallery"


def test_views_spread_across_the_track():
    """k views of one instant carry one view's worth of evidence."""
    t = {"f": list(range(20)), "conf": [0.5] * 20}
    v = _views(t, k=4)
    assert len(v) == 4 and v == sorted(v)
    assert max(v) - min(v) >= 12


def test_dets_exposes_corner_boxes_for_gmc():
    """byte_tracker asks a detection batch for .xyxy to mask moving
    objects out of its camera-motion estimate, inside a try/except that
    warns and falls back to an identity warp. Without the property the
    fallback fired on EVERY frame - silent, because a fixed-camera corpus
    cannot tell identity from a correct warp. This test is the tripwire a
    moving-camera corpus would otherwise have to find for us."""
    b = np.array([[10., 20., 50., 80.], [0., 0., 4., 6.]], np.float32)
    xywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                     b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
    d = _Dets(xywh.astype(np.float32), np.array([.9, .5], np.float32),
              np.array([0, 1]))
    assert np.allclose(d.xyxy, b), "xywh -> xyxy must round-trip"
    # the tracker slices high/low confidence before asking for xyxy
    assert np.allclose(d[np.array([True, False])].xyxy, b[:1])


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {name}: {e}")
    print("all identity tests pass" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
