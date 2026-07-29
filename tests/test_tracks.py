"""track_lookup contract: max over views, min across atoms, NaN absent.

Both combination rules were chosen against measurement, so they are
pinned here rather than left to be re-derived:

  max over views   pooling along the track LOST to best-single-view
                   (1.81s vs 0.90s on q09, again on q08) - the box
                   drifts and the mean blends object with background
  min across atoms a compound query scored whole against one small
                   crop is a bag-of-concepts match; every named thing
                   must be independently present (Winoground, ARO)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def _store(tmp_path, vecs, keys):
    from elidedb import Store
    db = Store.create(str(tmp_path / "s"), "trk-test")
    V = np.asarray(vecs, np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    db.table("track_vectors").append(pa.table({
        "ts": pa.array([k[1] for k in keys], pa.int64()),
        "t1": pa.array([k[1] + 1 for k in keys], pa.int64()),
        "stream": pa.array([k[0] for k in keys]),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(V.astype(np.float16)).reshape(-1),
                     pa.float16()), V.shape[1]),
        "track": pa.array([0] * len(keys), pa.int32()),
        "area": pa.array([0.1] * len(keys), pa.float32()),
    }), kind="embeddings", meta={"dim": int(V.shape[1])})
    return db


def test_max_over_views_and_min_over_atoms(tmp_path):
    from elidedb.tracks import track_lookup
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    # episode X: one view matches atom a, another matches atom b
    # episode Y: two views both match a, nothing matches b
    keys = [("s", 100), ("s", 100), ("s", 200), ("s", 200)]
    db = _store(tmp_path, [a, b, a, a], keys)
    look = track_lookup(db, np.stack([a, b]))
    x = look("s", 100, 101)
    y = look("s", 200, 201)
    # X has both atoms -> min of two strong maxima stays high
    assert x > 0.9
    # Y is missing atom b -> min collapses even though a is perfect
    assert y < 0.1
    # max over views, not mean: X's per-atom best is ~1, not ~0.5
    assert x > 0.9


def test_absent_episode_is_nan(tmp_path):
    from elidedb.tracks import track_lookup
    db = _store(tmp_path, [np.array([1.0, 0.0, 0.0])], [("s", 100)])
    look = track_lookup(db, np.array([[1.0, 0.0, 0.0]]))
    assert np.isnan(look("s", 999, 1000))
    assert np.isnan(look("other", 100, 101))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
