"""2.5D depth for trajectories - the user's call, with the number known.

Depth Pro was measured first, as the standing gate required: 1.61
s/frame on MPS fp16, i.e. one hour of compute per hour of video at even
4 frames/episode - 60x the online budget. Shown that number, the user
chose 2.5D outright (2026-08-01: "this is too much time for
experimentation. do 2.5D only"). Depth Pro is out of the roster.

THE PROXY. Perspective projection makes apparent size inversely
proportional to distance: w_px = f * W / Z. For one physical object
(one track), f and W are constants, so

    z_t = median(s) / s_t          s_t = sqrt(w_t * h_t)

is that track's RELATIVE depth: z > 1 farther than its typical
distance, z < 1 nearer, dimensionless. Geometry only - no model, no
dataset prior, no cross-track comparability claimed (two tracks of
different objects have different W and their z ratios do not compare;
within a track, which is what a trajectory is, they do).

s is median-3 smoothed first: detector jitter changes a box a few px
frame to frame, and dividing by a jittering s manufactures depth
oscillation an object never had. Stated limits: a deforming or rotating
object aliases into z (a towel being folded "approaches"), and an
occluded box shrinks without the object moving. Both are the price of
2.5D and both were accepted with the Depth Pro number on the table.

    python scripts/add_z25.py [--store lake/fresh_bench]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402


def z_of(series):
    """Relative depth per sample from a track's box-size series."""
    s = np.sqrt(np.maximum(series, 1.0))
    if len(s) >= 3:                              # median-3 smoothing
        sm = np.stack([np.r_[s[0], s[:-1]], s, np.r_[s[1:], s[-1]]])
        s = np.median(sm, 0)
    return np.median(s) / s


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    db = Store.open(str(store))
    tb = db.table("trajectories").scan()
    d = tb.to_pydict()
    n = len(d["ts"])
    area = ((np.asarray(d["x1"], np.float64) - np.asarray(d["x0"]))
            * (np.asarray(d["y1"], np.float64) - np.asarray(d["y0"])))
    # (stream, track_ts, t1) alone is NOT a track key: in a ~35-frame
    # segment, every track that spans the whole segment shares the same
    # start and end, so 133,912 tracks collapsed to 64,378 groups and z
    # was computed over MIXED box series of different objects. object_id
    # disambiguates; the only remaining collisions are double
    # detections of one object, whose sizes agree, so mixing them is
    # harmless where mixing distinct objects was not.
    key = np.array([hash((s, a, b, o)) for s, a, b, o in
                    zip(d["stream"], d["track_ts"], d["t1"],
                        d["object_id"])])
    ts = np.asarray(d["ts"], np.int64)
    pz = np.ones(n, np.float32)
    order = np.lexsort((ts, key))
    t0 = time.time()
    start = 0
    n_tracks = 0
    for i in range(1, n + 1):
        if i == n or key[order[i]] != key[order[start]]:
            idx = order[start:i]
            pz[idx] = z_of(area[idx])
            n_tracks += 1
            start = i
    print(f"{n:,} samples over {n_tracks:,} tracks  "
          f"({time.time()-t0:.0f}s)")
    print(f"pz: p05 {np.percentile(pz, 5):.3f}  med "
          f"{np.median(pz):.3f}  p95 {np.percentile(pz, 95):.3f}  "
          f"|pz-1|>5% of samples: {100.0*np.mean(np.abs(pz-1)>0.05):.1f}%")

    out = pa.table({**{c: pa.array(d[c], tb.schema.field(c).type)
                       for c in tb.column_names},
                    "pz": pa.array(pz, pa.float32())})
    db.table("trajectories").replace(
        out, kind="index", evolve=True,
        meta={"unit": "trajectory_sample",
              "point": "agent-contact overlap midpoint else centroid",
              "z": "2.5D box-scale proxy, per-track relative "
                   "(user decision 2026-08-01; Depth Pro measured "
                   "1.61 s/frame and declined)"})
    got = db.table("trajectories").scan()
    assert "pz" in got.schema.names and len(got) == n
    print(f"verified: pz on all {len(got):,} rows")


if __name__ == "__main__":
    main()
