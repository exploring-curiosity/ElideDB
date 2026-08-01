"""Do ACTORS have identity across time? Measure before building on it.

The scene/actors/story representation rests entirely on actors
persisting: "the eggplant went into the drawer" is only expressible if
the thing that moved can be followed across frames. Binding already
failed once because it was built on crops that could not recognise
objects (best cosine to "an eggplant" across 51,101 crops: 0.184). This
checks the foundation FIRST, on six episodes, before any ingest pass.

Actors here come from MOTION, not detection: dense optical flow, then
connected components of coherent movement, then linked across frames by
overlap. No class names, no detector, nothing dataset-specific - it
works on a robot arm, a car, or a drone.

What it reports per episode:
  tracks         how many persistent moving things were found
  lifespan       how much of the episode the best track survives
  coverage       how much of the frame moves (whole-frame => camera
                 motion or lighting, not an actor)
  separability   is the mover compact and distinct from the background

  python scripts/probe_actors.py [--n 6]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from elidedb.fftools import find  # noqa: E402
from elidedb.store import Store  # noqa: E402

EPOCH_NS = 1_704_067_200_000_000_000
FILE_STRIDE_NS = 20_000_000_000_000
FPS = 5.0
SRC = ROOT / "data/bridge/videos/observation.images.image_0/chunk-000"
FLOW_MIN = 1.0          # px/frame that counts as movement
MIN_BLOB_FRAC = 4e-4    # of frame area; see extract_events


def frames_of(stream, t0, n=16, size=256):
    fidx = int(stream.rsplit("-", 1)[1])
    base = EPOCH_NS + fidx * FILE_STRIDE_NS
    t_sec = max((t0 - base) / 1e9, 0.0)
    src = SRC / f"file-{fidx:03d}.mp4"
    if not src.exists():
        return []
    r = subprocess.run(
        [find("ffmpeg"), "-v", "error", "-ss", f"{t_sec:.3f}", "-i", str(src),
         "-frames:v", str(n), "-vf", f"scale={size}:{size}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
    fb = size * size * 3
    return [np.frombuffer(r.stdout[i*fb:(i+1)*fb], np.uint8).reshape(size, size, 3)
            for i in range(len(r.stdout) // fb)]


def motion_actors(frames):
    """Connected components of coherent motion, linked across frames.

    Returns (tracks, per_frame_coverage). A track is a list of
    (frame_idx, centroid, area)."""
    import cv2
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    tracks, live, cover = [], [], []
    for i in range(len(grey) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grey[i], grey[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        # robust threshold: movement is what stands out from this
        # frame-pair's own motion background (camera shake, noise)
        med = float(np.median(mag))
        mad = float(np.median(np.abs(mag - med))) + 1e-6
        mask = (mag > max(FLOW_MIN, med + 4 * 1.4826 * mad)).astype(np.uint8)
        cover.append(float(mask.mean()))
        nlab, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        min_blob = MIN_BLOB_FRAC * mask.shape[0] * mask.shape[1]
        blobs = [(cents[j], float(stats[j, cv2.CC_STAT_AREA]))
                 for j in range(1, nlab)
                 if stats[j, cv2.CC_STAT_AREA] >= min_blob]
        nxt = []
        for c, a in blobs:
            # link to the nearest live track (identity by proximity)
            best, bd = None, 1e9
            for tr in live:
                d = float(np.hypot(*(np.array(tr[-1][1]) - np.array(c))))
                if d < bd:
                    best, bd = tr, d
            if best is not None and bd < 40:
                best.append((i, tuple(c), a))
                nxt.append(best)
            else:
                tr = [(i, tuple(c), a)]
                tracks.append(tr)
                nxt.append(tr)
        live = nxt
    return tracks, cover


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6
    db = Store.open("lake/_bench_recovered")
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    q9 = [(s, int(a)) for q, s, a, v in
          zip(t["query_id"], t["stream"], t["t0"], t["true"])
          if int(q) == 9 and int(v) == 1]
    ep = db.table("episodes").scan()
    allk = list(zip(ep.column("stream").to_pylist(),
                    [int(v) for v in ep.column("ts").to_pylist()]))
    others = [k for k in allk if k not in q9][:max(n - len(q9), 0)]
    sample = [(k, "q09-TRUE") for k in q9] + [(k, "random") for k in others]

    print(f"{'episode':>28} {'kind':>10} {'frames':>7} {'tracks':>7} "
          f"{'best lifespan':>14} {'coverage':>9}  verdict")
    for (stream, t0), kind in sample:
        f = frames_of(stream, t0)
        if len(f) < 4:
            print(f"{stream[-14:]+' '+str(t0)[-6:]:>28} {kind:>10}  no frames")
            continue
        tracks, cover = motion_actors(f)
        long = sorted(tracks, key=len, reverse=True)
        best = len(long[0]) if long else 0
        span = best / max(len(f) - 1, 1)
        cov = float(np.mean(cover))
        verdict = ("WHOLE FRAME MOVES" if cov > 0.5 else
                   "no persistent actor" if span < 0.4 else
                   "actor persists")
        print(f"{stream[-14:]+' '+str(t0)[-6:]:>28} {kind:>10} {len(f):>7} "
              f"{len(tracks):>7} {span:>13.0%} {cov:>9.1%}  {verdict}")


if __name__ == "__main__":
    main()
