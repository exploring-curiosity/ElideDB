"""STORY: does a trajectory feature separate the true episodes?

Actors were validated first (scripts/probe_actors.py): motion-derived
tracks persist 40-100% of an episode. This asks the next question, and
asks it the only way that counts - does the feature DISCRIMINATE across
the whole corpus, not does it look good on the positives.

The story of a manipulation is: something was carried, and it ended
somewhere it did not start. Three variables, all from motion, none
naming an object or a dataset:

  carry      total displacement of the longest-lived track, in frame
             widths (was something transported, or just jiggled)
  settle     |end - start| / path length (did it END somewhere else,
             or return - "put into" ends displaced, "wipe" returns)
  focus      mover area / total moving area (one thing moved, or the
             whole scene shifted)

Reported as ranking quality against the truthset: if these separate the
true episodes, the ranks concentrate; if not, they scatter uniformly and
we stop rather than embed 1,122 episodes on a hunch.

  python scripts/probe_story.py [--q 9] [--n 120]
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
SIZE = 192
MIN_BLOB = 30


def frames_of(stream, t0, n=12, size=SIZE):
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


def story_features(frames):
    """(carry, settle, focus) - or None when nothing moves."""
    import cv2
    if len(frames) < 4:
        return None
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    tracks, live = [], []
    mover_area, total_area = 0.0, 0.0
    for i in range(len(grey) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grey[i], grey[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        med = float(np.median(mag))
        mad = float(np.median(np.abs(mag - med))) + 1e-6
        mask = (mag > max(1.0, med + 4 * 1.4826 * mad)).astype(np.uint8)
        nlab, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        blobs = [(cents[j], float(stats[j, cv2.CC_STAT_AREA]))
                 for j in range(1, nlab)
                 if stats[j, cv2.CC_STAT_AREA] >= MIN_BLOB]
        total_area += sum(a for _, a in blobs)
        nxt = []
        for c, a in blobs:
            best, bd = None, 1e9
            for tr in live:
                d = float(np.hypot(*(np.array(tr[-1][0]) - np.array(c))))
                if d < bd:
                    best, bd = tr, d
            if best is not None and bd < 30:
                best.append((tuple(c), a))
                nxt.append(best)
            else:
                tr = [(tuple(c), a)]
                tracks.append(tr)
                nxt.append(tr)
        live = nxt
    if not tracks:
        return None
    main = max(tracks, key=len)
    if len(main) < 3:
        return None
    pts = np.array([p for p, _ in main], float)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    path = float(steps.sum())
    net = float(np.linalg.norm(pts[-1] - pts[0]))
    mover_area = float(np.mean([a for _, a in main]))
    carry = path / SIZE
    settle = net / (path + 1e-6)
    focus = mover_area * len(main) / (total_area + 1e-6)
    return carry, settle, focus


def main():
    argv = sys.argv
    qi = int(argv[argv.index("--q") + 1]) if "--q" in argv else 9
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 120
    db = Store.open("lake/bench")
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    ep = db.table("episodes").scan()
    keys = list(zip(ep.column("stream").to_pylist(),
                    [int(v) for v in ep.column("ts").to_pylist()]))
    pos = [k for k in keys if truth.get((qi, k[0], k[1])) == 1]
    rng = np.random.default_rng(0)
    negs = [k for k in keys if k not in pos]
    negs = [negs[i] for i in rng.choice(len(negs), min(n - len(pos), len(negs)),
                                        replace=False)]
    sample = [(k, 1) for k in pos] + [(k, 0) for k in negs]
    print(f"q{qi:02d}: {len(pos)} true + {len(negs)} random  "
          f"(pool {len(sample)})", flush=True)

    rows = []
    for i, (k, lab) in enumerate(sample):
        f = frames_of(*k)
        feat = story_features(f)
        if feat is None:
            continue
        rows.append((lab, *feat))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)
    if not rows:
        print("no usable episodes")
        return
    A = np.array(rows, float)
    lab = A[:, 0].astype(bool)
    names = ["carry", "settle", "focus"]
    print(f"\n{'feature':>8} {'true mean':>10} {'rest mean':>10} "
          f"{'separation':>11}  {'true ranks (of %d)' % len(A):>22}")
    for j, nm in enumerate(names):
        x = A[:, j + 1]
        order = np.argsort(-x)
        ranks = [int(np.where(order == i)[0][0]) + 1
                 for i in np.where(lab)[0]]
        sep = ((x[lab].mean() - x[~lab].mean())
               / (x[~lab].std() + 1e-9))
        print(f"{nm:>8} {x[lab].mean():>10.3f} {x[~lab].mean():>10.3f} "
              f"{sep:>10.2f}s  {sorted(ranks)}")
    print("\nseparation is in standard deviations of the background;\n"
          "|sep| < 0.5 means the feature does not discriminate here.")


if __name__ == "__main__":
    main()
