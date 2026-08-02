"""STEP 1 gate: what does DINOv3 cost on THIS machine?

Nothing full-corpus runs until the projection is known. This downloads
the two candidate variants, times them batched on MPS at the canonical
size, and prints the projection against the numbers every later run
will be charged at:

    frames  18,014 / hour of video  (70,436 across fresh_bench's 3.91 h)
    crops   75,216 / hour           (73,521 tracks x ~4 exemplar views)

The output is minutes per hour of video and minutes for the full
corpus, per variant, per granularity. The A/B and the full builds are
scheduled from these numbers, not from hope.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

VARIANTS = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
]
FRAMES_PER_H = 18_014
CROPS_PER_H = 75_216
CORPUS_H = 3.91


def bench(mid):
    from elidedb import dinov3
    rng = np.random.default_rng(0)
    # frame-shaped (segments are 640x480-ish) and crop-shaped inputs;
    # the processor resizes both to the canonical square, so shape only
    # exercises the resize cost realistically
    frames = [rng.integers(0, 255, (480, 640, 3), np.uint8)
              for _ in range(96)]
    crops = [rng.integers(0, 255, (h, w, 3), np.uint8)
             for h, w in rng.integers(24, 220, (96, 2))]
    t = time.time()
    dinov3._load(mid)
    load_s = time.time() - t
    dinov3.embed(frames[:8], mid=mid)              # warm up MPS graphs
    out = {"model": mid.split("/")[-1], "load_s": round(load_s, 1)}
    for name, imgs in (("frame", frames), ("crop", crops)):
        for b in (32, 64):
            t = time.time()
            v = dinov3.embed(imgs, batch=b, mid=mid)
            ms = (time.time() - t) / len(imgs) * 1e3
            out[f"{name}_b{b}_ms"] = round(ms, 2)
        out[f"{name}_dim"] = int(v.shape[1])
    ms_f = out["frame_b64_ms"]
    ms_c = out["crop_b64_ms"]
    out["frames_min_per_h"] = round(FRAMES_PER_H * ms_f / 6e4, 2)
    out["crops_min_per_h"] = round(CROPS_PER_H * ms_c / 6e4, 2)
    out["corpus_frames_min"] = round(
        FRAMES_PER_H * CORPUS_H * ms_f / 6e4, 1)
    out["corpus_crops_min"] = round(
        CROPS_PER_H * CORPUS_H * ms_c / 6e4, 1)
    return out


def main():
    rows = []
    for mid in VARIANTS:
        print(f"== {mid}", flush=True)
        try:
            r = bench(mid)
        except Exception as e:
            r = {"model": mid.split("/")[-1], "error": str(e)[:300]}
        rows.append(r)
        print(json.dumps(r, indent=1), flush=True)
    (ROOT / "bench/bench_dinov3.json").write_text(
        json.dumps(rows, indent=1))
    print("wrote bench/bench_dinov3.json")


if __name__ == "__main__":
    main()
