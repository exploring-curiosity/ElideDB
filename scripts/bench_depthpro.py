"""USER GATE: Apple Depth Pro, measured on this machine before any
decision about z. The standing instruction is explicit - no 2.5D
fallback gets implemented on an assumption; the number gets measured,
reported, and the user chooses.

Depth Pro is metric monocular depth (absolute meters, not relative),
which is what a trajectory's z coordinate actually needs. It runs at a
fixed internal resolution (1536x1536 multi-scale), so it is expected to
be heavy; the open question is exactly how heavy on MPS, and what
sampling rate that cost permits:

    every frame        70,436 / corpus     the full-3D dream
    NGEOM per episode  ~9 x 2,097 = 18,873  the geometry sampling rate
    1 per event        15,175               z at transition time only
    4 per episode      8,388                start/end-ish anchors
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

MID = "apple/DepthPro-hf"
RATES = {"every frame": 70_436, "NGEOM/episode": 18_873,
         "1/event": 15_175, "4/episode": 8_388}


def main():
    import torch
    from transformers import (DepthProForDepthEstimation,
                              DepthProImageProcessorFast)
    from elidedb.device import pick
    dev, dtype = pick()
    print(f"loading {MID} on {dev} {dtype} ...", flush=True)
    t = time.time()
    proc = DepthProImageProcessorFast.from_pretrained(MID)
    model = DepthProForDepthEstimation.from_pretrained(
        MID, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    load_s = time.time() - t
    print(f"loaded in {load_s:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    im = rng.integers(0, 255, (480, 640, 3), np.uint8)
    px = proc(images=[im], return_tensors="pt").to(dev)
    if dtype != torch.float32:
        px["pixel_values"] = px["pixel_values"].to(dtype)

    with torch.no_grad():
        out = model(**px)                      # warm-up + sanity
    post = proc.post_process_depth_estimation(
        out, target_sizes=[(480, 640)])
    d = post[0]["predicted_depth"]
    finite = bool(torch.isfinite(d).all())
    print(f"depth map {tuple(d.shape)}  range "
          f"{float(d.min()):.2f}-{float(d.max()):.2f} m  finite={finite}",
          flush=True)
    if not finite and dtype != torch.float32:
        print("NaN under fp16 - retrying fp32 (the DINOv3 lesson)")
        model = DepthProForDepthEstimation.from_pretrained(
            MID, dtype=torch.float32, low_cpu_mem_usage=True).to(dev).eval()
        px["pixel_values"] = px["pixel_values"].float()
        dtype = torch.float32

    times = []
    for _ in range(8):
        a = time.time()
        with torch.no_grad():
            out = model(**px)
        if dev == "mps":
            torch.mps.synchronize()
        times.append(time.time() - a)
    s = float(np.median(times))
    rep = {"model": MID, "device": dev, "dtype": str(dtype),
           "load_s": round(load_s, 1), "s_per_frame": round(s, 2),
           "finite": finite}
    print(f"\n{ s:.2f} s/frame (median of 8, {dev}, {dtype})")
    print(f"{'sampling':<16}{'frames':>9}{'corpus':>12}{'per h video':>12}")
    for name, n in RATES.items():
        rep[name] = round(n * s / 60, 1)
        print(f"{name:<16}{n:>9,}{n*s/60:>10.0f} m{n*s/3600/3.91:>10.1f} h")
    (ROOT / "bench/bench_depthpro.json").write_text(json.dumps(rep, indent=1))
    print("wrote bench/bench_depthpro.json")


if __name__ == "__main__":
    main()
