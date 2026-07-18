#!/usr/bin/env python3
"""Oxford RobotCar sample -> StreetDex-ready artifacts (ETL adapter).

StreetDex's generic entry points are `sdx index-video` (any video + ns
timestamps) and `sdx ingest-csv` (ts_ns + numeric columns). This adapter maps
the RobotCar layout onto them, WITHOUT touching the raw files:

- camera dirs (raw Bayer PNGs, one file per frame, filename = epoch µs) ->
  demosaic -> JPEG -> ONE packed .mjpeg per camera + a .ts sidecar (ns per
  frame). A pile of small files has no random-access story; a packed MJPEG
  stream + SFI gives frame-exact byte-range reads with the existing decoder
  (identical to the lab captures' raw-MJPEG segments). The packed file is a
  derived immutable artifact and lives in the store, like SDX files do.
- gps/{gps,ins}.csv -> ts_ns + numeric columns (µs->ns, non-numeric dropped)
- lms_front / lms_rear (2D scans: x,y,reflectance f64) and ldmrs (3D x,y,z)
  -> per-scan summary rows (n_points, mean/min/max range) so lidar activity
  is queryable/alignable as a scalar stream.

Bayer patterns per the RobotCar SDK: stereo (Bumblebee XB3) = GBRG, mono
(Grasshopper2) = RGGB. OpenCV constant names refer to the 2nd row's 2nd/3rd
pixels, hence GBRG->COLOR_BayerGR, RGGB->COLOR_BayerBG (verified visually).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

CAMERAS = {  # dir -> (bayer cv2 code, stream id)
    "stereo/centre": (cv2.COLOR_BayerGR2BGR, "stereo/centre"),
    "stereo/left": (cv2.COLOR_BayerGR2BGR, "stereo/left"),
    "stereo/right": (cv2.COLOR_BayerGR2BGR, "stereo/right"),
    "mono_left": (cv2.COLOR_BayerBG2BGR, "mono_left"),
    "mono_right": (cv2.COLOR_BayerBG2BGR, "mono_right"),
    "mono_rear": (cv2.COLOR_BayerBG2BGR, "mono_rear"),
}


def pack_camera(src_dir: Path, code: int, out_base: Path, quality: int) -> int:
    frames = sorted(src_dir.glob("*.png"))
    t0 = time.time()
    raw_path = out_base.with_suffix(".mjpeg")
    with open(raw_path, "wb") as vid, open(out_base.with_suffix(".ts"),
                                           "w") as ts:
        for p in frames:
            raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            bgr = cv2.cvtColor(raw, code)
            ok, jpg = cv2.imencode(".jpg", bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError(f"jpeg encode failed: {p}")
            vid.write(jpg.tobytes())
            ts.write(f"{int(p.stem) * 1000}\n")  # µs filename -> ns
    # Remux (stream copy, no re-encode) into AVI: a bare JPEG concatenation
    # forces libav's raw-mjpeg parser to guess packet byte positions
    # (chunk-granular), while AVI chunk headers make av_packet->pos
    # frame-exact — which is what SFI byte-range reads are built on.
    import subprocess
    avi_path = out_base.with_suffix(".avi")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "mjpeg",
                    "-i", str(raw_path), "-c:v", "copy", str(avi_path)],
                   check=True)
    raw_path.unlink()
    print(f"  packed {src_dir.name}: {len(frames)} frames in "
          f"{time.time() - t0:.1f} s -> {avi_path.name}")
    return len(frames)


def convert_csv(src: Path, out: Path) -> int:
    with open(src) as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    # Keep ts + every column that parses as a number in the first data row.
    keep = [0]
    for c in range(1, len(header)):
        try:
            float(body[0][c])
            keep.append(c)
        except ValueError:
            pass
    body.sort(key=lambda r: int(r[0]))  # SDX requires non-decreasing ts
    with open(out, "w") as f:
        f.write(",".join(["ts_ns"] + [header[c] for c in keep[1:]]) + "\n")
        for r in body:
            f.write(",".join([str(int(r[0]) * 1000)] +
                             [r[c] for c in keep[1:]]) + "\n")
    return len(body)


def lidar_summary(src_dir: Path, dims: int, out: Path) -> int:
    files = sorted(src_dir.glob("*.bin"))
    with open(out, "w") as f:
        f.write("ts_ns,n_points,mean_range,min_range,max_range\n")
        for p in files:
            a = np.fromfile(p, np.float64)
            pts = a.reshape(3, -1)  # SDK layout: 3 x N (x, y, refl|z)
            r = np.hypot(pts[0], pts[1]) if dims == 2 else \
                np.sqrt(pts[0] ** 2 + pts[1] ** 2 + pts[2] ** 2)
            f.write(f"{int(p.stem) * 1000},{r.size},{r.mean():.3f},"
                    f"{r.min():.3f},{r.max():.3f}\n")
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()
    ds = Path(args.dataset)
    packed = Path(args.store) / "packed"
    packed.mkdir(parents=True, exist_ok=True)

    for rel, (code, _) in CAMERAS.items():
        if (ds / rel).is_dir():
            pack_camera(ds / rel, code, packed / rel.replace("/", "_"),
                        args.quality)
    for name in ("gps", "ins"):
        src = ds / "gps" / f"{name}.csv"
        if src.exists():
            n = convert_csv(src, packed / f"{name}.csv")
            print(f"  converted {name}.csv: {n} rows")
    for name, dims in (("lms_front", 2), ("lms_rear", 2), ("ldmrs", 3)):
        if (ds / name).is_dir():
            n = lidar_summary(ds / name, dims, packed / f"{name}.csv")
            print(f"  summarized {name}: {n} scans")
    print(f"prepared artifacts in {packed}")


if __name__ == "__main__":
    main()
