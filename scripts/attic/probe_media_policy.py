"""Measure the media rendition dial on REAL bench media, both ways.

The central tension (CLAUDE.md, and the Lance line of work): compression
fights random access. A longer GOP shrinks the file but forces a window
query to decode more frames it did not ask for. This measures BOTH ends
on one real source file, so the policy is chosen by numbers:

  size            bytes of the rendition
  window bytes    bytes a 2 s window must read (the GOP span covering
                  it) — the elision metric, applied to media

Nothing is written to any store. Usage:
  python scripts/probe_media_policy.py <source.mp4> [--fps 5]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb.fftools import find  # noqa: E402
from elidedb.video import scan_video_packets  # noqa: E402

WINDOW_S = 2.0


def window_bytes(rows, fps, window_s=WINDOW_S) -> tuple[float, float]:
    """Mean/max bytes a window read must fetch: from the keyframe at or
    before the window start, through the last packet inside it — exactly
    what the byte-range decode path reads."""
    size = rows["packet_size"]
    key = rows["keyframe"]
    n = len(size)
    span = max(int(round(window_s * fps)), 1)
    costs = []
    for start in range(0, max(n - span, 1), max(span // 2, 1)):
        k = start
        while k > 0 and not key[k]:
            k -= 1
        costs.append(sum(size[k:start + span]))
    return (sum(costs) / len(costs), max(costs)) if costs else (0.0, 0.0)


def probe(src: Path, codec: str, crf: int, gop_s: float, fps: float,
          tmp: Path) -> dict:
    g = max(1, round(gop_s * fps))
    enc = "libx265" if codec == "hevc" else "libx264"
    out = tmp / f"{codec}_crf{crf}_g{g}.{codec}"
    if not out.exists():
        subprocess.run(
            [find("ffmpeg"), "-v", "error", "-y", "-i", str(src),
             "-c:v", enc, "-preset", "fast", "-crf", str(crf),
             "-g", str(g), "-keyint_min", str(g), "-an",
             *(["-x265-params", "log-level=error"] if codec == "hevc"
               else []),
             "-f", codec, str(out)], check=True)
    rows = scan_video_packets(out)
    mean_w, max_w = window_bytes(rows, fps)
    return {"codec": codec, "crf": crf, "gop_s": gop_s,
            "bytes": os.path.getsize(out), "frames": len(rows["ts"]),
            "keyframes": sum(1 for k in rows["keyframe"] if k),
            "window_mean": mean_w, "window_max": max_w}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fps = 5.0
    if "--fps" in sys.argv:
        fps = float(sys.argv[sys.argv.index("--fps") + 1])
    src = Path(args[0])
    src_bytes = os.path.getsize(src)
    print(f"source {src.name}  {src_bytes/1e6:.1f} MB  fps {fps}")
    tmp = Path(tempfile.mkdtemp(prefix="mediapolicy_"))
    print(f"{'codec':6} {'crf':>4} {'gop_s':>6} {'MB':>8} {'vs src':>7} "
          f"{'keyfr':>6} {'win KB':>8} {'win max KB':>11}")
    for codec, crf, gop in [("h264", 26, 1.0),     # the current policy
                            ("hevc", 26, 1.0),
                            ("hevc", 28, 1.0),
                            ("hevc", 28, 2.0),
                            ("hevc", 30, 2.0),
                            ("hevc", 28, 4.0)]:
        r = probe(src, codec, crf, gop, fps, tmp)
        print(f"{r['codec']:6} {r['crf']:>4} {r['gop_s']:>6} "
              f"{r['bytes']/1e6:>8.1f} {src_bytes/r['bytes']:>6.2f}x "
              f"{r['keyframes']:>6} {r['window_mean']/1e3:>8.1f} "
              f"{r['window_max']/1e3:>11.1f}")
    print(f"\nscratch: {tmp}")


if __name__ == "__main__":
    main()
