"""Fixed-RATE windowing and the throughput that follows from it.

THE DEFECT THIS REPLACES. relmo.vjs.sample_clip takes N frames spread across
the WHOLE episode, so a 7-second clip is sampled at ~4.6 fps and a 60-second
clip at ~0.53 fps. Both cost the same to encode, which looked like a feature
and is actually a bug: the long clip is sampled worse, and any event shorter
than its sampling period is invisible. On an hour of video it is one frame per
~56 seconds.

It also makes "cost per episode" the wrong unit. Production is not episodes -
it is timestamped frames in a parquet, with episode boundaries only implied by
gaps in time. So the unit here is SECONDS OF VIDEO PER SECOND OF COMPUTE, and
the sampling rate is fixed rather than normalised away.

    window   WIN_FRAMES frames at SAMPLE_FPS  ->  WIN_FRAMES/SAMPLE_FPS seconds
    hop      HOP_S seconds between window starts
    windows per hour = 3600 / HOP_S

WHAT IS LOST BY FIXING THE RATE. Whole-episode normalisation gave duration
invariance for free - a 3 s and a 14 s version of the same event mapped to the
same 32 samples. A fixed rate does not, so invariance has to come back from
encoding at SEVERAL rates and letting the matcher align across them. That is a
real cost of doing this correctly and is not yet built; this module fixes the
cost model and the sampling, not the invariance.

    python -m relmo.vjstream --video <path> --sample-fps 4 --hop 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjrec4 import CTX, WIN, record  # noqa: E402
from relmo.vjs import MODEL, TUBELET, probe_dims, read_frames  # noqa: E402

WIN_FRAMES = 32          # what the encoder consumes; 32 -> 8 trace steps
SAMPLE_FPS = 4.0         # fixed, NOT derived from clip length
HOP_S = 4.0


def windows(n_frames, src_fps, win_frames=WIN_FRAMES, sample_fps=SAMPLE_FPS,
            hop_s=HOP_S):
    """-> [(t0_s, t1_s, frame_indices)] at a FIXED sample rate."""
    span_s = win_frames / sample_fps
    step = src_fps / sample_fps                      # source frames per sample
    out = []
    t = 0.0
    while (t + span_s) * src_fps <= n_frames:
        start = t * src_fps
        idx = (start + np.arange(win_frames) * step).round().astype(int)
        idx = np.clip(idx, 0, n_frames - 1)
        out.append((t, t + span_s, idx))
        t += hop_s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--dataset", default="lr_TurnOnMicrowave")
    ap.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--hop", type=float, default=HOP_S)
    ap.add_argument("--win-frames", type=int, default=WIN_FRAMES)
    ap.add_argument("--limit-windows", type=int, default=40)
    a = ap.parse_args()

    import torch
    from transformers import VJEPA2Model
    from relmo.vjeval import REC

    vid = Path(a.video) if a.video else Path(
        R.read_manifest(a.dataset)["episodes"][0]["video"])
    src_fps = float(R.read_manifest(a.dataset).get("fps", 20)) if not a.video \
        else 20.0
    w_, h_ = probe_dims(vid)
    F = read_frames(vid, w_, h_)
    wins = windows(len(F), src_fps, a.win_frames, a.sample_fps, a.hop)
    span = a.win_frames / a.sample_fps
    print(f"{vid.name}: {len(F)} frames @ {src_fps:g} fps "
          f"= {len(F)/src_fps:.1f} s")
    print(f"window {a.win_frames} frames @ {a.sample_fps:g} fps = {span:.1f} s "
          f"| hop {a.hop:g} s -> {len(wins)} windows "
          f"({3600/a.hop:.0f} per video-hour)")
    if not wins:
        raise SystemExit("clip shorter than one window")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16
    z = np.load(REC / "rcasa" / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    model = VJEPA2Model.from_pretrained(MODEL, dtype=dt).to(dev).eval()

    use = wins[:a.limit_windows]
    t0 = time.time()
    for _, _, idx in use:
        record(model, torch, dev, F[idx], a.win_frames, cal, 6, dt)
    if dev == "mps":
        torch.mps.synchronize()
    per = (time.time() - t0) / len(use)

    # THE UNIT THAT MATTERS: video-seconds covered per compute-second.
    covered_per_window = a.hop            # each window advances the stream by hop
    rt = covered_per_window / per
    print(f"\n{'per window':24s} {per*1000:8.1f} ms")
    print(f"{'video-s per compute-s':24s} {rt:8.2f}x realtime  (one stream)")
    print(f"{'compute-h per video-h':24s} {1/rt:8.3f}")
    print(f"{'streams at realtime':24s} {rt:8.1f}")
    print(f"\nsampling density: {a.sample_fps:g} fps everywhere - a "
          f"{span:.0f}s window and a 1h recording are sampled the SAME, which "
          f"is the whole point of the change")
    R.log("vjstream", sample_fps=a.sample_fps, hop=a.hop,
          win_frames=a.win_frames, ms_per_window=round(per*1000, 1),
          realtime=round(rt, 3))


if __name__ == "__main__":
    main()
