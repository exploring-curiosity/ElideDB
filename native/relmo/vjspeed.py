"""Where the ingest time goes, and what compression buys. Measured, per stage.

The read path must run FASTER THAN REALTIME per camera, because production
ingests several cameras at once. So the unit here is not seconds per episode -
it is the ratio of video-seconds processed to wall-seconds spent, per camera.
A 3-camera rig needs >3x to keep up in one process.

Stages, per episode:
  decode      ffmpeg -> frames
  vjepa_enc   ViT-L, 24 layers, 1024-d, 64 frames -> 32 temporal x 256 spatial
              = 8192 tokens. Expected to dominate everything else combined.
  vjepa_pred  12 layers, 384-d, called once per 4-step window
  siglip      base/16 at 224, 24 frames
  rank        the trained recurrence + pooled cosine, 337k params

    python -m relmo.vjspeed --episodes 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjrec4 import CTX, WIN  # noqa: E402
from relmo.vjs import (GRID, MODEL, TUBELET, probe_dims,  # noqa: E402
                       read_frames, sample_clip, to_tensor)


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, dt):
        self.t.setdefault(k, []).append(dt)

    def report(self, n_ep, video_seconds, extra=""):
        tot = sum(np.sum(v) for v in self.t.values())
        print(f"\n{'stage':14s} {'ms/ep':>9s} {'share':>7s}")
        for k, v in sorted(self.t.items(), key=lambda x: -np.sum(x[1])):
            s = float(np.sum(v))
            print(f"{k:14s} {1000*s/n_ep:9.1f} {s/tot:7.1%}")
        per = tot / n_ep
        print(f"{'TOTAL':14s} {1000*per:9.1f}")
        print(f"\nvideo seconds per episode : {video_seconds:.2f}")
        print(f"wall seconds per episode  : {per:.3f}")
        print(f"REALTIME FACTOR           : {video_seconds/per:.2f}x  {extra}")
        return video_seconds / per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--no-siglip", action="store_true")
    ap.add_argument("--batch", type=int, default=1,
                    help="episodes encoded per forward - the multi-camera case")
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, VJEPA2Model
    from relmo.vjsig import MODEL as SIG_MODEL

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    man = R.read_manifest("rcasa")
    eps = man["episodes"][:a.episodes]
    fps = 20.0

    print(f"loading models onto {dev} as {a.dtype}...", flush=True)
    enc = VJEPA2Model.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    sig = None if a.no_siglip else AutoModel.from_pretrained(
        SIG_MODEL, dtype=dt).to(dev).eval()
    n_t, n_sp = a.frames // TUBELET, GRID * GRID
    T = Timer()
    vid_s = []

    def sync():
        if dev == "mps":
            torch.mps.synchronize()

    for e in eps:
        d = R.dataset_dir("rcasa") / e["shard"] / e["id"] / "frames.mp4"
        t0 = time.time()
        w_, h_ = probe_dims(d)
        F = read_frames(d, w_, h_)
        T.add("decode", time.time() - t0)
        vid_s.append(len(F) / fps)
        clip = sample_clip(F, a.frames)

        t0 = time.time()
        px = to_tensor(clip, torch, dev, dt)
        if a.batch > 1:
            px = px.repeat(a.batch, 1, 1, 1, 1)
        with torch.no_grad():
            out = enc.encoder(pixel_values_videos=px)
            seq = out.last_hidden_state
        sync()
        T.add("vjepa_enc", (time.time() - t0) / a.batch)

        t0 = time.time()
        with torch.no_grad():
            for c in range(CTX, n_t, WIN):
                hi = min(c + WIN, n_t)
                ctx = torch.arange((c - CTX) * n_sp, c * n_sp,
                                   device=dev).unsqueeze(0)
                tgt = torch.arange(c * n_sp, hi * n_sp,
                                   device=dev).unsqueeze(0)
                enc.predictor(encoder_hidden_states=seq[:1],
                              context_mask=[ctx], target_mask=[tgt])
        sync()
        T.add("vjepa_pred", time.time() - t0)

        if sig is not None:
            t0 = time.time()
            import torch.nn.functional as Fn
            steps = list(range(CTX, n_t))
            x = torch.tensor(clip[[t * TUBELET for t in steps]]).permute(
                0, 3, 1, 2).float().div_(255.)
            x = Fn.interpolate(x, size=(224, 224), mode="bilinear",
                               align_corners=False)
            x = ((x - 0.5) / 0.5).to(dev, dt)
            with torch.no_grad():
                sig.get_image_features(pixel_values=x)
            sync()
            T.add("siglip", time.time() - t0)

    v = float(np.mean(vid_s))
    rt = T.report(len(eps), v,
                  f"({a.dtype}, {a.frames} frames, batch {a.batch}"
                  f"{', no siglip' if a.no_siglip else ''})")
    R.log("vjspeed", dtype=a.dtype, frames=a.frames, batch=a.batch,
          siglip=not a.no_siglip, realtime=round(rt, 3))


if __name__ == "__main__":
    main()
