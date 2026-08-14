"""Is the negative R2 a systematic OFFSET, or is the predictor useless here?

vjs_r2 measured R2 < 0 on all four arms while cosine stayed POSITIVE (+0.12).
Right direction, wrong magnitude. Two candidate causes, with opposite
consequences, and this separates them.

CAUSE A - single-encoder mismatch (fixable, and not my bug)
  V-JEPA trains the predictor to match a separate EMA TARGET encoder. The
  transformers implementation runs ONE encoder and slices its output for both
  context and target; the source even comments on this. Predictor output and
  target then come from systematically different distributions, which shows up
  as exactly this signature. If so, one GLOBAL affine correction - fitted on
  held-out episodes, not per clip - should push R2 positive, and the residual
  AFTER correction is the surprise signal the design needs.

CAUSE B - the corpus is off-distribution (not fixable by calibration)
  These are 320x240 synthetic kitchen renders upscaled to 256. V-JEPA 2 was
  pretrained on real internet video. If the predictor works on real footage
  and fails here, the problem is the evaluation substrate, not the model - a
  finding that matters far beyond this test, because every retrieval number
  this project has produced was measured on that same synthetic corpus.

PROTOCOL
  Fit the affine on FIT episodes, score on EVAL episodes. A per-clip fit would
  be cheating - it would absorb the very signal we are trying to measure.
  alpha and b are ONE scalar and ONE vector for the whole corpus.

      raw          R2 of the predictor as it comes out
      scaled       R2 after a single global scalar alpha
      affine       R2 after global alpha and offset b  (fitted / held out)
  and the same three on a REAL video for comparison.

    python -m relmo.vjs_cal --episodes 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import GRID, MODEL, TUBELET, read_frames, to_tensor  # noqa: E402
from relmo.vjs_r2 import clip_native, m_block, m_temporal  # noqa: E402

REAL = Path("/Users/sudharshanramesh/Studies/MyProjects/StreetDex/"
            "Data/oxford_clip/drive.mp4")


def dims(mp4):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of",
                        "csv=p=0:s=x", str(mp4)],
                       stdout=subprocess.PIPE, check=True)
    return (int(v) for v in p.stdout.decode().strip().split("x"))


def collect(model, torch, dev, mp4, n_frames, mask, n_t, n_sp, n_clips=1):
    """Return (P, T) float32 numpy for one video."""
    w, h = dims(mp4)
    F = read_frames(mp4, w, h)
    if len(F) < n_frames:
        return None
    outP, outT = [], []
    starts = np.linspace(0, len(F) - n_frames, n_clips).round().astype(int)
    for s in starts:
        px = to_tensor(F[s:s + n_frames], torch, dev, torch.float32)
        ctx, tgt = mask(n_t, n_sp, dev, torch) if mask is m_temporal else \
            mask(n_t, n_sp, dev, torch, np.random.default_rng(1))
        with torch.no_grad():
            po = model(pixel_values_videos=px, context_mask=[ctx],
                       target_mask=[tgt]).predictor_output
        outP.append(po.last_hidden_state.float()[0].cpu().numpy())
        outT.append(po.target_hidden_state.float()[0].cpu().numpy())
    return np.concatenate(outP), np.concatenate(outT)


def r2(P, T):
    mu = T.mean(0, keepdims=True)
    return float(1 - ((P - T) ** 2).sum() / (((T - mu) ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--dataset", default="rcasa")
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    print("  loaded", flush=True)

    n_t, n_sp = a.frames // TUBELET, GRID * GRID
    d = R.dataset_dir(a.dataset) / "shard_0000"
    eps = sorted(p for p in d.iterdir() if (p / "frames.mp4").exists())
    rng = np.random.default_rng(0)
    eps = [eps[i] for i in rng.choice(len(eps), min(a.episodes, len(eps)),
                                      replace=False)]
    half = len(eps) // 2
    res = {}

    for mname, mask in (("block", m_block), ("temporal", m_temporal)):
        fitP, fitT, evP, evT = [], [], [], []
        for i, p in enumerate(tqdm(eps, unit="ep", desc=f"sim/{mname}")):
            g = collect(model, torch, dev, p / "frames.mp4", a.frames, mask,
                        n_t, n_sp)
            if g is None:
                continue
            (fitP if i < half else evP).append(g[0])
            (fitT if i < half else evT).append(g[1])
        Pf, Tf = np.concatenate(fitP), np.concatenate(fitT)
        Pe, Te = np.concatenate(evP), np.concatenate(evT)
        # ONE global scalar, fitted on FIT episodes only
        alpha = float((Pf * Tf).sum() / ((Pf * Pf).sum() + 1e-12))
        b = (Tf - alpha * Pf).mean(0, keepdims=True)
        res[f"sim/{mname}"] = dict(raw=r2(Pe, Te),
                                   scaled=r2(alpha * Pe, Te),
                                   affine=r2(alpha * Pe + b, Te),
                                   alpha=alpha)

    if REAL.exists():
        for mname, mask in (("block", m_block), ("temporal", m_temporal)):
            g = collect(model, torch, dev, REAL, a.frames, mask, n_t, n_sp,
                        n_clips=6)
            if g is None:
                continue
            P, T = g
            n = len(P) // 2
            alpha = float((P[:n] * T[:n]).sum() / ((P[:n] ** 2).sum() + 1e-12))
            b = (T[:n] - alpha * P[:n]).mean(0, keepdims=True)
            res[f"REAL/{mname}"] = dict(raw=r2(P[n:], T[n:]),
                                        scaled=r2(alpha * P[n:], T[n:]),
                                        affine=r2(alpha * P[n:] + b, T[n:]),
                                        alpha=alpha)

    print("\n" + "=" * 70)
    print(f"{'video/mask':18s} {'raw R2':>9s} {'scaled':>9s} {'affine':>9s} "
          f"{'alpha':>8s}")
    print("-" * 70)
    for k in sorted(res):
        v = res[k]
        print(f"{k:18s} {v['raw']:+9.4f} {v['scaled']:+9.4f} "
              f"{v['affine']:+9.4f} {v['alpha']:8.3f}")
    print("=" * 70)
    print("affine is fitted on held-out episodes, never per clip.")
    print("If affine >> raw: systematic offset (CAUSE A, fixable).")
    print("If REAL >> sim:   synthetic corpus is off-distribution (CAUSE B).")
    R.log("vjs_cal", dataset=a.dataset, episodes=len(eps), frames=a.frames,
          **{f"{k}_{kk}".replace("/", "_"): round(vv, 4)
             for k, d_ in res.items() for kk, vv in d_.items()})


if __name__ == "__main__":
    main()
