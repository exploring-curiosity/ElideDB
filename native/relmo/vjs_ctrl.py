"""POSITIVE CONTROL for vjs.py: is the predictor predicting anything at all?

vjs.py returned ratio 1.01 - residual identical on static and moving patches -
which reads as "my scene-robustness claim is refuted". But a flat residual of
~95 everywhere is also exactly what you get if the predictor is contributing
NOTHING and the residual is just the norm of the target latent. Those two
situations produce the same headline number and opposite conclusions.

The F-gate incident on this project failed in both directions until a positive
control was built. Same discipline here: prove the instrument works before
trusting what it says.

Baselines, all on the SAME target tokens:
    ||target||              residual if the predictor output were exactly 0
    ||ctx_mean - target||   predict the mean context token (no structure)
    ||persist - target||    predict each target token = the co-located token
                            in the LAST context timestep (a persistence /
                            "nothing changes" baseline - the video equivalent
                            of const-velocity, which is the baseline nothing
                            in this project has beaten yet)
    ||pred - target||       the actual predictor

And two mask shapes:
    TEMPORAL  context = first half of time, target = second half
              (what vjs.py does; FAR off-distribution for V-JEPA, whose
              training masks are 3D multi-blocks, not a causal time split)
    BLOCK     context = everything except a few random spatial blocks spanning
              all time; target = those blocks
              (close to V-JEPA's actual pretraining objective)

READ IT LIKE THIS:
  If BLOCK residual << ||target|| but TEMPORAL residual ~= ||target||, the
  predictor works and simply cannot do causal future prediction. vjs.py's
  verdict is then VOID, not refuted - wrong tool, wrong question.
  If BOTH ~= ||target||, the predictor is inert as loaded and the whole
  V-JEPA-2-as-predictor plan is dead.
  If TEMPORAL beats persistence, future prediction is real and vjs.py's
  refutation stands.

    python -m relmo.vjs_ctrl --episodes 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import (CROP, GRID, MODEL, TUBELET, read_frames,  # noqa: E402
                       sample_clip, to_tensor)


def masks_temporal(n_t, n_sp, dev, torch):
    half = n_t // 2
    ctx = torch.arange(0, half * n_sp, device=dev).unsqueeze(0)
    tgt = torch.arange(half * n_sp, n_t * n_sp, device=dev).unsqueeze(0)
    return ctx, tgt


def masks_block(n_t, n_sp, dev, torch, rng, n_blocks=4, bh=4, bw=4):
    """Random spatial blocks spanning ALL time - V-JEPA's own mask shape."""
    keep = np.ones((n_t, GRID, GRID), bool)
    for _ in range(n_blocks):
        y = rng.integers(0, GRID - bh + 1)
        x = rng.integers(0, GRID - bw + 1)
        keep[:, y:y + bh, x:x + bw] = False
    flat = keep.reshape(-1)
    ctx = torch.tensor(np.nonzero(flat)[0], device=dev).unsqueeze(0)
    tgt = torch.tensor(np.nonzero(~flat)[0], device=dev).unsqueeze(0)
    return ctx, tgt


def persistence(seq, ctx_idx, tgt_idx, n_sp, torch):
    """Each target token <- co-located token at the last context timestep."""
    last_t = int(ctx_idx.max().item()) // n_sp
    base = seq[0, last_t * n_sp:(last_t + 1) * n_sp]        # (n_sp, D)
    sp = (tgt_idx[0] % n_sp).long()
    return base[sp].unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
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

    d = R.dataset_dir(a.dataset) / "shard_0000"
    eps = sorted(p for p in d.iterdir() if (p / "frames.mp4").exists())
    rng = np.random.default_rng(0)
    eps = [eps[i] for i in rng.choice(len(eps), min(a.episodes, len(eps)),
                                      replace=False)]
    n_t = a.frames // TUBELET
    n_sp = GRID * GRID
    acc = {}

    for p in tqdm(eps, unit="ep", desc="ctrl"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
             str(p / "frames.mp4")], stdout=subprocess.PIPE, check=True)
        w, h = (int(v) for v in probe.stdout.decode().strip().split("x"))
        F = read_frames(p / "frames.mp4", w, h)
        if len(F) < a.frames:
            continue
        px = to_tensor(sample_clip(F, a.frames), torch, dev, torch.float32)
        with torch.no_grad():
            seq = model(pixel_values_videos=px,
                        skip_predictor=True).last_hidden_state
        for name, mk in (("TEMPORAL", masks_temporal(n_t, n_sp, dev, torch)),
                         ("BLOCK", masks_block(n_t, n_sp, dev, torch,
                                               np.random.default_rng(1)))):
            ctx, tgt = mk
            with torch.no_grad():
                po = model(pixel_values_videos=px, context_mask=[ctx],
                           target_mask=[tgt]).predictor_output
            T = po.target_hidden_state.float()
            P = po.last_hidden_state.float()
            ctx_mean = seq[0, ctx[0].long()].float().mean(0, keepdim=True)
            pers = persistence(seq.float(), ctx, tgt, n_sp, torch)
            row = dict(
                target_norm=float(T.norm(dim=-1).mean()),
                pred=float((P - T).norm(dim=-1).mean()),
                ctx_mean=float((ctx_mean - T).norm(dim=-1).mean()),
                persist=float((pers - T).norm(dim=-1).mean()),
            )
            acc.setdefault(name, []).append(row)

    print("\n" + "=" * 72)
    print(f"{'mask':10s} {'||target||':>11s} {'predictor':>11s} "
          f"{'ctx-mean':>11s} {'persistence':>12s}   {'verdict'}")
    print("-" * 72)
    out = {}
    for name, rows in acc.items():
        g = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        beats_zero = g["pred"] < 0.95 * g["target_norm"]
        beats_pers = g["pred"] < 0.95 * g["persist"]
        v = ("predicts + beats persistence" if beats_zero and beats_pers else
             "predicts, LOSES to persistence" if beats_zero else
             "INERT - no better than predicting nothing")
        print(f"{name:10s} {g['target_norm']:11.2f} {g['pred']:11.2f} "
              f"{g['ctx_mean']:11.2f} {g['persist']:12.2f}   {v}")
        out[name] = dict(g, verdict=v)
    print("=" * 72)
    R.log("vjs_ctrl", dataset=a.dataset, episodes=len(eps),
          frames=a.frames, **{f"{k}_{kk}": vv for k, d_ in out.items()
                              for kk, vv in d_.items()})


if __name__ == "__main__":
    main()
