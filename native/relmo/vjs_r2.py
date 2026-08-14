"""Does the V-JEPA 2 predictor explain ANY variance? R2, not raw L2.

vjs_ctrl reported raw L2 norms and I read a verdict off them. That was the
wrong metric: L2 distance in a 1024-d anisotropic latent space is dominated by
the common mean component, so every number lands near 100 and differences of a
few percent are unreadable. It also let me print "predicts + beats
persistence" for a predictor that was LOSING to a constant.

The right measure is the one used everywhere else in this project:

    R2 = 1 - SS_res / SS_tot
    SS_res = sum || pred   - target ||^2
    SS_tot = sum || mu_tgt - target ||^2      mu_tgt = mean of TARGET tokens

R2 = 0 means "no better than predicting the mean target token". R2 < 0 means
worse than that. Pooled SSE/SST across the clip, never an average of ratios.

Two things vary, because either could be my bug rather than the model's:

  SAMPLING   uniform  64 frames spread over the WHOLE episode (what vjs.py
                      did - consecutive tokens ~10 real frames apart, far
                      off the frame rate V-JEPA 2 was trained at)
             native   64 CONTIGUOUS frames from the middle of the episode
                      (the clip shape the checkpoint expects)

  MASK       temporal context = first half of time -> predict second half
             block    random spatial blocks spanning all time (V-JEPA's own
                      pretraining mask shape; the in-distribution control)

Also reported: cosine between centred pred and centred target, which is scale
free and says whether the DIRECTION is right even if the magnitude is not.

DECISION RULE, fixed before running:
  block/native R2 <= 0   -> the predictor is unusable as shipped through this
                            API. The V-JEPA-2-as-predictor plan is dead and no
                            amount of downstream cleverness rescues it.
  block R2 > 0, temporal R2 <= 0
                        -> it predicts, but cannot do causal future
                            prediction. The design needs a different predictor
                            or a trained temporal head; vjs.py's scene verdict
                            stays VOID either way.
  temporal R2 > 0       -> future prediction is real; re-run vjs.py and the
                            scene claim can finally be tested for real.

    python -m relmo.vjs_r2 --episodes 6
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


def clip_uniform(F, n):
    return F[np.linspace(0, len(F) - 1, n).round().astype(int)]


def clip_native(F, n):
    """n CONTIGUOUS frames from the middle - the shape the model expects."""
    s = max((len(F) - n) // 2, 0)
    return F[s:s + n]


def m_temporal(n_t, n_sp, dev, torch):
    half = n_t // 2
    return (torch.arange(0, half * n_sp, device=dev).unsqueeze(0),
            torch.arange(half * n_sp, n_t * n_sp, device=dev).unsqueeze(0))


def m_block(n_t, n_sp, dev, torch, rng, n_blocks=4, bh=4, bw=4):
    keep = np.ones((n_t, GRID, GRID), bool)
    for _ in range(n_blocks):
        y, x = rng.integers(0, GRID - bh + 1), rng.integers(0, GRID - bw + 1)
        keep[:, y:y + bh, x:x + bw] = False
    f = keep.reshape(-1)
    return (torch.tensor(np.nonzero(f)[0], device=dev).unsqueeze(0),
            torch.tensor(np.nonzero(~f)[0], device=dev).unsqueeze(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
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
    n_t, n_sp = a.frames // TUBELET, GRID * GRID
    # pooled sums per arm, never an average of per-clip ratios
    acc = {}

    for p in tqdm(eps, unit="ep", desc="r2"):
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
             str(p / "frames.mp4")], stdout=subprocess.PIPE, check=True)
        w, h = (int(v) for v in pr.stdout.decode().strip().split("x"))
        F = read_frames(p / "frames.mp4", w, h)
        if len(F) < a.frames:
            continue
        for sname, fn in (("uniform", clip_uniform), ("native", clip_native)):
            px = to_tensor(fn(F, a.frames), torch, dev, torch.float32)
            for mname, mk in (("temporal", m_temporal(n_t, n_sp, dev, torch)),
                              ("block", m_block(n_t, n_sp, dev, torch,
                                                np.random.default_rng(1)))):
                ctx, tgt = mk
                with torch.no_grad():
                    po = model(pixel_values_videos=px, context_mask=[ctx],
                               target_mask=[tgt]).predictor_output
                T = po.target_hidden_state.float()[0]
                P = po.last_hidden_state.float()[0]
                mu = T.mean(0, keepdim=True)
                sse = float(((P - T) ** 2).sum())
                sst = float(((T - mu) ** 2).sum())
                Tc, Pc = T - mu, P - mu
                cos = float((Tc * Pc).sum(-1).div(
                    Tc.norm(dim=-1) * Pc.norm(dim=-1) + 1e-9).mean())
                k = f"{sname}/{mname}"
                s = acc.setdefault(k, [0.0, 0.0, []])
                s[0] += sse
                s[1] += sst
                s[2].append(cos)

    print("\n" + "=" * 66)
    print(f"{'sampling/mask':22s} {'R2':>9s} {'cos(centred)':>14s}  note")
    print("-" * 66)
    out = {}
    for k in sorted(acc):
        sse, sst, cs = acc[k]
        r2 = 1 - sse / (sst + 1e-12)
        c = float(np.mean(cs))
        note = ("explains variance" if r2 > 0.05 else
                "at the mean" if r2 > -0.05 else "WORSE than the mean")
        print(f"{k:22s} {r2:+9.4f} {c:+14.4f}  {note}")
        out[k.replace("/", "_")] = round(r2, 4)
    print("=" * 66)
    print("R2=0 is 'predict the mean target token'. Anything <=0 means the")
    print("predictor adds nothing over a constant on this corpus.")
    R.log("vjs_r2", dataset=a.dataset, episodes=len(eps), frames=a.frames,
          **out)


if __name__ == "__main__":
    main()
