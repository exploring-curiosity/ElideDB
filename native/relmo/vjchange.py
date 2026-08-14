"""PREDICTED CHANGE, not prediction error. Owner's correction, 2026-08-14.

The design so far localised action with the residual

    residual(t,i) = || pred(t,i) - actual(t,i) ||          "where I was WRONG"

That is the training loss. A perfectly trained V-JEPA drives it to ZERO
everywhere, so it can never be a reliable locator of action - it measures the
model's ignorance, and improving the model destroys the signal. Measured
symptoms, all consistent with that: concentration 0.468 against a 0.253 noise
floor and a 0.832 pixel-motion reference, flat across a 16x range of horizons,
and only 1.07x hotter on moving patches than static ones.

The quantity that actually says where the action is:

    change(t,i)   = || pred(t,i) - actual(LAST CONTEXT STEP, i) ||
                    "where do I expect this patch to BECOME DIFFERENT"

This does not vanish as the model improves - it converges to the true change.
A static wall has pred ~= current and contributes nothing; a swinging door
does. Same forward pass, different subtraction.

THREE MAPS, all from ONE forward, so the comparison is like for like
  residual   || pred - actual_future ||          what the design used
  change     || pred - actual_now    ||          the proposal
  oracle     || actual_future - actual_now ||    what a PERFECT predictor would
                                                 give. The ceiling. If change
                                                 lands near oracle the
                                                 predictor is good enough; if
                                                 oracle is sharp and change is
                                                 not, the predictor is the
                                                 problem, not the idea.

SCORED two ways, both against references already measured on this corpus
  concentration   mass in the hottest 10% of patches
                  noise floor 0.253 | residual 0.468 | pixel motion 0.832
  moving/static   median map value on high-motion vs low-motion patches,
                  quartile split from pixel differencing. Residual scored 1.07,
                  i.e. a still kitchen is 93% as "surprising" as the action.

    python -m relmo.vjchange --episodes 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import (GRID, MODEL, TUBELET, patch_motion, probe_dims,  # noqa: E402
                       read_frames, sample_clip, to_tensor)
from relmo.vjeval import REC  # noqa: E402


def conc(M):
    """Fraction of positive mass in the hottest 10% of patches, per step."""
    P = np.clip(M, 0, None).reshape(len(M), -1)
    n = max(1, int(0.1 * P.shape[1]))
    return float((np.sort(P, 1)[:, ::-1][:, :n].sum(1)
                  / (P.sum(1) + 1e-9)).mean())


def ratio(M, mot):
    r, mf = M.ravel(), mot.ravel()
    lo, hi = np.quantile(mf, 0.25), np.quantile(mf, 0.75)
    return float(np.median(r[mf >= hi]) / (np.median(r[mf <= lo]) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--frames", type=int, default=64)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d = REC / a.dataset
    files = sorted(p for p in d.glob("*.npz") if not p.name.startswith("_"))
    z = np.load(d / "_calib.npz")
    alpha, b = float(z["alpha"]), z["b"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    print("  loaded", flush=True)

    n_t, n_sp = a.frames // TUBELET, GRID * GRID
    half = n_t // 2
    S = n_t - half
    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), min(a.episodes, len(files)), replace=False)
    acc = {k: {"c": [], "r": []} for k in ("residual", "change", "oracle")}

    for i in tqdm(pick, unit="ep", desc="change"):
        ep = R.dataset_dir(a.dataset) / "shard_0000" / files[i].stem / "frames.mp4"
        if not ep.exists():
            continue
        w_, h_ = probe_dims(ep)
        F = read_frames(ep, w_, h_)
        if len(F) < a.frames:
            continue
        clip = sample_clip(F, a.frames)
        px = to_tensor(clip, torch, dev, torch.float32)
        ctx = torch.arange(0, half * n_sp, device=dev).unsqueeze(0)
        tgt = torch.arange(half * n_sp, n_t * n_sp, device=dev).unsqueeze(0)
        with torch.no_grad():
            out = model(pixel_values_videos=px, context_mask=[ctx],
                        target_mask=[tgt])
        # ONE forward gives the encoder sequence AND the predictor pair, so
        # all three maps are computed on identical activations.
        seq = out.last_hidden_state.float()[0].cpu().numpy()
        po = out.predictor_output
        P = alpha * po.last_hidden_state.float()[0].cpu().numpy() + b
        T = po.target_hidden_state.float()[0].cpu().numpy()
        now = seq[(half - 1) * n_sp: half * n_sp]          # (n_sp, D)
        P, T = P.reshape(S, n_sp, -1), T.reshape(S, n_sp, -1)
        maps = dict(residual=np.linalg.norm(P - T, axis=-1),
                    change=np.linalg.norm(P - now[None], axis=-1),
                    oracle=np.linalg.norm(T - now[None], axis=-1))
        mot = patch_motion(clip, n_t)[half:].reshape(S, -1)
        for k, M in maps.items():
            Mc = M - np.median(M, 0, keepdims=True)
            acc[k]["c"].append(conc(Mc.reshape(S, GRID, GRID)))
            acc[k]["r"].append(ratio(M, mot))

    print("\n" + "=" * 60)
    print(f"{'map':12s} {'concentration':>15s} {'moving/static':>15s}")
    print("-" * 60)
    out_ = {}
    for k in ("residual", "change", "oracle"):
        c, r = float(np.mean(acc[k]["c"])), float(np.mean(acc[k]["r"]))
        print(f"{k:12s} {c:15.3f} {r:15.3f}")
        out_[f"{k}_conc"] = round(c, 4)
        out_[f"{k}_ratio"] = round(r, 4)
    print("-" * 60)
    print(f"{'pixel motion':12s} {0.832:15.3f} {'-':>15s}   reference")
    print(f"{'noise floor':12s} {0.253:15.3f} {1.000:15.3f}   floor")
    print("=" * 60)
    print("oracle = what a PERFECT predictor would give. If change ~ oracle,")
    print("the predictor is good enough and the subtraction was the bug.")
    R.log("vjchange", dataset=a.dataset, episodes=len(pick),
          frames=a.frames, **out_)


if __name__ == "__main__":
    main()
