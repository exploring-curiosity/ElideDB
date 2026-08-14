"""Let PIXEL MOTION say WHERE, and the residual say WHAT KIND.

Measured, and this is why the spatial route stalled:

  concentration of positive mass in the hottest 10% of patches
    pixel-motion map          0.832
    V-JEPA residual map       0.468      (horizon 1: 0.426, horizon 16: 0.475)
    pure noise floor          0.253

The residual is genuinely informative - well above noise - but it is DIFFUSE,
and changing the prediction horizon over a 16x range barely moves it. That is
structural, not a tuning problem: the encoder runs 24 layers of full
self-attention over 8192 tokens, so a token at (t,y,x) is a global mixture
rather than a description of that patch, and error in one place bleeds
everywhere. Geometry computed on such a map cannot say where the door swung,
which is exactly what relmo/vjspace.py found (geom lift 1.05/1.00 on the
event-vs-object control - chance on both).

So stop asking the residual where to look. relmo/vjcache.py pools the residual
weighted by its OWN magnitude, which lets a blurry signal choose its own
location. Weight it by MOTION instead: motion is computed from raw frames by
differencing, needs no model, no labels and no dataset prior, and is 1.8x
sharper. The residual then only has to answer "what kind of surprise", which
is the part it is actually good at.

ARMS, all pooled over the same target timesteps, scored on the same subset
  self    weight = the residual's own positive magnitude   (current design)
  motion  weight = pixel motion in that patch              (proposed)
  both    weight = product of the two
  flat    uniform weight                                   (floor - if this
          ties the others, the spatial weighting is decorative)

Reported next to scene-only on the same episodes, because a number without the
baseline it has to beat is not a result.

    python -m relmo.vjgate --episodes 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import (GRID, MODEL, TUBELET, forward_pair, patch_motion,  # noqa: E402
                       probe_dims, read_frames, sample_clip)
from relmo.vjeval import REC, l2, parse  # noqa: E402


def variants(P, T, mot, cal, n_t):
    alpha, b = cal
    S = n_t - n_t // 2
    res = (alpha * P + b - T).reshape(S, GRID * GRID, -1)
    mag = np.linalg.norm(res, axis=-1)
    magc = np.clip(mag - np.median(mag, 0, keepdims=True), 0, None)
    mo = mot[n_t // 2:].reshape(S, -1)
    moc = np.clip(mo - np.median(mo, 0, keepdims=True), 0, None)
    out = {}
    for k, w in (("self", magc), ("motion", moc), ("both", magc * moc),
                 ("flat", np.ones_like(magc))):
        out[k] = (res * w[..., None]).sum(1) / (w.sum(1, keepdims=True) + 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d = REC / a.dataset
    files = sorted(p for p in d.glob("*.npz") if not p.name.startswith("_"))
    z = np.load(d / "_calib.npz")
    cal = (float(z["alpha"]), z["b"])
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(files), min(a.episodes, len(files)),
                             replace=False))

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    print(f"  loaded | {len(pick)} episodes (~{len(pick)*5.4/60:.0f} min)",
          flush=True)

    n_t = a.frames // TUBELET
    keep, acc, ff = [], {k: [] for k in ("self", "motion", "both", "flat")}, []
    for i in tqdm(pick, unit="ep", desc="gate"):
        ep = R.dataset_dir(a.dataset) / "shard_0000" / files[i].stem / "frames.mp4"
        if not ep.exists():
            continue
        w_, h_ = probe_dims(ep)
        F = read_frames(ep, w_, h_)
        if len(F) < a.frames:
            continue
        clip = sample_clip(F, a.frames)
        P, T, _ = forward_pair(model, torch, dev, torch.float32, clip, a.frames)
        v = variants(P, T, patch_motion(clip, n_t), cal, n_t)
        for k in acc:
            acc[k].append(v[k])
        ff.append(np.load(files[i])["f_first"])
        keep.append(i)

    meta = [parse(files[i].stem) for i in keep]
    obj = np.array([m["obj"] for m in meta])
    verb = np.array([m["verb"] for m in meta])
    epid = np.array([f"{m['task']}#{m['epnum']}" for m in meta])
    A = {k: l2(np.stack(v)) for k, v in acc.items()}
    FF = l2(np.stack(ff))
    rows = {k: [] for k in list(A) + ["scene-only"]}
    base = []
    for i in range(len(keep)):
        cand = np.where((obj != obj[i]) & (epid != epid[i]))[0]
        if len(cand) < a.k:
            continue
        y = (verb[cand] == verb[i])
        if y.sum() == 0:
            continue
        base.append(y.mean())
        for k, M in A.items():
            s = np.einsum("sd,nsd->n", M[i], M[cand]) / M.shape[1]
            rows[k].append(y[np.argsort(-s)[:a.k]].mean())
        rows["scene-only"].append(
            y[np.argsort(-(FF[cand] @ FF[i]))[:a.k]].mean())

    b = float(np.mean(base))
    print(f"\n{len(base)} queries on a {len(keep)}-episode subset "
          f"(NOT comparable to the 447-episode numbers - smaller pool)")
    print(f"{'weighting':14s} {'P@k':>8s} {'lift':>7s}")
    print("-" * 32)
    for k in ("flat", "self", "motion", "both", "scene-only"):
        v = float(np.mean(rows[k]))
        print(f"{k:14s} {v:8.3f} {v/b:7.2f}")
    print(f"{'random':14s} {b:8.3f} {1.0:7.2f}")
    R.log("vjgate", dataset=a.dataset, episodes=len(keep), queries=len(base),
          k=a.k, base_rate=round(b, 4),
          **{k: round(float(np.mean(v)), 4) for k, v in rows.items()})


if __name__ == "__main__":
    main()
