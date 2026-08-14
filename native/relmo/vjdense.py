"""v3: a DENSE descriptor over the WHOLE clip, not one vector for half of it.

v2 (relmo/vjrec2.py) used a single split: context = temporal steps 0-15,
target = 16-31, and the record covered only the target half. The first half of
every episode contributed nothing to the descriptor. That is the largest
outright waste in the design - half the evidence, discarded.

It also blocks the thing the owner actually wants. A result must be a located
SPAN inside a long recording, with boundaries that come from the match rather
than from a fixed grid. One vector per episode cannot express a span; a dense
per-timestep sequence can, and subsequence alignment over it returns the span
directly (relmo/vjspan.py).

WHAT CHANGES
  v2   one split      context [0,16) -> target [16,32)      16 steps, horizon 1-16
  v3   sliding        context [0,c)  -> target [c,c+4)      28 steps, horizon 1-4
                      for c = 4, 8, ... 28

Steps 0-3 can never be predicted - a predictor needs some context - so 28 of 32
steps are covered rather than 16. The horizon also drops from 1-16 to 1-4,
which matters: a short horizon is a confident prediction, so its residual is
about what actually surprised the model rather than about general uncertainty.

COST IS NOT 7x. The encoder is 24 layers at 1024-d over 8192 tokens and
dominates; the predictor is 12 layers at 384-d. So this runs the encoder ONCE
and calls the predictor seven times against the cached sequence, which is why
vjrec2's build_record could not be reused - it goes through VJEPA2Model.forward,
which re-encodes every time.

The WHERE gate is per-split: 'now' is the LAST CONTEXT STEP of that split, not
a fixed midpoint, so every target step is compared against what was actually
observed just before it.

    python -m relmo.vjdense --episodes 120 --layer 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import (GRID, MODEL, TUBELET, probe_dims, read_frames,  # noqa: E402
                       sample_clip, to_tensor)
from relmo.vjeval import REC, group_key, l2, parse  # noqa: E402

OUT3 = R.BASE / "vjrec3"
MINCTX = 4          # steps needed before anything can be predicted
WIN = 4             # target steps per split -> horizon 1..4


def dense_record(model, torch, dev, clip, n_frames, cal, layer):
    n_t, n_sp = n_frames // TUBELET, GRID * GRID
    px = to_tensor(clip, torch, dev, torch.float32)
    with torch.no_grad():
        enc = model.encoder(pixel_values_videos=px, output_hidden_states=True)
        seq = enc.last_hidden_state                       # (1,N,1024)
        h = enc.hidden_states[layer].float()[0]
        alpha, b = cal
        bT = torch.tensor(b, device=dev, dtype=torch.float32)
        wheres, whats = [], []
        for c in range(MINCTX, n_t, WIN):
            hi = min(c + WIN, n_t)
            ctx = torch.arange(0, c * n_sp, device=dev).unsqueeze(0)
            tgt = torch.arange(c * n_sp, hi * n_sp, device=dev).unsqueeze(0)
            po = model.predictor(encoder_hidden_states=seq,
                                 context_mask=[ctx], target_mask=[tgt])
            S = hi - c
            P = alpha * po.last_hidden_state.float()[0] + bT
            T = seq.float()[0, c * n_sp:hi * n_sp]
            now = h[(c - 1) * n_sp:c * n_sp]              # last OBSERVED step
            w = (h[c * n_sp:hi * n_sp].reshape(S, n_sp, -1)
                 - now.unsqueeze(0)).norm(dim=-1)         # (S,n_sp) change
            res = (P - T).reshape(S, n_sp, -1)
            wheres.append(w.cpu().numpy())
            whats.append(res.cpu().numpy())
    W = np.concatenate(wheres)                            # (28,n_sp)
    Rw = np.concatenate(whats)                            # (28,n_sp,1024)
    # temporal centring over the WHOLE clip now, not half of it
    Wc = np.clip(W - np.median(W, 0, keepdims=True), 0, None)
    what = (Rw * Wc[..., None]).sum(1) / (Wc.sum(1, keepdims=True) + 1e-9)
    return (Wc.reshape(-1, GRID, GRID).astype(np.float32),
            what.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d = REC / a.dataset
    d2 = R.BASE / "vjrec2" / f"{a.dataset}_L{a.layer}"
    files = [p for p in sorted(d.glob("*.npz"))
             if not p.name.startswith("_") and (d2 / f"{p.stem}.npz").exists()]
    z = np.load(d / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(files), min(a.episodes, len(files)),
                             replace=False))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    n_t = a.frames // TUBELET
    steps = len(range(MINCTX, n_t, WIN)) * WIN
    print(f"  loaded | v3 sliding: {steps} steps of {n_t} covered "
          f"(v2 covered {n_t//2}), horizon 1-{WIN}", flush=True)

    cache = OUT3 / f"{a.dataset}_L{a.layer}"
    if a.save:
        cache.mkdir(parents=True, exist_ok=True)
    keep, NEW, OLD, t0 = [], [], [], time.time()
    for i in tqdm(pick, unit="ep", desc=f"v3/L{a.layer}"):
        ep = R.dataset_dir(a.dataset) / "shard_0000" / files[i].stem / "frames.mp4"
        if not ep.exists():
            continue
        cf = cache / f"{files[i].stem}.npz"
        if a.save and cf.exists():
            zc = np.load(cf)
            m, w = zc["where_map"], zc["what_seq"]
        else:
            w_, h_ = probe_dims(ep)
            F = read_frames(ep, w_, h_)
            if len(F) < a.frames:
                continue
            m, w = dense_record(model, torch, dev, sample_clip(F, a.frames),
                                a.frames, cal, a.layer)
            if a.save:
                tmp = cache / f".w_{files[i].stem}.npz"
                np.savez_compressed(tmp, where_map=m, what_seq=w)
                tmp.rename(cf)
        NEW.append(w)
        OLD.append(np.load(d2 / f"{files[i].stem}.npz")["what_seq"])
        keep.append(i)
    per_ep = (time.time() - t0) / max(len(keep), 1)

    meta = [parse(files[i].stem) for i in keep]
    epid = np.array([f"{m['task']}#{m['epnum']}" for m in meta])
    G = np.array([group_key(m) for m in meta])
    arms = {"v2 (half clip, 16 steps)": l2(np.stack(OLD)),
            "v3 (whole clip, %d steps)" % NEW[0].shape[0]: l2(np.stack(NEW))}
    tot = {k: [0, 0] for k in list(arms) + ["random"]}
    for i in range(len(keep)):
        full = np.where(epid != epid[i])[0]
        sv = G[full] == G[i]
        sup = int(sv.sum())
        if sup < 5:
            continue
        for k, M in arms.items():
            s = np.einsum("sd,nsd->n", M[i], M[full]) / M.shape[1]
            tot[k][0] += int(sv[np.argsort(-s)[:sup]].sum())
            tot[k][1] += sup
        tot["random"][0] += sup * sup / len(full)
        tot["random"][1] += sup
    if tot["random"][1] == 0:
        raise SystemExit(f"no query had >=5 same-group candidates among "
                         f"{len(keep)} episodes - run more episodes")
    r = tot["random"][0] / tot["random"][1]
    print(f"\n{len(keep)}-episode subset, k=support, group-aware grading")
    print(f"{'arm':28s} {'true/returned':>16s} {'prec':>8s} {'lift':>7s}")
    print("-" * 62)
    for k in ["random"] + list(arms):
        c, n = tot[k]
        print(f"{k:28s} {f'{c:.0f}/{n}':>16s} {c/n:8.3f} {(c/n)/r:6.2f}x")
    print(f"\n{per_ep:.2f} s/episode (v2 was 6.35)")
    R.log("vjdense", dataset=a.dataset, layer=a.layer, episodes=len(keep),
          steps=int(NEW[0].shape[0]), sec_per_ep=round(per_ep, 2),
          **{k.split(" ")[0]: round(tot[k][0] / tot[k][1], 4) for k in arms})


if __name__ == "__main__":
    main()
