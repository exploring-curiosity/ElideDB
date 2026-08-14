"""v5: MULTI-REGION pooling. One vector per timestep cannot hold two motions.

The family table says this plainly. Ranked by precision:

    OpenDrawer      0.712    one panel slides
    PickPlaceSink   0.643    one object is carried
    PickPlaceDrawer 0.597
    OpenCabinet     0.480
    CloseMicrowave  0.474
    CloseDrawer     0.447
    CloseCabinet    0.339    a hand AND a panel, doing different things

The top is where one thing moves. The bottom is where two things interact. v4
pools the whole frame into ONE vector per timestep, so a hand travelling left
while a door swings right is averaged into a single blurred direction that
describes neither. No matcher downstream can recover what pooling destroyed.

v5 emits K vectors per timestep instead of one:

  1. cluster the patches by POSITION, weighted by the gate, K regions per
     timestep, seeded at the K strongest gate maxima with spatial suppression
     so the seeds do not collapse onto one blob
  2. pool the content channel within each region separately
  3. match with late interaction (MaxSim): for each query region take its best
     matching reference region, average over query regions. Order within the
     set does not matter, which is what lets region identity drift between
     clips - the hand need not be region 0 in both.

Clustering is PER TIMESTEP, not once per clip: the disturbed area moves as the
door swings, so a fixed spatial partition would smear it back together.

Content is pred_change; the error channel is barred (see vjrec4).

    python -m relmo.vjrec5 --episodes 120 --regions 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import (GRID, MODEL, TUBELET, probe_dims, read_frames,  # noqa: E402
                       sample_clip, to_tensor)
from relmo.vjeval import REC  # noqa: E402
from relmo.vjrec4 import CTX, WIN  # noqa: E402

OUT5 = R.BASE / "vjrec5"
YY, XX = np.mgrid[0:GRID, 0:GRID]
POS = np.stack([XX.ravel(), YY.ravel()], -1).astype(np.float64)   # (256,2)


def regions(gate, K, iters=6, supp=3.0):
    """(256,) gate -> (K,256) soft-ish assignment weights.

    Weighted k-means on patch POSITION. Seeds are the K strongest gate maxima
    with a suppression radius, because seeding at the top-K patches outright
    puts every seed inside the same blob and collapses the partition.
    """
    w = gate.astype(np.float64)
    if w.sum() <= 1e-9:
        w = np.ones_like(w)
    seeds, cand = [], w.copy()
    for _ in range(K):
        j = int(np.argmax(cand))
        seeds.append(POS[j])
        cand = cand * (np.linalg.norm(POS - POS[j], axis=1) > supp)
        if cand.sum() <= 0:
            cand = w.copy()
    C = np.stack(seeds)
    for _ in range(iters):
        d = ((POS[:, None, :] - C[None]) ** 2).sum(-1)     # (256,K)
        a = np.argmin(d, 1)
        for k in range(K):
            m = (a == k) & (w > 0)
            if m.sum():
                C[k] = (POS[m] * w[m, None]).sum(0) / w[m].sum()
    d = ((POS[:, None, :] - C[None]) ** 2).sum(-1)
    a = np.argmin(d, 1)
    A = np.zeros((K, len(POS)))
    for k in range(K):
        A[k] = w * (a == k)
    return A


def record(model, torch, dev, clip, n_frames, cal, layer, K):
    n_t, n_sp = n_frames // TUBELET, GRID * GRID
    px = to_tensor(clip, torch, dev, torch.float32)
    alpha, b = cal
    bT = torch.tensor(b, device=dev)
    with torch.no_grad():
        enc = model.encoder(pixel_values_videos=px, output_hidden_states=True)
        seq = enc.last_hidden_state
        h = enc.hidden_states[layer].float()[0]
        gates, vals = [], []
        for c in range(CTX, n_t, WIN):
            hi = min(c + WIN, n_t)
            S = hi - c
            ctx = torch.arange((c - CTX) * n_sp, c * n_sp, device=dev).unsqueeze(0)
            tgt = torch.arange(c * n_sp, hi * n_sp, device=dev).unsqueeze(0)
            po = model.predictor(encoder_hidden_states=seq,
                                 context_mask=[ctx], target_mask=[tgt])
            P = (alpha * po.last_hidden_state.float()[0] + bT).reshape(S, n_sp, -1)
            NOW = seq.float()[0, (c - 1) * n_sp:c * n_sp].unsqueeze(0)
            now6 = h[(c - 1) * n_sp:c * n_sp].unsqueeze(0)
            gates.append((h[c * n_sp:hi * n_sp].reshape(S, n_sp, -1)
                          - now6).norm(dim=-1).cpu().numpy())
            vals.append((P - NOW).cpu().numpy())           # pred_change
    G = np.concatenate(gates)
    V = np.concatenate(vals)                                # (S,256,D)
    Gc = np.clip(G - np.median(G, 0, keepdims=True), 0, None)
    S_, D = len(Gc), V.shape[-1]
    out = np.zeros((S_, K, D), np.float32)
    for t in range(S_):
        A = regions(Gc[t], K)                               # (K,256)
        den = A.sum(1, keepdims=True) + 1e-9
        out[t] = (A @ V[t]) / den
    return dict(multi=out, where_map=Gc.reshape(-1, GRID, GRID).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--regions", type=int, default=3)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=0)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d = REC / a.dataset
    files = [p for p in sorted(d.glob("*.npz")) if not p.name.startswith("_")]
    if a.episodes:
        rng = np.random.default_rng(0)
        files = [files[i] for i in sorted(rng.choice(len(files), a.episodes,
                                                     replace=False))]
    z = np.load(d / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    print(f"  loaded | v5: K={a.regions} regions per timestep", flush=True)

    out = OUT5 / f"{a.dataset}_L{a.layer}_K{a.regions}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in files if not (out / f"{p.stem}.npz").exists()]
    print(f"{len(files)} episodes, {len(todo)} to do "
          f"(~{len(todo)*11/60:.0f} min)", flush=True)
    t0, done = time.time(), 0
    for p in tqdm(todo, unit="ep", desc=f"v5/K{a.regions}"):
        ep = R.dataset_dir(a.dataset) / "shard_0000" / p.stem / "frames.mp4"
        if not ep.exists():
            continue
        try:
            w_, h_ = probe_dims(ep)
            F = read_frames(ep, w_, h_)
            if len(F) < a.frames:
                continue
            rec = record(model, torch, dev, sample_clip(F, a.frames),
                         a.frames, cal, a.layer, a.regions)
        except Exception as e:                              # noqa: BLE001
            tqdm.write(f"  {p.stem}: {type(e).__name__}: {e}")
            continue
        tmp = out / f".w_{p.stem}.npz"
        np.savez_compressed(tmp, **rec)
        tmp.rename(out / f"{p.stem}.npz")
        done += 1
    have = len(list(out.glob("*.npz")))
    rep = dict(dataset=a.dataset, layer=a.layer, K=a.regions,
               episodes=len(files), written=done, on_disk=have,
               minutes=round((time.time() - t0) / 60, 1))
    print("\n" + json.dumps(rep, indent=1))
    print(f"VERIFIED on disk: {have}/{len(files)} v5 records")
    R.log("vjrec5", **rep)


if __name__ == "__main__":
    main()
