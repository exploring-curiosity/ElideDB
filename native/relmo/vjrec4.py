"""v4: FIXED-LENGTH context, and all three content channels cached at once.

TWO defects in v3 that the owner exposed, fixed here.

1. CONTEXT-LENGTH CONFOUND
   v3 slid a GROWING context: [0,c) -> [c,c+4). Step 4 was predicted from one
   step of history, step 28 from twenty-seven. Prediction error falls as
   context grows, so the trace carried a systematic ramp that had nothing to do
   with the event - early steps looked "surprising" merely because almost
   nothing had been seen yet. v4 uses a FIXED window [c-L, c), so every step is
   predicted from exactly L steps of history and the values are comparable
   along the trace. This is a prerequisite for any temporal comparison: DTW
   aligning two traces that both ramp will match the ramps, not the events.

2. CONTENT IS pred_change. THE ERROR CHANNEL IS BARRED.
       pred_change pred(t+k) - actual(t)    what the model expected to change
       obs_change  actual(t+k) - actual(t)  what actually changed
       error       pred(t+k) - actual(t+k)  FORBIDDEN - do not revive

   Standing owner rule, given repeatedly and finally as absolute: "Dont use the
   error channel. Youre dependent on error as the signal will never understand
   the experience in complex scenarios and it will just get carried forward."

   The reason is structural and no benchmark answers it. `error` is a function
   of (event, MODEL PRIOR), not of the event. Its magnitude reports what the
   model happened to guess: actual=open/guessed=close gives a large error,
   actual=close/guessed=close a small one. So the same event yields different
   descriptors as the prior shifts, an index built from it drifts against
   itself under any adaptation, and it is structurally worst on exactly the
   open/close distinction the product exists to make.
   It scored higher on this ONE frozen checkpoint - 0.584 [0.566,0.602] against
   pred_change 0.525 [0.508,0.540]. That is evidence about the checkpoint, not
   about the design, and I was wrong to keep re-raising it as a counter.
   The 0.06 is the accepted cost of a signal that generalises.

The gate is unchanged: layer-6 observed change, which converges on truth rather
than on zero as the model improves.

    python -m relmo.vjrec4 --dataset rcasa            # full corpus, resumable
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

OUT4 = R.BASE / "vjrec4"
CTX = 8             # FIXED history per prediction, in temporal steps
WIN = 4             # target steps per split
# error = pred(t+k)-actual(t+k) is BARRED by standing owner rule and is not
# cached, benchmarked or reported. See docs/memory/decisions.md 2026-08-14.
CHANNELS = ("pred_change", "obs_change")


def record(model, torch, dev, clip, n_frames, cal, layer):
    n_t, n_sp = n_frames // TUBELET, GRID * GRID
    px = to_tensor(clip, torch, dev, torch.float32)
    alpha, b = cal
    bT = torch.tensor(b, device=dev)
    with torch.no_grad():
        enc = model.encoder(pixel_values_videos=px, output_hidden_states=True)
        seq = enc.last_hidden_state
        h = enc.hidden_states[layer].float()[0]
        gates, out = [], {k: [] for k in CHANNELS}
        for c in range(CTX, n_t, WIN):
            hi = min(c + WIN, n_t)
            S = hi - c
            # FIXED-length context window, not [0,c)
            ctx = torch.arange((c - CTX) * n_sp, c * n_sp,
                               device=dev).unsqueeze(0)
            tgt = torch.arange(c * n_sp, hi * n_sp, device=dev).unsqueeze(0)
            po = model.predictor(encoder_hidden_states=seq,
                                 context_mask=[ctx], target_mask=[tgt])
            P = (alpha * po.last_hidden_state.float()[0] + bT).reshape(S, n_sp, -1)
            T = seq.float()[0, c * n_sp:hi * n_sp].reshape(S, n_sp, -1)
            NOW = seq.float()[0, (c - 1) * n_sp:c * n_sp].unsqueeze(0)
            now6 = h[(c - 1) * n_sp:c * n_sp].unsqueeze(0)
            gates.append((h[c * n_sp:hi * n_sp].reshape(S, n_sp, -1)
                          - now6).norm(dim=-1).cpu().numpy())
            out["obs_change"].append((T - NOW).cpu().numpy())
            out["pred_change"].append((P - NOW).cpu().numpy())
    G = np.concatenate(gates)
    Gc = np.clip(G - np.median(G, 0, keepdims=True), 0, None)
    den = Gc.sum(1, keepdims=True) + 1e-9
    rec = {"where_map": Gc.reshape(-1, GRID, GRID).astype(np.float32)}
    for k in CHANNELS:
        V = np.concatenate(out[k])
        rec[k] = ((V * Gc[..., None]).sum(1) / den).astype(np.float32)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d = REC / a.dataset
    files = [p for p in sorted(d.glob("*.npz")) if not p.name.startswith("_")]
    if a.limit:
        files = files[:a.limit]
    z = np.load(d / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    n_t = a.frames // TUBELET
    steps = len(range(CTX, n_t, WIN)) * WIN
    print(f"  loaded | v4: fixed context {CTX} steps, {steps} trace steps, "
          f"channels {CHANNELS}", flush=True)

    out = OUT4 / f"{a.dataset}_L{a.layer}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in files if not (out / f"{p.stem}.npz").exists()]
    print(f"{len(files)} episodes, {len(files)-len(todo)} cached, "
          f"{len(todo)} to do (~{len(todo)*11/60:.0f} min)", flush=True)
    t0, done, failed = time.time(), 0, 0
    for p in tqdm(todo, unit="ep", desc=f"v4/L{a.layer}"):
        ep = R.dataset_dir(a.dataset) / "shard_0000" / p.stem / "frames.mp4"
        if not ep.exists():
            failed += 1
            continue
        try:
            w_, h_ = probe_dims(ep)
            F = read_frames(ep, w_, h_)
            if len(F) < a.frames:
                failed += 1
                continue
            rec = record(model, torch, dev, sample_clip(F, a.frames),
                         a.frames, cal, a.layer)
        except Exception as e:                                # noqa: BLE001
            tqdm.write(f"  {p.stem}: {type(e).__name__}: {e}")
            failed += 1
            continue
        tmp = out / f".w_{p.stem}.npz"
        np.savez_compressed(tmp, **rec)
        tmp.rename(out / f"{p.stem}.npz")
        done += 1
    have = len(list(out.glob("*.npz")))
    rep = dict(dataset=a.dataset, layer=a.layer, ctx=CTX, steps=steps,
               episodes=len(files), written=done, failed=failed, on_disk=have,
               minutes=round((time.time() - t0) / 60, 1))
    print("\n" + json.dumps(rep, indent=1))
    print(f"VERIFIED on disk: {have}/{len(files)} v4 records")
    R.log("vjrec4", **rep)


if __name__ == "__main__":
    main()
