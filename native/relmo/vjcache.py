"""Cache one EXPERIENCE RECORD per episode. Video in, nothing else.

This is the write path of the design: watch the clip, predict forward, and keep
what the prediction got WRONG - as a series over time, not a single vector.

Per episode, from the mp4 alone:

  res_seq  (S, 1024)  the surprise, step by step. At each future timestep, the
                      residual vectors of all 256 patches pooled with their own
                      magnitude as weight, so a patch the model got badly wrong
                      dominates the step and a patch it nailed contributes
                      nothing. This is "what kind of thing surprised me, and
                      when".
  res_map  (S,16,16)  where the surprise landed, per patch. Drives the viewer
                      and any later spatial analysis.
  f_all    (1024,)    mean encoder feature over the whole clip   -> APPEARANCE
  f_first  (1024,)    mean encoder feature of the FIRST timestep -> SCENE-ONLY
  f_mid    (1024,)    mean encoder feature of a middle timestep  -> ONE-FRAME

The last three exist only to be baselines. If the surprise record cannot beat
"what does this kitchen look like", it is not doing anything and should be
abandoned - that is the whole reason they are cached alongside.

TWO CORRECTIONS ARE APPLIED, both measured, neither optional:

1. GLOBAL AFFINE (relmo/vjs_cal.py). transformers runs one encoder for context
   and target; V-JEPA trained the predictor against a separate EMA target
   encoder. Raw predictor output is ~2.7x too small, so ||P-T|| ~= ||T|| and
   every residual reads as the same near-uniform noise. Fitted once on held-out
   episodes, reused for the whole corpus.

2. PER-PATCH TEMPORAL CENTERING. The measured failure of my scene-robustness
   claim (relmo/vjs.py: static patches 83.3, moving 88.3, ratio 1.07) says a
   still room produces almost as much prediction error as the action. But the
   room's error is CONSTANT over the clip while the event's error is not, so
   subtracting each patch's own temporal median removes the room without any
   room label, any motion detector, or any hand-chosen prior. What survives is
   deviation from that clip's own steady state, which is also exactly what
   "the series of expectations and outcomes" means - we care how surprise
   CHANGES, not its absolute level.

Sim state is never opened. Episode names are never read here; grading happens
elsewhere.

    python -m relmo.vjcache --dataset rcasa
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
from relmo.vjs import (GRID, MODEL, TUBELET, forward_pair, probe_dims,  # noqa: E402
                       read_frames, sample_clip, to_tensor)

OUT = R.BASE / "vjrec"


def encoder_feats(model, torch, dev, clip, n_frames):
    px = to_tensor(clip, torch, dev, torch.float32)
    with torch.no_grad():
        seq = model(pixel_values_videos=px,
                    skip_predictor=True).last_hidden_state.float()[0]
    n_t, n_sp = n_frames // TUBELET, GRID * GRID
    s = seq.reshape(n_t, n_sp, -1).cpu().numpy()
    return (s.reshape(-1, s.shape[-1]).mean(0), s[0].mean(0),
            s[n_t // 2].mean(0))


def record(model, torch, dev, mp4, n_frames, cal):
    w, h = probe_dims(mp4)
    F = read_frames(mp4, w, h)
    if len(F) < n_frames:
        return None
    clip = sample_clip(F, n_frames)
    P, T, _ = forward_pair(model, torch, dev, torch.float32, clip, n_frames)
    alpha, b = cal
    n_t = n_frames // TUBELET
    S = n_t - n_t // 2
    res = (alpha * P + b - T).reshape(S, GRID * GRID, -1)     # (S, 256, D)
    mag = np.linalg.norm(res, axis=-1)                        # (S, 256)
    # per-patch temporal centering: remove this clip's own steady state
    magc = mag - np.median(mag, axis=0, keepdims=True)
    w_ = np.clip(magc, 0, None)                               # only surprises
    denom = w_.sum(1, keepdims=True) + 1e-9
    seq = (res * w_[..., None]).sum(1) / denom                # (S, D)
    f_all, f_first, f_mid = encoder_feats(model, torch, dev, clip, n_frames)
    return dict(res_seq=seq.astype(np.float32),
                res_map=magc.reshape(S, GRID, GRID).astype(np.float32),
                f_all=f_all.astype(np.float32),
                f_first=f_first.astype(np.float32),
                f_mid=f_mid.astype(np.float32),
                n_src_frames=np.int32(len(F)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--fit", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev} (~1.3 GB)...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    print("  loaded", flush=True)

    d = R.dataset_dir(a.dataset) / "shard_0000"
    eps = sorted(p for p in d.iterdir() if (p / "frames.mp4").exists())
    if a.limit:
        eps = eps[:a.limit]
    out = OUT / a.dataset
    out.mkdir(parents=True, exist_ok=True)

    # calibration, fitted on the FIRST --fit episodes and then frozen
    calf = out / "_calib.npz"
    if calf.exists():
        z = np.load(calf)
        cal = (float(z["alpha"]), z["b"])
        print(f"calibration loaded: alpha {cal[0]:.3f}", flush=True)
    else:
        Pf, Tf = [], []
        for p in tqdm(eps[:a.fit], unit="ep", desc="calib"):
            w, h = probe_dims(p / "frames.mp4")
            F = read_frames(p / "frames.mp4", w, h)
            if len(F) < a.frames:
                continue
            P, T, _ = forward_pair(model, torch, dev, torch.float32,
                                   sample_clip(F, a.frames), a.frames)
            Pf.append(P)
            Tf.append(T)
        Pf, Tf = np.concatenate(Pf), np.concatenate(Tf)
        alpha = float((Pf * Tf).sum() / ((Pf * Pf).sum() + 1e-12))
        b = (Tf - alpha * Pf).mean(0, keepdims=True).astype(np.float32)
        np.savez(calf, alpha=alpha, b=b)
        cal = (alpha, b)
        print(f"calibration fitted on {a.fit} episodes: alpha {alpha:.3f}",
              flush=True)

    todo = [p for p in eps if not (out / f"{p.name}.npz").exists()]
    print(f"{len(eps)} episodes, {len(eps)-len(todo)} cached, {len(todo)} to do "
          f"(~{len(todo)*5.4/60:.0f} min)", flush=True)
    t0, done, failed = time.time(), 0, 0
    for p in tqdm(todo, unit="ep", desc=f"vjrec/{a.dataset}"):
        try:
            rec = record(model, torch, dev, p / "frames.mp4", a.frames, cal)
        except Exception as e:                                # noqa: BLE001
            tqdm.write(f"  {p.name}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if rec is None:
            failed += 1
            continue
        tmp = out / f".w_{p.name}.npz"
        np.savez_compressed(tmp, **rec)
        tmp.rename(out / f"{p.name}.npz")
        done += 1
    # verify against the FILESYSTEM, not against the loop counter
    have = len(list(out.glob("*.npz"))) - (1 if calf.exists() else 0)
    rep = dict(dataset=a.dataset, episodes=len(eps), written=done,
               failed=failed, on_disk=have, frames=a.frames,
               alpha=round(cal[0], 4), minutes=round((time.time()-t0)/60, 1))
    print("\n" + json.dumps(rep, indent=1))
    print(f"VERIFIED on disk: {have}/{len(eps)} experience records")
    R.log("vjcache", **rep)


if __name__ == "__main__":
    main()
