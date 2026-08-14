"""Is V-JEPA 2's prediction residual concentrated on MOTION, or on the ROOM?

This tests ONE claim, which I asserted without evidence and which the whole
error-based-retrieval design rests on:

    "a still kitchen is trivially predictable, so it generates almost no
     surprise; a representation made of error cannot be dominated by the room."

That was a hypothesis I invented in conversation, not something measured. It
matters because the retrieval numbers on this corpus have been beaten by a
SCENE-ONLY baseline (0.219 vs 0.202) - the appearance path is doing kitchen
recognition. If the residual is ALSO spread over static background, the pivot
to error-based representation fixes nothing and should be abandoned here,
cheaply, before anything is built on it.

The counter-argument is well known and must be taken seriously: prediction
residuals in video are routinely dominated by camera motion, lighting change,
compression artefacts and high-frequency texture rather than by anything
semantic. This corpus is static-camera, which removes the biggest confound,
but leaves texture: a patch of wood grain is hard to predict pixel-wise even
when nothing happens to it. V-JEPA predicts in LATENT space, which is exactly
the claimed defence against that - unpredictable detail is supposed to be
dropped by the encoder before the predictor ever sees it. So this is a fair
test of the specific thing being claimed.

METHOD
  context = temporal token steps [0, T/2)      (the first half of the clip)
  target  = temporal token steps [T/2, T)      (predict the second half)
  residual r[t,y,x] = || predicted_latent - actual_latent ||   per target token
  motion   m[t,y,x] = mean |frame difference| inside that token's spacetime
                      extent - computed from RAW PIXELS ONLY, no sim state

  report  spearman(r, m)                     does error track motion at all
          median r on static patches (bottom quartile of m)
          median r on moving patches (top quartile of m)
          the ratio moving/static             <- THE NUMBER

INTERPRETATION, fixed before running so it cannot be moved afterwards:
  ratio <= 1.2   claim REFUTED. Error is as large on the still room as on the
                 action. Error-based retrieval inherits the scene confound and
                 buys nothing. Stop.
  1.2 - 2.0      claim WEAK. Some concentration, probably not enough to carry
                 retrieval on its own.
  > 2.0          claim SUPPORTED at the patch level. Necessary, NOT sufficient
                 - it says error avoids the static room, not that it encodes
                 the event.

Nothing here reads sim state. Input is the mp4 and nothing else.

    python -m relmo.vjs --episodes 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

MODEL = "facebook/vjepa2-vitl-fpc64-256"
CROP = 256
PATCH = 16
TUBELET = 2
GRID = CROP // PATCH                      # 16 x 16 spatial tokens


def probe_dims(mp4: Path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of",
                        "csv=p=0:s=x", str(mp4)],
                       stdout=subprocess.PIPE, check=True)
    return (int(v) for v in p.stdout.decode().strip().split("x"))


def read_frames(mp4: Path, w: int, h: int) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-f",
                        "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def sample_clip(F: np.ndarray, n: int) -> np.ndarray:
    """n frames spread over the WHOLE episode.

    Uniform over the episode, not a fixed stride: a 3 s clip and a 14 s clip
    must both be covered end to end, which is the objection raised against the
    old fixed 24-frame stride. The temporal spacing therefore differs between
    episodes; that is deliberate - the unit is the episode, not the second.
    """
    idx = np.linspace(0, len(F) - 1, n).round().astype(int)
    return F[idx]


def to_tensor(clip: np.ndarray, torch, device, dtype):
    """(T,H,W,3) uint8 -> (1,T,3,CROP,CROP) normalised."""
    import torch.nn.functional as Fn
    x = torch.tensor(clip).permute(0, 3, 1, 2).float().div_(255.)   # T,3,H,W
    x = Fn.interpolate(x, size=(CROP, CROP), mode="bilinear",
                       align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0).to(device=device, dtype=dtype)


def patch_motion(clip: np.ndarray, n_t: int) -> np.ndarray:
    """(n_t, GRID, GRID) mean |frame diff| per token, from pixels only.

    A token covers TUBELET frames and PATCH x PATCH pixels of the RESIZED
    frame, so the motion reference is computed on the same resized grid the
    model sees - otherwise the two are not aligned and the correlation is
    meaningless.
    """
    import torch
    import torch.nn.functional as Fn
    x = torch.tensor(clip).permute(0, 3, 1, 2).float().div_(255.)
    x = Fn.interpolate(x, size=(CROP, CROP), mode="bilinear",
                       align_corners=False).mean(1)          # T,CROP,CROP
    d = (x[1:] - x[:-1]).abs()
    d = torch.cat([d[:1], d], 0)                             # T, keep length
    # pool each tubelet's frames, then each patch's pixels
    d = d.reshape(n_t, TUBELET, CROP, CROP).mean(1)
    d = d.reshape(n_t, GRID, PATCH, GRID, PATCH).mean((2, 4))
    return d.numpy()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def forward_pair(model, torch, dev, dtype, clip, n_frames):
    """(pred, target) latents for the second half of the clip."""
    px = to_tensor(clip, torch, dev, dtype)
    n_t = n_frames // TUBELET
    n_sp = GRID * GRID
    half = n_t // 2
    ctx = torch.arange(0, half * n_sp, device=dev).unsqueeze(0)
    tgt = torch.arange(half * n_sp, n_t * n_sp, device=dev).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        po = model(pixel_values_videos=px, context_mask=[ctx],
                   target_mask=[tgt]).predictor_output
    return (po.last_hidden_state.float()[0].cpu().numpy(),
            po.target_hidden_state.float()[0].cpu().numpy(),
            time.time() - t0)


def run_episode(model, torch, dev, dtype, mp4: Path, w: int, h: int,
                n_frames: int, cal):
    F = read_frames(mp4, w, h)
    if len(F) < n_frames:
        return None
    clip = sample_clip(F, n_frames)
    P, T, dt = forward_pair(model, torch, dev, dtype, clip, n_frames)
    # CALIBRATION IS NOT OPTIONAL. Measured in relmo/vjs_cal.py: the raw
    # predictor output is systematically ~3x too small against this target,
    # because transformers runs ONE encoder for context and target while
    # V-JEPA trained the predictor against a separate EMA target encoder.
    # Uncalibrated, ||P - T|| ~= ||T||, i.e. the residual is just the norm of
    # the target and is therefore near-uniform over every patch in the frame.
    # The first version of this file did exactly that and reported ratio 1.01
    # - a flat number that looked like a refutation and measured nothing.
    alpha, b = cal
    n_t = n_frames // TUBELET
    half = n_t // 2
    r = np.linalg.norm(alpha * P + b - T, axis=-1)
    r = r.reshape(n_t - half, GRID, GRID)
    m = patch_motion(clip, n_t)[half:]
    return r, m, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--fit", type=int, default=6,
                    help="episodes used ONLY to fit the global calibration")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--dataset", default="rcasa")
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32          # fp16 on MPS has produced NaNs here before
    # Load BEFORE the progress bar exists and say so: a lazy multi-GB load
    # inside iteration one hides behind a bar reading 0/N and looks hung.
    print(f"loading {MODEL} onto {dev} (~1.1 GB, first call is slow)...",
          flush=True)
    t0 = time.time()
    model = VJEPA2Model.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
    npar = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time()-t0:.1f}s | {npar/1e6:.0f}M params "
          f"| {a.frames} frames -> {a.frames//TUBELET} temporal x {GRID}x{GRID} "
          f"spatial = {a.frames//TUBELET*GRID*GRID} tokens", flush=True)

    d = R.dataset_dir(a.dataset) / "shard_0000"
    allep = sorted(p for p in d.iterdir() if (p / "frames.mp4").exists())
    rng = np.random.default_rng(0)
    pick = rng.choice(len(allep), min(a.fit + a.episodes, len(allep)),
                      replace=False)
    fit_eps = [allep[i] for i in pick[:a.fit]]
    eps = [allep[i] for i in pick[a.fit:]]

    # Fit ONE global (alpha, b) on episodes DISJOINT from the scored ones.
    # A per-clip fit would absorb the very signal being measured.
    Pf, Tf = [], []
    for p in tqdm(fit_eps, unit="ep", desc="calib"):
        w, h = probe_dims(p / "frames.mp4")
        F = read_frames(p / "frames.mp4", w, h)
        if len(F) < a.frames:
            continue
        P, T, _ = forward_pair(model, torch, dev, dtype,
                               sample_clip(F, a.frames), a.frames)
        Pf.append(P)
        Tf.append(T)
    Pf, Tf = np.concatenate(Pf), np.concatenate(Tf)
    alpha = float((Pf * Tf).sum() / ((Pf * Pf).sum() + 1e-12))
    b = (Tf - alpha * Pf).mean(0, keepdims=True)
    cal = (alpha, b)
    print(f"calibration fitted on {len(fit_eps)} held-out episodes: "
          f"alpha {alpha:.3f}", flush=True)

    rows, lat = [], []
    for p in tqdm(eps, unit="ep", desc="vjs"):
        try:
            w, h = probe_dims(p / "frames.mp4")
            got = run_episode(model, torch, dev, dtype, p / "frames.mp4",
                              w, h, a.frames, cal)
        except Exception as e:                       # noqa: BLE001
            tqdm.write(f"  {p.name}: {type(e).__name__}: {e}")
            continue
        if got is None:
            continue
        r, m, dt = got
        lat.append(dt)
        rf, mf = r.ravel(), m.ravel()
        lo, hi = np.quantile(mf, 0.25), np.quantile(mf, 0.75)
        stat, mov = rf[mf <= lo], rf[mf >= hi]
        rows.append(dict(ep=p.name, task=p.name.split("_episode_")[0],
                         rho=spearman(rf, mf),
                         r_static=float(np.median(stat)),
                         r_moving=float(np.median(mov)),
                         ratio=float(np.median(mov) / (np.median(stat) + 1e-9))))
        tqdm.write(f"  {p.name[:44]:44s} rho {rows[-1]['rho']:+.3f}  "
                   f"ratio {rows[-1]['ratio']:.2f}  {dt:.2f}s")

    if not rows:
        raise SystemExit("no episodes scored")
    rho = float(np.median([x["rho"] for x in rows]))
    ratio = float(np.median([x["ratio"] for x in rows]))
    ms = float(np.median(lat))
    print("\n" + "=" * 62)
    print(f"episodes scored     {len(rows)}")
    print(f"median spearman(residual, motion)   {rho:+.3f}")
    print(f"median r on STATIC patches          "
          f"{np.median([x['r_static'] for x in rows]):.3f}")
    print(f"median r on MOVING patches          "
          f"{np.median([x['r_moving'] for x in rows]):.3f}")
    print(f"median ratio moving/static          {ratio:.2f}   <- THE NUMBER")
    print(f"median latency per episode          {ms:.2f} s")
    verdict = ("REFUTED - error is as big on the still room as on the action"
               if ratio <= 1.2 else
               "WEAK - some concentration, likely not enough on its own"
               if ratio <= 2.0 else
               "SUPPORTED at patch level - necessary, not sufficient")
    print(f"\nverdict: {verdict}")
    print(f"latency budget is 1 s/episode: "
          f"{'PASS' if ms <= 1.0 else f'FAIL by {ms:.1f}x'}")
    print("=" * 62)
    R.log("vjs_scene", dataset=a.dataset, episodes=len(rows),
          frames=a.frames, rho=round(rho, 4), ratio=round(ratio, 4),
          sec_per_episode=round(ms, 3), verdict=verdict, device=dev)


if __name__ == "__main__":
    main()
