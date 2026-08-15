"""v6: STREAM-TIME records. `t` is absolute time, not a position inside a clip.

WHAT v4 GOT WRONG, in the owner's words: "t doesnt make sense with this logic
at all. t means time. this doesnt signify any history. just an variable length
encoder of smaller vs bigger window."

v4 sampled 64 (or 32) frames spread across the WHOLE episode and emitted a
fixed-length trace. Two consequences, both fatal to the stated formula:

  * the sample rate was a function of episode duration, so a 7 s clip and a
    60 s clip were encoded at 4.6 fps and 0.53 fps. `t` indexed a normalised
    position, not a time.
  * z was re-initialised to zero for every clip, so the recurrence in the core
    formula had no history to carry. It was a pooling operator over one window.

v6 fixes both. Sampling is at a FIXED rate; windows tile the recording; the
per-window descriptor blocks concatenate into ONE continuous, time-indexed
trace per recording, and z runs over that trace with a single initial
condition (see relmo/vjrank.py).

GEOMETRY, and why these numbers.

    step        dt = TUBELET / STREAM_FPS = 0.25 s
    window      WIN_FRAMES frames at STREAM_FPS  = 4.0 s  (encoder input)
    context     CTX steps                        = 2.0 s  (per prediction)
    descriptors steps CTX..n_t-1                 = 2.0 s  (second half)
    hop         HOP_S                            = 2.0 s

Hop EQUALS the descriptor span, so consecutive windows tile the stream exactly
- no gap, no overlap - and global step T = 8j + i for window j, in-clip step i.
That identity is what makes concatenation legitimate; it is asserted at
runtime, not assumed.

The obvious alternative, 32 frames at 4 fps (8 s window, 4 s hop), was measured
against this corpus and rejected: 58 of 447 rcasa episodes are shorter than one
8 s window and would be silently DROPPED, and 242 of 447 would yield a single
window - so the one thing v6 exists to test, carrying z across a window
boundary, would never happen for the majority of the corpus. At 4 s / 2 s no
episode is dropped and the median episode spans 4 windows. The cost is 2x the
windows per second of video.

CHANNELS ARE STEP-DIFFERENCED, not anchor-differenced. v4 measured both
pred_change and obs_change against NOW = the last context step, so the horizon
k sawtoothed 1,2,3,4 phase-locked to the window grid. Under a per-clip reset
that cancelled (every clip had identical phase); in a continuous stream it does
not, and two recordings of the same event offset by 1 s relative to their
window grids get differently-phased descriptors. Here every descriptor is a
ONE-STEP change:

    a_t = x^_t - x^_{t-1}     expected change over one step, along the model's
                              own forecast trajectory
    b_t = x_t   - x_{t-1}     the change that occurred

    INVARIANT: every difference is between two quantities OF THE SAME KIND
    (forecast-forecast, or observation-observation) produced by the same
    predictor call on the same window. No forecast is ever differenced against
    an observation, and nothing is differenced across windows.

That invariant is not decoration; it was arrived at by measurement. The first
draft referenced a block's opening step to the OBSERVED anchor, because no
forecast of the preceding step existed. |a| then spiked by ~2x at every block
start (23.6, 13.9, 18.5, 14.6 | 26.8, 13.2, 14.4, 14.1 | ...) - a period-4
sawtooth phase-locked to the window grid, i.e. exactly the artefact this file
exists to remove. Forecast and observation are not interchangeable: the affine
calibration aligns them globally but leaves a systematic offset, so a mixed
difference carries that offset and a pure one does not.

The fix costs nothing. Each block predicts ONE EXTRA step - target
[c-1, c+WIN) rather than [c, c+WIN) - from a context of L_CTX steps ending at
c-2, so the reference x^_{c-1} is a forecast like every other term. Horizons
run 1..5 in every block; in-block position p uses x^(k=p+2) - x^(k=p+1) for
every block alike.

Nothing is lost by differencing: z is a recurrence and can integrate. The
anchored form pre-integrates with a periodic reset, which is strictly worse.

The error channel pred(t)-x_t remains BARRED by standing owner rule. It is not
computed, cached or reported.

    python -m relmo.vjrec6 --dataset rcasa --fp16
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
from relmo.vjeval import REC  # noqa: E402
from relmo.vjrec4 import CTX, WIN  # noqa: E402
from relmo.vjs import (GRID, MODEL, TUBELET, probe_dims, read_frames,  # noqa: E402
                       to_tensor)

OUT6 = R.BASE / "vjrec6"
STREAM_FPS = 8.0          # FIXED sample rate. Not derived from clip length.
WIN_FRAMES = 32           # encoder input
HOP_S = 2.0               # == descriptor span, so windows tile exactly
# CTX (=8, from vjrec4) is the first in-clip step that emits a descriptor.
# L_CTX is the prediction context LENGTH, one shorter, so that the block's
# target can start at c-1 and every a_t is a forecast-forecast difference.
# The identity that must hold is  CTX - 1 - L_CTX >= 0.
L_CTX = CTX - 1
CHANNELS = ("pred_change", "obs_change")


def geometry(win_frames=WIN_FRAMES, stream_fps=STREAM_FPS, hop_s=HOP_S,
             ctx=CTX):
    """-> (dt, n_t, desc_steps, hop_steps). Asserts the tiling identity."""
    dt = TUBELET / stream_fps
    n_t = win_frames // TUBELET
    desc = n_t - ctx                       # in-clip steps that emit a descriptor
    hop_steps = hop_s / dt
    assert hop_steps == desc, (
        f"windows do not tile: hop {hop_s}s = {hop_steps} steps but each "
        f"window emits {desc} descriptor steps. Set hop_s = {desc*dt}.")
    return dt, n_t, desc, int(hop_steps)


def frame_change(frames, grid=48):
    """Per-frame change magnitude from PIXELS ONLY. No model, no labels, and
    causal enough to run on a live stream. Downsampled hard because only the
    coarse rate of change matters, not its content."""
    st = max(1, frames.shape[1] // grid)
    F = frames[:, ::st, ::st].astype(np.float32).mean(-1)
    d = np.abs(np.diff(F.reshape(len(F), -1), axis=0)).mean(1)
    return np.concatenate([[0.0], d])


def windows_tempo(frames, ds_win, win_frames=WIN_FRAMES, ctx=CTX,
                  n_t=None):
    """TEMPO-ADAPTIVE grid: each window spans a constant amount of CHANGE
    rather than a constant amount of TIME.

    This is the surviving hypothesis from the clip-vs-stream mechanism hunt.
    v4 spread a fixed frame budget over the whole episode, so its prediction
    horizon was a FRACTION of the event and scaled with the event's tempo; a
    fixed-rate stream asks "what happens in the next 0.25 s" of every event
    regardless. Arc-length reparameterisation is the post-hoc approximation of
    tempo adaptation and is the largest single frozen gain measured; this
    applies it at the ENCODER INPUT instead, where it can also change what the
    predictor is asked to forecast.

    Unlike v4 it needs no episode extent - the change curve is available from
    the pixels of a running stream - so it is deployable on continuous video.
    """
    n_t = n_t or win_frames // TUBELET
    desc = n_t - ctx
    hop_ds = ds_win * desc / n_t        # descriptors cover the last `desc` steps
    c = np.cumsum(frame_change(frames))
    total = float(c[-1])
    out, u = [], 0.0
    while u + ds_win <= total:
        tgt = u + np.linspace(0.0, ds_win, win_frames)
        idx = np.interp(tgt, c, np.arange(len(c))).round().astype(int)
        out.append((u, np.clip(idx, 0, len(frames) - 1)))
        u += hop_ds
    return out


def windows(n_frames, src_fps, win_frames=WIN_FRAMES, stream_fps=STREAM_FPS,
            hop_s=HOP_S, offset_s=0.0):
    """-> [(t0_s, frame_indices)] at a FIXED sample rate.

    offset_s shifts the whole window GRID. In this corpus every episode begins
    at the start of its demonstration, so grid phase is locked to event onset -
    a coincidence that does not hold for a continuous stream, where an event
    starting at 13.7 s lands on whatever phase it lands on. Re-ingesting the
    query side with a shifted grid is the test for whether a model has learned
    the event or the grid.
    """
    span_s = win_frames / stream_fps
    step = src_fps / stream_fps
    out, t = [], float(offset_s)
    while (t + span_s) * src_fps <= n_frames:
        idx = (t * src_fps + np.arange(win_frames) * step).round().astype(int)
        out.append((t, np.clip(idx, 0, n_frames - 1)))
        t += hop_s
    return out


def encode_window(model, torch, dev, clip, cal, layer, n_t, dt_torch):
    """One window -> (a, b, g, n_steps). Step-differenced, fp32 out."""
    n_sp = GRID * GRID
    px = to_tensor(clip, torch, dev, dt_torch)
    alpha, b_cal = cal
    bT = torch.tensor(b_cal, device=dev)
    with torch.no_grad():
        enc = model.encoder(pixel_values_videos=px, output_hidden_states=True)
        seq = enc.last_hidden_state
        X = seq.float()[0].reshape(n_t, n_sp, -1)
        H = enc.hidden_states[layer].float()[0].reshape(n_t, n_sp, -1)
        A, B, G = [], [], []
        for c in range(CTX, n_t, WIN):
            hi = min(c + WIN, n_t)
            S = hi - c
            # context of L_CTX steps ending at c-2; target starts at c-1 so the
            # reference for a_c is itself a FORECAST, not an observation
            assert c - 1 - L_CTX >= 0
            ctx_m = torch.arange((c - 1 - L_CTX) * n_sp, (c - 1) * n_sp,
                                 device=dev).unsqueeze(0)
            tgt_m = torch.arange((c - 1) * n_sp, hi * n_sp,
                                 device=dev).unsqueeze(0)
            po = model.predictor(encoder_hidden_states=seq,
                                 context_mask=[ctx_m], target_mask=[tgt_m])
            P = (alpha * po.last_hidden_state.float()[0]
                 + bT).reshape(S + 1, n_sp, -1)
            # expected one-step change, forecast against forecast throughout
            A.append((P[1:] - P[:-1]).cpu().numpy().astype(np.float16))
            B.append((X[c:hi] - X[c - 1:hi - 1]).cpu().numpy()
                     .astype(np.float16))
            G.append((H[c:hi] - H[c - 1:hi - 1]).norm(dim=-1).cpu().numpy())
    return (np.concatenate(A), np.concatenate(B),
            np.concatenate(G).astype(np.float32))


def record_stream(model, torch, dev, frames, src_fps, cal, layer, dt_torch,
                  win_frames=WIN_FRAMES, stream_fps=STREAM_FPS, hop_s=HOP_S,
                  offset_s=0.0, tempo_ds=0.0):
    """A whole recording -> one continuous trace, indexed by TIME or by CHANGE."""
    dt, n_t, desc, hop_steps = geometry(win_frames, stream_fps, hop_s)
    if tempo_ds > 0:
        wins = windows_tempo(frames, tempo_ds, win_frames, CTX, n_t)
        off_steps = 0
    else:
        wins = windows(len(frames), src_fps, win_frames, stream_fps, hop_s,
                       offset_s)
        off_steps = int(round(offset_s / dt))
    if not wins:
        return None
    A, B, G, step, frame0 = [], [], [], [], []
    for j, (_, idx) in enumerate(wins):
        a, b, g = encode_window(model, torch, dev, frames[idx], cal, layer,
                                n_t, dt_torch)
        A.append(a)
        B.append(b)
        G.append(g)
        step.append(hop_steps * j + off_steps + np.arange(CTX, n_t))
        frame0.append(idx[np.arange(CTX, n_t) * TUBELET])
    step = np.concatenate(step)
    # the tiling identity, checked rather than trusted
    assert (np.diff(step) == 1).all() and step[0] == CTX + off_steps, \
        f"stream steps are not contiguous: {step[:20]}"
    A, B = np.concatenate(A), np.concatenate(B)
    G = np.concatenate(G)
    # gate normalised over the WHOLE recording, not per window: a per-window
    # median would make the spatial pooling depend on window phase, which is
    # the artefact this file exists to remove
    Gc = np.clip(G - np.median(G, 0, keepdims=True), 0, None)
    den = Gc.sum(1, keepdims=True) + 1e-9
    rec = {"where_map": Gc.reshape(-1, GRID, GRID).astype(np.float32),
           "step": step.astype(np.int32),
           "frame0": np.concatenate(frame0).astype(np.int32),
           "dt": np.float32(dt), "n_windows": np.int32(len(wins))}
    for k, V in zip(CHANNELS, (A, B)):
        rec[k] = ((V.astype(np.float32) * Gc[..., None]).sum(1)
                  / den).astype(np.float32)
    return rec


def episode_paths(dataset):
    """-> [(id, mp4_path, src_fps)] straight off the manifest."""
    man = R.read_manifest(dataset)
    fps = float(man.get("fps", 20))
    out = []
    for e in man["episodes"]:
        p = (Path(e["video"]) if e.get("video") else
             R.dataset_dir(dataset) / e.get("shard", "shard_0000") / e["id"]
             / "frames.mp4")
        out.append((e["id"], p, float(e.get("fps", fps))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--stream-fps", type=float, default=STREAM_FPS)
    ap.add_argument("--win-frames", type=int, default=WIN_FRAMES)
    ap.add_argument("--hop", type=float, default=HOP_S)
    ap.add_argument("--calib", default="rcasa")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tempo-ds", type=float, default=0.0,
                    help="tempo-adaptive grid: change-units per window. >0 "
                         "replaces the fixed-rate time grid entirely.")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="shift the window GRID by this many seconds - the "
                         "grid-alignment leakage test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", default="",
                    help="comma-separated split names to restrict to "
                         "(train/val/test) - used by the grid-offset control, "
                         "which only needs the QUERY side re-ingested")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    dt, n_t, desc, hop_steps = geometry(a.win_frames, a.stream_fps, a.hop)
    eps = episode_paths(a.dataset)
    if a.splits:
        from relmo.vjsplit import load as load_split
        sp = load_split()
        keep = set().union(*(sp[k.strip()] for k in a.splits.split(",")
                             if k.strip()))
        eps = [e for e in eps if e[0] in keep]
    if a.limit:
        eps = eps[:a.limit]
    z = np.load(REC / a.calib / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt_torch = torch.float16 if a.fp16 else torch.float32
    print(f"loading {MODEL} onto {dev} in "
          f"{'fp16' if a.fp16 else 'fp32'}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=dt_torch).to(dev).eval()
    print(f"  loaded | step {dt:g}s | window {a.win_frames}f @ "
          f"{a.stream_fps:g}fps = {a.win_frames/a.stream_fps:g}s | ctx "
          f"{CTX*dt:g}s | {desc} descriptor steps/window | hop {a.hop:g}s = "
          f"{hop_steps} steps -> windows TILE", flush=True)

    out = OUT6 / f"{a.dataset}_L{a.layer}{a.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eps if not (out / f"{e[0]}.npz").exists()]
    print(f"{len(eps)} episodes, {len(eps)-len(todo)} cached, {len(todo)} to do",
          flush=True)
    t0, done, failed, short, steps_tot = time.time(), 0, 0, 0, 0
    for eid, mp4, src_fps in tqdm(todo, unit="ep", desc=f"v6/L{a.layer}"):
        if not mp4.exists():
            failed += 1
            continue
        try:
            w_, h_ = probe_dims(mp4)
            F = read_frames(mp4, w_, h_)
            rec = record_stream(model, torch, dev, F, src_fps, cal, a.layer,
                                dt_torch, a.win_frames, a.stream_fps, a.hop,
                                a.offset, a.tempo_ds)
        except Exception as e:                                # noqa: BLE001
            tqdm.write(f"  {eid}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if rec is None:
            short += 1
            continue
        steps_tot += len(rec["step"])
        tmp = out / f".w_{eid}.npz"
        np.savez_compressed(tmp, **rec)
        tmp.rename(out / f"{eid}.npz")
        done += 1
    have = sorted(p for p in out.glob("*.npz") if not p.name.startswith("."))
    lens = [len(np.load(p)["step"]) for p in have]
    rep = dict(dataset=a.dataset, layer=a.layer, stream_fps=a.stream_fps,
               win_frames=a.win_frames, hop_s=a.hop, dt=dt, ctx_steps=CTX,
               offset_s=a.offset,
               episodes=len(eps), written=done, failed=failed,
               too_short=short, on_disk=len(have),
               steps_min=int(min(lens)) if lens else 0,
               steps_med=int(np.median(lens)) if lens else 0,
               steps_max=int(max(lens)) if lens else 0,
               steps_total=int(sum(lens)),
               minutes=round((time.time() - t0) / 60, 1))
    print("\n" + json.dumps(rep, indent=1))
    print(f"VERIFIED on disk: {len(have)}/{len(eps)} stream records, "
          f"{sum(lens)} total steps ({sum(lens)*dt/60:.1f} video-minutes "
          f"described)")
    R.log("vjrec6", **rep)


if __name__ == "__main__":
    main()
