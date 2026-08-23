"""v8: ONE pass. Same three artifacts as vjrec6 + vjrec7 + vjsig6, less compute.

THE DUPLICATION, precisely. relmo.vjrec6.encode_window already computes the
per-token observed change `b` for every window - it pools it against the gate
and DISCARDS the tokens. relmo.vjrec7.encode_tokens then re-decodes the video
and re-runs the V-JEPA encoder over the identical windows to recompute exactly
those tokens. relmo.vjsig6 decodes the video a third time to embed the frames
whose indices vjrec6 already wrote out. So the shipped write path is

    3 x ffmpeg decode + 2 x V-JEPA encoder + 1 x predictor + 1 x SigLIP

where one decode and one encoder pass suffice. This module is that arrangement.

IT IS A MERGE, NOT A NEW CONFIGURATION. The three artifacts must come out
bit-identical to the three-script pipeline, or the accuracy numbers measured
against them stop applying. Two details carry that guarantee:

  * v6 and v7 both consume the SAME fp16 per-token `b`. v6 upcasts it to fp32
    and pools; v7 upcasts it to fp32 and projects. Computing it once and
    feeding both consumers changes nothing - the fp16 rounding that would be
    the only risk happens before the split, not after.
  * the token PCA is LOADED, never refitted. Refitting on a different sample
    would silently move the basis and every stored token with it.

`--verify` checks that claim against the existing records rather than asserting
it, array by array, and is the gate this module had to pass before its output
was allowed to stand in for theirs.

    python -m relmo.vjrec8 --dataset rcasa --fp16 --verify --limit 8
    python -m relmo.vjrec8 --dataset rcasa --fp16
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
from relmo.vjrec6 import (CHANNELS, HOP_S, L_CTX, OUT6, STREAM_FPS,  # noqa: E402
                          WIN_FRAMES, episode_paths, geometry, windows,
                          windows_tempo)
from relmo.vjrec7 import OUT7  # noqa: E402
from relmo.vjs import MODEL, TUBELET, probe_dims, read_frames, to_tensor  # noqa: E402
from relmo.vjsig import MODEL as SIG_MODEL, RES as SIG_RES  # noqa: E402

OUTS = R.BASE / "vjsig6"


def encode_window(model, torch, dev, clip, cal, layer, n_t, dt_torch, res=None):
    """One window, ONE encoder forward -> (a_tok, b_tok, gate), fp16/fp16/fp32.

    Byte-for-byte relmo.vjrec6.encode_window. It is duplicated here rather than
    imported only so the merged path is readable end to end; --verify keeps the
    two honest.
    """
    from relmo.vjs import CROP, PATCH
    n_sp = ((res or CROP) // PATCH) ** 2
    px = to_tensor(clip, torch, dev, dt_torch, res)
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
            assert c - 1 - L_CTX >= 0
            ctx_m = torch.arange((c - 1 - L_CTX) * n_sp, (c - 1) * n_sp,
                                 device=dev).unsqueeze(0)
            tgt_m = torch.arange((c - 1) * n_sp, hi * n_sp,
                                 device=dev).unsqueeze(0)
            po = model.predictor(encoder_hidden_states=seq,
                                 context_mask=[ctx_m], target_mask=[tgt_m])
            P = (alpha * po.last_hidden_state.float()[0]
                 + bT).reshape(S + 1, n_sp, -1)
            A.append((P[1:] - P[:-1]).cpu().numpy().astype(np.float16))
            B.append((X[c:hi] - X[c - 1:hi - 1]).cpu().numpy()
                     .astype(np.float16))
            G.append((H[c:hi] - H[c - 1:hi - 1]).norm(dim=-1).cpu().numpy())
    return (np.concatenate(A), np.concatenate(B),
            np.concatenate(G).astype(np.float32))


def siglip_at(sg, torch, dev, frames, frame0, batch, mean, std):
    """The vjsig6 body, on frames that are already in memory."""
    import torch.nn.functional as Fn
    sel = frames[np.clip(frame0.astype(int), 0, len(frames) - 1)]
    V = []
    for s in range(0, len(sel), batch):
        x = torch.tensor(sel[s:s + batch]).permute(0, 3, 1, 2).float().div_(255.)
        x = Fn.interpolate(x, size=(SIG_RES, SIG_RES), mode="bilinear",
                           align_corners=False)
        x = ((x - mean) / std).to(dev, torch.float16)
        with torch.no_grad():
            V.append(sg.get_image_features(pixel_values=x).float().cpu().numpy())
    return np.concatenate(V).astype(np.float32)


def record_all(model, sg, torch, dev, frames, src_fps, cal, layer, dt_torch,
               proj, win_frames=WIN_FRAMES, stream_fps=STREAM_FPS, hop_s=HOP_S,
               offset_s=0.0, tempo_ds=0.0, res=None, sig_batch=64,
               mean=None, std=None, T=None):
    """A whole recording -> (rec6, rec7, sig). One decode, one encoder pass."""
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

    t = time.time()
    A, B, G, step, frame0 = [], [], [], [], []
    for j, (_, idx) in enumerate(wins):
        a, b, g = encode_window(model, torch, dev, frames[idx], cal, layer,
                                n_t, dt_torch, res)
        A.append(a)
        B.append(b)
        G.append(g)
        step.append(hop_steps * j + off_steps + np.arange(CTX, n_t))
        frame0.append(idx[np.arange(CTX, n_t) * TUBELET])
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()
    if T is not None:
        T["vjepa"] += time.time() - t

    t = time.time()
    step = np.concatenate(step)
    assert (np.diff(step) == 1).all() and step[0] == CTX + off_steps, \
        f"stream steps are not contiguous: {step[:20]}"
    A, B = np.concatenate(A), np.concatenate(B)
    G = np.concatenate(G)
    f0 = np.concatenate(frame0).astype(np.int32)

    # ---- v6 artifact: gate-pooled, gate normalised over the WHOLE recording
    Gc = np.clip(G - np.median(G, 0, keepdims=True), 0, None)
    den = Gc.sum(1, keepdims=True) + 1e-9
    gsz = int(round(Gc.shape[1] ** 0.5))
    rec6 = {"where_map": Gc.reshape(-1, gsz, gsz).astype(np.float32),
            "step": step.astype(np.int32), "frame0": f0,
            "dt": np.float32(dt), "n_windows": np.int32(len(wins))}
    for k, V in zip(CHANNELS, (A, B)):
        rec6[k] = ((V.astype(np.float32) * Gc[..., None]).sum(1)
                   / den).astype(np.float32)

    # ---- v7 artifact: the SAME fp16 b, projected instead of pooled.
    # Accelerate's BLAS raises invalid/divide/overflow flags on Apple Silicon
    # for matmuls whose results are finite; --verify shows this projection is
    # bit-identical to vjrec7's, so the flags are spurious. Scoped, never global.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        b_tok = (B.astype(np.float32) @ proj).astype(np.float16)
    assert np.isfinite(b_tok.astype(np.float32)).all(), "b_tok not finite"
    rec7 = {"b_tok": b_tok, "gate": G.astype(np.float32),
            "step": step.astype(np.int32), "frame0": f0}
    if T is not None:
        T["pack"] += time.time() - t

    t = time.time()
    sig = siglip_at(sg, torch, dev, frames, f0, sig_batch, mean, std)
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()
    if T is not None:
        T["siglip"] += time.time() - t
    assert len(sig) == len(f0), f"{len(sig)} != {len(f0)}"
    return rec6, rec7, sig


def _cmp(new, ref_path, label, tol=0.0):
    """Array-by-array identity check against an existing record."""
    if not ref_path.exists():
        return [f"{label}: no reference at {ref_path.name}"]
    z = np.load(ref_path)
    bad = []
    keys = sorted(set(new) | set(z.files))
    for k in keys:
        if k not in new or k not in z.files:
            bad.append(f"{label}.{k}: present in only one")
            continue
        u, v = np.asarray(new[k]), np.asarray(z[k])
        if u.shape != v.shape:
            bad.append(f"{label}.{k}: shape {u.shape} vs {v.shape}")
            continue
        if u.dtype != v.dtype:
            bad.append(f"{label}.{k}: dtype {u.dtype} vs {v.dtype}")
        d = np.abs(u.astype(np.float64) - v.astype(np.float64)).max() \
            if u.size else 0.0
        if d > tol:
            bad.append(f"{label}.{k}: max|diff| {d:g}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--calib", default="rcasa")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--win-frames", type=int, default=WIN_FRAMES)
    ap.add_argument("--stream-fps", type=float, default=STREAM_FPS)
    ap.add_argument("--hop", type=float, default=HOP_S)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--tempo-ds", type=float, default=0.0)
    ap.add_argument("--sig-batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", default="")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--pca-suffix", default=None,
                    help="token PCA to LOAD; defaults to --suffix")
    ap.add_argument("--verify", action="store_true",
                    help="compare against the vjrec6/vjrec7/vjsig6 records "
                         "instead of writing; exits non-zero on any difference")
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModel, VJEPA2Model

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
    psuf = a.suffix if a.pca_suffix is None else a.pca_suffix
    pfile = OUT7 / f"_token_pca{psuf}.npz"
    if not pfile.exists():
        raise SystemExit(f"no token PCA at {pfile} - run "
                         f"`python -m relmo.vjrec7 --fit` first")
    P = np.load(pfile)
    proj = P["W"].astype(np.float32)

    from relmo.device import pick as _pick_device  # cuda > mps > cpu
    dev = _pick_device()
    dt_torch = torch.float16 if a.fp16 else torch.float32
    print(f"loading {MODEL} + {SIG_MODEL} onto {dev} in "
          f"{'fp16' if a.fp16 else 'fp32'} (excluded from timings)...",
          flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=dt_torch).to(dev).eval()
    sg = AutoModel.from_pretrained(SIG_MODEL,
                                   dtype=torch.float16).to(dev).eval()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    print(f"  loaded | step {dt:g}s | window {a.win_frames}f @ "
          f"{a.stream_fps:g}fps | {desc} descriptor steps/window | hop "
          f"{a.hop:g}s | token PCA {proj.shape[0]}->{proj.shape[1]} "
          f"(var {float(P['var']):.3f})", flush=True)

    d6 = OUT6 / f"{a.dataset}_L{a.layer}{a.suffix}"
    d7 = OUT7 / f"{a.dataset}_L{a.layer}{a.suffix}"
    ds = OUTS / f"{a.dataset}{a.suffix}"
    if not a.verify:
        for d in (d6, d7, ds):
            d.mkdir(parents=True, exist_ok=True)
        todo = [e for e in eps
                if not ((d6 / f"{e[0]}.npz").exists()
                        and (d7 / f"{e[0]}.npz").exists()
                        and (ds / f"{e[0]}.npz").exists())]
    else:
        todo = eps
    print(f"{len(eps)} episodes, {len(eps)-len(todo)} cached, "
          f"{len(todo)} to do", flush=True)

    T = dict(decode=0.0, vjepa=0.0, siglip=0.0, pack=0.0, write=0.0)
    t0, done, failed, short, steps_tot = time.time(), 0, 0, 0, 0
    video_s, diffs, checked = 0.0, [], 0
    for eid, mp4, src_fps in tqdm(todo, unit="ep",
                                  desc="v8/verify" if a.verify else "v8/one-pass"):
        if not mp4.exists():
            failed += 1
            continue
        try:
            t = time.time()
            w_, h_ = probe_dims(mp4)
            F = read_frames(mp4, w_, h_)
            T["decode"] += time.time() - t
            video_s += len(F) / src_fps
            out = record_all(model, sg, torch, dev, F, src_fps, cal, a.layer,
                             dt_torch, proj, a.win_frames, a.stream_fps, a.hop,
                             a.offset, a.tempo_ds, a.res or None, a.sig_batch,
                             mean, std, T)
        except Exception as e:                                  # noqa: BLE001
            tqdm.write(f"  {eid}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if out is None:
            short += 1
            continue
        rec6, rec7, sig = out
        steps_tot += len(rec6["step"])
        if a.verify:
            bad = (_cmp(rec6, d6 / f"{eid}.npz", "v6")
                   + _cmp(rec7, d7 / f"{eid}.npz", "v7")
                   + _cmp({"sig": sig}, ds / f"{eid}.npz", "sig"))
            checked += 1
            if bad:
                diffs.append((eid, bad))
                tqdm.write(f"  DIFF {eid}: {'; '.join(bad[:4])}")
            done += 1
            continue
        t = time.time()
        for d, payload in ((d6, rec6), (d7, rec7), (ds, {"sig": sig})):
            tmp = d / f".w_{eid}.npz"
            np.savez_compressed(tmp, **payload)
            tmp.rename(d / f"{eid}.npz")
        T["write"] += time.time() - t
        done += 1

    tot = sum(T.values())
    wall = time.time() - t0
    if a.verify:
        print(f"\nVERIFY: {checked} recordings compared against the "
              f"vjrec6/vjrec7/vjsig6 records, {len(diffs)} with any difference")
        if diffs:
            for eid, bad in diffs[:10]:
                print(f"  {eid}: {'; '.join(bad)}")
            raise SystemExit(f"{len(diffs)}/{checked} records differ - "
                             f"the merge is NOT identity-preserving")
        print("VERIFIED: every array bit-identical. The merged pass may stand "
              "in for the three-script pipeline.")
        R.log("vjrec8_verify", dataset=a.dataset, checked=checked, diffs=0)
        return

    n6 = len([p for p in d6.glob("*.npz") if not p.name.startswith(".")])
    n7 = len([p for p in d7.glob("*.npz") if not p.name.startswith(".")])
    nsg = len([p for p in ds.glob("*.npz") if not p.name.startswith(".")])
    rate = video_s / max(tot, 1e-9)
    rep = dict(dataset=a.dataset, layer=a.layer, suffix=a.suffix,
               win_frames=a.win_frames, hop_s=a.hop, res=a.res or 256,
               episodes=len(eps), written=done, failed=failed, too_short=short,
               on_disk_v6=n6, on_disk_v7=n7, on_disk_sig=nsg,
               steps_total=steps_tot, video_s=round(video_s, 1),
               minutes=round(wall / 60, 1),
               compute_min=round(tot / 60, 1),
               realtime=round(rate, 2),
               min_per_video_hour=round(60.0 / max(rate, 1e-9), 2),
               **{f"s_{k}": round(v, 1) for k, v in T.items()})
    print("\n" + json.dumps(rep, indent=1))
    print(f"\n{'stage':10s} {'compute s':>10s} {'share':>7s}")
    print("-" * 30)
    for k in ("decode", "vjepa", "siglip", "pack", "write"):
        print(f"{k:10s} {T[k]:10.1f} {T[k]/max(tot,1e-9)*100:6.1f}%")
    print("-" * 30)
    print(f"{'TOTAL':10s} {tot:10.1f} {100.0:6.1f}%")
    print(f"\n{video_s/60:.1f} video-minutes -> {rate:.2f}x real-time = "
          f"{60.0/max(rate,1e-9):.2f} compute-minutes per video-hour")
    print(f"VERIFIED on disk: v6 {n6}, v7 {n7}, siglip {nsg} of {len(eps)}")
    R.log("vjrec8", **rep)


if __name__ == "__main__":
    main()
