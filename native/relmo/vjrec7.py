"""v7: PER-TOKEN records, so the spatial pooling can be LEARNED.

v6 collapses V-JEPA's 256 spatial tokens into one vector per step with a fixed
gate: layer-6 observed change magnitude, median-subtracted and clipped. That
gate was never trained and never compared against anything. This module keeps
the tokens so a head can learn which of them matter.

WHAT IS STORED, and why only one channel. The barred residual
pred(t+1)-act(t+1) = a - b is formable by any linear layer that receives both as
vectors, so at most one may be one. v6 chose `a`; the frozen sweep says that was
backwards - `b` alone outscores `a` alone (0.442 vs 0.420 prec) and `a` is what
the model GUESSED, hence partly a function of the prior, which is the owner's
own objection to the error channel. So `b` is the primary channel here and the
only one stored per token. `g`'s two scalars and `sig` come from the v6 records
unchanged.

TOKEN DIMENSION. 1024-d per token is (T, 256, 1024) - 46 MB for a median
recording. A PCA fitted on TRAIN tokens cuts it to 96-d at ~2 MB. This loses
nothing for the pooling itself: softmax-weighted pooling is a convex combination
of tokens, hence linear, and PCA commutes with linear pooling. It costs only in
the attention WEIGHTS, which are computed from the projected features.

    python -m relmo.vjrec7 --fit          # fit the token PCA on train
    python -m relmo.vjrec7 --dataset rcasa --fp16
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
from relmo.vjrec6 import (HOP_S, L_CTX, STREAM_FPS, WIN_FRAMES,  # noqa: E402
                          episode_paths, geometry, windows)
from relmo.vjs import GRID, MODEL, TUBELET, probe_dims, read_frames, to_tensor  # noqa: E402

OUT7 = R.BASE / "vjrec7"
TOK_DIM = 96


def encode_tokens(model, torch, dev, clip, layer, n_t, dt_torch):
    """One window -> (b_tok (S,256,1024), gate (S,256)). Observation only."""
    n_sp = GRID * GRID
    px = to_tensor(clip, torch, dev, dt_torch)
    with torch.no_grad():
        enc = model.encoder(pixel_values_videos=px, output_hidden_states=True)
        X = enc.last_hidden_state.float()[0].reshape(n_t, n_sp, -1)
        H = enc.hidden_states[layer].float()[0].reshape(n_t, n_sp, -1)
        B, G = [], []
        for c in range(CTX, n_t, WIN):
            hi = min(c + WIN, n_t)
            B.append((X[c:hi] - X[c - 1:hi - 1]).cpu().numpy().astype(np.float16))
            G.append((H[c:hi] - H[c - 1:hi - 1]).norm(dim=-1).cpu().numpy())
    return np.concatenate(B), np.concatenate(G).astype(np.float32)


def record(model, torch, dev, frames, src_fps, layer, dt_torch, proj=None):
    dt, n_t, desc, hop_steps = geometry(WIN_FRAMES, STREAM_FPS, HOP_S)
    wins = windows(len(frames), src_fps, WIN_FRAMES, STREAM_FPS, HOP_S)
    if not wins:
        return None
    B, G, step, frame0 = [], [], [], []
    for j, (_, idx) in enumerate(wins):
        b, g = encode_tokens(model, torch, dev, frames[idx], layer, n_t,
                             dt_torch)
        if proj is not None:
            b = (b.astype(np.float32) @ proj).astype(np.float16)
        B.append(b)
        G.append(g)
        step.append(hop_steps * j + np.arange(CTX, n_t))
        frame0.append(idx[np.arange(CTX, n_t) * TUBELET])
    step = np.concatenate(step)
    assert (np.diff(step) == 1).all() and step[0] == CTX
    return dict(b_tok=np.concatenate(B), gate=np.concatenate(G),
                step=step.astype(np.int32),
                frame0=np.concatenate(frame0).astype(np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--fit", action="store_true",
                    help="fit the token PCA on TRAIN episodes and exit")
    ap.add_argument("--fit-episodes", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    _, n_t, _, _ = geometry(WIN_FRAMES, STREAM_FPS, HOP_S)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt_torch = torch.float16 if a.fp16 else torch.float32
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=dt_torch).to(dev).eval()
    OUT7.mkdir(parents=True, exist_ok=True)
    pfile = OUT7 / "_token_pca.npz"

    if a.fit:
        from relmo.vjsplit import load as load_split
        sp = load_split()
        eps = [e for e in episode_paths(a.dataset) if e[0] in sp["train"]]
        eps = eps[:a.fit_episodes]
        toks = []
        for eid, mp4, fps in tqdm(eps, unit="ep", desc="pca-fit"):
            if not mp4.exists():
                continue
            w_, h_ = probe_dims(mp4)
            F = read_frames(mp4, w_, h_)
            wins = windows(len(F), fps, WIN_FRAMES, STREAM_FPS, HOP_S)
            for _, idx in wins[:4]:
                b, _ = encode_tokens(model, torch, dev, F[idx], a.layer, n_t,
                                     dt_torch)
                toks.append(b.reshape(-1, b.shape[-1]).astype(np.float32))
        X = np.concatenate(toks)
        rng = np.random.default_rng(0)
        if len(X) > 300_000:
            X = X[rng.choice(len(X), 300_000, replace=False)]
        mu = X.mean(0, keepdims=True)
        _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        var = float((S[:TOK_DIM] ** 2).sum() / (S ** 2).sum())
        np.savez(pfile, mu=mu, W=Vt[:TOK_DIM].T.astype(np.float32), var=var)
        print(f"\ntoken PCA on {len(X)} token vectors from {len(eps)} TRAIN "
              f"recordings -> {TOK_DIM}d, variance kept {var:.3f}")
        print(f"VERIFIED: wrote {pfile}")
        R.log("vjrec7_pca", dim=TOK_DIM, var=round(var, 4), n_tokens=len(X))
        return

    if not pfile.exists():
        raise SystemExit("run --fit first")
    P = np.load(pfile)
    proj = P["W"].astype(np.float32)
    print(f"  token PCA {proj.shape[0]} -> {proj.shape[1]}, "
          f"variance {float(P['var']):.3f}", flush=True)

    eps = episode_paths(a.dataset)
    if a.limit:
        eps = eps[:a.limit]
    out = OUT7 / f"{a.dataset}_L{a.layer}{a.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eps if not (out / f"{e[0]}.npz").exists()]
    print(f"{len(eps)} episodes, {len(todo)} to do", flush=True)
    t0, done, failed, short = time.time(), 0, 0, 0
    for eid, mp4, fps in tqdm(todo, unit="ep", desc="v7/tokens"):
        if not mp4.exists():
            failed += 1
            continue
        try:
            w_, h_ = probe_dims(mp4)
            rec = record(model, torch, dev, read_frames(mp4, w_, h_), fps,
                         a.layer, dt_torch, proj)
        except Exception as e:                                # noqa: BLE001
            tqdm.write(f"  {eid}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if rec is None:
            short += 1
            continue
        tmp = out / f".w_{eid}.npz"
        np.savez_compressed(tmp, **rec)
        tmp.rename(out / f"{eid}.npz")
        done += 1
    have = [p for p in out.glob("*.npz") if not p.name.startswith(".")]
    gb = sum(p.stat().st_size for p in have) / 1e9
    rep = dict(dataset=a.dataset, layer=a.layer, tok_dim=TOK_DIM,
               episodes=len(eps), written=done, failed=failed, too_short=short,
               on_disk=len(have), gb=round(gb, 2),
               minutes=round((time.time() - t0) / 60, 1))
    print("\n" + json.dumps(rep, indent=1))
    print(f"VERIFIED on disk: {len(have)}/{len(eps)} per-token records, "
          f"{gb:.2f} GB")
    R.log("vjrec7", **rep)


if __name__ == "__main__":
    main()
