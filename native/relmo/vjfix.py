"""Build every variant the audit needs to test its fixes, in one pass.

The audit (relmo/vjaudit.py) found that almost nothing the pipeline claims to
recover is actually recovered:

    A WHERE   gate peak on the object    0.241 vs 0.114 chance   2.12x  weak
    B WHEN    corr(gate mass, speed)     +0.180                         weak
    C WHAT    content -> displacement    R2 0.008                       nothing
    D CONTACT corr(gate mass, contact)   -0.017                         nothing
    E SPAN    active vs moving IoU       0.390 vs 0.333                 chance

while retrieval still reaches 0.544 at 2.45x chance. So retrieval works for
reasons other than the stated mechanism, and each claim needs its own fix
tested rather than the whole thing swapped for a bigger model.

THE DECISIVE ONE IS C, and it needs a control the cached records cannot give.
Content is pooled over patches WEIGHTED BY THE GATE. If the gate is wrong, the
content is pooled from the wrong place, and C fails for A's reason rather than
its own. Pooling under an ORACLE gate - the true segmentation mask of the
manipulated object - separates the two:
    oracle pooling recovers displacement  -> the CONTENT is fine, the GATE is
                                             the bug, fix A and C follows
    oracle pooling still fails            -> the content genuinely does not
                                             carry it, and no gate fixes that

CACHED PER EPISODE, all aligned to the same 24 trace steps:
    g3, g6, g12   gate maps at three depths (localisation degrades with depth:
                  moving/static 6.64 / 2.62 / 1.40)
    gmot          pixel-motion map, the sharpest localiser measured (0.832
                  concentration vs the layer-6 gate's 0.596)
    gorc          ORACLE gate from seg + target_bodies. SCORING ONLY - it is
                  never available at serve and must never enter an index.
    c_<gate>      pred_change pooled under each of the above, plus uniform

    python -m relmo.vjfix --episodes 150
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
from relmo.vjs import (GRID, MODEL, TUBELET, patch_motion, probe_dims,  # noqa: E402
                       read_frames, sample_clip, to_tensor)
from relmo.vjeval import REC  # noqa: E402
from relmo.vjrec4 import CTX, WIN  # noqa: E402

OUTF = R.BASE / "vjfix"
LAYERS = (3, 6, 12)
GATES = ("g3", "g6", "g12", "gmot", "gorc", "guni")


def oracle_gate(seg, tb, idx, steps):
    """True mask of the manipulated object on the 16x16 token grid.

    SCORING ONLY. seg holds BODY ids directly (an earlier version of this test
    indirected through geom_bodyid and produced an all-zero mask).
    """
    import torch
    import torch.nn.functional as Fn
    out = np.zeros((len(steps), GRID * GRID), np.float32)
    T = len(seg)
    for si, t in enumerate(steps):
        f = min(idx[t * TUBELET], T - 1)
        m = np.isin(seg[f], tb).astype(np.float32)
        g = Fn.interpolate(torch.tensor(m)[None, None], size=(GRID, GRID),
                           mode="area")[0, 0].numpy()
        out[si] = g.ravel()
    return out


def build(model, torch, dev, clip, cal, seg, tb, idx):
    n_t, n_sp = 32, GRID * GRID
    px = to_tensor(clip, torch, dev, torch.float32)
    alpha, b = cal
    bT = torch.tensor(b, device=dev)
    steps = list(range(CTX, n_t))
    with torch.no_grad():
        enc = model.encoder(pixel_values_videos=px, output_hidden_states=True)
        seq = enc.last_hidden_state
        H = {L: enc.hidden_states[L].float()[0] for L in LAYERS}
        gl = {L: [] for L in LAYERS}
        cont = []
        for c in range(CTX, n_t, WIN):
            hi = min(c + WIN, n_t)
            S = hi - c
            ctx = torch.arange((c - CTX) * n_sp, c * n_sp, device=dev).unsqueeze(0)
            tgt = torch.arange(c * n_sp, hi * n_sp, device=dev).unsqueeze(0)
            po = model.predictor(encoder_hidden_states=seq,
                                 context_mask=[ctx], target_mask=[tgt])
            P = (alpha * po.last_hidden_state.float()[0] + bT).reshape(S, n_sp, -1)
            NOW = seq.float()[0, (c - 1) * n_sp:c * n_sp].unsqueeze(0)
            cont.append((P - NOW).cpu().numpy())            # pred_change per patch
            for L in LAYERS:
                now = H[L][(c - 1) * n_sp:c * n_sp].unsqueeze(0)
                gl[L].append((H[L][c * n_sp:hi * n_sp].reshape(S, n_sp, -1)
                              - now).norm(dim=-1).cpu().numpy())
    V = np.concatenate(cont)                                 # (24,256,1024)
    rec = {}
    for L in LAYERS:
        g = np.concatenate(gl[L])
        rec[f"g{L}"] = np.clip(g - np.median(g, 0, keepdims=True), 0, None)
    mot = patch_motion(clip, n_t).reshape(n_t, -1)[CTX:]
    rec["gmot"] = np.clip(mot - np.median(mot, 0, keepdims=True), 0, None)
    rec["gorc"] = oracle_gate(seg, tb, idx, steps)
    rec["guni"] = np.ones_like(rec["g6"])
    for k in GATES:
        w = rec[k]
        rec["c_" + k] = ((V * w[..., None]).sum(1)
                         / (w.sum(1, keepdims=True) + 1e-9)).astype(np.float32)
        rec[k] = rec[k].astype(np.float32)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--frames", type=int, default=64)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d = REC / a.dataset
    man = R.read_manifest(a.dataset)
    by = {e["id"]: e for e in man["episodes"]}
    files = [p for p in sorted(d.glob("*.npz"))
             if not p.name.startswith("_") and p.stem in by]
    rng = np.random.default_rng(0)
    files = [files[i] for i in sorted(rng.choice(len(files), min(a.episodes,
                                                                len(files)),
                                                 replace=False))]
    z = np.load(d / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    print(f"  loaded | gates {GATES} | {len(files)} episodes", flush=True)

    out = OUTF / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in files if not (out / f"{p.stem}.npz").exists()]
    print(f"{len(todo)} to do (~{len(todo)*11/60:.0f} min)", flush=True)
    t0, done = time.time(), 0
    for p in tqdm(todo, unit="ep", desc="vjfix"):
        e = by[p.stem]
        dd = R.dataset_dir(a.dataset) / e["shard"] / p.stem
        try:
            st = np.load(dd / "state.npz")
            if "seg" not in st.files:
                continue
            w_, h_ = probe_dims(dd / "frames.mp4")
            F = read_frames(dd / "frames.mp4", w_, h_)
            if len(F) < a.frames:
                continue
            idx = np.linspace(0, len(F) - 1, a.frames).round().astype(int)
            rec = build(model, torch, dev, sample_clip(F, a.frames), cal,
                        st["seg"], st["target_bodies"], idx)
        except Exception as ex:                              # noqa: BLE001
            tqdm.write(f"  {p.stem}: {type(ex).__name__}: {ex}")
            continue
        tmp = out / f".w_{p.stem}.npz"
        np.savez_compressed(tmp, **rec)
        tmp.rename(out / f"{p.stem}.npz")
        done += 1
    have = len(list(out.glob("*.npz")))
    rep = dict(dataset=a.dataset, episodes=len(files), written=done,
               on_disk=have, minutes=round((time.time() - t0) / 60, 1))
    print("\n" + json.dumps(rep, indent=1))
    print(f"VERIFIED on disk: {have} records")
    R.log("vjfix", **rep)


if __name__ == "__main__":
    main()
