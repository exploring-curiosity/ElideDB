"""Domain-blind write path: pretrained video-encoder WINDOW SEQUENCES.

The design that survived every measured failure: no events, no
elements, no fitted semantic cuts - a frozen pretrained video encoder
(V-JEPA 2, motion-sensitive, no text) embeds overlapping 2s windows,
and an episode IS its sequence of window vectors on the timeline.
Retrieval = temporal alignment of sequences (native/seqbench.py).
Works identically on tabletop sim, kitchen video, driving - nothing
here knows what an arm is.

Per-episode npz cache; decode once per episode at 256px, windows are
batched through the encoder.

    python native/seq.py --store lake/sim_chains --eps 2   (timing)
    python native/seq.py --store lake/sim_chains           (full)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

MID = "facebook/vjepa2-vitl-fpc64-256"
WIN_S = 2.0
STRIDE_S = 1.0
NFRAMES = 16
BATCH = 4

SCRATCH = Path(os.environ.get(
    "ELIDEDB_SCRATCH",
    "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
    "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
    "scratchpad"))


def cache_dir(store_name):
    d = SCRATCH / f"seq_vjepa_v1_{store_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def arg(name, default, cast):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def episode_windows(ts):
    """[(t_center_ns, frame indices)] on the WIN/STRIDE grid."""
    t0, t1 = int(ts[0]), int(ts[-1])
    out = []
    c = t0 + int(WIN_S / 2 * 1e9)
    while c <= t1 - int(WIN_S / 2 * 1e9) + int(STRIDE_S * 1e9):
        a = c - int(WIN_S / 2 * 1e9)
        b = c + int(WIN_S / 2 * 1e9)
        idx = np.where((ts >= a) & (ts <= b))[0]
        if len(idx) >= 4:
            pick = idx[np.linspace(0, len(idx) - 1, NFRAMES)
                       .round().astype(int)]
            out.append((c, pick))
        c += int(STRIDE_S * 1e9)
    return out


def main():
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoVideoProcessor
    from elidedb import Store
    import chain_delta as cd

    store = ROOT / arg("--store", "lake/sim_chains", str)
    n_eps = arg("--eps", 0, int)
    stream = arg("--stream", "", str)

    db = Store.open(str(store))
    views = cd.episode_views(db)
    if stream:
        views = [v for v in views if v[1] == stream]
    else:
        # one stream per episode is the sequence (simA convention)
        first = {}
        for v in views:
            first.setdefault(v[0], v)
        views = [first[k] for k in sorted(first)]
    if n_eps:
        views = views[:n_eps]
    cdir = cache_dir(store.name)
    todo = [v for v in views
            if not (cdir / f"{v[0]:05d}.npz").exists()]
    print(f"{store.name}: {len(views)} episodes, {len(todo)} to embed",
          flush=True)
    if not todo:
        return

    print(f"loading {MID} (fp16, mps)...", flush=True)
    proc = AutoVideoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(MID, dtype=torch.float16) \
        .to("mps").eval()

    t_all = time.time()
    for ep, sv, sl in tqdm(todo, desc="embed", unit="ep"):
        t0 = time.time()
        ts, F = cd.decode_view(db, sl)
        t_dec = time.time() - t0
        wins = episode_windows(np.asarray(ts, np.int64))
        vecs, cents = [], []
        t0 = time.time()
        for i0 in range(0, len(wins), BATCH):
            chunk = wins[i0:i0 + BATCH]
            vids = [[F[j] for j in pick] for _, pick in chunk]
            inp = proc([v for v in vids], return_tensors="pt")
            pv = inp["pixel_values_videos"].to("mps", torch.float16)
            with torch.no_grad():
                out = model(pixel_values_videos=pv)
            V = out.last_hidden_state.mean(1).float().cpu().numpy()
            for (c, _), v in zip(chunk, V):
                vecs.append(v / (np.linalg.norm(v) + 1e-8))
                cents.append(c)
        t_emb = time.time() - t0
        np.savez_compressed(cdir / f"{ep:05d}.npz",
                            V=np.stack(vecs).astype(np.float16),
                            ts=np.array(cents, np.int64))
        if ep == todo[0][0]:
            print(f"  first ep: {len(wins)} windows  decode "
                  f"{t_dec:.0f}s embed {t_emb:.0f}s -> projected "
                  f"{(t_dec+t_emb)*len(todo)/60:.0f} min total",
                  flush=True)
    print(f"embedded {len(todo)} eps in {(time.time()-t_all)/60:.1f} "
          f"min -> {cdir}", flush=True)


if __name__ == "__main__":
    main()
