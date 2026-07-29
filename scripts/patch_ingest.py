"""LATE INTERACTION: store the patch grid, not the pooled vector.

One 1152-d vector per frame has to answer both "what is this scene"
and "is there an eggplant in it", and it answers the first. That is
why every channel scores near zero on queries naming a small object
while scoring well on generic manipulation. Cutting the object out
and embedding it alone was measured dead at corpus scale (crops lose
the context that identifies the thing). Late interaction is the third
option: keep the PATCH tokens from the same single full-frame pass and
score MaxSim per query token - ColBERT's operator, applied to video by
Video-ColBERT (arXiv 2503.19009).

Measured before building (scripts/probe_patch_align.py):
  patch-max separates an eggplant frame from a banana frame 0.173 vs
  0.063, where the POOLED embedding manages only 0.107 vs 0.071, and
  each query's best patch lands in a different place on the grid.

Four measured decisions, in the order they were forced:

  patch tokens need the head   raw last_hidden_state is pre-pooling
                               and not in text space; each patch is
                               pushed through the MAP head's own
                               value/output path (MaskCLIP construction)
  no random projection         JL noise is what MaxSim maximizes:
                               contrast 0.110 -> 0.002 at d=256
  PCA helps, pooling does not  whitening RAISES contrast (0.110 ->
                               0.157); spatial pooling destroys the
                               smallest object (0.157 -> 0.030 at 13x13)
  1 bit is enough              signs of the top-256 patches hold
                               contrast at 0.125 vs fp16's 0.113, at
                               8 KB per frame instead of 364 KB

Storage: 256 patches x 256 bits = 8 KB/frame, ~36 MB for the corpus.
The PCA basis is fitted unsupervised on corpus patches and stored with
the table, so a query can project into the same space.

  python scripts/patch_ingest.py [store] [--limit N]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402

NF = 4                  # frames per episode
DIM = 256               # PCA dims kept
KEEP = 256              # patches kept per frame
FIT_EPISODES = 40       # episodes sampled to fit the PCA basis
BATCH = 8


def patch_vectors(model, proc, dev, images):
    """(n, P, 1152) patch tokens in the POOLED embedding's space.

    SigLIP pools with an attention (MAP) head, so last_hidden_state is
    pre-head and not comparable to text. Each patch is pushed through
    the head's value/output projection and residual MLP on its own -
    what the head would produce if that patch were the only one
    attended."""
    import torch
    with torch.no_grad():
        px = proc(images=images, return_tensors="pt").to(dev)
        vm = model.vision_model
        H = vm(**px).last_hidden_state
        head, attn = vm.head, vm.head.attention
        D = H.shape[-1]
        V = H @ attn.in_proj_weight[2 * D:].T + attn.in_proj_bias[2 * D:]
        V = attn.out_proj(V)
        V = V + head.mlp(head.layernorm(V))
    return V.float().cpu().numpy()


def pack(X):
    """(P, DIM) float -> (P, DIM/8) uint8 sign bits, MSB first."""
    return np.packbits((X > 0).astype(np.uint8), axis=-1)


def main():
    from PIL import Image

    from elidedb.device import pick
    from elidedb.sig2 import MID
    from elidedb.video import FrameSet
    from transformers import AutoModel, AutoProcessor

    argv = sys.argv[1:]
    limit = None
    if "--limit" in argv:
        i = argv.index("--limit")
        limit = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    store = Store.open(argv[0] if argv else "lake/bench")

    dev, dtype = pick()
    proc = AutoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(MID, dtype=dtype,
                                      low_cpu_mem_usage=True).to(dev).eval()

    ep = store.table("episodes").scan()
    recs = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    if limit:
        recs = recs[:limit]
    frames_tbl = store.table("frames").scan()

    def frames_of(s, a, b, n=NF):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < n:
            return []
        pi = np.linspace(0, len(sel) - 1, n).round().astype(int)
        try:
            dec = FrameSet(store, "frames", sel.take(pi)).decode()
        except Exception:
            return []
        return [Image.fromarray(f) for _, f in sorted(dec)]

    # ---- pass 1: fit the PCA basis, unsupervised, on corpus patches --
    t0 = time.time()
    step = max(1, len(recs) // FIT_EPISODES)
    fit = []
    for s, a, b in recs[::step][:FIT_EPISODES]:
        ims = frames_of(s, a, b, n=2)
        if ims:
            fit.append(patch_vectors(model, proc, dev, ims))
    FP = np.concatenate([f.reshape(-1, f.shape[-1]) for f in fit])
    mu = FP.mean(0)
    rng = np.random.default_rng(0)
    sub = FP[rng.choice(len(FP), min(60000, len(FP)), replace=False)] - mu
    _, _, Vt = np.linalg.svd(sub, full_matrices=False)
    R = np.ascontiguousarray(Vt[:DIM].T.astype(np.float32))
    print(f"PCA basis from {len(FP):,} patches, {time.time() - t0:.0f}s",
          flush=True)

    # ---- pass 2: code every episode ---------------------------------
    rows_s, rows_a, rows_b, rows_f, codes = [], [], [], [], []
    for ri, (s, a, b) in enumerate(recs):
        ims = frames_of(s, a, b)
        if not ims:
            continue
        P = patch_vectors(model, proc, dev, ims)
        for fi in range(len(P)):
            X = (P[fi] - mu) @ R
            X /= np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8
            # write-time pruning, query-independent: patches that differ
            # most from this frame's own mean carry its distinct content
            d = np.linalg.norm(X - X.mean(0, keepdims=True), axis=-1)
            X = X[np.argsort(-d)[:KEEP]]
            rows_s.append(s); rows_a.append(a); rows_b.append(b)
            rows_f.append(fi)
            codes.append(pack(X).reshape(-1).tobytes())
        if (ri + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {ri + 1}/{len(recs)}  {el:.0f}s  "
                  f"ETA {el / (ri + 1) * len(recs) / 60:.0f}min", flush=True)

    nbytes = KEEP * DIM // 8
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "frame_no": pa.array(rows_f, pa.int32()),
        "code": pa.array(codes, pa.binary(nbytes)),
    })
    tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
    store.table("patch_codes").append(
        tbl, kind="index",
        meta={"model": MID, "dim": DIM, "keep": KEEP,
              "frames_per_episode": NF, "code_bytes": nbytes})
    # the projection travels WITH the store - a query cannot enter the
    # code space without it, and a store must answer on its own
    np.savez(Path(store.dir) / "_patch_basis.npz", mu=mu.astype(np.float32),
             R=R)
    print(json.dumps({"frames": len(tbl), "episodes": len(set(rows_a)),
                      "bytes_per_frame": nbytes,
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
