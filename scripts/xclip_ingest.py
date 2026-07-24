"""Ingest the adopted stock video-text encoder: X-CLIP-large clip
embeddings per recording -> `xclip_vectors`. Adoption basis (measured,
balanced sample, identical grading): relational queries 1/40 -> 10/40 vs
the SigLIP appearance incumbent; overall 32 -> 39/140. Pretrained by
Microsoft on generic video-text pairs — nothing of ours; every store,
any upload, day one."""
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

MID = "microsoft/xclip-large-patch14"


def main():
    import torch
    from transformers import AutoProcessor, XCLIPModel
    from elidedb.video import FrameSet
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = XCLIPModel.from_pretrained(MID).to(dev).eval()
    vproc = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")

    store = Store.open(sys.argv[1] if len(sys.argv) > 1 else "lake/bridge4h")
    ep = store.table("episodes").scan()
    recs = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    frames_tbl = store.table("frames").scan()
    rows_s, rows_a, rows_b, vecs = [], [], [], []
    t0 = time.time()
    buf, meta_buf = [], []

    # get_video_features is broken under transformers 5.x (returns a
    # tuple internally); the full forward works — pay one dummy token
    dummy = AutoProcessor.from_pretrained(MID)(
        text=["x"], return_tensors="pt", padding=True)
    dummy = {k: v.to(dev) for k, v in dummy.items()}

    def flush():
        if not buf:
            return
        with torch.no_grad():
            pv = torch.cat([vproc(images=list(v),
                                  return_tensors="pt")["pixel_values"]
                            for v in buf]).to(dev)
            out = model(pixel_values=pv, **dummy)
        V = out.video_embeds.cpu().float().numpy()
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
        for (s, a, b), v in zip(meta_buf, V):
            rows_s.append(s); rows_a.append(a); rows_b.append(b)
            vecs.append(v.astype(np.float32))
        buf.clear(); meta_buf.clear()

    for ri, (s, a, b) in enumerate(recs):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 8:
            continue
        pick = np.linspace(0, len(sel) - 1, 8).round().astype(int)
        dec = FrameSet(store, "frames", sel.take(pick)).decode(width=224)
        if len(dec) < 8:
            continue
        buf.append([d[1] for d in sorted(dec)])
        meta_buf.append((s, a, b))
        if len(buf) >= 4:
            flush()
        if (ri + 1) % 200 == 0:
            print(f"  {ri + 1}/{len(recs)} (t={time.time() - t0:.0f}s)",
                  flush=True)
    flush()
    V = np.stack(vecs)
    dim = V.shape[1]
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(V).reshape(-1)), dim),
    })
    tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
    store.table("xclip_vectors").append(
        tbl, kind="embeddings",
        meta={"model": MID, "dim": dim, "frames_per_clip": 8,
              "text_tower": MID})
    print(json.dumps({"recordings": len(tbl), "dim": dim,
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
