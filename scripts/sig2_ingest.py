"""Ingest SigLIP 2 (google/siglip2-so400m-patch14-384) frame vectors:
8 frames per episode -> sig2_vectors. Successor to SigLIP 1 with
improved fine-grained understanding and localization (arXiv
2502.14786) — serves as a content channel and as the frame space for
CONJUNCTIVE atom scoring (every query noun phrase must find its own
frame match). Pretrained weights only; any corpus."""
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

MID = "google/siglip2-so400m-patch14-384"


def main():
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    from elidedb.video import FrameSet
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(MID,
                                      dtype=torch.float16).to(dev).eval()

    store = Store.open(sys.argv[1] if len(sys.argv) > 1
                       else "lake/bench")
    ep = store.table("episodes").scan()
    recs = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    frames_tbl = store.table("frames").scan()
    rows_s, rows_a, rows_b, vecs = [], [], [], []
    t0 = time.time()
    # PER-ITERATION progress. The old "every 200th recording"
    # print moved a bar three times over a 600-item run, which
    # tells you nothing about whether it is alive between them.
    from tqdm import tqdm
    # tqdm wraps the ITERABLE, so the count advances when an iteration
    # COMPLETES. Updating at the top of the body instead reports work
    # that has not happened yet.
    _bar = tqdm(recs, desc="sig2", unit="rec", dynamic_ncols=True,
                mininterval=0.3)
    for ri, (s, a, b) in enumerate(_bar):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 8:
            continue
        pick = np.linspace(0, len(sel) - 1, 8).round().astype(int)
        try:
            dec = FrameSet(store, "frames",
                           sel.take(pick)).decode(width=384)
        except Exception:
            continue        # stream-boundary episode in filtered store
        if len(dec) < 8:
            continue
        imgs = [Image.fromarray(d[1]) for d in sorted(dec)]
        with torch.no_grad():
            px = proc(images=imgs, return_tensors="pt")[
                "pixel_values"].to(dev, torch.float16)
            f = model.get_image_features(pixel_values=px)
        f = (f / f.norm(dim=-1, keepdim=True)).cpu().float().numpy()
        for v in f:
            rows_s.append(s); rows_a.append(a); rows_b.append(b)
            vecs.append(v.astype(np.float32))
    _bar.close()
    V = np.stack(vecs)
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(V).reshape(-1)), V.shape[1]),
    })
    tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
    store.table("sig2_vectors").append(
        tbl, kind="embeddings",
        meta={"model": MID, "dim": int(V.shape[1]),
              "frames_per_episode": 8})
    print(json.dumps({"rows": len(tbl), "dim": int(V.shape[1]),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
