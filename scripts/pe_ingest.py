"""Ingest Meta Perception Encoder (PE-Core-L) — the published SOTA
zero-shot video-text encoder — as a stock channel: 8 frame vectors per
recording -> `pe_vectors`. Adoption basis (balanced sample, identical
grading): best single-encoder precision@10 ever measured on this bench
(48/140 vs shipping fusion 40), with complementary wins (put-into-drawer
10/10, take-out 6/10, towel 4/10). Naive equal fusion measured WORSE
than PE alone — routing happens through the learned per-query-type
weights instead. Pretrained weights, nothing of ours, any upload."""
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


def main():
    import torch
    import open_clip
    from PIL import Image
    from elidedb.video import FrameSet
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(
        "PE-Core-L-14-336", pretrained="meta")
    model = model.to(dev).eval()

    store = Store.open(sys.argv[1] if len(sys.argv) > 1
                       else "lake/bridge4h")
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
    _bar = tqdm(recs, desc="pe", unit="rec", dynamic_ncols=True,
                mininterval=0.3)
    for ri, (s, a, b) in enumerate(_bar):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 8:
            continue
        pick = np.linspace(0, len(sel) - 1, 8).round().astype(int)
        dec = FrameSet(store, "frames", sel.take(pick)).decode(width=336)
        if len(dec) < 8:
            continue
        imgs = torch.stack([pre(Image.fromarray(d[1]))
                            for d in sorted(dec)]).to(dev)
        with torch.no_grad():
            f = model.encode_image(imgs)
        f = (f / f.norm(dim=-1, keepdim=True)).cpu().float().numpy()
        for v in f:
            rows_s.append(s); rows_a.append(a); rows_b.append(b)
            vecs.append(v.astype(np.float32))
    _bar.close()
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
    store.table("pe_vectors").append(
        tbl, kind="embeddings",
        meta={"model": "PE-Core-L-14-336", "dim": dim,
              "frames_per_recording": 8, "pool": "top5-mean at query"})
    print(json.dumps({"rows": len(tbl), "dim": dim,
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
