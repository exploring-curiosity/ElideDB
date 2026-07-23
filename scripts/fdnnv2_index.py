"""Materialize the V2 context-event index: `context_events` table.

Event-first retrieval (docs/FDNNV2_PLAN.md): the encoder's update gate
already says WHEN the scene model is being rewritten, so events — not
fixed windows — are the retrieval unit. Each row is one event span with a
novelty-pooled 256-d ctx vector; a text query becomes
adapter(embed_text(q)) and ranking is ONE matmul, index-only.

Frames come from the training cache (data/cache/b4h) — it holds exactly
the store's decoded frames at model resolution, so this pass costs a
model forward (~0.2 ms/frame), not a decode.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                     # noqa: E402
from elidedb.fdnnv2 import load_v2, pool_event, segment_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="lake/bridge4h")
    ap.add_argument("--cache", default="data/cache/b4h")
    args = ap.parse_args()
    t_start = time.time()

    store = Store.open(args.store)
    try:
        if store.table("context_events").state().files:
            print("context_events already exists — skipping (remove the "
                  "table to rebuild)")
            return
    except Exception:
        pass

    model, _, meta = load_v2(Path(args.store) / "models" / "fdnnv2")
    cache = Path(args.cache)
    px = np.load(cache / "px.npy", mmap_mode="r")
    sid = np.load(cache / "sid.npy")
    ts = np.load(cache / "ts.npy")
    streams = list(np.load(cache / "streams.npy"))

    rows_s, rows_a, rows_b, rows_v = [], [], [], []
    n_frames = 0
    for si, s in enumerate(streams):
        m = np.where(sid == si)[0]
        m = m[np.argsort(ts[m])]
        if len(m) < 4:
            continue
        _, ctx, gates = model.embed_stream_np(px[m])
        n_frames += len(m)
        for lo, hi in segment_events(gates, ts[m]):
            rows_s.append(str(s))
            rows_a.append(int(ts[m[lo]]))
            rows_b.append(int(ts[m[hi - 1]]))
            rows_v.append(pool_event(ctx, gates, lo, hi))
        print(f"  {s}: {len(m)} frames -> {len(rows_s)} events total",
              flush=True)

    dim = len(rows_v[0])
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.array([v.tolist() for v in rows_v],
                           pa.list_(pa.float32(), dim)),
    })
    version = store.table("context_events").append(
        tbl, kind="embeddings",
        meta={"model": "fdnnv2", "dim": dim,
              "report": meta.get("report", {})})
    dt = time.time() - t_start
    print({"events": len(tbl), "frames": n_frames, "dim": dim,
           "version": version, "seconds": round(dt, 1),
           "ms_per_frame": round(dt / max(n_frames, 1) * 1e3, 3)})


if __name__ == "__main__":
    main()
