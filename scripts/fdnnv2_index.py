"""Materialize the V2 context-event index: `context_events` table.

Event-first retrieval (docs/FDNNV2_PLAN.md): the encoder's update gate
already says WHEN the scene model is being rewritten, so events — not
fixed windows — are the retrieval unit. Each row is one event span with a
novelty-pooled 256-d ctx vector; a text query becomes
adapter(embed_text(q)) and ranking is ONE matmul, index-only.

Events are computed PER RECORDING (episodes table = structural ingest
metadata): packed corpora butt recordings together with no time gap, and
both the recurrent state and an event span must never cross a cut — a
"scene change" at a cut is an artifact, not an event.

Frames come from the training cache (data/cache/b4h) — it holds exactly
the store's decoded frames at model resolution, so this pass costs a
model forward (~0.2 ms/frame), not a decode.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                     # noqa: E402
from elidedb.fdnnv2 import load_v2, pool_event, segment_events  # noqa: E402


def recording_spans(store, stream):
    """[(t0, t1)] for one stream, from the episodes table; whole-stream
    fallback when the store has no recording metadata."""
    try:
        ep = store.table("episodes").scan()
        spans = [(int(a), int(b)) for s, a, b in
                 zip(ep.column("stream").to_pylist(),
                     ep.column("ts").to_pylist(),
                     ep.column("t1").to_pylist()) if s == stream]
        if spans:
            return sorted(spans)
    except Exception:
        pass
    return [(-(1 << 62), 1 << 62)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="lake/bridge4h")
    ap.add_argument("--cache", default="data/cache/b4h")
    ap.add_argument("--rebuild", action="store_true",
                    help="atomic replace of the existing index (one commit, "
                         "old files removed, time travel keeps them)")
    args = ap.parse_args()
    t_start = time.time()

    store = Store.open(args.store)
    tab = store.table("context_events")
    try:
        existing = tab.state().files
    except Exception:
        existing = []
    if existing and not args.rebuild:
        print("context_events already exists — pass --rebuild to replace")
        return

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
        for r0, r1 in recording_spans(store, str(s)):
            r = m[(ts[m] >= r0) & (ts[m] <= r1)]
            if len(r) < 4:
                continue
            _, ctx, gates = model.embed_stream_np(px[r])
            n_frames += len(r)
            for lo, hi in segment_events(gates, ts[r]):
                rows_s.append(str(s))
                rows_a.append(int(ts[r[lo]]))
                rows_b.append(int(ts[r[hi - 1]]))
                rows_v.append(pool_event(ctx, gates, lo, hi))
        print(f"  {s}: -> {len(rows_s)} events total", flush=True)

    dim = len(rows_v[0])
    order = np.argsort(np.array(rows_a))
    tbl = pa.table({
        "ts": pa.array([rows_a[i] for i in order], pa.int64()),
        "t1": pa.array([rows_b[i] for i in order], pa.int64()),
        "stream": pa.array([rows_s[i] for i in order]),
        "vector": pa.array([rows_v[i].tolist() for i in order],
                           pa.list_(pa.float32(), dim)),
    })
    kind_meta = {"model": "fdnnv2", "dim": dim, "per_recording": True,
                 "report": meta.get("report", {})}
    if existing:
        from elidedb.log import FileEntry
        from elidedb.store import write_parquet
        fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
        path = tab.dir / fname
        write_parquet(tbl, path)
        tsv = tbl.column("ts")
        version = tab.log.commit(
            op="replace", kind="embeddings", schema=str(tbl.schema),
            add=[FileEntry(fname, len(tbl), path.stat().st_size,
                           tsv[0].as_py(), tsv[-1].as_py())],
            remove=[f.path for f in existing], meta=kind_meta)
    else:
        version = tab.append(tbl, kind="embeddings", meta=kind_meta)
    dt = time.time() - t_start
    print({"events": len(tbl), "frames": n_frames, "dim": dim,
           "version": version, "seconds": round(dt, 1),
           "ms_per_frame": round(dt / max(n_frames, 1) * 1e3, 3)})


if __name__ == "__main__":
    main()
