"""The motion vector channel: motion = temporal DIFFERENCE of appearance.

The Marengo lesson is that motion must be its own vector, never averaged
into appearance. The obvious implementations were measured and lost:
  X-CLIP (video-native, cross-frame attention)    close/open AUC 0.67
  VLM swap-contrast (query-time, 2 model calls)   0.86 (2B) / 0.91 (7B)
  ctx adapter (caption-trained)                   ~chance
  THIS: delta-appearance + query-swap difference  0.98 / 0.98

An event's motion vector is normalize(app(end) - app(start)) in SigLIP
space, from the per-frame vectors ALREADY on disk — ingest cost is a
numpy mean. A directional query's motion vector is
normalize(embed(q) - embed(swap(q))) — the same swap-contrast idea, moved
from VLM logits into embedding space, so it runs index-only in
microseconds. Both sides subtract their shared appearance component, so
what remains on each side is the DIRECTION of state change, and they
were trained (by SigLIP, incidentally) to point the same way.

Limits, measured: works when the state change is visually large
(drawer open<->close: 0.98); chance when a small object moves
(put-in/take-out: 0.46-0.50) — those queries are carried by the
appearance and caption-lexical channels instead. Non-directional queries
have no swap and the channel ABSTAINS (RRF median rank, no veto).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa


def index_motion(store, events_table="context_events", edge_s=1.5,
                 frames_table="frame_vectors"):
    """One motion vector per event span -> `motion_vectors` table.

    Event spans come from the gate-segmented event index so every retrieval
    unit carries both an appearance-family vector and a motion vector —
    the multi-vector layout, aligned. Rebuild = atomic replace commit.
    """
    import time
    import uuid

    from .embeddings import _vec_table
    from .log import FileEntry
    from .store import write_parquet
    t_start = time.time()

    ev = store.table(events_table).scan()
    fv, fvecs = _vec_table(store, frames_table)
    fs = np.asarray(fv.column("stream").to_pylist())
    ft = np.asarray([int(v) for v in fv.column("ts").to_pylist()])
    order = np.argsort(ft, kind="stable")
    fs, ft, fvecs = fs[order], ft[order], fvecs[order]
    edge = int(edge_s * 1e9)

    rows_s, rows_a, rows_b, rows_v = [], [], [], []
    for s, a, b in zip(ev.column("stream").to_pylist(),
                       (int(v) for v in ev.column("ts").to_pylist()),
                       (int(v) for v in ev.column("t1").to_pylist())):
        m = (fs == s) & (ft >= a) & (ft <= b)
        if m.sum() < 6:
            continue
        tt, vv = ft[m], fvecs[m]
        v0 = vv[tt <= tt[0] + edge].mean(0)
        v1 = vv[tt >= tt[-1] - edge].mean(0)
        d = v1 - v0
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            continue
        rows_s.append(s)
        rows_a.append(a)
        rows_b.append(b)
        rows_v.append((d / n).astype(np.float32))
    if not rows_v:
        return {"events": 0}

    dim = len(rows_v[0])
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.array([v.tolist() for v in rows_v],
                           pa.list_(pa.float32(), dim)),
    })
    tab = store.table("motion_vectors")
    meta = {"model": "delta-appearance", "dim": dim, "edge_s": edge_s,
            "events_table": events_table}
    try:
        existing = tab.state().files
    except Exception:
        existing = []
    if existing:
        fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
        path = tab.dir / fname
        write_parquet(tbl, path)
        tsv = tbl.column("ts")
        version = tab.log.commit(
            op="replace", kind="embeddings", schema=str(tbl.schema),
            add=[FileEntry(fname, len(tbl), path.stat().st_size,
                           tsv[0].as_py(), tsv[-1].as_py())],
            remove=[f.path for f in existing], meta=meta)
    else:
        version = tab.append(tbl, kind="embeddings", meta=meta)
    return {"events": len(tbl), "dim": dim, "version": version,
            "seconds": round(time.time() - t_start, 2)}


def motion_scores(store, qv, qv_swap):
    """(stream, t0, t1, score) per event: motion vectors against the
    query-direction vector. Pure matmul over a version-cached matrix."""
    from .embeddings import _vec_table
    tbl, vecs = _vec_table(store, "motion_vectors")
    qd = np.asarray(qv, np.float32) - np.asarray(qv_swap, np.float32)
    n = float(np.linalg.norm(qd))
    if n < 1e-6:
        return []
    sc = vecs @ (qd / n)
    return list(zip(tbl.column("stream").to_pylist(),
                    (int(v) for v in tbl.column("ts").to_pylist()),
                    (int(v) for v in tbl.column("t1").to_pylist()),
                    sc.tolist()))
