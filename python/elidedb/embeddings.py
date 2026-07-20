"""Semantic layer over Parquet: embeddings and centroids are tables like
everything else — versioned by the log, readable by DuckDB, no sidecar
binary formats. Vectors are fixed_size_list<float32>[d] columns.

Retrieval is two-stage (learned-cell IVF): rank cluster centroids, scan the
top nprobe cells exactly in full space; HDBSCAN noise is always scanned so
the prune can cost recall nothing. All ranking is one numpy matmul — exact,
simple, and fast far past 100k windows."""
from __future__ import annotations

import time

import numpy as np
import pyarrow as pa

_MODEL_CACHE = {}
DEFAULT_MODEL = "mlx-community/siglip-so400m-patch14-384"


def _load_model(model_id):
    if model_id not in _MODEL_CACHE:
        from mlx_embeddings.utils import load
        _MODEL_CACHE[model_id] = load(model_id)
    return _MODEL_CACHE[model_id]


def _embed_images(images, model_id):
    import mlx.core as mx
    model, processor = _load_model(model_id)
    iv = processor(images=images, return_tensors="np")
    out = np.array(model.get_image_features(mx.array(iv["pixel_values"])),
                   dtype=np.float32)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def embed_text(text, model_id=DEFAULT_MODEL):
    import mlx.core as mx
    model, processor = _load_model(model_id)
    ti = processor(text=[text], padding="max_length", max_length=64,
                   truncation=True, return_tensors="np")
    v = np.array(model.get_text_features(mx.array(ti["input_ids"])),
                 dtype=np.float32)[0]
    return v / np.linalg.norm(v)


def _vec_table(store, name="embeddings", version=None):
    t = store.table(name).scan(version=version)
    if len(t) == 0:
        raise RuntimeError(
            f"store '{store.name}' has no embeddings — run "
            "store.embed_windows() first")
    vecs = np.stack([np.asarray(v, dtype=np.float32)
                     for v in t.column("vector").to_pylist()])
    return t, vecs


def embed_windows(store, frame_table="frames", window_s=2.0,
                  frames_per_window=2, model=None, batch=16, stride_s=None,
                  incremental=True):
    """Tumbling windows over every video stream → mean-pooled SigLIP vectors
    → one commit to the `embeddings` table. Frames come through the same
    byte-range path queries use."""
    from PIL import Image  # noqa: F401 (decode happens in FrameSet)
    model = model or DEFAULT_MODEL
    tab = store.table(frame_table)
    st = tab.state()
    win_ns = int(window_s * 1e9)
    stride_ns = int((stride_s or window_s) * 1e9)
    frames = tab.scan()
    streams = sorted(set(frames.column("stream").to_pylist()))
    # Incremental: only embed windows past what the embeddings table already
    # covers per stream — adding a new day of footage costs a new day of
    # embedding, not a re-run of history.
    done_until = {}
    if incremental:
        try:
            prev = store.table("embeddings").scan()
            if len(prev):
                s_arr = prev.column("stream").to_pylist()
                t1_arr = prev.column("t1").to_pylist()
                for s_, e_ in zip(s_arr, t1_arr):
                    done_until[s_] = max(done_until.get(s_, 0), e_)
        except Exception:
            pass
    jobs = []  # (stream, t0, t1)
    import pyarrow.compute as pc
    for s in streams:
        rows = frames.filter(pc.equal(frames.column("stream"), s))
        ts = rows.column("ts").to_numpy()
        t = (int(ts[0]) // win_ns) * win_ns
        while t <= ts[-1]:
            lo, hi = np.searchsorted(ts, [t, t + win_ns])
            if hi > lo and t >= done_until.get(s, -1):
                jobs.append((s, max(t, int(ts[0])),
                             min(t + win_ns - 1, int(ts[-1]))))
            t += stride_ns
    from .video import FrameSet
    t_start = time.time()
    if not jobs:
        return {"windows": 0, "dim": None, "version": None, "seconds": 0.0,
                "note": "nothing new to embed (incremental)"}
    recs = {"ts": [], "t1": [], "stream": [], "vector": []}
    imgs, owners = [], []

    def flush():
        nonlocal imgs, owners
        if not imgs:
            return
        vecs = _embed_images(imgs, model)
        for (key, v) in zip(owners, vecs):
            pooled.setdefault(key, []).append(v)
        imgs, owners = [], []

    pooled = {}
    for (s, t0, t1) in jobs:
        fs = FrameSet(store, frame_table,
                      frames.filter(pc.and_(
                          pc.equal(frames.column("stream"), s),
                          pc.and_(pc.greater_equal(frames.column("ts"), t0),
                                  pc.less_equal(frames.column("ts"), t1)))))
        n = len(fs)
        picks = np.linspace(0, n - 1, min(frames_per_window, n)).round().astype(int)
        decoded = fs.decode(width=512)
        for p in picks:
            if p < len(decoded):
                from PIL import Image as PILImage
                imgs.append(PILImage.fromarray(decoded[p][1]))
                owners.append((s, t0, t1))
        if len(imgs) >= batch:
            flush()
    flush()
    for (s, t0, t1), vs in pooled.items():
        v = np.mean(vs, axis=0)
        v /= np.linalg.norm(v)
        recs["stream"].append(s)
        recs["ts"].append(t0)
        recs["t1"].append(t1)
        recs["vector"].append(v)
    dim = len(recs["vector"][0])
    t = pa.table({
        "ts": pa.array(recs["ts"], pa.int64()),
        "t1": pa.array(recs["t1"], pa.int64()),
        "stream": pa.array(recs["stream"]),
        "vector": pa.array([v.tolist() for v in recs["vector"]],
                           pa.list_(pa.float32(), dim)),
    })
    version = store.table("embeddings").append(
        t, kind="embeddings",
        meta={"model": model, "dim": dim, "window_s": window_s,
              "source_table": frame_table,
              "embedded_in_s": round(time.time() - t_start, 1)})
    return {"windows": len(t), "dim": dim, "version": version,
            "seconds": round(time.time() - t_start, 1)}


def cluster(store, pca_dims=50, min_cluster_size=8):
    """PCA → HDBSCAN over the embeddings table → cluster ids written back as
    a new embeddings version + a `centroids` table (full-space, normalized:
    the coarse stage must rank in the space the fine stage scores in)."""
    t, vecs = _vec_table(store)
    from sklearn.decomposition import PCA
    import hdbscan
    red = PCA(n_components=min(pca_dims, len(vecs), vecs.shape[1]),
              random_state=0).fit_transform(vecs)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(red)
    out = t.drop_columns(["cluster"]) if "cluster" in t.column_names else t
    out = out.append_column("cluster", pa.array(labels.astype("int32")))
    # replace = remove old files + add the re-clustered ones, one commit
    st = store.table("embeddings").state()
    log = store.table("embeddings").log
    import pyarrow.parquet as pq
    import uuid as _uuid
    fname = f"part-{_uuid.uuid4().hex[:12]}.parquet"
    from .store import write_parquet
    write_parquet(out, store.dir / "tables" / "embeddings" / fname)
    from .log import FileEntry
    p = store.dir / "tables" / "embeddings" / fname
    tsv = out.column("ts").to_numpy()
    log.commit(op="recluster", kind="embeddings", schema=str(out.schema),
               add=[FileEntry(fname, len(out), p.stat().st_size,
                              int(tsv.min()), int(tsv.max()))],
               remove=[f.path for f in st.files],
               meta={**st.meta, "clusters": int(labels.max() + 1),
                     "noise": int((labels < 0).sum())})
    cents = []
    for c in range(labels.max() + 1):
        m = vecs[labels == c].mean(axis=0)
        cents.append(m / np.linalg.norm(m))
    if cents:
        ct = pa.table({
            "ts": pa.array([0] * len(cents), pa.int64()),
            "cluster": pa.array(range(len(cents)), pa.int32()),
            "vector": pa.array([c.tolist() for c in cents],
                               pa.list_(pa.float32(), vecs.shape[1])),
        })
        cst = store.table("centroids").state()
        store.table("centroids").log.commit(
            op="replace", kind="centroids", schema=str(ct.schema),
            add=[], remove=[f.path for f in cst.files])
        store.table("centroids").append(ct, kind="centroids")
    return {"clusters": int(labels.max() + 1),
            "noise": int((labels < 0).sum()), "windows": len(out)}


def _rank(store, q, k, nprobe, merge=True, t0=None, t1=None, streams=None,
          method="auto"):
    t, vecs = _vec_table(store)
    all_t0 = t.column("ts").to_numpy()
    all_t1 = t.column("t1").to_numpy()
    all_s = t.column("stream").to_numpy(zero_copy_only=False)

    # ---- hybrid retrieval: predicates pushed INTO candidate selection ------
    # Time and stream are first-class dimensions of this database; vector
    # search composes with them instead of post-filtering a global top-k
    # (which silently starves filtered queries of results).
    pred = np.ones(len(vecs), bool)
    if t0 is not None:
        pred &= all_t1 >= t0
    if t1 is not None:
        pred &= all_t0 <= t1
    if streams:
        pred &= np.isin(all_s, list(streams))

    labels = (t.column("cluster").to_numpy()
              if "cluster" in t.column_names else None)
    probed = total_clusters = 0
    used = "exact"

    idx = scores = None
    if method in ("auto", "hnsw"):
        from . import ann
        hx = ann.load_hnsw(store)
        if hx is not None:
            # overfetch beyond k so predicate filtering and segment merging
            # still see the event's neighborhood, then score exactly
            fetch = int(min(len(vecs), max(k * 8, 64)))
            hx.set_ef(max(fetch, 64))
            cand, _ = hx.knn_query(q, k=fetch)
            cand = cand[0]
            cand = cand[pred[cand]]
            if len(cand) >= min(k, pred.sum()):
                idx = np.asarray(cand)
                scores = vecs[idx] @ q
                used = "hnsw"
    if idx is None and method in ("auto", "ivfpq"):
        from . import ann
        r = ann.search_ivfpq(store, q, k=max(k * 4, 32), nprobe=max(nprobe, 8),
                             mask=pred) if method == "ivfpq" else None
        if r is not None and r[0]:
            idx = np.array([i for i, _ in r[0]])
            scores = np.array([s for _, s in r[0]])
            used = "ivfpq"
    if idx is None:
        mask = pred.copy()
        if labels is not None and nprobe > 0:
            try:
                _, cents = _vec_table(store, "centroids")
                total_clusters = len(cents)
                order = np.argsort(cents @ q)[::-1][:nprobe]
                probed = len(order)
                mask &= np.isin(labels, order) | (labels < 0)  # noise stays
                used = "ivf"
            except RuntimeError:
                pass
        idx = np.where(mask)[0]
        scores = vecs[idx] @ q
    scanned = len(idx)
    streams_sel = all_s[idx]
    w_t0 = all_t0[idx]
    w_t1 = all_t1[idx]

    stats = {"scanned": scanned, "total": len(vecs), "method": used,
             "clusters_probed": probed, "clusters_total": total_clusters,
             "predicate_candidates": int(pred.sum())}

    if not merge:
        order = np.argsort(scores)[::-1][:k]
        hits = [{"stream": str(streams_sel[i]), "t0": int(w_t0[i]),
                 "t1": int(w_t1[i]), "score": float(scores[i]),
                 "windows": 1} for i in order]
        return hits, stats

    # ---- dynamic segments: merge, don't chunk -------------------------------
    # Fixed embedding windows are an INDEXING granularity, not an answer
    # granularity. A result is the maximal run of consecutive qualifying
    # windows on one stream: a 20 s event comes back as ONE 20 s hit (its
    # sub-windows are never returned separately), while a query that only
    # matches 2 s of it comes back as that tight 2 s. "Qualifying" is decided
    # per query from the score distribution — an absolute cutoff cannot work
    # because SigLIP cosines live on different scales per query.
    med = float(np.median(scores))
    top = float(scores.max())
    thr = med + 0.55 * (top - med)
    stats["threshold"] = round(thr, 4)
    qual = np.where(scores >= thr)[0]
    order = np.lexsort((w_t0[qual], streams_sel[qual]))
    qual = qual[order]

    gap_ns = int(np.median(w_t1[qual] - w_t0[qual])) + 1 if len(qual) else 0
    segs = []
    for i in qual:
        s, a, b, sc = (str(streams_sel[i]), int(w_t0[i]), int(w_t1[i]),
                       float(scores[i]))
        last = segs[-1] if segs else None
        if last and last["stream"] == s and a - last["t1"] <= gap_ns:
            last["t1"] = max(last["t1"], b)
            last["score"] = max(last["score"], sc)   # peak represents the segment
            last["mean"] = (last["mean"] * last["windows"] + sc) / (last["windows"] + 1)
            last["windows"] += 1
        else:
            segs.append({"stream": s, "t0": a, "t1": b, "score": sc,
                         "mean": sc, "windows": 1})
    segs.sort(key=lambda g: -g["score"])
    stats["qualifying_windows"] = len(qual)
    stats["segments"] = len(segs)
    return segs[:k], stats


def search_text(store, text, k=10, nprobe=3, merge=True, t0=None, t1=None,
                streams=None, method="auto"):
    st = store.table("embeddings").state()
    q = embed_text(text, st.meta.get("model", DEFAULT_MODEL))
    return _rank(store, q, k, nprobe, merge=merge, t0=t0, t1=t1,
                 streams=streams, method=method)


def search_clip(store, stream, t0, t1, k=10, nprobe=3, merge=True,
                pt0=None, pt1=None, pstreams=None, method="auto"):
    t, vecs = _vec_table(store)
    s = t.column("stream").to_numpy(zero_copy_only=False)
    a = t.column("ts").to_numpy()
    b = t.column("t1").to_numpy()
    sel = (s == stream) & (a <= t1) & (b >= t0)
    if not sel.any():
        raise ValueError(f"no embedded windows overlap {stream} [{t0},{t1}]")
    q = vecs[sel].mean(axis=0)
    q /= np.linalg.norm(q)
    hits, stats = _rank(store, q, k + 8, nprobe, merge=merge,
                        t0=pt0, t1=pt1, streams=pstreams, method=method)
    hits = [h for h in hits
            if not (h["stream"] == stream and h["t0"] <= t1 and h["t1"] >= t0)]
    return hits[:k], stats
