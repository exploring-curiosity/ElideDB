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
                  frames_per_window=2, model=None, batch=16):
    """Tumbling windows over every video stream → mean-pooled SigLIP vectors
    → one commit to the `embeddings` table. Frames come through the same
    byte-range path queries use."""
    from PIL import Image  # noqa: F401 (decode happens in FrameSet)
    model = model or DEFAULT_MODEL
    tab = store.table(frame_table)
    st = tab.state()
    win_ns = int(window_s * 1e9)
    frames = tab.scan()
    streams = sorted(set(frames.column("stream").to_pylist()))
    jobs = []  # (stream, t0, t1)
    import pyarrow.compute as pc
    for s in streams:
        rows = frames.filter(pc.equal(frames.column("stream"), s))
        ts = rows.column("ts").to_numpy()
        t = (int(ts[0]) // win_ns) * win_ns
        while t <= ts[-1]:
            lo, hi = np.searchsorted(ts, [t, t + win_ns])
            if hi > lo:
                jobs.append((s, max(t, int(ts[0])),
                             min(t + win_ns - 1, int(ts[-1]))))
            t += win_ns
    from .video import FrameSet
    t_start = time.time()
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
    pq.write_table(out, store.dir / "tables" / "embeddings" / fname,
                   compression="zstd")
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


def _rank(store, q, k, nprobe):
    t, vecs = _vec_table(store)
    labels = (t.column("cluster").to_numpy()
              if "cluster" in t.column_names else None)
    scanned = len(vecs)
    mask = np.ones(len(vecs), bool)
    probed = total_clusters = 0
    if labels is not None and nprobe > 0:
        try:
            _, cents = _vec_table(store, "centroids")
            total_clusters = len(cents)
            order = np.argsort(cents @ q)[::-1][:nprobe]
            probed = len(order)
            mask = np.isin(labels, order) | (labels < 0)  # noise always scanned
            scanned = int(mask.sum())
        except RuntimeError:
            pass
    scores = vecs[mask] @ q
    rows = np.where(mask)[0][np.argsort(scores)[::-1][:k]]
    hits = [{"stream": t.column("stream")[int(i)].as_py(),
             "t0": t.column("ts")[int(i)].as_py(),
             "t1": t.column("t1")[int(i)].as_py(),
             "score": float(vecs[int(i)] @ q)} for i in rows]
    return hits, {"scanned": scanned, "total": len(vecs),
                  "clusters_probed": probed, "clusters_total": total_clusters}


def search_text(store, text, k=10, nprobe=3):
    st = store.table("embeddings").state()
    q = embed_text(text, st.meta.get("model", DEFAULT_MODEL))
    return _rank(store, q, k, nprobe)


def search_clip(store, stream, t0, t1, k=10, nprobe=3):
    t, vecs = _vec_table(store)
    s = t.column("stream").to_numpy(zero_copy_only=False)
    a = t.column("ts").to_numpy()
    b = t.column("t1").to_numpy()
    sel = (s == stream) & (a <= t1) & (b >= t0)
    if not sel.any():
        raise ValueError(f"no embedded windows overlap {stream} [{t0},{t1}]")
    q = vecs[sel].mean(axis=0)
    q /= np.linalg.norm(q)
    hits, stats = _rank(store, q, k + 8, nprobe)
    hits = [h for h in hits
            if not (h["stream"] == stream and h["t0"] <= t1 and h["t1"] >= t0)]
    return hits[:k], stats
