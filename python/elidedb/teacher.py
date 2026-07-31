"""TEACHER OUTPUT CONTRACT: vectors the storage layer can prune on.

A vector cannot be pruned. Parquet keeps min/max per column chunk, and
min/max over 1024 floats is meaningless - measured directly: a full
vector scan reads 99.2% of the store and elides 0.777%, because there is
no statistic a planner can refuse on. Every other table in this engine
gets three layers of pruning (commit-log zone map, row-group statistics,
page index); the vector tables get none, and they are ~87% of the bytes.

So a teacher writes FOUR things, not one, and three of them are scalars
the footer can hold statistics for:

    vector    fp16, opaque. What ranking actually consumes. Unprunable
              by construction and that is fine, because nothing should
              reach it until the other three have narrowed the field.

    code      int32. The vector's cell in a codebook fitted on the
              corpus. This is the PRUNABLE PROJECTION of the vector:
              cluster the teacher's output space, store which cell each
              row fell in, and CLUSTER THE TABLE ON IT. A query embeds,
              finds its nearest cells, and the commit log drops files
              whose code range excludes them - before a footer is
              opened, let alone a vector decompressed.

              This is IVF, but used for STORAGE LAYOUT rather than only
              for search. The coarse quantiser has always been in this
              design; what is new is that its cell id is a sorted column
              so the row-group statistics enforce it.

    energy    float32. The vector's norm before normalisation, i.e. how
              much the encoder had to say about this interval. A range
              predicate on it ("only intervals the teacher was confident
              about") prunes without touching the vector.

    margin    float32. Distance to the SECOND nearest cell, minus the
              first. Low margin means the row sits on a cell boundary
              and a code-only prune would be unsafe for it. Storing it
              lets a reader widen to neighbouring cells exactly where it
              must, instead of always or never.

FILE-LEVEL METADATA carries the codebook itself, in the Parquet footer's
key-value map. A reader can therefore map a query into cell ids from the
footer alone - kilobytes - without scanning a codes column, and without
a side file that can drift out of sync with the data it describes.

INTERVALS, not points. ts..t1 is the interval the teacher describes.
Raw capture is continuous; episodes are something the engine produces.
The store's overlap predicate and max_end zone map already handle this
correctly, and a teacher that emits point rows silently loses the
"was it there" question.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

# Rows per cell, targeted. The cell COUNT is derived from corpus size
# rather than fixed: a 500-row channel and a 5-million-row channel do
# not want the same number of cells, and picking one constant for both
# is how a coarse quantiser stops being coarse.
ROWS_PER_CELL = 256
MIN_CELLS, MAX_CELLS = 4, 4096


def fit_codebook(V, rows_per_cell=ROWS_PER_CELL, seed=0):
    """Cluster a teacher's output space into cells. Returns centroids.

    k-means, not HDBSCAN: this is a QUANTISER, so every row must land
    somewhere. Discovery elsewhere in the engine leaves outliers
    unlabelled on purpose; a codebook that refuses to place a row would
    make that row unprunable and therefore unreachable.
    """
    V = np.asarray(V, np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    k = int(np.clip(len(V) // rows_per_cell, MIN_CELLS, MAX_CELLS))
    k = min(k, len(V))
    if k < 2:
        return V[:1].copy()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(V)
    C = km.cluster_centers_.astype(np.float32)
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)


def assign(V, C):
    """Rows -> (code, margin). Margin is the gap to the runner-up cell:
    small means the row is near a boundary and a code-only prune would
    need its neighbour too."""
    V = np.asarray(V, np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    S = V @ np.asarray(C, np.float32).T
    if S.shape[1] == 1:
        return np.zeros(len(V), np.int32), np.ones(len(V), np.float32)
    idx = np.argsort(-S, axis=1)[:, :2]
    top = S[np.arange(len(S)), idx[:, 0]]
    second = S[np.arange(len(S)), idx[:, 1]]
    return idx[:, 0].astype(np.int32), (top - second).astype(np.float32)


def probe(query_vec, C, n_cells=None, min_frac=0.05):
    """Query vector -> the cell ids worth reading.

    Returns the nearest cells whose similarity is within `min_frac` of
    the best. Not a fixed nprobe: how many cells a query needs depends
    on how sharply it lands, and a constant would over-read confident
    queries while under-reading ambiguous ones.
    """
    q = np.asarray(query_vec, np.float32).ravel()
    q = q / (np.linalg.norm(q) + 1e-8)
    s = np.asarray(C, np.float32) @ q
    order = np.argsort(-s)
    keep = [int(order[0])]
    for i in order[1:]:
        if s[i] >= s[order[0]] - min_frac:
            keep.append(int(i))
        else:
            break
    return keep[:n_cells] if n_cells else keep


def table(ts, t1, stream, V, C=None, extra=None):
    """Build a teacher channel's Arrow table in the contract's shape.

    The column ORDER matters for readability of the footer, not for
    correctness: ts/t1 first because they are the interval, then the
    scalars a planner reads, then the vector nothing should reach until
    the scalars have done their work.
    """
    V = np.asarray(V, np.float32)
    energy = np.linalg.norm(V, axis=1).astype(np.float32)
    if C is None:
        C = fit_codebook(V)
    code, margin = assign(V, C)
    Vn = (V / (energy[:, None] + 1e-8)).astype(np.float16)
    cols = {
        "ts": pa.array(np.asarray(ts, np.int64), pa.int64()),
        "t1": pa.array(np.asarray(t1, np.int64), pa.int64()),
        "stream": pa.array(list(stream)),
        "code": pa.array(code, pa.int32()),
        "energy": pa.array(energy, pa.float32()),
        "margin": pa.array(margin, pa.float32()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(Vn).reshape(-1), pa.float16()),
            V.shape[1]),
    }
    for k, v in (extra or {}).items():
        cols[k] = v
    return pa.table(cols), C


# MEASURED knee, 20,515 rows / 80 cells. Average bytes a one-cell probe
# touches, and the footer's share of the file:
#     32 rows/group  80 groups  footer 0.2%   860,702 B   2.0% of full
#     64             78         0.2%          862,305     2.0%   <- here
#    128             68         0.2%          918,615     2.1%
#    256             47         0.1%        1,271,317     2.9%
#    512             28         0.1%        1,685,366     3.8%
# Finer than 64 buys nothing because a cell is already its own group;
# coarser merges cells and a probe drags in rows it cannot use. Recall
# is 0.918 throughout - group size moves BYTES, never the answer.
CODE_GROUP_ROWS = 64


def write(store, name, tbl, C, meta=None, min_group_rows=CODE_GROUP_ROWS):
    """Commit a teacher channel, CLUSTERED ON ITS CODE.

    Clustering on `code` is what turns the codebook from a search trick
    into a storage one: rows sharing a cell become physically adjacent,
    so their row-group min/max is narrow and a probe for a few cells
    reads a few row groups. Sorted second by ts so a time window still
    prunes within a cell.

    The codebook rides in the table's own metadata. A side file would
    drift; the footer cannot.
    """
    t = store.table(name)
    t.set_layout("code", sort_by=["code", "ts"],
                 min_group_rows=min_group_rows)
    return t.append(tbl, kind="embeddings", meta=dict(
        meta or {}, codebook=json.dumps(np.asarray(C, np.float32)
                                        .round(4).tolist()),
        cells=int(len(C)), dim=int(tbl.column("vector").type.list_size)))


def codebook(store, name):
    """Read a channel's codebook back out of its metadata."""
    m = store.table(name).state().meta
    cb = m.get("codebook")
    return np.asarray(json.loads(cb), np.float32) if cb else None
