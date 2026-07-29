"""TRACK channel: what the moving thing IS, at native resolution.

The store's earlier region channel (`objects.py` / object_vectors) was
built from crops cut out of frames that had already been downscaled for
decode, so a table-top object reached SigLIP as roughly 30 px and the
channel measured nothing — best cosine to "an eggplant" across 51,101
crops was 0.184, and the two episodes containing one ranked 75th and
110th corpus-wide.

Rebuilt at NATIVE resolution over Gestalt common-fate tracks
(scripts/track_ingest.py), the same idea separates on a pool of 80:

    query            shipped yield@10   track channel alone
    q10 banana            0.00               1.00
    q09 eggplant          0.00               1.00
    q00 green object      0.20               0.75
    q08 spoon             0.10               0.50

Two measured design facts, both against my prior:
  * views are kept SEPARATE, not pooled. Pooling along the track (the
    Treisman/Kahneman object-file prediction) lost on both queries it
    was tried on — the box drifts, and the mean blends object with
    background. Max-over-views wins.
  * per-atom min (soft AND) across query atoms, matching objects.py:
    a compound query scored whole against one small crop is a
    bag-of-concepts mismatch (Winoground, ARO).

Rows are keyed by the EPISODE span, so lookup is an exact (stream, ts)
hit — no run-walking as object_vectors needs.
"""
from __future__ import annotations

import numpy as np

_TRK_IDX: dict = {}


def track_lookup(store, qv):
    """lookup(stream, t0, t1) -> best crop cosine for that episode.

    `qv` may be a matrix of atom embeddings: each atom takes its best
    crop (max over views and tracks), then the atoms are combined with
    min — every named thing must actually be present."""
    from .embeddings import _vec_table
    ver = store.table("track_vectors").state().version
    key = (str(store.dir), ver)
    if key not in _TRK_IDX:
        tbl, _ = _vec_table(store, "track_vectors")
        ss = np.asarray(tbl.column("stream").to_pylist())
        sa = np.asarray([int(v) for v in tbl.column("ts").to_pylist()])
        idx: dict = {}
        for i, (s, a) in enumerate(zip(ss, sa)):
            idx.setdefault((s, int(a)), []).append(i)
        idx = {k: np.asarray(v, np.int64) for k, v in idx.items()}
        if len(_TRK_IDX) > 8:
            _TRK_IDX.clear()
        _TRK_IDX[key] = idx
    idx = _TRK_IDX[key]
    _, vecs = _vec_table(store, "track_vectors")
    Q = np.atleast_2d(np.asarray(qv, np.float32))
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    S = np.asarray(vecs, np.float32) @ Q.T          # (crops, atoms)

    def lookup(s, a, b):
        rows = idx.get((s, int(a)))
        if rows is None or len(rows) == 0:
            return float("nan")
        return float(S[rows].max(axis=0).min())
    return lookup
