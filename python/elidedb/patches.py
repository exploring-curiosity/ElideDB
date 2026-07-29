"""LATE INTERACTION channel: MaxSim over stored patch sign-codes.

Built by scripts/patch_ingest.py, which carries the measurement that
justifies it. Here is only the scoring, and it has one trick worth
reading.

The codes are one bit per PCA dimension, so a patch is a vector of
+-1 and its score against a query q is sum_d sign_d * q_d. Unpacking
1.15M patches x 256 bits into floats to run that as a matmul costs
294 MB of RAM and throws away the reason for storing bits. Instead,
for each of the 32 code BYTES, precompute the partial sum of the 8 q
dimensions it covers, for all 256 possible byte values: a (32, 256)
table. Scoring is then one gather per byte - 32 lookups per patch, no
unpacking, and the codes stay exactly as they sit on disk. Building
the table costs 32 x 256 adds, once per query.

Query side stays full precision (asymmetric quantization): there is
one query and a million patches, so precision is free where it is
scarce and paid where it is cheap.
"""
from __future__ import annotations

import numpy as np

_PIDX: dict = {}


def _load(store):
    """(codes uint8 (rows, 32*keep), episode row-groups, mu, R)."""
    ver = store.table("patch_codes").state().version
    key = (str(store.dir), ver)
    if key in _PIDX:
        return _PIDX[key]
    from pathlib import Path
    tbl = store.table("patch_codes").scan()
    meta = store.table("patch_codes").state().meta or {}
    dim = int(meta.get("dim", 256))
    keep = int(meta.get("keep", 256))
    raw = tbl.column("code").to_pylist()
    C = np.frombuffer(b"".join(raw), np.uint8).reshape(
        len(raw), keep, dim // 8)
    idx: dict = {}
    for i, (s, a) in enumerate(zip(tbl.column("stream").to_pylist(),
                                   tbl.column("ts").to_pylist())):
        idx.setdefault((str(s), int(a)), []).append(i)
    idx = {k: np.asarray(v, np.int64) for k, v in idx.items()}
    b = np.load(Path(store.dir) / "_patch_basis.npz")
    if len(_PIDX) > 4:
        _PIDX.clear()
    _PIDX[key] = (C, idx, b["mu"], b["R"], dim)
    return _PIDX[key]


def _tables(q, dim):
    """(dim/8, 256) partial sums of q over every possible code byte."""
    nb = dim // 8
    # bit j of a byte is dimension 8*b + (7 - j) — packbits is MSB first
    bits = ((np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1)
    T = np.empty((nb, 256), np.float32)
    for b in range(nb):
        w = q[b * 8:(b + 1) * 8]
        # code bit 1 means +1 on that dimension, 0 means -1
        T[b] = bits @ w - (1 - bits) @ w
    return T


def patch_lookup(store, texts):
    """lookup(stream, t0, t1) -> MaxSim score for the episode.

    `texts` is a list of query atoms. Each atom takes its best patch
    over every patch of every frame of the episode (MaxSim), then the
    atoms are combined with min — the same soft-AND the other binding
    channels use, so a compound query needs each named thing to be
    somewhere in the episode rather than one strong match to carry it.
    """
    from .sig2 import _text_vec
    C, idx, mu, R, dim = _load(store)
    nb = dim // 8
    Ts = []
    for t in texts:
        v = (np.asarray(_text_vec(t), np.float32) - mu) @ R
        v /= np.linalg.norm(v) + 1e-8
        Ts.append(_tables(v, dim))
    ar = np.arange(nb)

    def lookup(s, a, b):
        rows = idx.get((str(s), int(a)))
        if rows is None or len(rows) == 0:
            return float("nan")
        codes = C[rows].reshape(-1, nb)           # (frames*patches, nb)
        best = None
        for T in Ts:
            sc = T[ar, codes].sum(1).max() / np.sqrt(dim)
            best = sc if best is None else min(best, sc)
        return float(best)
    return lookup
