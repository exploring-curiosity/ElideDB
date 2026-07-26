"""SELF-RECOGNIZED action contrast — pseudo-relevance feedback in the
video-native feature space. No class names, no verb maps, no dataset
priors in code (user rule: priors must be recognized from the corpus,
not typed — a self-driving corpus must work with this exact file).

Mechanism (Rocchio, 1971 — the oldest trick in retrieval, applied
cross-modally): the TEXT channels' top-K for the query are weak
positives; the swap-query's top-K are weak negatives; the action
contrast IS the difference of their mean V-JEPA vectors. V-JEPA
features are self-supervised and domain-general — the same anchors
form for "closing the drawer"/"opening the drawer" on bridge and for
"merging left"/"merging right" on driving footage. The corpus itself
defines what the direction looks like here.

Also here: per-query CHANNEL INFORMATIVENESS — a channel that ranks
episodes identically for the query and its swap carries no direction
information for THIS query (measured, not declared)."""
from __future__ import annotations

import numpy as np

_VJ = {}


def _vjepa_matrix(store, keys):
    """Episode-aligned V-JEPA vectors (normalized), cached per store
    version. NaN rows for episodes without a vector."""
    from .embeddings import _vec_table
    ver = store.table("vjepa_vectors").state().version
    ck = (str(store.dir), ver, len(keys))
    if ck in _VJ:
        return _VJ[ck]
    tbl, vecs = _vec_table(store, "vjepa_vectors")
    idx = {}
    for r, (s, a) in enumerate(zip(tbl.column("stream").to_pylist(),
                                   (int(v) for v in
                                    tbl.column("ts").to_pylist()))):
        idx[(str(s), a)] = r
    V = np.asarray(vecs)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    M = np.full((len(keys), V.shape[1]), np.nan, np.float32)
    for i, (s, a, b) in enumerate(keys):
        r = idx.get((s, a))
        if r is not None:
            M[i] = V[r]
    if len(_VJ) > 4:
        _VJ.clear()
    _VJ[ck] = M
    return M


def prf_contrast(store, keys, seed_scores, swap_seed_scores=None,
                 k=16):
    """Per-episode contrast scores from corpus-derived anchors.

    seed_scores: text-channel scores for the query (weak relevance);
    swap_seed_scores: same for the swapped query, or None (falls back
    to the corpus mean as the neutral anchor). Returns scores aligned
    to keys (NaN where no video-native vector exists)."""
    M = _vjepa_matrix(store, keys)
    have = np.isfinite(M[:, 0])

    def anchor(sc):
        s = np.where(np.isfinite(sc) & have, sc, -np.inf)
        top = np.argsort(-s)[:k]
        top = top[np.isfinite(s[top])]
        if len(top) == 0:
            return None
        a = M[top].mean(0)
        return a / (np.linalg.norm(a) + 1e-8)

    a_pos = anchor(np.asarray(seed_scores, float))
    if a_pos is None:
        return np.full(len(keys), np.nan)
    if swap_seed_scores is not None:
        a_neg = anchor(np.asarray(swap_seed_scores, float))
    else:
        a_neg = None
    if a_neg is None:
        a_neg = np.nanmean(M[have], 0)
        a_neg = a_neg / (np.linalg.norm(a_neg) + 1e-8)
    d = a_pos - a_neg
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.full(len(keys), np.nan)
    out = M @ (d / n)
    out[~have] = np.nan
    return out


def informativeness(scores_text, scores_swap):
    """1 - |rank correlation| between the channel's view of the query
    and of its swap: ~0 means the channel cannot tell the directions
    apart FOR THIS QUERY on THIS corpus. Measured in microseconds,
    needs no truth."""
    a = np.asarray(scores_text, float)
    b = np.asarray(scores_swap, float)
    fin = np.isfinite(a) & np.isfinite(b)
    if fin.sum() < 8:
        return 0.0
    ra = np.argsort(np.argsort(a[fin]))
    rb = np.argsort(np.argsort(b[fin]))
    r = np.corrcoef(ra, rb)[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(1.0 - abs(r))


def anchor_support(store, keys, seed_scores, k=16, trials=64):
    """Self-recognized no-match signal: how much more coherent is the
    pseudo-positive set than random corpus draws, in video-native
    space? z << 1 means the text channels found no distinctive
    neighborhood — the queried thing plausibly does not happen here."""
    rng = np.random.default_rng(0)
    M = _vjepa_matrix(store, keys)
    have = np.where(np.isfinite(M[:, 0]))[0]
    s = np.where(np.isfinite(np.asarray(seed_scores, float)),
                 seed_scores, -np.inf)
    top = np.argsort(-s)[:k]
    top = top[np.isfinite(np.asarray(s)[top])]
    top = [t for t in top if np.isfinite(M[t, 0])]
    if len(top) < 4 or len(have) < 4 * k:
        return None
    def coh(rows):
        X = M[rows]
        g = X @ X.T
        n = len(rows)
        return float((g.sum() - n) / max(1, n * n - n))
    obs = coh(np.asarray(top))
    null = np.array([coh(rng.choice(have, size=len(top),
                                    replace=False))
                     for _ in range(trials)])
    return float((obs - null.mean()) / (null.std() + 1e-9))
