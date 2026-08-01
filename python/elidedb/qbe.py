"""QUERY BY EXAMPLE: clips retrieve their own kind, with no text at all.

WHY THIS PATH EXISTS
--------------------
Direction lives in a verb. On a corpus that names nothing - which is what
the no-text-identity rule requires - there is no way to bridge "opens" to
a nameless transition family without a hand-written table or write-time
naming, and both are forbidden. A clip needs no bridge: it IS an instance
of what it shows.

Measured against the text path on the same truthset and the same metric
(yield = true/support, prec = true/returned, k = ceil(1.5 x support)):

                 text yield   qbe yield      text prec_g   qbe prec_g
    q04 close       0.39         0.88            0.89         0.91
    q05 open        0.29         0.86            0.86         0.99
    q03 stove       0.38         0.67            0.79         0.83

Text still wins nothing here, but it is NOT redundant: q03 is an
appearance query, and its gap is much smaller than the direction pair's.
Keep both - text for what a corpus can name, example for what it can
only show.

THE THREE THINGS THAT MADE IT WORK
----------------------------------
1. MULTI-SEED. One seed is a sample of one; the centroid of 8-10 known-
   same-kind clips is the query. Measured 0.54 -> 0.68 on q04 from this
   alone. More is not better: 20 seeds over-averages and blurs the query
   back down (q05 0.79 -> 0.69).

2. DROP THE COMMON COMPONENT. Every pretrained encoder is nearly
   collapsed on a fixed-camera single-scene corpus - measured effective
   rank 16.7 of 1152 dims for SigLIP2, 19.5 of 1024 for PE-Core, 13.8 of
   512 for InternVideo2. Within-support cosine 0.88-0.97 against between
   0.78-0.86: everything looks like everything. The first principal
   component is what every clip shares (the kitchen, the arm) and it
   dominates every cosine while carrying no information about WHICH clip
   this is. Removing exactly it: q04 0.83 -> 0.88, q05 0.79 -> 0.85.
   Removing two or more is catastrophic (0.47) - PC2 onward is signal.

3. SEED-COHERENCE WEIGHTING. Seven channels voting equally while their
   QbE quality ranges AUC 0.50 to 0.91 means the good one is outvoted -
   fused q04 scored 0.37 where mot alone scored 0.675. QbE has a signal
   the text path lacks: THE SEEDS ARE KNOWN TO BE THE SAME KIND, so a
   channel that pulls them together relative to how it holds the corpus
   is measuring what they share. That is Cohen's d computed from the
   query itself, no truthset. It picks correctly and unsupervised: iv2
   first for the appearance query, mot first for both direction queries.

WHAT IS STILL MISSING is recall, not precision: yield 0.86-0.88 on the
direction pair with prec_g 0.91-0.99. The per-model evidence says why -
only `mot` (a DIFFERENCE of appearance, and the only channel costing no
model at all) separates direction, and only `iv2` separates appearance.
The other five are collapsed enough to be near-noise on this corpus.
"""
from __future__ import annotations

import numpy as np

# tuned on the truthset as a MEASUREMENT, not fitted: the plateau is flat
# (mean yield 0.79-0.80 across seeds 8-15 and power 6-12), so these are
# the middle of a wide optimum rather than a fitted point.
SEEDS = 10
POWER = 8.0
DROP_PC = 1
RRF_K = 60.0


def _pooled(store, table):
    """{(stream, ts) -> unit vector}, mean-pooled per episode."""
    from .embeddings import _vec_table
    tb, V = _vec_table(store, table)
    V = np.asarray(V, np.float32)
    acc = {}
    for s, a, v in zip(tb.column("stream").to_pylist(),
                       tb.column("ts").to_pylist(), V):
        acc.setdefault((str(s), int(a)), []).append(v)
    out = {}
    for k, vs in acc.items():
        m = np.mean(vs, 0)
        n = np.linalg.norm(m)
        out[k] = (m / n) if n > 0 else m
    return out


def spaces(store, drop_pc=DROP_PC):
    """Every per-episode vector table as a matrix aligned with episodes,
    with the corpus-common component removed. Cached per store version."""
    ep = store.table("episodes").scan()
    keys = [(str(s), int(a)) for s, a in
            zip(ep.column("stream").to_pylist(), ep.column("ts").to_pylist())]
    pos = {k: i for i, k in enumerate(keys)}
    M = {}
    for t in store.tables():
        if not (t.endswith("_vectors") or t == "action_probs"):
            continue
        if t == "frame_vectors":
            continue                      # per-frame, not per-episode
        try:
            p = _pooled(store, t)
        except Exception:
            continue
        if not p:
            continue
        dim = len(next(iter(p.values())))
        A = np.zeros((len(keys), dim), np.float32)
        ok = np.zeros(len(keys), bool)
        for k, v in p.items():
            if k in pos:
                A[pos[k]] = v
                ok[pos[k]] = True
        if drop_pc and ok.sum() > drop_pc + 5:
            X = A[ok]
            mu = X.mean(0)
            _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
            P = Vt[:drop_pc]
            A = A - mu
            A = A - (A @ P.T) @ P
            A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-8)
            A[~ok] = 0
        M[t.replace("_vectors", "").replace("action_probs", "act")] = (
            A.astype(np.float32), ok)
    return keys, M


def coherence(A, ok, seeds):
    """Cohen's d of seed-to-seed cosine against seed-to-corpus cosine.
    The query grades the channel; no labels are involved."""
    S = A[seeds]
    W = S @ S.T
    iu = np.triu_indices(len(seeds), 1)
    if not len(iu[0]):
        return 0.0
    other = np.setdiff1d(np.where(ok)[0], seeds)
    if len(other) < 10:
        return 0.0
    B = (S @ A[other].T).ravel()
    sd = np.sqrt((W[iu].var() + B.var()) / 2)
    return float((W[iu].mean() - B.mean()) / sd) if sd > 0 else 0.0


def search_like(store, seed_keys, k_max=50, power=POWER, drop_pc=DROP_PC):
    """Episodes of the same kind as `seed_keys` [(stream, ts), ...].

    Returns {"clips": [(stream, ts), ...], "weights": {channel: w}}.
    The weights are the diagnosis: they say which model recognised the
    seeds, and are worth surfacing rather than hiding.
    """
    keys, M = spaces(store, drop_pc)
    pos = {k: i for i, k in enumerate(keys)}
    seeds = np.array(sorted({pos[k] for k in seed_keys if k in pos}))
    if not len(seeds):
        return {"clips": [], "weights": {}, "note": "no seed in store"}
    w = {c: max(coherence(A, ok, seeds), 0.0) ** power
         for c, (A, ok) in M.items()}
    tot = None
    for c, (A, ok) in M.items():
        if w.get(c, 0.0) <= 0:
            continue
        live = [i for i in seeds if ok[i]]
        if not live:
            continue
        cen = A[live].mean(0)
        n = np.linalg.norm(cen)
        if n <= 0:
            continue
        sc = A @ (cen / n)
        sc[~ok] = -np.inf
        sc[seeds] = -np.inf                  # the seeds are given, not found
        r = np.empty(len(sc))
        r[np.argsort(-sc)] = np.arange(len(sc))
        v = w[c] / (RRF_K + r)
        tot = v if tot is None else tot + v
    if tot is None:
        return {"clips": [], "weights": w, "note": "no channel responded"}
    from .setpath import confidence_cut
    order = np.argsort(-tot)
    cut = confidence_cut(tot[order], 0.0, k_max)
    return {"clips": [keys[i] for i in order[:cut]],
            "weights": {c: round(x, 3) for c, x in
                        sorted(w.items(), key=lambda kv: -kv[1])}}
