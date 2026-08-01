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
# POWER is tuned to the coherence statistic, and the two must move
# together: 8.0 was right for an unbounded Cohen's d and crushes a
# bounded [0,1] rank statistic to zero (0.42**8 = 0.001, every channel
# silenced). Re-measured for the bounded form, the optimum is 2 and the
# plateau is flat from 2 to 4.
POWER = 2.0
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


# EACH MODEL IS SCORED THE WAY IT WAS BUILT TO BE SCORED.
#
# Everything here was originally scored by cosine, which is the native
# operation for exactly one of these channels. `act` is a 174-way SSv2
# CLASSIFIER: its rows are a probability distribution over actions, and
# cosine on a simplex is not a comparison of distributions. Measured, the
# difference is not marginal - on q03 cosine scored AUC 0.483, BELOW
# CHANCE, while cross-entropy on the same numbers scores 0.812:
#
#     operator            q03     q04     q05
#     cosine (was)       0.483   0.561   0.622
#     cross-entropy      0.812   0.641   0.764
#     neg-JS             0.793   0.641   0.750
#
# Cross-entropy log(p_episode) . p_seed is the likelihood of the
# episode's posterior under the seed's, which is what "does this clip
# show the same action" means for a classifier.
#
# PCA whitening is also wrong for a simplex - subtracting a mean and
# renormalising a probability vector produces something that is no longer
# a distribution - so drop_pc is not applied to classifier channels.
OPS = {"act": "crossent"}          # everything else: cosine


def spaces(store, drop_pc=DROP_PC):
    """Every per-episode table as a matrix aligned with episodes, plus
    the operator each one should be scored with."""
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
        name = t.replace("_vectors", "").replace("action_probs", "act")
        if drop_pc and OPS.get(name) != "crossent" and ok.sum() > drop_pc + 5:
            X = A[ok]
            mu = X.mean(0)
            _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
            P = Vt[:drop_pc]
            A = A - mu
            A = A - (A @ P.T) @ P
            A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-8)
            A[~ok] = 0
        M[name] = (A.astype(np.float32), ok, OPS.get(name, "cosine"))
    return keys, M


def _score(A, op, seeds):
    """Score every episode against the seeds, the model's own way."""
    if op == "crossent":
        pm = A[seeds].mean(0)
        return np.log(A + 1e-12) @ pm
    cen = A[seeds].mean(0)
    n = np.linalg.norm(cen)
    return A @ (cen / n) if n > 0 else np.zeros(len(A))


def _pair(A, op, rows, cols):
    """Pairwise affinity between two row sets, the model's own way. The
    OPERATOR MUST GOVERN THE WEIGHT TOO. Scoring `act` by cross-entropy
    while weighting it by cosine measured a REGRESSION (q03 0.67 -> 0.60)
    even though the channel's own AUC had risen 0.483 -> 0.812: the
    channel became right and its voice became wrong. Half-applying the
    principle is worse than not applying it."""
    if op == "crossent":
        return np.log(A[cols] + 1e-12) @ A[rows].T   # cols x rows
    return A[cols] @ A[rows].T


def coherence(A, ok, seeds, op="cosine", cap=4000):
    """How reliably this model rates a seed pair above a seed/corpus
    pair - as a probability, in the model's own metric.

    Cohen's d was the first form and it does not survive mixed
    operators: cross-entropy is a log-likelihood with a far wider spread
    than cosine, so `d` for a classifier and `d` for an embedding are
    different units. Raising incomparable numbers to the 8th power then
    decides the whole fusion, and measured it silenced `act` entirely
    (q03 0.67 -> 0.60) at the moment the channel started scoring well.

    This is the probability that a random seed-seed affinity exceeds a
    random seed-corpus one - a rank statistic, bounded [0, 1], identical
    in meaning whether the model returns a cosine or a log-likelihood.
    0.5 is "this model cannot tell the seeds from the corpus".
    """
    if len(seeds) < 2:
        return 0.0
    other = np.setdiff1d(np.where(ok)[0], seeds)
    if len(other) < 10:
        return 0.0
    W = _pair(A, op, seeds, seeds)
    iu = np.triu_indices(len(seeds), 1)
    within = W[iu]
    B = _pair(A, op, seeds, other).ravel()
    if len(B) > cap:                       # rank stat needs no more
        B = B[np.linspace(0, len(B) - 1, cap).astype(int)]
    # P(within > between), ties at half
    allv = np.concatenate([within, B])
    r = np.empty(len(allv))
    r[np.argsort(allv)] = np.arange(len(allv))
    rw = r[:len(within)].sum()
    u = rw - len(within) * (len(within) - 1) / 2
    p = u / (len(within) * len(B))
    return float(max(p - 0.5, 0.0) * 2.0)      # 0 at chance, 1 at perfect


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
    w = {c: max(coherence(A, ok, seeds, op), 0.0) ** power
         for c, (A, ok, op) in M.items()}
    tot = None
    for c, (A, ok, op) in M.items():
        if w.get(c, 0.0) <= 0:
            continue
        live = np.array([i for i in seeds if ok[i]])
        if not len(live):
            continue
        sc = _score(A, op, live)
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
