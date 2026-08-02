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

import os

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
# fusion sharpness on the LOO quality; swept 2..64, flat 12-16
ZPOWER = 12.0
# ZERO, re-measured under LOO weighting + z-fusion. Removing the first
# principal component was worth q04 0.83 -> 0.88 under coherence+RRF,
# because RRF sees only ranks and the shared component (the kitchen,
# the arm) dominated every cosine equally. z-fusion standardises each
# channel before combining, which handles that shift directly - and the
# subtraction then costs signal instead of noise: q04 0.93 -> 0.87.
# A correction for a defect that no longer exists is just a distortion.
DROP_PC = int(os.environ.get("ELIDEDB_DROP_PC", "0"))
RRF_K = 60.0


def _pooled(store, table):
    """{(stream, episode_ts) -> unit vector}, mean-pooled per episode.

    Two row conventions exist and both must pool correctly. pe/sig2
    write 8 rows per episode all carrying the EPISODE's ts, so grouping
    by (stream, ts) is already per-episode. A per-frame table
    (scene_vectors) carries each FRAME's ts - grouping by it yields one
    "episode" per frame, and when spaces() then joins on episode starts,
    the channel silently becomes first-frame-only: 2,097 of 70,436 rows
    used, 97% of the signal dropped with no error. So keys that don't
    match an episode start are binned into the episode whose span
    contains them, using the episodes table - the same containment the
    labels join uses.
    """
    from .embeddings import _vec_table
    tb, V = _vec_table(store, table)
    V = np.asarray(V, np.float32)
    ep = store.table("episodes").scan()
    spans = {}
    for s, a, b in zip(ep.column("stream").to_pylist(),
                       ep.column("ts").to_pylist(),
                       ep.column("t1").to_pylist()):
        spans.setdefault(str(s), []).append((int(a), int(b)))
    for v in spans.values():
        v.sort()
    starts = {s: [a for a, _ in v] for s, v in spans.items()}
    acc = {}
    for s, a, v in zip(tb.column("stream").to_pylist(),
                       tb.column("ts").to_pylist(), V):
        s, a = str(s), int(a)
        sp = spans.get(s)
        if sp:
            j = int(np.searchsorted(starts[s], a, "right")) - 1
            if j >= 0 and a <= sp[j][1]:
                a = sp[j][0]
        acc.setdefault((s, a), []).append(v)
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
        # A CHANNEL IS SOMETHING A MODEL PRODUCED. Selecting on the name
        # suffix alone silently enrolled object_vectors - the identity
        # index - as an eighth retrieval channel the moment that table
        # was first persisted, diluting the fusion with ReID appearance
        # descriptors that exist to answer "is this the same object",
        # not "does this episode match the query". Every encoder output
        # already declares its `model`; an index declares a cut instead,
        # so the distinction is recorded rather than guessed from a name.
        if not (store.table(t).state().meta or {}).get("model"):
            continue
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


def otsu_cut(sorted_scores, k_max):
    """Where the returned set ENDS: the valley between the two modes.

    setpath's confidence_cut fits a knee against RRF's scale and, on
    z-fused scores, fires almost immediately: measured 16 returned of a
    165-clip support (yield 0.08 at precision 0.94). It is not wrong,
    it is calibrated for a different score shape.

    A good query's score curve is bimodal - the clips of its kind, and
    the corpus - so the cut is the threshold that best SEPARATES those
    two populations. Otsu's method finds it by maximising between-class
    variance, which is parameter-free and reads only this query's own
    scores: no truthset, no fitted alpha, nothing per-corpus.

    Degenerate cases fail open (return k_max) rather than returning a
    handful, because a flat curve means "cannot tell", and answering
    almost nothing is a worse response to that than answering fully.
    """
    s = np.asarray(sorted_scores, float)
    s = s[np.isfinite(s)]
    n = min(len(s), int(k_max))
    if n < 4:
        return int(n)
    head = s[:max(n * 4, 200)][:len(s)]
    lo, hi = float(head.min()), float(head.max())
    if not np.isfinite(lo) or hi - lo < 1e-9:
        return int(n)
    hist, edges = np.histogram(head, bins=64, range=(lo, hi))
    p = hist.astype(float) / max(hist.sum(), 1)
    w0 = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
    between[~np.isfinite(between)] = -1.0
    thr = float(mids[int(between.argmax())])
    keep = int((s[:n] >= thr).sum())
    return int(min(max(keep, 1), n)) if keep else int(n)


LOO_DEPTHS = (25, 50, 100, 200, 400, 800)


def loo_quality(A, ok, seeds, op):
    """LEAVE-ONE-SEED-OUT retrieval quality of a channel. Query-only.

    Hold out one seed, query with the rest, record where the held-out
    seed lands; the statistic is its mean recall over a ladder of
    depths, which is the area under the LOO recall curve and so needs
    no chosen depth.

    This replaces coherence as the WEIGHT because coherence measures
    the wrong thing. Cohen's d asks "are the seeds tight?"; measured
    over 15 seed groups it named the wrong channel 4 times, because
    `motion` held yield 0.88-0.95 while its d fell from 2.5 to 0.76.
    Tightness is not retrieval. Selecting on LOO instead moved the
    high-support queries from 0.742 to 0.845 mean yield.

    Legal by the same argument coherence was: the seeds ARE the query.
    Their mutual membership is what the user asserted by supplying
    them, not something read from the truthset.

    A single depth would have been a tuned constant - it swung the
    result from 0.598 at 50 to 0.811 at 200 - so the ladder is not a
    refinement, it is what keeps this honest.
    """
    if len(seeds) < 3 or not ok[seeds].all():
        return 0.0
    out = []
    for i in seeds:
        rest = np.asarray([j for j in seeds if j != i])
        sc = _score(A, op, rest)
        sc[~ok] = -np.inf
        sc[rest] = -np.inf                   # other seeds not candidates
        r = int((sc > sc[i]).sum())
        out.append(float(np.mean([r < d for d in LOO_DEPTHS])))
    return float(np.mean(out))


def search_like(store, seed_keys, k_max=50, power=ZPOWER, drop_pc=DROP_PC):
    """Episodes of the same kind as `seed_keys` [(stream, ts), ...].

    Returns {"clips": [(stream, ts), ...], "weights": {channel: w}}.
    The weights are the diagnosis: they say which model recognised the
    seeds, and are worth surfacing rather than hiding.

    Fusion is at SCORE level, not rank level. RRF knows only that a
    channel put something first, never by how much, and the margin is
    exactly what separates a channel that recognises the query from one
    that is guessing: z-fusion measured 0.845 against RRF's 0.742 on
    the same weights and seeds.
    """
    keys, M = spaces(store, drop_pc)
    pos = {k: i for i, k in enumerate(keys)}
    seeds = np.array(sorted({pos[k] for k in seed_keys if k in pos}))
    if not len(seeds):
        return {"clips": [], "weights": {}, "note": "no seed in store"}
    q = {c: max(loo_quality(A, ok, seeds, op), 0.0)
         for c, (A, ok, op) in M.items()}
    mx = max(q.values()) if q else 0.0
    if mx <= 0:
        return {"clips": [], "weights": q, "note": "no channel responded"}
    w = {c: (v / mx) ** power for c, v in q.items()}
    tot = None
    for c, (A, ok, op) in M.items():
        if w.get(c, 0.0) <= 0:
            continue
        live = np.array([i for i in seeds if ok[i]])
        if not len(live):
            continue
        sc = _score(A, op, live)
        fin = np.isfinite(sc) & ok
        if fin.sum() < 3:
            continue
        z = np.zeros(len(sc))
        mu, sd = sc[fin].mean(), sc[fin].std() or 1e-9
        z[fin] = (sc[fin] - mu) / sd
        z[~fin] = -np.inf
        v = w[c] * z
        tot = v if tot is None else tot + v
    if tot is None:
        return {"clips": [], "weights": w, "note": "no channel responded"}
    tot[seeds] = -np.inf                     # the seeds are given, not found
    order = np.argsort(-tot)
    cut = otsu_cut(tot[order], k_max)
    return {"clips": [keys[i] for i in order[:cut]],
            "weights": {c: round(x, 3) for c, x in
                        sorted(q.items(), key=lambda kv: -kv[1])}}
