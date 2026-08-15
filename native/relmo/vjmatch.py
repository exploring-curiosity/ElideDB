"""Matchers for ragged, stream-time traces - and the length bias they must not have.

THE DEFECT THIS EXISTS TO FIX. relmo.vjeval5.dtw_from_cost has free endpoints
and divides the accumulated path cost by the QUERY length only. A longer
candidate therefore offers more sub-spans in which to find a cheap match, at no
extra cost, so long candidates win mechanically. Measured on v6 frozen records,
val split:

    Spearman(DTW score, candidate trace length)  =  +0.710,
        positive for 65 of 65 queries
    a DURATION-ONLY ranker, no pixels involved   =  0.315 precision@support
    the frozen CONTENT matcher                   =  0.314
    chance                                       =  0.215

Under v4 every trace was exactly 24 steps, so the bias was a constant and
cancelled. Stream time made every trace a different length and exposed it.

THE FIX is the symmetric2 step pattern (Sakoe & Chiba), which is the standard
answer and is exact rather than a heuristic:

    D[i,j] = min( D[i-1,j-1] + 2*c[i,j],      diagonal
                  D[i-1,j]   + 1*c[i,j],      query advances, reference holds
                  D[i,j-1]   + 1*c[i,j] )     reference advances, query holds

Every complete path from (0,0) to (Q-1,R-1) accumulates weight exactly Q+R
regardless of its shape, so dividing by Q+R normalises without bias. There is
no `pen` to tune: the weighting IS the penalty, and it is the one that makes
the normaliser exact.

TWO QUERY TYPES, previously conflated under one matcher:

    full   both endpoints anchored. "Is this whole recording the same event as
           that whole recording?" - the query the duration-invariance work is
           about, and the one graded by precision@support.
    sub    free start and end on the REFERENCE. "Where inside this stream does
           the query occur?" - span localisation. Normalised by Q + the
           reference span actually consumed, tracked along the chosen path, so
           a long candidate gains nothing from being long.

ARC-LENGTH REPARAMETERISATION lives here too because it is the other half of
the same problem. A trace indexed by TIME makes a 7 s and a 20 s instance of
one event different lengths. Re-indexing by CUMULATIVE CHANGE - emitting a
descriptor every fixed increment of how much has happened rather than every
0.25 s - makes them the same length by construction, compresses idle stretches,
and preserves order. That last point matters: relmo/vjtime already established
that "a descriptor that survives warping by discarding order has not solved
anything", which rules out pooling as an answer.
"""
from __future__ import annotations

import numpy as np

# Arc-length increment, in units of the layer-6 gate energy that vjrec6 already
# computes. SELECTED ON THE TRAIN SPLIT ONLY (100 queries vs the 291-recording
# train pool): 0.8x/1.0x/1.3x/1.7x/2.2x of the train median gave 0.388 / 0.401 /
# 0.392 / 0.382 / 0.361 against 0.349 for no arc. val and test never saw it.
ARC_DS = 1716.64


def dtw(C, free_ends=False, lens=None, band=0.0):
    """band: Sakoe-Chiba half-width as a FRACTION of the reference length. 0
    disables it. The DP is a Python scan over (query x reference), so it is the
    bottleneck in every evaluation - banding it cuts the inner loop from R to
    2*band*R. Paths outside the band are forbidden, which is the standard
    constraint and is exact for any alignment that does not warp by more than
    the band allows."""
    return _dtw(C, free_ends, lens, band)


def _dtw(C, free_ends=False, lens=None, band=0.0):
    """C: (N, Q, R) cost -> (N,) length-normalised distance. Lower is better.

    free_ends=False anchors both ends and divides by Q+R exactly.
    free_ends=True lets the path enter and leave the reference anywhere, and
    divides by Q + (reference span consumed), tracked along the path.

    lens: (N,) true reference lengths when C is RIGHT-PADDED. This is not
    optional for the anchored matcher - it has to finish at the reference's
    last real step, and reading the padded last column instead scored 0.158
    against a chance of 0.215, i.e. below chance, with rho(score, length)
    = +0.98. Horizontal moves only ever run left to right, so padding on the
    right cannot influence any real column and the same DP serves both.
    """
    C = np.asarray(C, np.float64)
    n, q, r = C.shape
    lens = (np.full(n, r, np.int64) if lens is None
            else np.asarray(lens, np.int64))
    if free_ends:
        D = 2.0 * C[:, 0, :]                    # enter the reference anywhere
        St = np.tile(np.arange(r, dtype=np.int64), (n, 1))
    else:
        D = np.empty((n, r))
        D[:, 0] = 2.0 * C[:, 0, 0]
        for j in range(1, r):                   # only horizontals in row 0
            D[:, j] = D[:, j - 1] + C[:, 0, j]
        St = np.zeros((n, r), np.int64)

    INF = np.float64(1e18)
    # band limits per query row, around the diagonal of the LONGEST reference
    if band > 0:
        w = max(2, int(round(band * r)))
        slope = (r - 1) / max(q - 1, 1)
        lo_all = np.maximum(0, (np.arange(q) * slope - w).astype(int))
        hi_all = np.minimum(r - 1, (np.arange(q) * slope + w).astype(int))
        if not free_ends:
            D[:, hi_all[0] + 1:] = INF
    for i in range(1, q):
        c = C[:, i, :]
        diag = np.concatenate([np.full((n, 1), INF), D[:, :-1]], 1) + 2.0 * c
        vert = D + c                            # query advances, reference holds
        if not free_ends:
            vert[:, 0] = D[:, 0] + c[:, 0]
            diag[:, 0] = INF                    # nothing to the left of j=0
        take_d = diag <= vert
        run = np.where(take_d, diag, vert)
        runS = np.where(take_d,
                        np.concatenate([np.zeros((n, 1), np.int64),
                                        St[:, :-1]], 1), St)
        j0, j1 = (1, r) if band <= 0 else (max(1, lo_all[i]), hi_all[i] + 1)
        for j in range(j0, j1):                 # reference advances, query holds
            alt = run[:, j - 1] + c[:, j]
            better = alt < run[:, j]
            run[:, j] = np.where(better, alt, run[:, j])
            runS[:, j] = np.where(better, runS[:, j - 1], runS[:, j])
        if band > 0:                            # outside the band is forbidden
            run[:, :lo_all[i]] = INF
            run[:, hi_all[i] + 1:] = INF
        D, St = run, runS

    rows = np.arange(n)
    if not free_ends:
        return D[rows, lens - 1] / (q + lens)
    span = np.arange(r, dtype=np.int64)[None, :] - St + 1
    out = D / (q + span)
    out[np.arange(r)[None, :] >= lens[:, None]] = np.inf   # padded endpoints
    return out.min(1)


def multirate(qs, cs, cost=None, free_ends=False):
    """Best-explaining rate pair. qs, cs are {rate_name: descriptor or None}.

    WHY RATES AND NOT JUST ALIGNMENT. Warping a recording 3x slower and asking
    it to retrieve its own original gave MRR 0.009 against a chance of 0.0149 -
    the time-indexed system cannot find its own video. Arc-length
    reparameterisation lifted that to 0.079, an 8.8x gain that is still close to
    nothing, and the reason is that the descriptor is itself a RATE. At 8 fps a
    3x-slow event moves a third as far per step, so V-JEPA's one-step forecast
    is a different quantity, not the same quantity sampled differently. No
    re-indexing of a fixed-rate trace can undo that; the event has to be
    re-encoded at a rate that makes its per-step motion comparable.

    A recording is only encodable at a rate whose window fits inside it - the
    4 fps window spans 8 s and the 8/3 fps window 12 s - so coarse rates exist
    only for long recordings. That is not a gap: it is exactly the case where a
    coarse rate is what a short query needs to match against.

    Returns the min distance over rate pairs that both sides have.
    """
    best = None
    for rq, q in qs.items():
        if q is None or len(q) < 2:
            continue
        for rc, c in cs.items():
            if c is None or len(c) < 2:
                continue
            d = cost(q, c, free_ends)
            best = d if best is None else np.minimum(best, d)
    return best


def pack(desc_by_rate, ids, pad_cost=1e6):
    """{rate: {id: (T,D)}} x ids -> {rate: (P, ok, L, present)} for scoring.

    A recording missing at a rate gets present=False rather than a placeholder,
    so it simply does not compete at that rate.
    """
    out = {}
    for rate, d in desc_by_rate.items():
        present = np.array([i in d for i in ids])
        if not present.any():
            continue
        seqs = [np.asarray(d[i], np.float32) if i in d
                else np.zeros((2, 1), np.float32) for i in ids]
        dim = max(x.shape[-1] for x in seqs)
        seqs = [x if x.shape[-1] == dim else np.zeros((2, dim), np.float32)
                for x in seqs]
        tmax = max(len(x) for x in seqs)
        P = np.zeros((len(ids), tmax, dim), np.float32)
        ok = np.zeros((len(ids), tmax), bool)
        for k, x in enumerate(seqs):
            P[k, :len(x)] = x
            ok[k, :len(x)] = True
        out[rate] = (P, ok, np.array([len(x) for x in seqs]), present)
    return out


def score_rates(q_by_rate, packed, keep=None, free_ends=False, pad_cost=1e6,
                mode="match"):
    """Distance under the rate pair that best explains the candidate.

    mode="min"    take the minimum over every rate pair both sides have.
    mode="match"  take the ONE pair whose trace lengths are closest in log
                  ratio, and use only that distance.

    "min" is the obvious rule and it is measurably worse. Nine chances to find
    a low distance help a wrong candidate more than a right one: recall of true
    positives at duration ratio <1.25x fell 0.704 -> 0.571 under "min" while
    the >1.75x bin rose 0.091 -> 0.192. "match" asks a different and better
    question - encode both sides so their step counts agree, THEN compare -
    and gets the far-duration gain without handing out free attempts.
    """
    any_rate = next(iter(packed.values()))
    n = len(any_rate[3])
    sel = np.ones(n, bool) if keep is None else np.asarray(keep, bool)
    k = int(sel.sum())
    best = np.full(k, np.inf)
    mism = np.full(k, np.inf)
    for q in q_by_rate.values():
        if q is None or len(q) < 2:
            continue
        for P, ok, L, present in packed.values():
            m = present & sel
            if not m.any():
                continue
            C = 1.0 - np.einsum("sd,nkd->nsk", q, P[m])
            C = np.where(ok[m][:, None, :], C, pad_cost)
            d = dtw(C, free_ends, L[m])
            at = m[sel]
            if mode == "min":
                best[at] = np.minimum(best[at], d)
            else:
                mm = np.abs(np.log(len(q) / np.maximum(L[m], 1)))
                take = mm < mism[at]
                idx = np.where(at)[0]
                best[idx[take]] = d[take]
                mism[idx[take]] = mm[take]
    return best


def fit_pca(seqs, dim=256, sample=200_000, seed=0, whiten=False):
    """PCA over pooled STEP vectors. Label-free, fitted on train only.

    The cost matrix einsum is 93-98% of evaluation time and scales linearly in
    descriptor dimension (measured: 1024-d costs 107ms/query at Q=R=44 against
    8ms for the DP; 2816-d costs 417ms). Projecting to 256 dims is a ~7x
    end-to-end speedup for a 1024-d channel and ~25x for a 2816-d concat, and
    it shrinks whatever index this eventually feeds.
    """
    X = np.concatenate([np.asarray(s, np.float32) for s in seqs])
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        X = X[rng.choice(len(X), sample, replace=False)]
    mu = X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    dim = min(dim, Vt.shape[0])
    var = float((S[:dim] ** 2).sum() / (S ** 2).sum())
    W = Vt[:dim].T
    if whiten:
        # divide each component by its singular value. Cosine retrieval on
        # pooled deep features is usually dominated by a few high-variance
        # directions; whitening equalises them so the tail of the spectrum -
        # where the discriminative structure often sits - can contribute.
        W = W / (S[:dim] / np.sqrt(max(len(X) - 1, 1)) + 1e-6)
    return dict(mu=mu, W=W.astype(np.float32), var=var, whiten=whiten)


def apply_pca(x, p):
    # Accelerate's BLAS sets spurious divide-by-zero / overflow FP status flags
    # on Apple Silicon even when every input and output is finite - verified
    # against a float64 einsum reference (max abs diff 1e-6, relative 1e-7).
    # Silence it here rather than globally, so a REAL non-finite value elsewhere
    # still surfaces.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return ((np.asarray(x, np.float32) - p["mu"]) @ p["W"]).astype(np.float32)


def _brute(c, free_ends):
    """Reference implementation for tests. Exponential; tiny inputs only."""
    q, r = c.shape
    best = np.inf
    starts = range(r) if free_ends else [0]
    ends = range(r) if free_ends else [r - 1]

    def walk(i, j, acc, start):
        nonlocal best
        if i == q - 1:
            if j in ends:
                span = j - start + 1
                best = min(best, acc / (q + (span if free_ends else r)))
        if i + 1 < q:
            walk(i + 1, j, acc + c[i + 1, j], start)               # vertical
            if j + 1 < r:
                walk(i + 1, j + 1, acc + 2 * c[i + 1, j + 1], start)  # diagonal
        if j + 1 < r:
            walk(i, j + 1, acc + c[i, j + 1], start)               # horizontal

    for s in starts:
        walk(0, s, 2 * c[0, s], s)
    return best


def arc_resample(seq, mag, ds, aux=None, min_len=4, max_len=512):
    """Re-index a trace by CUMULATIVE CHANGE instead of by time.

    seq  (T, D) descriptors
    mag  (T,)   per-step change magnitude - how much happened at that step
    ds          arc-length per output step; the corpus median of `mag` keeps
                output length comparable to input length for a typical event
    aux         optional list of (T, *) arrays resampled on the same grid

    A 7 s and a 20 s execution of one motion accumulate the same total change,
    so they emit the SAME number of steps. Idle stretches contribute little and
    compress. Order is preserved.
    """
    seq = np.asarray(seq, np.float32)
    m = np.maximum(np.asarray(mag, np.float64), 0.0)
    s = np.concatenate([[0.0], np.cumsum(m)])          # (T+1,) knot positions
    total = float(s[-1])
    if total <= 0 or ds <= 0:
        return (seq, list(aux or []))
    n_out = int(np.clip(round(total / ds), min_len, max_len))
    # sample at step CENTRES so the first and last steps are represented
    want = (np.arange(n_out) + 0.5) * (total / n_out)
    src = np.interp(want, s[1:], np.arange(len(seq), dtype=np.float64))
    lo = np.floor(src).astype(int).clip(0, len(seq) - 1)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (src - lo).astype(np.float32)[:, None]

    def take(x):
        x = np.asarray(x, np.float32)
        ww = w if x.ndim == 2 else w[:, 0]
        return ((1 - ww) * x[lo] + ww * x[hi]).astype(np.float32)

    return take(seq), [take(x) for x in (aux or [])]
