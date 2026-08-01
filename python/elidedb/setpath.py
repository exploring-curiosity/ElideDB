"""Shared set-path selection: ONE code path for the live query
(scenario.search_set) and the fitter (scripts/fit_set_weights.py).

The fit-live divergences of the acceptance sprint (the knee reshaping
the fitted cut, the obj-boost overriding fitted weights) were measured
regressions: what the fit optimized was not what the query executed.
Anything the fit searches over must run through these functions."""
from __future__ import annotations

import numpy as np


def rankfrac(v):
    """Rank fraction in [0,1]; NaN (abstaining) entries vote a neutral
    0.5 so a channel that cannot score an episode never executes it."""
    v = np.asarray(v, float)
    r = np.full(len(v), 0.5)
    fin = np.isfinite(v)
    if fin.sum() > 1:
        r[fin] = np.argsort(np.argsort(v[fin])) / (fin.sum() - 1)
    return r


def confidence_cut(scores, alpha, k_max):
    """Where the returned set ENDS: keep clips while the fused score
    stays >= alpha x the query's own top-5 mean, bounded by k_max.

    The product contract is "up to K, and everything returned is
    true" — so the cut is where the system stops being confident, not
    a fixed count. alpha is FITTED per query type (0 => plain top-K,
    the pre-cut behavior); the ratio-to-own-top-mass form is
    scale-invariant, so uncalibrated RRF scores from different channel
    counts compare safely across queries. This is the fitted successor
    of the hand-tuned 0.62 knee floor the fit-live unification
    removed. `scores` must be sorted descending."""
    s = np.asarray(scores, float)
    n = min(len(s), int(k_max))
    if n == 0:
        return 0
    bg = _unsupervised_cut(s, n)      # outliers above the background
    if alpha <= 0:
        # alpha=0 used to mean "return k_max". That is not abstention,
        # it is the absence of it: measured at k_max=100 the directional
        # queries (whose fitted alpha is 0) returned 100 results for a
        # support of 2. k is a CEILING, not a target, so with no fitted
        # alpha we fall back to the UNSUPERVISED knee of the query's own
        # score curve — a property of the scores, needing no truthset.
        return _unsupervised_cut(s, n)
    top = float(s[:min(5, len(s))].mean())
    if top <= 0:
        return n
    head = int(np.searchsorted(-s[:n], -alpha * top, side="right"))
    # The fitted alpha is a fraction of the query's OWN HEAD, so it stops
    # early when the head is peaky: q03 returned 10 while 65 of its top
    # 100 were true. Taking the larger of the two never returns less than
    # the confidence floor allows, and no longer discards matches that
    # clearly stand out from the corpus.
    return max(head, bg)


def _unsupervised_cut(s, n):
    """Return everything that stands out from the BACKGROUND.

    Two earlier rules both failed, in opposite directions, and the
    reason is the same: they looked only at the head of the curve.

      - largest relative drop (knee): the biggest gap on an RRF curve is
        at the very top, so it returned 3 clips for a query with 196
        true episodes.
      - fraction of the top-5 mean (fitted alpha): a fixed fraction of
        the head, so q03 returned 10 while 65 of its own top-100 were
        true - it discarded 55 correct answers it had already ranked.

    The right question is not "where does the head end" but "which
    episodes score unlike the corpus". Most episodes are irrelevant to
    any given query, so they form a background distribution; a match is
    an outlier above it. Median + 3*MAD is the standard robust test for
    that, and it is scale-free, so it works on uncalibrated RRF scores.
    It also adapts to what is actually there: hundreds of outliers means
    hundreds returned (up to the ceiling), two means two.

    Uses only this query's own scores - no evaluation data informs it."""
    x = np.asarray(s, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0
    if len(x) < 4:
        return min(len(x), n)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 0:
        # a degenerate spread (most scores identical): fall back to the
        # positives above the median rather than returning everything
        keep = int((x > med).sum())
        return max(1, min(keep, n))
    thr = med + 3.0 * 1.4826 * mad          # 1.4826 -> sigma-equivalent
    keep = int((x >= thr).sum())
    return max(1, min(keep, n))
def event_positions(keys):
    """(stream_id, position-in-stream) per episode key — store
    geometry only (episodes are consecutive windows; position
    distance is the unit of temporal adjacency)."""
    by = {}
    for i, (s, a, _b) in enumerate(keys):
        by.setdefault(s, []).append((a, i))
    sid = np.empty(len(keys), int)
    pos = np.empty(len(keys), int)
    for j, s in enumerate(sorted(by)):
        for p, (_a, i) in enumerate(sorted(by[s])):
            sid[i] = j
            pos[i] = p
    return sid, pos


def nms_keep(order, sid, pos, r):
    """Temporal non-max suppression over the ranked order: a clip
    within r episode positions of an already-kept SAME-STREAM clip is
    the same event seen again — suppress it (r=0 disables; r is
    FITTED per query type). Measured motivation: the eggplant query
    returned five windows of one event while the graded action moment
    sat at rank ~15 — the product wants each true event once, and
    every suppressed duplicate frees a slot for the next-ranked
    DISTINCT event."""
    if r <= 0:
        return order
    kept = []
    taken = {}
    for i in order:
        ps = taken.setdefault(int(sid[i]), [])
        p = int(pos[i])
        if any(abs(p - q) <= r for q in ps):
            continue
        ps.append(p)
        kept.append(int(i))
    return np.asarray(kept, int)


def filter_mask(ch, names, q):
    """UNANIMOUS rank-fraction filter: an episode dies only when EVERY
    named filter channel puts it in the bottom q.

    Quantile, never sign: AUC-validated channels have uncalibrated zero
    points (a sign test executed 130/183 true closes, measured).
    All-alive when no named channel has evidence or q is 0.

    WHY UNANIMITY, NOT THE MEDIAN. scenario.py's contract has always read
    "an episode BOTH direction-aware channels score negative is dropped"
    - and this computed a MEDIAN, which is not that. The difference is
    not cosmetic: deletion happens BEFORE ranking, so nothing downstream
    can recover an episode this drops, and the median lets a majority of
    weak channels delete on evidence no single trustworthy channel
    supports.

    Measured on fresh_bench, where q05's contrast channels are act
    (AUC 0.330 - inverted, it ranks true episodes BELOW false ones) and
    mot (0.529 - chance): the median rule destroyed 68 of 196 known
    positives before ranking, capping achievable yield at 0.65 no matter
    what the ranking or the cut did afterwards. q04 lost 18 of 165.

    Unanimity is the conservative reading and the one the docstring
    promised: a single channel that disagrees is enough to spare an
    episode, so an inverted channel can no longer carry a deletion on
    its own. It cannot make the filter delete MORE than the median did,
    only less, which is the right direction for an irreversible step.

    An earlier attempt fixed this by dropping vetoed channels from
    `names` instead. That REGRESSED (q05 yield 0.27 -> 0.21) because the
    median over fewer channels is more decisive, not less - the fix has
    to change the rule, not the membership.
    """
    con = [rankfrac(ch[c]) for c in names
           if c in ch and np.isfinite(ch[c]).any()]
    size = len(next(iter(ch.values()))) if ch else 0
    if not con or q <= 0:
        return np.ones(size, bool)
    return ~np.all(np.stack(con) < q, axis=0)
