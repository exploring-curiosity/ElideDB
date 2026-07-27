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
    if alpha <= 0 or n == 0:
        return n
    top = float(s[:min(5, len(s))].mean())
    if top <= 0:
        return n
    return int(np.searchsorted(-s[:n], -alpha * top, side="right"))


def filter_mask(ch, names, q):
    """Median rank-fraction over the named filter channels; episodes
    in the bottom-q die. Quantile, never sign: AUC-validated channels
    have uncalibrated zero points (a sign test executed 130/183 true
    closes, measured). All-alive when no named channel has evidence or
    q is 0."""
    con = [rankfrac(ch[c]) for c in names
           if c in ch and np.isfinite(ch[c]).any()]
    size = len(next(iter(ch.values()))) if ch else 0
    if not con or q <= 0:
        return np.ones(size, bool)
    return ~(np.median(np.stack(con), 0) < q)
