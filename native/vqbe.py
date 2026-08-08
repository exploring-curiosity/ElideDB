"""THE READ PATH. One example clip in, spans out. Pixels are the query.

    "I give you a sample, I want all samples like that."

That sentence is the whole specification, and it contains a problem that
every earlier version of this system dodged: ONE example gives no way to
calibrate anything. With a single vector there is no leave-one-out, so
there is no threshold, so the system must return exactly k things, so
precision is capped at support/k by construction and the number is
meaningless before the search even runs.

The resolution is that an example is not one vector. It is a SET - its
own overlapping sub-windows - and the members of that set are positives
of one another by construction, for free, from the pixels the user
already handed over. That set supplies three things:

  THE CUT           leave-one-out over the query's own sub-windows.
                    cut_c = min_i max_{j!=i} sim_c(q_i, q_j). If the
                    query cannot recognise itself at that similarity,
                    nothing at that similarity is a match. Returning
                    fewer than k results is correct behaviour, not a
                    failure to fill a quota.

  THE WEIGHTS       how informative is each channel FOR THIS QUERY.
                    w_c = (agreement inside Q) - (agreement of Q with a
                    random sample of the store). A channel where the
                    query is not self-consistent, or where everything in
                    the corpus already looks alike, earns nothing. This
                    is decided per query from pixels; there is no fitted
                    fusion weight anywhere, which matters because the
                    best single channel was measured to beat any fixed
                    fusion.

  THE QUERY SIDE OF CSLS   its own mean similarity to the store, which
                    is the term that makes the hubness correction
                    symmetric.

Scoring is CSLS rather than plain cosine (Conneau et al., ICLR 2018):

    s_c(x) = 2 * max_i cos_c(x, q_i) - hub_c(x) - hub_c(Q)

A vector that is close to everything is now close to nothing in
particular. The shape channel needs this badly - its similarity between
unrelated windows was measured at p99 = 0.898, so under plain cosine it
would drown every result list in hubs.

Then: probe coarse cells, exact rerank inside, abstain at the cut,
collapse overlapping survivors into spans (five views of one moment are
one answer, not five), and decode only what survives.

    python native/vqbe.py --store sim --media sim/ep0003 --t0 4 --t1 12
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "native"))

import vcore                                             # noqa: E402
import vsrc                                              # noqa: E402
from vstore import CHANNELS, STORES, Store               # noqa: E402

import os

# The coarse prune must not buy elision with answers, and a fixed
# probe fraction cannot promise that: measured recall against an exact
# scan was 0.908 on bridge but 0.617 on car and 0.550 on drone, because
# the right number of cells to open depends on how the corpus clusters,
# not on a constant. Instead, open cells in order of their best match to
# the query set until the OPENED MASS covers CAND_FRAC of the store, or
# the centroid similarity has fallen a full standard deviation below the
# best one - whichever comes first. Both terms are read off this store
# and this query; neither is fitted on a benchmark.
CAND_FRAC = 0.25   # candidate budget as a fraction of the store
WMODE = os.environ.get("SDX_WMODE", "pos")   # "pos" | "z" (see plan())
# Ablation only: force a channel's weight to zero at query time. Lets a
# channel be tested for its keep WITHOUT rebuilding the store, which is
# the difference between a 15-minute question and a 4-hour one.
DROP = set(x for x in os.environ.get("SDX_DROP", "").split(",") if x)
# note: with SDX_CHANNELS defaulting to "c2" the store no longer holds
# c1/c3 at all, so DROP is now only an ablation tool for stores that do.
REF_SAMPLE = 2048  # store rows used as the "what does everything look
                   # like" reference for weights and for CSLS


def _l2(V):
    V = np.asarray(V, np.float32)
    if V.ndim == 1:
        return V / max(float(np.linalg.norm(V)), 1e-8)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)


class Query:
    """An example, encoded. Nothing but frames goes in."""

    def __init__(self, frames):
        wins, e = vcore.encode_clip(frames)
        if wins is None or not e["valid"].any():
            raise ValueError("clip too short to form a single unit")
        ok = e["valid"]
        self.wins = [w for w, v in zip(wins, ok) if v]
        self.q = {c: e[c][ok] for c in CHANNELS}
        self.m = len(self.wins)

    @classmethod
    def from_media(cls, src, t0, t1):
        return cls(src.cut(t0, t1))


def otsu(x):
    """Maximum between-class variance split of a 1-D score array."""
    v = np.sort(np.asarray(x, float))
    n = len(v)
    if n < 4:
        return -np.inf
    cw = np.arange(1, n)
    c1 = np.cumsum(v)[:-1] / cw
    c2 = np.cumsum(v[::-1])[::-1][1:] / (n - cw)
    return float(v[int(np.argmax(cw * (n - cw) * (c1 - c2) ** 2))])


def score_cut(sc, cap=8):
    """THE ABSTENTION CUT, read off this query's own score distribution.

    What it replaced, and why. The cut used to be a leave-one-out floor
    among the query's own sub-windows. That premise does not hold: the
    sub-windows either OVERLAP (sharing half their frames, so their
    similarity is inflated and says nothing about a match) or are
    DISJOINT (different moments of one clip, so on the order channel
    they are near-uncorrelated and the floor collapses toward zero).
    Neither is "another instance of this event", which is the quantity a
    match threshold has to be calibrated against. Measured: the old
    threshold passed 866 of 873 non-matches, so abstention never fired
    and precision was pinned at yield/1.5 by arithmetic.

    Three rules that beat it were rejected before this one, each for a
    contamination that looked clean in code:
      - otsu over the top k*4 scores with k from the SUPPORT: the
        support is the answer key. Scored 0.873/0.815 by reading it.
      - otsu over the top 5%: the 5% won a sweep over these four
        corpora, which makes it a constant fitted on the evaluation set.
      - recursive otsu at depth 3: the depth is the same fitted
        constant, one level up.

    So the recursion stops on the data: keep splitting while each split
    explains MORE of the surviving variance than the last (eta^2 =
    between-class / total), stop when the survivors are homogeneous.
    No percentage, no depth, no support. `cap` only bounds runtime.

    Measured against the alternatives (4 stores, 8 queries each):
        loo   0.915/0.595   <- what this replaces
        this  0.898/0.781
        gmm2  0.910/0.643   two-component mixture crossover: the
                            non-match component dominates, so posterior
                            0.5 lands inside the false mass
    The fitted rules reach 0.836/0.901. Refusing to fit costs about
    0.12 precision, and that is the honest price, not a rounding error.
    """
    v = np.asarray(sc, float)
    t, prev = -np.inf, -1.0
    for _ in range(cap):
        if len(v) < 8:
            break
        c = otsu(v)
        a, b = v[v < c], v[v >= c]
        if len(a) == 0 or len(b) == 0 or v.var() <= 0:
            break
        eta = (len(a) * len(b) / len(v) ** 2) * (a.mean() - b.mean()) ** 2 \
            / v.var()
        if eta < prev:
            break
        t, prev, v = c, eta, b
    return float(t)


def self_cut(q, wins=None):
    """Leave-one-out floor inside the query's own sub-windows.

    Only NON-OVERLAPPING pairs count. The sub-windows are cut at stride
    = half scale, so neighbours literally share half their frames -
    scoring one against the other is not a leave-one-out at all, it is a
    window compared with itself twice, and the floor it produces is a
    statement about the grid rather than about the event. Falls back to
    all pairs only when the clip is too short to contain a disjoint
    pair, which is reported by returning the looser value rather than
    pretending otherwise.
    """
    if len(q) < 2:
        return -np.inf
    S = q @ q.T
    np.fill_diagonal(S, -np.inf)
    if wins is not None and len(wins) == len(q):
        ok = np.zeros_like(S, bool)
        for i, (a0, a1, _s) in enumerate(wins):
            for j, (b0, b1, _t) in enumerate(wins):
                ok[i, j] = i != j and (b0 >= a1 - 1e-6 or b1 <= a0 + 1e-6)
        if ok.any(1).sum() >= 2:
            S = np.where(ok, S, -np.inf)
            rows = S.max(1)
            return float(rows[np.isfinite(rows)].min())
    return float(S.max(1).min())


def plan(st, Q, ref=None, rs=None):
    """Per-channel cut, weight and query-side hubness. No labels, no
    fitted constants: everything below comes from the query's pixels and
    the store's own vectors."""
    rs = rs or np.random.RandomState(0)
    if ref is None:
        sel = np.sort(rs.choice(st.n, min(REF_SAMPLE, st.n),
                                replace=False))
        ref = {c: st.col[c].take(sel) for c in CHANNELS}
    cut, w, qhub = {}, {}, {}
    for c in CHANNELS:
        q = Q.q[c]
        cut[c] = self_cut(q, Q.wins)
        R = q @ ref[c].T
        qhub[c] = float(np.sort(R, axis=1)[:, -min(16, R.shape[1]):]
                        .mean())
        inside = (float(np.mean((q @ q.T)[~np.eye(len(q), dtype=bool)]))
                  if len(q) > 1 else 0.0)
        w[c] = inside - float(R.mean())
    v = np.array([w[c] for c in CHANNELS], np.float64)
    if WMODE == "z":
        # z-score then clip. Observed to zero the weakest channel on
        # essentially EVERY query across all four corpora - which is an
        # artifact, not a finding: z-scoring three numbers puts their
        # mean at 0, so the minimum is always negative and always
        # clipped. "Selection over fusion" should mean selection on
        # evidence, not dropping one channel by construction.
        s = v.std()
        v = (v - v.mean()) / s if s > 1e-9 else np.ones_like(v)
    # Default: keep a channel when the query agrees with ITSELF more
    # than with random corpus content. That is a real per-query test
    # with a meaningful zero, it can keep any number of channels, and
    # the differencing cancels each channel's own baseline similarity
    # (which matters: c3's unrelated pairs already sit near 0.9).
    if DROP:
        v = np.array([0.0 if c in DROP else v[i]
                      for i, c in enumerate(CHANNELS)])
    v = np.clip(v, 0.0, None)
    if v.sum() <= 1e-9:
        v = np.ones_like(v)
    v = v / v.sum()
    return cut, {c: float(v[i]) for i, c in enumerate(CHANNELS)}, qhub


def candidates(st, Q, probe=CAND_FRAC, exclude=None, w=None):
    """Coarse prune over c1, then exact rerank of everything probed.

    `exclude` drops rows within a time separation of the query in the
    SAME media. That rule is the difference between measuring retrieval
    and measuring that a clip resembles the seconds either side of it.
    """
    keep = np.ones(st.n, bool)
    if exclude is not None:
        mid, a, b, sep = exclude
        near = (st.media == mid) & (st.t1 > a - sep) & (st.t0 < b + sep)
        keep &= ~near
    if not st.centroids or probe >= 1.0:
        return np.flatnonzero(keep)
    # Probe with the SET, not with its mean. The query's sub-windows are
    # separate points and they routinely fall in different cells; the
    # mean is a point that may sit in none of them. Probing the mean
    # measured 0.533 recall against an exact scan, which is elision
    # bought with answers - exactly what gate P exists to catch. Ranking
    # cells by their best match to ANY sub-window is both the correct
    # reading of "the example is a set" and the fix.
    # Probe the cells of the channels this query actually weights, and
    # union them. A row the ranking would score highly on c2 must not be
    # discarded because it sits far away in c1.
    live = [c for c in CHANNELS if w is None or w.get(c, 1.0) > 0]
    live = [c for c in live if c in st.centroids] or list(st.centroids)
    budget = max(int(probe * st.n / max(len(live), 1)), 64)
    sel = np.zeros(st.n, bool)
    for c in live:
        d = (_l2(st.centroids[c]) @ _l2(Q.q[c]).T).max(1)
        order = np.argsort(-d)
        size = np.bincount(st.cell[c], minlength=len(d))
        floor = float(d[order[0]] - d.std())
        hot, got = [], 0
        for j in order:
            hot.append(j)
            got += int(size[j])
            if got >= budget and d[j] < floor:
                break
        sel |= np.isin(st.cell[c], np.asarray(hot))
    sel &= keep
    if sel.sum() < 8:
        return np.flatnonzero(keep)
    return np.flatnonzero(sel)


def score(st, Q, idx, cut, w, qhub):
    """CSLS-corrected, per-channel, max over the query's sub-windows.

    The threshold must carry the SAME correction terms a candidate does.
    It previously subtracted the query's hubness twice while a candidate
    was charged its own hubness plus the query's - so whenever corpus
    rows are less hubby than the query (the normal case) every candidate
    was scored on a more generous scale than the bar it had to clear.
    Measured: the threshold landed at -0.4 to -0.7 while the FALSE score
    median was -0.11 to -0.45, so 866 of 873 non-matches passed and
    abstention never fired. Precision was then pinned at yield/1.5 by
    arithmetic, which is exactly the trap k-as-a-max-bound exists to
    avoid.
    """
    tot = np.zeros(len(idx), np.float64)
    thr = 0.0
    for i, c in enumerate(CHANNELS):
        if w[c] <= 0:
            continue
        A = st.col[c].take(idx)
        s = (A @ Q.q[c].T).max(1)
        h = st.hub[idx, i]
        tot += w[c] * (2.0 * s - h - qhub[c])
        # the hubness a typical candidate actually carries, read off this
        # query's own candidate pool - corpus-derived, nothing fitted
        thr += w[c] * (2.0 * cut[c] - float(np.median(h)) - qhub[c])
    return tot, thr


def spans(st, idx, sc, order):
    """Collapse overlapping survivors in one media into single answers.

    Without this a 2 s, a 4 s and an 8 s window covering one moment are
    three results, and both yield and precision become a statement about
    the window grid rather than about retrieval.
    """
    out = []
    for j in order:
        r = int(idx[j])
        m, a, b, s = st.media[r], st.t0[r], st.t1[r], float(sc[j])
        for o in out:
            if o[0] == m and a < o[2] and b > o[1]:
                o[1], o[2] = min(o[1], a), max(o[2], b)
                o[3] = max(o[3], s)
                break
        else:
            out.append([m, a, b, s])
    return [(m, float(a), float(b), float(s)) for m, a, b, s in out]


def search(st, Q, k=None, abstain=True, probe=CAND_FRAC, exclude=None,
           ref=None, merge=True):
    """One example -> ranked spans. The product entry point.

    `merge=False` returns the surviving store ROWS instead of merged
    spans. The benchmark needs both: rows carry real support (several
    windows cover one moment, across three scales), while spans are what
    a user is actually handed. Grading only spans would force support to
    1 and cap precision at 1/ceil(1.5) = 0.5 before the search even ran.
    """
    st.reset_bytes()
    cut, w, qhub = plan(st, Q, ref=ref)
    idx = candidates(st, Q, probe, exclude, w)
    if not len(idx):
        return [], dict(w=w, cut=cut, bytes=st.bytes_read, cand=0,
                        kept=0, thr=0.0)
    sc, _old = score(st, Q, idx, cut, w, qhub)
    thr = score_cut(sc)
    order = np.argsort(-sc)
    if abstain:
        order = order[sc[order] >= thr]
    if k is not None:
        order = order[:k]
    info = dict(w=w, cut=cut, thr=float(thr), cand=int(len(idx)),
                bytes=st.bytes_read, kept=int(len(order)))
    if not merge:
        return [(int(idx[j]), float(sc[j])) for j in order], info
    return spans(st, idx, sc, order), info


def main():
    from flowgebd import arg
    name = arg("--store", "sim")
    st = Store(STORES / name)
    mid = arg("--media", str(st.media[0]))
    t0, t1 = arg("--t0", 0.0, float), arg("--t1", 8.0, float)
    k = arg("--k", 10, int)
    src = {s.id: s for s in vsrc.sources(name)}[mid]
    Q = Query.from_media(src, t0, t1)
    res, info = search(st, Q, k=k,
                       exclude=(mid, t0, t1, vcore.CTX_MULT * max(vcore.SCALES)))
    print(f"{st!r}")
    print(f"query {mid} [{t0:.1f},{t1:.1f}]  {Q.m} sub-windows")
    print("weights " + "  ".join(f"{c} {info['w'][c]:.3f}"
                                 for c in CHANNELS)
          + "   cut " + "  ".join(f"{c} {info['cut'][c]:.3f}"
                                  for c in CHANNELS))
    print(f"candidates {info['cand']}/{st.n}  kept {info['kept']}  "
          f"read {info['bytes'] / 1e6:.2f} MB\n")
    for m, a, b, s in res:
        print(f"  {s:+.3f}  {m}  [{a:.1f}, {b:.1f}]")


if __name__ == "__main__":
    main()
