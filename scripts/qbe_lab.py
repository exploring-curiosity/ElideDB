"""QbE iteration lab: one harness, many strategies, identical metric.

Every lever aimed at the 0.90 target gets measured here, on the same
disjoint seed groups (RandomState(0), as bench_qbe_weighted uses), the
same k = ceil(1.5 x support), the same three high-support queries. A
strategy is a function (ctx, seeds, k) -> chosen episode indices; the
harness knows nothing else about it.

NO TRUTHSET FITTING. Strategies may read the corpus and the seeds; the
grades are consulted only after a strategy has committed its ranking.
That is why coherence and PRF are legal here and a per-query tuned
constant would not be.

    python scripts/qbe_lab.py --headroom
    python scripts/qbe_lab.py --run coh4,coh4_prf
    python scripts/qbe_lab.py --list
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                     # noqa: E402
from elidedb.qbe import spaces                                # noqa: E402

QUERIES = tuple(int(x) for x in __import__("os").environ.get(
    "ELIDEDB_QUERIES", "3,4,5").split(","))
NSEED = 5
NGROUP = 5
# k = KMULT x support. At 1.5 the metric is capped: prec <= yield/1.5,
# so "yield AND precision both 0.90" is arithmetically impossible there
# and only reachable at KMULT = 1, where returning exactly `support`
# items makes yield and precision the same number.
KMULT = float(__import__("os").environ.get("ELIDEDB_KMULT", "1.5"))
AGG = __import__("os").environ.get("ELIDEDB_AGG", "centroid")


class Ctx:
    """Everything a strategy may see: channels, corpus, seeds."""

    def __init__(self, store="lake/fresh_bench"):
        self.db = Store.open(str(ROOT / store))
        ep = self.db.table("episodes").scan()
        self.keys = [(str(s), int(a)) for s, a in
                     zip(ep.column("stream").to_pylist(),
                         ep.column("ts").to_pylist())]
        self.eidx = [int(i) for i in
                     ep.column("episode_index").to_pylist()]
        self.t1 = [int(v) for v in ep.column("t1").to_pylist()]
        skeys, SM = spaces(self.db, drop_pc=0)
        spos = {k: i for i, k in enumerate(skeys)}
        take = np.array([spos[k] for k in self.keys])
        self.M = {}
        for c, (A, _, op) in SM.items():
            A = np.asarray(A, np.float32)[take]
            self.M[c] = (A, np.abs(A).sum(1) > 0, op)
        self.n = len(self.keys)
        self.extra = {}
        self._fine()

    # ---------------------------------------------------------------
    def _fine(self):
        """FINE-GRAINED channels from tables the store already has.

        An episode-pooled vector is a room average: DINOv3 is the best
        appearance model here and `scene` scores 0.37 alone, because
        mean-pooling ~34 frames dissolves the object that the query is
        about. The frames and the per-track descriptors are both stored
        per element, so the finer granularity costs a matmul, not a
        model - this is the element investment expressed as retrieval.
        """
        import numpy as _np
        pos = {k: i for i, k in enumerate(self.keys)}
        ep = self.db.table("episodes").scan().to_pydict()
        spans = {}
        for s_, a, b in zip(ep["stream"], ep["ts"], ep["t1"]):
            spans.setdefault(str(s_), []).append((int(a), int(b)))
        for v in spans.values():
            v.sort()

        def _owner(stream, ts):
            lst = spans.get(str(stream))
            if not lst:
                return -1
            st = [a for a, _ in lst]
            j = int(_np.searchsorted(st, int(ts), "right")) - 1
            if j < 0 or int(ts) > lst[j][1]:
                return -1
            return pos.get((str(stream), lst[j][0]), -1)

        # sig2_vectors already holds 8 rows per episode - spaces()
        # mean-pools them, which is the same averaging the fine
        # channels exist to avoid, so it also gets a set-matched twin.
        for name, table in (("frame", "scene_vectors"),
                            ("objset", "object_vectors"),
                            ("sig2f", "sig2_vectors"),
                            ("iv2w", "iv2_win_vectors")):
            if table not in self.db.tables():
                continue
            d = self.db.table(table).scan().to_pydict()
            V = _np.asarray(d["vector"], _np.float32).reshape(
                len(d["ts"]), -1)
            V /= _np.maximum(_np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
            own = _np.array([_owner(s_, t) for s_, t in
                             zip(d["stream"], d["ts"])])
            keep = own >= 0
            V, own = V[keep], own[keep]
            ok = _np.zeros(self.n, bool)
            ok[_np.unique(own)] = True
            self.extra[name] = (V, own)
            self.M[name] = (None, ok, "fine")

        if "objset" in self.extra:
            m = (self.db.table("object_vectors").state().meta or {})
            self._objcut = float(m.get("match_cut", 0.85))
            ok = self.M["objset"][1]
            self.M["objcons"] = (None, ok, "cons")

        # participants that an EVENT bound - the significant objects,
        # not every blob the detector ever saw. In a one-kitchen corpus
        # "contains similar objects" is true of every pair, which is why
        # the unrestricted object set measured 0.35.
        if "objset" in self.extra and "events" in self.db.tables():
            ev = self.db.table("events").scan().to_pydict()
            sig = {(str(s_), int(o)) for s_, o in
                   zip(ev["stream"], ev["object_id"]) if int(o) >= 0}
            d = self.db.table("object_vectors").scan().to_pydict()
            V, own = self.extra["objset"]
            keep = _np.array([(str(s_), int(o)) in sig for s_, o in
                              zip(d["stream"], d["object_id"])])
            own_all = _np.array([_owner(s_, t) for s_, t in
                                 zip(d["stream"], d["ts"])])
            m = keep & (own_all >= 0)
            if m.sum() > 100:
                Vs = _np.asarray(d["vector"], _np.float32).reshape(
                    len(d["ts"]), -1)[m]
                Vs /= _np.maximum(_np.linalg.norm(Vs, axis=1, keepdims=True),
                                  1e-8)
                ok2 = _np.zeros(self.n, bool)
                ok2[_np.unique(own_all[m])] = True
                self.extra["objsig"] = (Vs, own_all[m])
                self.M["objsig"] = (None, ok2, "fine")
                self.M["objcons2"] = (None, ok2, "cons2")

    def objcons_score(self, seeds):
        """CONSENSUS OBJECTS: what the seeds share and the corpus does not.

        The low-support queries are object queries, and QbE "learns a
        kind" from seeds - but an episode-level embedding of a cluttered
        scene barely moves when one small object changes, which is why
        every appearance channel sits at 0.34-0.41. The OBJECT INDEX
        changes the question: with N seed episodes, the query's object
        is whichever object family appears in MOST of the seeds while
        being RARE across the corpus. Consensus x rarity - the lift of
        the object given the seeds - which is corpus statistics end to
        end: the match threshold is the store's own recurrence-fitted
        identity cut, read from table meta, and no color, name, or
        class appears anywhere.

        Episode score = best (lift-weighted) match to any consensus
        object. One matching spoon outranks thirty frames of the same
        kitchen, because nothing here ever averages over the scene.
        """
        V, own = self.extra["objset"]
        theta = self._objcut
        S = list(int(x) for x in seeds)
        seed_mask = np.isin(own, np.asarray(S))
        Q = V[seed_mask]
        q_ep = own[seed_mask]
        if not len(Q):
            return np.full(self.n, -np.inf)
        sims = V @ Q.T                          # all objects x seed objects
        hit = sims >= theta
        n_ep = max(len(set(own.tolist())), 1)
        # per seed-object: how many OTHER seed episodes hold a match,
        # and how much of the corpus does
        ep_of_col = q_ep
        seed_hits = np.zeros(Q.shape[0])
        corpus_eps = np.zeros(Q.shape[0])
        for j in range(Q.shape[0]):
            eps_j = set(own[hit[:, j]].tolist())
            corpus_eps[j] = len(eps_j)
            seed_hits[j] = len(eps_j & set(S) - {int(ep_of_col[j])})
        need = max(1, (len(S) - 1) // 2 + 1)    # majority of other seeds
        lift = ((seed_hits + 1) / len(S)) / (corpus_eps / n_ep + 1.0 / n_ep)
        keep = seed_hits >= need
        if not keep.any():
            keep = seed_hits >= 1               # fail soft, not silent
        if not keep.any():
            return np.full(self.n, -np.inf)
        w = lift[keep] / max(lift[keep].max(), 1e-9)
        sub = sims[:, keep] * w[None, :]        # lift-weighted similarity
        best = sub.max(1)
        out = np.full(self.n, -np.inf)
        order = np.argsort(own, kind="stable")
        o_sorted, s_sorted = own[order], best[order]
        bounds = np.searchsorted(o_sorted, np.arange(self.n + 1))
        for e in range(self.n):
            a, b = bounds[e], bounds[e + 1]
            if b > a:
                out[e] = float(s_sorted[a:b].max())
        return out

    def objcons2_score(self, seeds):
        """Rank-based consensus over EVENT-BOUND objects.

        The thresholded version failed for a measured reason: the
        identity cut (0.894) is an INSTANCE bar and object KINDS in one
        kitchen live at 0.7-0.85, below it, so the consensus set was
        empty and fail-soft admitted noise. Diagnosed gaps after
        restricting BOTH sides to event-bound (manipulated) objects:
        cross-seed 0.77-0.85 vs background 0.71-0.72. Thin, but real -
        so no threshold at all: each seed object ranks every episode by
        its best event-bound match, its WEIGHT is how highly it ranks
        the other seed episodes (pairwise-LOO at object level), and the
        episode score is the best weighted rank. Statistics of the
        query and corpus only.
        """
        Vs, owns = self.extra.get("objsig", (None, None))
        if Vs is None:
            return np.full(self.n, -np.inf)
        S = list(int(x) for x in seeds)
        sm = np.isin(owns, np.asarray(S))
        Q = Vs[sm]
        q_ep = owns[sm]
        if not len(Q):
            return np.full(self.n, -np.inf)
        sims = Vs @ Q.T
        order = np.argsort(owns, kind="stable")
        o_sorted = owns[order]
        bounds = np.searchsorted(o_sorted, np.arange(self.n + 1))
        ep_max = np.full((self.n, Q.shape[0]), -np.inf)
        s_sorted = sims[order]
        for e in range(self.n):
            a, b = bounds[e], bounds[e + 1]
            if b > a:
                ep_max[e] = s_sorted[a:b].max(0)
        # percentile rank of each episode under each seed object
        R = np.full_like(ep_max, np.nan)
        for j in range(Q.shape[0]):
            col = ep_max[:, j]
            f = np.isfinite(col)
            if f.sum() > 2:
                r = col[f].argsort().argsort() / (f.sum() - 1)
                R[f, j] = r
        # weight: how highly does this object rank its SIBLING seeds
        w = np.zeros(Q.shape[0])
        for j in range(Q.shape[0]):
            sib = [e for e in S if e != int(q_ep[j])]
            vals = [R[e, j] for e in sib if np.isfinite(R[e, j])]
            w[j] = np.mean(vals) if vals else 0.0
        if w.max() <= 0:
            return np.full(self.n, -np.inf)
        w = w / w.max()
        with np.errstate(invalid="ignore"):
            sc = np.nanmax(R * w[None, :], 1)
        sc = np.where(np.isnan(sc), -np.inf, sc)
        return sc

    def _fine_score(self, c, seeds, topk=5, mode="max"):
        """SET-TO-SET: each candidate part scores against its NEAREST
        seed part, then the episode takes the top-k mean.

        The centroid version of this measured 0.34/0.30/0.57 - no
        better than the pooled channel it was meant to replace, because
        averaging every seed frame into one query vector reproduces the
        room average on the QUERY side. A query is a set of parts, not
        their mean; matching set to set is what keeps one object from
        being averaged away by thirty frames of background.
        """
        V, own = self.extra[c]
        sel = np.isin(own, np.asarray(seeds))
        if sel.sum() < 1:
            return np.full(self.n, -np.inf)
        Q = V[sel]
        if mode == "max":
            sim = (V @ Q.T).max(1)
        else:
            q = Q.mean(0)
            nq = np.linalg.norm(q)
            if nq <= 0:
                return np.full(self.n, -np.inf)
            sim = V @ (q / nq)
        out = np.full(self.n, -np.inf)
        order = np.argsort(own, kind="stable")
        o_sorted, s_sorted = own[order], sim[order]
        bounds = np.searchsorted(o_sorted, np.arange(self.n + 1))
        for e in range(self.n):
            a, b = bounds[e], bounds[e + 1]
            if b > a:
                v = s_sorted[a:b]
                m = min(topk, len(v))
                out[e] = float(np.sort(v)[-m:].mean())
        return out

    def score(self, c, seeds, agg=None):
        """A channel's similarity of every episode to a seed set.

        agg="centroid" averages the seeds into one query vector;
        agg="max" scores against the NEAREST seed. A query of five
        clips is a set, and the centroid of a set whose members differ
        (five ways of closing a drawer) is a vector describing none of
        them - the same averaging that made the fine-grained channels
        useless until they matched set to set.
        """
        A, ok, op = self.M[c]
        agg = agg or AGG
        if op == "cons":
            s = self.objcons_score(seeds)
            s[~ok] = -np.inf
            return s
        if op == "cons2":
            s = self.objcons2_score(seeds)
            s[~ok] = -np.inf
            return s
        if op == "fine":
            s = self._fine_score(c, seeds)
            s[~ok] = -np.inf
            return s
        if op == "crossent":
            L = np.log(np.maximum(A, 1e-12))
            s = (L @ A[seeds].T).max(1) if agg == "max" \
                else L @ A[seeds].mean(0)
        elif agg == "max":
            S = A[seeds]
            S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-8)
            s = (A @ S.T).max(1)
        else:
            cen = A[seeds].mean(0)
            nrm = np.linalg.norm(cen)
            s = A @ (cen / nrm) if nrm > 0 else np.zeros(len(A))
        s = np.asarray(s, np.float64).copy()
        s[~ok] = -np.inf
        return s

    def coherence(self, c, seeds):
        """Cohen's d of seed-seed vs seed-corpus similarity. Query-only."""
        A, ok, _ = self.M[c]
        # coherence needs an episode-vector space; the fine-grained
        # channels have no episode vector by design (that pooling is
        # what they exist to avoid), so they are simply not selectable
        # by the older statistic. LOO scores them fine - it only needs
        # score().
        if A is None or not ok[seeds].all():
            return 0.0
        S = A[seeds]
        W = S @ S.T
        iu = np.triu_indices(len(seeds), 1)
        if not len(iu[0]):
            return 0.0
        within = W[iu]
        other = np.setdiff1d(np.where(ok)[0], seeds)
        if len(other) < 10:
            return 0.0
        across = (S @ A[other].T).ravel()
        sd = np.sqrt((within.var() + across.var()) / 2) or 1e-9
        return float((within.mean() - across.mean()) / sd)


def loo_quality(ctx, seeds, c, kind="mrr"):
    """LEAVE-ONE-SEED-OUT retrieval quality of a channel. Query-only.

    Coherence (Cohen's d) asks "are the seeds tight?", which is a proxy
    and a noisy one: measured over 15 seed groups it named the wrong
    channel 4 times, and each miss cost ~0.4 yield because motion kept
    yield 0.88-0.95 while its d fell from 2.5 to 0.76. Tightness is not
    retrieval.

    This asks the question we actually care about, and the seed set can
    answer it: hold out one seed, query with the rest, and see where the
    held-out seed lands. A channel that ranks the query's own members
    highly is the channel that retrieves this query type. No grade is
    read - the seeds ARE the query, and their mutual membership is
    given by the user, not by the truthset.
    """
    _, ok, op = ctx.M[c]
    if not ok[seeds].all() or len(seeds) < 3:
        return 0.0
    if kind == "pair":
        # PAIRWISE: every ordered seed pair, not every hold-one-out.
        # With 5 seeds LOO has 5 samples to judge a channel on and
        # mis-selects roughly one group in five; pairs give 20 from the
        # same query at the same cost per sample.
        out = []
        for i in seeds:
            for j in seeds:
                if i == j:
                    continue
                sc = ctx.score(c, np.asarray([j]))
                sc = sc.copy()
                sc[np.asarray([x for x in seeds if x != i])] = -np.inf
                r = int((sc > sc[i]).sum())
                from elidedb.qbe import loo_depths
                out.append(float(np.mean([r < d for d in
                                          loo_depths(ctx.n)])))
        return float(np.mean(out))
    out = []
    for i in seeds:
        rest = np.asarray([j for j in seeds if j != i])
        s = ctx.score(c, rest).copy()
        s[rest] = -np.inf                       # other seeds not candidates
        r = int((s > s[i]).sum())               # rank of the held-out seed
        if kind == "mrr":
            out.append(1.0 / (1.0 + r))
        elif kind == "log":
            out.append(-np.log1p(r))
        elif kind == "auc":
            # DEPTH-FREE: mean recall over a geometric ladder of depths
            # is the area under the LOO recall curve; the ladder itself
            # is the SHARED corpus-relative one from qbe (audit: the
            # absolute 25..800 form was 2,097-episode shape knowledge)
            from elidedb.qbe import loo_depths
            out.append(float(np.mean([r < d for d in loo_depths(ctx.n)])))
        else:                                   # recall at a FIXED depth
            # deliberately not k: k is 1.5 x support and support comes
            # from the grades, so selecting on it would leak the answer
            # key into the query. A constant depth is a property of the
            # harness, not of the query's truth.
            out.append(float(r < int(kind)))
    return float(np.mean(out))


def rrf(ranklists, k=60.0):
    tot = None
    for r in ranklists:
        s = 1.0 / (k + r)
        tot = s if tot is None else tot + s
    return tot


def ranks_of(s):
    r = np.empty(len(s))
    r[np.argsort(-s)] = np.arange(len(s))
    return r


def weighted_fuse(ctx, seeds, w):
    """Weighted RRF over channels; weights from coherence (query-only)."""
    lists, ws = [], []
    for c, weight in w.items():
        if weight <= 0:
            continue
        s = ctx.score(c, seeds)
        lists.append(ranks_of(s))
        ws.append(weight)
    if not lists:
        return np.zeros(ctx.n)
    tot = None
    for r, weight in zip(lists, ws):
        v = weight / (60.0 + r)
        tot = v if tot is None else tot + v
    return tot


def coh_weights(ctx, seeds, power):
    return {c: max(ctx.coherence(c, seeds), 0.0) ** power for c in ctx.M}


# ---------------------------------------------------------------- strategies
def S_uniform(ctx, seeds, k):
    f = weighted_fuse(ctx, seeds, {c: 1.0 for c in ctx.M})
    f[seeds] = -np.inf
    return np.argsort(-f)[:k]


def _coh(power):
    def f(ctx, seeds, k):
        fused = weighted_fuse(ctx, seeds, coh_weights(ctx, seeds, power))
        fused[seeds] = -np.inf
        return np.argsort(-fused)[:k]
    return f


def _prf(power, m, rounds=1):
    """Rocchio: the head of pass one becomes additional seeds.

    Corpus-derived by construction - it never sees a grade, only the
    system's own top results. Standard IR, and the one lever that
    addresses 'the truth is in the candidate set but mis-ordered'.
    """
    def f(ctx, seeds, k):
        cur = np.asarray(seeds)
        for _ in range(rounds):
            fused = weighted_fuse(ctx, cur, coh_weights(ctx, cur, power))
            fused[seeds] = -np.inf
            top = np.argsort(-fused)[:m]
            cur = np.unique(np.concatenate([np.asarray(seeds), top]))
        # RESCORE WITH THE EXPANSION. The first cut of this returned the
        # ranking computed BEFORE the expansion - the feedback was built
        # and discarded, and every PRF row printed its baseline's exact
        # numbers. Identical output across a swept parameter is the tell.
        fused = weighted_fuse(ctx, cur, coh_weights(ctx, cur, power))
        fused[seeds] = -np.inf
        return np.argsort(-fused)[:k]
    return f


def _best_single(power=1):
    """The single most COHERENT channel, alone.

    Measured: the best channel beats the 5-channel fusion on every high
    -support query - iv2 0.77 vs fused 0.64 on q03, motion 0.93 vs 0.76
    on q04, 0.88 vs 0.83 on q05 - and coherence names it correctly each
    time (iv2 for the appearance query, motion for the two direction
    queries). RRF dilutes: four wrong voters outrank one right one.
    """
    def f(ctx, seeds, k):
        coh = {c: ctx.coherence(c, seeds) for c in ctx.M}
        b = max(coh, key=coh.get)
        s = ctx.score(b, seeds).copy()
        s[seeds] = -np.inf
        return np.argsort(-s)[:k]
    return f


def _topn(n, power=4):
    """The n most coherent channels, RRF'd, coherence-weighted."""
    def f(ctx, seeds, k):
        coh = {c: max(ctx.coherence(c, seeds), 0.0) for c in ctx.M}
        keep = sorted(coh, key=coh.get, reverse=True)[:n]
        w = {c: coh[c] ** power for c in keep}
        fused = weighted_fuse(ctx, seeds, w)
        fused[seeds] = -np.inf
        return np.argsort(-fused)[:k]
    return f


def _best_prf(m, rounds=1):
    """Best-single, then Rocchio on that channel alone."""
    def f(ctx, seeds, k):
        cur = np.asarray(seeds)
        for _ in range(rounds):
            coh = {c: ctx.coherence(c, cur) for c in ctx.M}
            b = max(coh, key=coh.get)
            s = ctx.score(b, cur).copy()
            s[seeds] = -np.inf
            cur = np.unique(np.concatenate([np.asarray(seeds),
                                            np.argsort(-s)[:m]]))
        coh = {c: ctx.coherence(c, cur) for c in ctx.M}
        b = max(coh, key=coh.get)
        s = ctx.score(b, cur).copy()
        s[seeds] = -np.inf
        return np.argsort(-s)[:k]
    return f


def _loo_best(kind="mrr"):
    def f(ctx, seeds, k):
        q = {c: loo_quality(ctx, seeds, c, kind) for c in ctx.M}
        b = max(q, key=q.get)
        s = ctx.score(b, seeds).copy()
        s[seeds] = -np.inf
        return np.argsort(-s)[:k]
    return f


def _loo_w(power, kind="mrr"):
    def f(ctx, seeds, k):
        q = {c: max(loo_quality(ctx, seeds, c, kind), 0.0) for c in ctx.M}
        mx = max(q.values()) or 1.0
        w = {c: (v / mx) ** power for c, v in q.items()}
        fused = weighted_fuse(ctx, seeds, w)
        fused[seeds] = -np.inf
        return np.argsort(-fused)[:k]
    return f


def zfuse(ctx, seeds, w):
    """SCORE-level fusion: z-normalise each channel, then weighted sum.

    RRF throws the margin away - it only knows a channel put something
    first, not by how much. That cost the text path 0.38 -> 0.27 when
    ITM was made a voter. For QbE the margin is exactly the evidence
    that separates a confident channel from a guessing one.
    """
    tot = None
    for c, weight in w.items():
        if weight <= 0:
            continue
        s = ctx.score(c, seeds)
        f = np.isfinite(s)
        z = np.full(len(s), 0.0)
        if f.sum() > 2:
            mu, sd = s[f].mean(), s[f].std() or 1e-9
            z[f] = (s[f] - mu) / sd
        tot = weight * z if tot is None else tot + weight * z
    return tot if tot is not None else np.zeros(ctx.n)


def _loo_z(power, kind="auc"):
    def f(ctx, seeds, k):
        q = {c: max(loo_quality(ctx, seeds, c, kind), 0.0) for c in ctx.M}
        mx = max(q.values()) or 1.0
        w = {c: (v / mx) ** power for c, v in q.items()}
        fused = zfuse(ctx, seeds, w)
        fused[seeds] = -np.inf
        return np.argsort(-fused)[:k]
    return f


def _two_stage(pool_mult, power=8, kind="auc"):
    """Best channel proposes a deep pool; the rest RERANK inside it.

    Recall at 3x support is 0.89-0.96 while yield at 1.5x is 0.77-0.93,
    so the truth is in a reachable pool and mis-ordered inside it. The
    weak channels are bad at FINDING but may still be good at ORDERING
    a pool the strong channel already made relevant.
    """
    def f(ctx, seeds, k):
        q = {c: max(loo_quality(ctx, seeds, c, kind), 0.0) for c in ctx.M}
        b = max(q, key=q.get)
        s = ctx.score(b, seeds).copy()
        s[seeds] = -np.inf
        pool = np.argsort(-s)[:int(k * pool_mult)]
        mx = max(q.values()) or 1.0
        w = {c: (q[c] / mx) ** power for c in ctx.M}
        fused = zfuse(ctx, seeds, w)
        order = pool[np.argsort(-fused[pool])]
        return order[:k]
    return f


def _loo_prf(m, power=8, rounds=1):
    """LOO-weighted fusion, then Rocchio with the same weighting."""
    def f(ctx, seeds, k):
        cur = np.asarray(seeds)
        for _ in range(rounds + 1):
            q = {c: max(loo_quality(ctx, seeds, c), 0.0) for c in ctx.M}
            mx = max(q.values()) or 1.0
            w = {c: (v / mx) ** power for c, v in q.items()}
            fused = zfuse(ctx, cur, w)
            fused[seeds] = -np.inf
            cur = np.unique(np.concatenate([np.asarray(seeds),
                                            np.argsort(-fused)[:m]]))
        return np.argsort(-fused)[:k]
    return f


_ANS = {}


def answer_scores(ctx, seed):
    """The structural join's score for one seed episode, cached."""
    if seed not in _ANS:
        from elidedb.answer import answer_like
        st, a = ctx.keys[int(seed)]
        b = ctx.t1[int(seed)]
        _, sc, shared = answer_like(ctx.db, st, a, b)
        _ANS[seed] = (np.asarray(sc, np.float64), np.asarray(shared))
    return _ANS[seed]


def _join_rerank(pool_mult, power=16, use_partition=True):
    """High-recall pool from the channels, RERANKED by the element join.

    The two engines fail in opposite directions, and that is the whole
    argument for composing them: the channels reach recall 0.89-0.96 at
    3x support but mis-order inside it, while the join reaches only
    0.37 recall yet is right about what it does return (prec_g
    0.90-0.94). Precision applied where recall already exists.
    """
    def f(ctx, seeds, k):
        q = {c: max(loo_quality(ctx, seeds, c, "auc"), 0.0) for c in ctx.M}
        mx = max(q.values()) or 1.0
        w = {c: (v / mx) ** power for c, v in q.items()}
        base = zfuse(ctx, seeds, w)
        base[seeds] = -np.inf
        pool = np.argsort(-base)[:int(k * pool_mult)]
        acc = np.zeros(ctx.n)
        part = np.zeros(ctx.n)
        for sd in seeds:
            sc, shared = answer_scores(ctx, sd)
            acc += sc
            part += shared.astype(float)
        acc /= max(len(seeds), 1)
        part /= max(len(seeds), 1)
        # rank inside the pool: the join's conjunction, with its event
        # -kind partition as the primary key when it is available
        if use_partition:
            order = pool[np.lexsort((-acc[pool], -part[pool]))]
        else:
            order = pool[np.argsort(-acc[pool])]
        return order[:k]
    return f


def _join_blend(pool_mult, power=16):
    """Pool from channels; order by channel-z + join-z (both scaled)."""
    def f(ctx, seeds, k):
        q = {c: max(loo_quality(ctx, seeds, c, "auc"), 0.0) for c in ctx.M}
        mx = max(q.values()) or 1.0
        w = {c: (v / mx) ** power for c, v in q.items()}
        base = zfuse(ctx, seeds, w)
        base[seeds] = -np.inf
        pool = np.argsort(-base)[:int(k * pool_mult)]
        acc = np.zeros(ctx.n)
        for sd in seeds:
            acc += answer_scores(ctx, sd)[0]
        acc /= max(len(seeds), 1)
        def z(v, idx):
            x = v[idx]
            return (x - x.mean()) / (x.std() or 1e-9)
        blend = z(base, pool) + z(acc, pool)
        return pool[np.argsort(-blend)][:k]
    return f


STRATEGIES = {
    "uniform": S_uniform,
    "join_rr2": _join_rerank(2), "join_rr3": _join_rerank(3),
    "join_rr5": _join_rerank(5),
    "join_rr3_np": _join_rerank(3, use_partition=False),
    "join_bl2": _join_blend(2), "join_bl3": _join_blend(3),
    "join_bl5": _join_blend(5),
    "auc_z12": _loo_z(12), "auc_z20": _loo_z(20), "auc_z24": _loo_z(24),
    "z16_prf25": _loo_prf(25, 16), "z16_prf50": _loo_prf(50, 16),
    "loo_best": _loo_best(), "loo_best_log": _loo_best("log"),
    "loo_best_r50": _loo_best("50"), "loo_best_r100": _loo_best("100"),
    "loo_best_r200": _loo_best("200"), "loo_best_r400": _loo_best("400"),
    "loo_best_auc": _loo_best("auc"),
    "pair_best": _loo_best("pair"),
    "pair_z8": _loo_z(8, "pair"), "pair_z12": _loo_z(12, "pair"),
    "pair_z16": _loo_z(16, "pair"), "pair_z24": _loo_z(24, "pair"),
    "auc_w4": _loo_w(4, "auc"), "auc_w8": _loo_w(8, "auc"),
    "auc_w16": _loo_w(16, "auc"),
    "auc_z2": _loo_z(2), "auc_z4": _loo_z(4), "auc_z8": _loo_z(8),
    "two_stage2": _two_stage(2), "two_stage3": _two_stage(3),
    "two_stage5": _two_stage(5),
    "auc_z16": _loo_z(16), "auc_z32": _loo_z(32), "auc_z64": _loo_z(64),
    "ts5_z16": _two_stage(5, 16), "ts5_z32": _two_stage(5, 32),
    "ts3_z16": _two_stage(3, 16),
    "loo_log_w2": _loo_w(2, "log"), "loo_log_w4": _loo_w(4, "log"),
    "loo_log_w8": _loo_w(8, "log"),
    "r100_w4": _loo_w(4, "100"), "r100_w8": _loo_w(8, "100"),
    "loo_w4": _loo_w(4), "loo_w8": _loo_w(8), "loo_w16": _loo_w(16),
    "loo_w32": _loo_w(32),
    "loo_prf25": _loo_prf(25), "loo_prf50": _loo_prf(50),
    "loo_prf100": _loo_prf(100),
    "best_single": _best_single(),
    "top2": _topn(2), "top3": _topn(3),
    "best_prf10": _best_prf(10), "best_prf25": _best_prf(25),
    "best_prf50": _best_prf(50), "best_prf100": _best_prf(100),
    "best_prf50x2": _best_prf(50, rounds=2),
    "coh2": _coh(2), "coh4": _coh(4), "coh8": _coh(8), "coh16": _coh(16),
    "coh4_prf10": _prf(4, 10), "coh4_prf25": _prf(4, 25),
    "coh4_prf50": _prf(4, 50), "coh4_prf100": _prf(4, 100),
    "coh4_prf25x2": _prf(4, 25, rounds=2),
    "coh8_prf25": _prf(8, 25), "coh8_prf50": _prf(8, 50),
}


# ---------------------------------------------------------------- harness
def truth(ctx):
    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    G, have = {}, set(ctx.eidx)
    sup = {}
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        ei = int(ei)
        G[(int(q), ei)] = int(v)
        if ei in have:
            sup[int(q)] = sup.get(int(q), 0) + int(v)
    return G, sup


def seed_groups(ctx, G, qi):
    truths = [i for i, e in enumerate(ctx.eidx) if G.get((qi, e)) == 1]
    rs = np.random.RandomState(0)
    ns = min(NSEED, len(truths))
    return [np.asarray(sorted(set(int(x) for x in
                                  rs.choice(truths, ns, replace=False))))
            for _ in range(NGROUP)], len(truths)


def evaluate(ctx, G, name, fn):
    rows = []
    for qi in QUERIES:
        groups, sup = seed_groups(ctx, G, qi)
        k = int(np.ceil(sup * KMULT))
        ys, ps, pgs = [], [], []
        for seeds in groups:
            chosen = fn(ctx, seeds, k)
            lab = [G.get((qi, ctx.eidx[int(i)])) for i in chosen]
            tr = sum(1 for x in lab if x == 1)
            ng = sum(1 for x in lab if x is not None)
            ys.append(tr / sup)
            ps.append(tr / max(len(chosen), 1))
            if ng:
                pgs.append(tr / ng)
        rows.append({"q": qi, "support": sup, "k": k,
                     "yield": float(np.mean(ys)),
                     "prec": float(np.mean(ps)),
                     "prec_g": float(np.mean(pgs)) if pgs else None})
    return {"strategy": name, "per_q": rows,
            "mean_yield": float(np.mean([r["yield"] for r in rows])),
            "mean_prec_g": float(np.mean([r["prec_g"] or 0 for r in rows]))}


def headroom(ctx, G):
    """Is the truth reachable at all? Recall at growing k, per channel
    and fused. A ranking problem and a representation problem need
    different fixes, and this is what tells them apart."""
    print("RECALL AT GROWING k (fused coh^4) - the ceiling")
    print(f"  {'q':<5}{'sup':>5}" + "".join(f"{f'{m}x':>8}"
                                            for m in (1, 1.5, 3, 5, 10, 20)))
    for qi in QUERIES:
        groups, sup = seed_groups(ctx, G, qi)
        line = f"  q{qi:02d}{sup:>6}"
        for mult in (1, 1.5, 3, 5, 10, 20):
            k = int(np.ceil(sup * mult))
            rec = []
            for seeds in groups:
                fused = weighted_fuse(ctx, seeds, coh_weights(ctx, seeds, 4))
                fused[seeds] = -np.inf
                ch = np.argsort(-fused)[:k]
                tr = sum(1 for i in ch if G.get((qi, ctx.eidx[int(i)])) == 1)
                rec.append(tr / sup)
            line += f"{np.mean(rec):>8.2f}"
        print(line, flush=True)

    print("\nPER-CHANNEL yield at k=1.5x (which channel actually carries)")
    cols = sorted(ctx.M)
    print(f"  {'q':<5}" + "".join(f"{c[:9]:>11}" for c in cols))
    for qi in QUERIES:
        groups, sup = seed_groups(ctx, G, qi)
        k = int(np.ceil(sup * 1.5))
        line = f"  q{qi:02d} "
        for c in cols:
            ys = []
            for seeds in groups:
                s = ctx.score(c, seeds).copy()
                s[seeds] = -np.inf
                ch = np.argsort(-s)[:k]
                ys.append(sum(1 for i in ch
                              if G.get((qi, ctx.eidx[int(i)])) == 1) / sup)
            line += f"{np.mean(ys):>11.2f}"
        print(line, flush=True)


def main():
    argv = sys.argv
    ctx = Ctx()
    G, _ = truth(ctx)
    print(f"channels: {sorted(ctx.M)}   episodes: {ctx.n:,}\n", flush=True)
    if "--list" in argv:
        print("strategies:", ", ".join(STRATEGIES))
        return
    if "--headroom" in argv:
        headroom(ctx, G)
        return
    want = (argv[argv.index("--run") + 1].split(",") if "--run" in argv
            else list(STRATEGIES))
    out = []
    print(f"{'strategy':<16}" + "".join(f"{f'q{q:02d}':>13}" for q in QUERIES)
          + f"{'mean_y':>9}{'mean_pg':>9}")
    for name in want:
        r = evaluate(ctx, G, name, STRATEGIES[name])
        out.append(r)
        cells = "".join(f"{x['yield']:>7.2f}/{(x['prec_g'] or 0):<5.2f}"
                        for x in r["per_q"])
        print(f"{name:<16}{cells}{r['mean_yield']:>9.3f}"
              f"{r['mean_prec_g']:>9.2f}", flush=True)
    (ROOT / "bench/qbe_lab.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
