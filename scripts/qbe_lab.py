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

QUERIES = (3, 4, 5)
NSEED = 5
NGROUP = 5
# k = KMULT x support. At 1.5 the metric is capped: prec <= yield/1.5,
# so "yield AND precision both 0.90" is arithmetically impossible there
# and only reachable at KMULT = 1, where returning exactly `support`
# items makes yield and precision the same number.
KMULT = float(__import__("os").environ.get("ELIDEDB_KMULT", "1.5"))


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
        skeys, SM = spaces(self.db, drop_pc=0)
        spos = {k: i for i, k in enumerate(skeys)}
        take = np.array([spos[k] for k in self.keys])
        self.M = {}
        for c, (A, _, op) in SM.items():
            A = np.asarray(A, np.float32)[take]
            self.M[c] = (A, np.abs(A).sum(1) > 0, op)
        self.n = len(self.keys)

    def score(self, c, seeds):
        """A channel's similarity of every episode to a seed set."""
        A, ok, op = self.M[c]
        if op == "crossent":
            pm = A[seeds].mean(0)
            s = np.log(np.maximum(A, 1e-12)) @ pm
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
        if not ok[seeds].all():
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
    A, ok, op = ctx.M[c]
    if not ok[seeds].all() or len(seeds) < 3:
        return 0.0
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
            # is the area under the LOO recall curve. Picking one depth
            # scored 0.598 at 50 and 0.811 at 200 - a 0.2 swing on a
            # constant I would otherwise be choosing by looking at the
            # answer, which is the definition of fitting the truthset.
            out.append(float(np.mean([r < d for d in
                                      (25, 50, 100, 200, 400, 800)])))
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
            fused = weighted_fuse(ctx, cur, w)
            fused[seeds] = -np.inf
            cur = np.unique(np.concatenate([np.asarray(seeds),
                                            np.argsort(-fused)[:m]]))
        return np.argsort(-fused)[:k]
    return f


STRATEGIES = {
    "uniform": S_uniform,
    "loo_best": _loo_best(), "loo_best_log": _loo_best("log"),
    "loo_best_r50": _loo_best("50"), "loo_best_r100": _loo_best("100"),
    "loo_best_r200": _loo_best("200"), "loo_best_r400": _loo_best("400"),
    "loo_best_auc": _loo_best("auc"),
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
    return [np.asarray(sorted(set(int(x) for x in
                                  rs.choice(truths, NSEED, replace=False))))
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
