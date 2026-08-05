"""ABSTENTION: k is a MAX BOUND, not the number returned.

Returning exactly k forces prec = yield/1.5 by arithmetic - that was a
harness bug, not a property of the metric. With a cut, a system that
returns ~support items that are nearly all true scores yield ~ prec ~
1.0, which is the actual target.

The cut needs no truth. The seed examples ARE labelled positives, so
leave-one-seed-out says what a TRUE match scores under this very
query; anything scoring below that is not worth returning. Rules
measured here, all query-time:

    fullk    return k                      (the old harness behaviour)
    loo      score >= min LOO seed score
    loo_p25  score >= 25th pct of LOO seed scores
    gap      cut at the largest relative score drop before k
    mad      score >= median + 3*MAD of the candidate tail

    python native/abstain.py --store lake/sim_chains
    python native/abstain.py --store lake/fresh_bench --q 3,4,5
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from mem import load_channels, build_reps, arg                  # noqa: E402
from memfuse import per_seed_scores, combine                    # noqa: E402

RULES = ("fullk", "loo", "loo_p25", "gap", "mad")


def loo_seed_scores(reps, eps, seeds):
    """Score each seed using the OTHER seeds - a truth-free sample of
    what a real positive scores for this query."""
    pos = {e: i for i, e in enumerate(eps)}
    out = []
    for h in seeds:
        rest = [s for s in seeds if s != h]
        if not rest or h not in pos:
            continue
        S = per_seed_scores(reps, eps, rest)
        tot = combine(S, rest, eps, "mean", "all_rrf")
        out.append(float(tot[pos[h]]))
    return np.array(out, np.float64)


def apply_rule(rule, order_scores, k, loo):
    """-> how many of the top-k to actually return."""
    s = order_scores[:k]
    if rule == "fullk" or len(s) == 0:
        return len(s)
    if rule.startswith("loo"):
        if len(loo) == 0:
            return len(s)
        bar = loo.min() if rule == "loo" else np.percentile(loo, 25)
        n = int((s >= bar).sum())
        return max(n, 1)
    if rule == "gap":
        if len(s) < 4:
            return len(s)
        d = s[:-1] - s[1:]
        lo = max(int(0.15 * len(s)), 2)
        j = lo + int(np.argmax(d[lo:]))
        return j + 1
    if rule == "mad":
        med = np.median(s)
        mad = np.median(np.abs(s - med)) + 1e-12
        n = int((s >= med + 3 * mad).sum())
        return max(n, 1)
    return len(s)


def evaluate(reps, eps, tasks, kmult=1.5):
    """tasks: [(label, seeds, positives_set, support)] -> per-rule
    mean yield / prec."""
    res = {r: ([], []) for r in RULES}
    per_task = {r: [] for r in RULES}
    for name, sd, truth, support in tasks:
        k = math.ceil(kmult * support)
        S = per_seed_scores(reps, eps, sd)
        tot = combine(S, sd, eps, "mean", "all_rrf")
        pos = {e: i for i, e in enumerate(eps)}
        for x in sd:
            if x in pos:
                tot[pos[x]] = -1e9
        order = np.argsort(-tot)
        ranked = [eps[i] for i in order]
        scores = tot[order]
        loo = loo_seed_scores(reps, eps, sd)
        for rule in RULES:
            n = apply_rule(rule, scores, k, loo)
            got = ranked[:n]
            tr = sum(1 for e in got if e in truth)
            y = tr / support
            p = tr / max(len(got), 1)
            res[rule][0].append(y)
            res[rule][1].append(p)
            per_task[rule].append((name, y, p, len(got), support))
    return res, per_task


def main():
    from elidedb import Store
    import pyarrow.parquet as pq
    store = ROOT / arg("--store", "lake/sim_chains")
    db = Store.open(str(store))
    reps = build_reps(load_channels(db, store.name))
    eps = sorted({e for R in reps.values() for e in R["pool"]})
    print(f"{store.name}: {len(eps)} episodes", flush=True)

    tasks = []
    rs = np.random.RandomState(0)
    if store.name == "sim_chains":
        t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
            .to_pydict()
        lab = {int(e): tm for e, tm in zip(t["episode"],
                                           t["template"])}
        groups = {}
        for e, tm in lab.items():
            if e in set(eps):
                groups.setdefault(tm, []).append(e)
        for tm, pool in sorted(groups.items()):
            if len(pool) < 6:
                continue
            sd = sorted(int(x) for x in rs.choice(pool, 5,
                                                  replace=False))
            truth = set(pool) - set(sd)
            tasks.append((tm, sd, truth, len(truth)))
    else:
        want = [int(x) for x in arg("--q", "3,4,5").split(",")]
        ep = db.table("episodes").scan()
        eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
        t = pq.read_table(ROOT / "eval/truthsets/graded.parquet") \
            .to_pydict()
        G = {}
        for q, ei, v in zip(t["query_id"], t["episode_index"],
                            t["true"]):
            G[(int(q), int(ei))] = int(v)
        for qi in want:
            truths = [e for e in eidx if G.get((qi, e)) == 1]
            if len(truths) < 6:
                continue
            sd = sorted(int(x) for x in rs.choice(truths, 5,
                                                  replace=False))
            truth = set(truths) - set(sd)
            tasks.append((f"q{qi}", sd, truth, len(truth)))

    for kmult in (1.5,):
        res, per_task = evaluate(reps, eps, tasks, kmult)
        print(f"\nk = {kmult} x support (MAX bound; abstention allowed)")
        print(f"  {'rule':<9}{'yield':<9}{'prec':<9}")
        for rule in RULES:
            ys, ps = res[rule]
            print(f"  {rule:<9}{np.mean(ys):<9.3f}{np.mean(ps):<9.3f}")
        best = max(RULES, key=lambda r: min(np.mean(res[r][0]),
                                            np.mean(res[r][1])))
        print(f"  best-by-min-metric: {best}")
        for name, y, p, n, sup in per_task[best]:
            print(f"     {name:<20} yield {y:.2f} prec {p:.2f} "
                  f"(returned {n} of support {sup})")


if __name__ == "__main__":
    main()
