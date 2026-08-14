"""One evaluator, used by every arm, over the frozen scene-grouped split.

Protocol is unchanged from the frozen system so the numbers stay comparable:
k = support, group-aware grading via vjeval.group_key, chance = support/pool,
and a query never retrieves its own rollout (both camera variants are excluded,
since they are the same physical event).

TWO POOLS, both always reported:

  primary      query in test, pool in val+test. Fully train-disjoint - nothing
               in the pool was ever a gradient.
  deployment   query in test, pool in everything. The index holds history, the
               query is new. This is the realistic setting and the optimistic
               one; quoting it alone would be dishonest, quoting only the other
               would understate what a deployed index does.

Callers pass {episode_id: (T, D) float32}. That is the only contract, so the
frozen baseline and a trained recurrence go through identical code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo.vjeval import group_key, l2, parse  # noqa: E402
from relmo.vjeval5 import dtw_from_cost  # noqa: E402


def _epid(name):
    m = parse(name)
    return f"{m['task']}#{m['epnum']}"


def evaluate(desc, query_ids, pool_ids, min_support=5, boot=0, seed=0):
    """-> dict(overall, per_group, per_task, chance, n_queries).

    desc: {episode_id: (T, D)}. Sequences are matched with subsequence DTW on
    cosine cost, exactly as the shipped read path does.
    """
    pool = [i for i in sorted(pool_ids) if i in desc]
    qry = [i for i in sorted(query_ids) if i in desc]
    if not pool or not qry:
        raise SystemExit("empty pool or query set")
    meta = {i: parse(i) for i in set(pool) | set(qry)}
    grp = {i: group_key(meta[i]) for i in meta}
    P = np.stack([l2(desc[i]) for i in pool])            # (N, T, D)
    pool_grp = np.array([grp[i] for i in pool])
    pool_ep = np.array([_epid(i) for i in pool])

    per_q = []
    for q in qry:
        keep = pool_ep != _epid(q)
        if keep.sum() < min_support:
            continue
        sv = pool_grp[keep] == grp[q]
        sup = int(sv.sum())
        if sup < min_support:
            continue
        C = 1.0 - np.einsum("sd,nkd->nsk", l2(desc[q]), P[keep])
        s = -dtw_from_cost(C)
        hit = int(sv[np.argsort(-s)[:sup]].sum())
        per_q.append((meta[q]["task"], grp[q], hit, sup,
                      sup * sup / int(keep.sum())))
    if not per_q:
        raise SystemExit("no gradeable queries")

    hit = np.array([r[2] for r in per_q], float)
    sup = np.array([r[3] for r in per_q], float)
    rnd = np.array([r[4] for r in per_q], float)
    out = dict(overall=hit.sum() / sup.sum(), chance=rnd.sum() / sup.sum(),
               n_queries=len(per_q))

    for field, key in (("per_group", 1), ("per_task", 0)):
        d = {}
        for r in per_q:
            a = d.setdefault(r[key], [0.0, 0.0, 0.0])
            a[0] += r[2]
            a[1] += r[3]
            a[2] += r[4]
        out[field] = {k: dict(prec=v[0] / v[1], chance=v[2] / v[1],
                              support=int(v[1]))
                      for k, v in sorted(d.items())}
    if boot:
        # resample QUERIES, not items - the query is the unit of replication
        rng = np.random.default_rng(seed)
        b = [(hit[k].sum() / sup[k].sum())
             for k in (rng.integers(0, len(hit), (boot, len(hit))))]
        out["ci95"] = (float(np.percentile(b, 2.5)),
                       float(np.percentile(b, 97.5)))
    return out


def report(name, res, per="per_group"):
    print(f"\n=== {name} ===")
    ci = res.get("ci95")
    ci_s = f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
    print(f"overall {res['overall']:.3f}  chance {res['chance']:.3f}  "
          f"lift {res['overall']/res['chance']:.2f}x  "
          f"n={res['n_queries']}{ci_s}")
    print(f"{'group':22s} {'prec':>6s} {'chance':>7s} {'lift':>6s} {'sup':>5s}")
    for k, v in sorted(res[per].items(), key=lambda x: -x[1]["prec"]):
        flag = "  <- 0.70" if v["prec"] >= 0.70 else ""
        print(f"{k:22s} {v['prec']:6.3f} {v['chance']:7.3f} "
              f"{v['prec']/v['chance']:5.2f}x {v['support']:5d}{flag}")
