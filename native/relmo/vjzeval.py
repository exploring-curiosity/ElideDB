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

T MAY DIFFER PER EPISODE. Under stream-time records (relmo/vjrec6) the trace
length is proportional to duration, so the pool is ragged. Candidates are
right-padded with an unreachable cost rather than stacked, which leaves
subsequence DTW's free endpoints intact: a padded column can never be the
argmin, and padding sits only at the right end so it cannot shorten a path
through real columns either. Verified against a per-item loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo.vjeval import group_key, l2, parse  # noqa: E402
from relmo.vjeval5 import dtw_from_cost  # noqa: E402
from relmo.vjmatch import dtw as dtw_sym2  # noqa: E402
from relmo import vjmatch  # noqa: E402
from relmo import vjrel  # noqa: E402


PAD_COST = 1e6            # unreachable, so padded columns never win the argmin


def _epid(name):
    m = parse(name)
    return f"{m['task']}#{m['epnum']}"


def _pad(seqs):
    """[(T_i,D)] -> (N,Tmax,D) padded, (N,Tmax) bool valid."""
    tmax = max(len(s) for s in seqs)
    out = np.zeros((len(seqs), tmax, seqs[0].shape[-1]), np.float32)
    m = np.zeros((len(seqs), tmax), bool)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
        m[i, :len(s)] = True
    return out, m


MATCHERS = {
    # both endpoints anchored, symmetric2 weights, exact 1/(Q+R) normalisation.
    # THE DEFAULT: the free-end variant scores candidates by how LONG they are
    # (Spearman +0.685 with candidate length, positive on 65/65 queries).
    "full": lambda C, L: dtw_sym2(C, False, L),
    # free start and end on the reference - span localisation, a different
    # question. Still length-biased (+0.624); use it to FIND a span, not to
    # decide which recording is the same event.
    "sub": lambda C, L: dtw_sym2(C, True, L),
    # the pre-2026-08-15 matcher, kept so old numbers stay reproducible
    "legacy": lambda C, L: dtw_from_cost(C),
}


def evaluate(desc, query_ids, pool_ids, min_support=5, boot=0, seed=0,
             qdesc=None, matcher="full", multirate=False, mr_mode="match"):
    """-> dict(overall, per_group, per_task, chance, n_queries).

    desc: {episode_id: (T, D)}. Sequences are matched with subsequence DTW on
    cosine cost, exactly as the shipped read path does.
    """
    qdesc = qdesc if qdesc is not None else desc
    # multirate: desc is {rate_suffix: {id: (T,D)}} and a candidate is scored by
    # the BEST-explaining rate pair. A recording too short to hold one window at
    # a coarse rate simply does not compete there.
    base = (desc.get("", next(iter(desc.values()))) if multirate else desc)
    qbase = (qdesc.get("", next(iter(qdesc.values()))) if multirate else qdesc)
    pool = [i for i in sorted(pool_ids) if i in base]
    qry = [i for i in sorted(query_ids) if i in qbase]
    if not pool or not qry:
        raise SystemExit("empty pool or query set")
    meta = {i: parse(i) for i in set(pool) | set(qry)}
    grp = {i: group_key(meta[i]) for i in meta}
    # The positive class is EVENT-level. Duration declines the rank rather than
    # gating membership, so it lives in the graded relevance and is read by
    # NDCG - a binary metric cannot express a passive decline.
    dur = {i: vjrel.all_meta().get(i, {}).get("dur", 0.0) for i in meta}
    pool_dur = np.array([dur[i] for i in pool])
    # graded relevance, computed once over the union and indexed per query.
    # This is where the duration decay lives.
    AM = vjrel.all_meta()
    uni = list(dict.fromkeys(list(qry) + list(pool)))
    upos = {i: k for k, i in enumerate(uni)}
    REL, _ = vjrel.relevance(uni, AM)
    rel_q = REL[np.array([upos[i] for i in qry])[:, None],
                np.array([upos[i] for i in pool])[None, :]]
    q_at = {i: k for k, i in enumerate(qry)}
    nd, nd_sup = [], []
    if multirate:
        packed = vjmatch.pack({r: {i: l2(v[i]) for i in v}
                               for r, v in desc.items()}, pool)
    else:
        P, P_ok = _pad([l2(base[i]) for i in pool])       # (N, Tmax, D)
        P_len = np.array([len(base[i]) for i in pool])
    match = MATCHERS[matcher]
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
        if multirate:
            qb = {r: l2(v[q]) for r, v in qdesc.items() if q in v}
            s = -vjmatch.score_rates(qb, packed, keep,
                                     free_ends=(matcher == "sub"),
                                     mode=mr_mode)
        else:
            C = 1.0 - np.einsum("sd,nkd->nsk", l2(qbase[q]), P[keep])
            C = np.where(P_ok[keep][:, None, :], C, PAD_COST)
            s = -match(C, P_len[keep])
        hit = int(sv[np.argsort(-s)[:sup]].sum())
        rq = rel_q[q_at[q]][keep]
        if rq.max() > 0:
            ones = np.ones((1, len(s)), bool)
            nd.append(vjrel.ndcg(s[None, :], rq[None, :], ones))
            # NDCG@support: cut at exactly the k that prec@support returns, so
            # the pair describes the SAME returned list - one says how many
            # belong, the other whether they are in the right order and whether
            # near-relevant beat far-relevant. Its random floor is 0.150,
            # against 0.528 for the full-list variant, which is why the
            # full-list number makes real differences look like rounding.
            nd_sup.append(vjrel.ndcg(s[None, :], rq[None, :], ones, k=sup))
        per_q.append((meta[q]["task"], grp[q], hit, sup,
                      sup * sup / int(keep.sum())))
    if not per_q:
        raise SystemExit("no gradeable queries")

    hit = np.array([r[2] for r in per_q], float)
    sup = np.array([r[3] for r in per_q], float)
    rnd = np.array([r[4] for r in per_q], float)
    out = dict(overall=hit.sum() / sup.sum(), chance=rnd.sum() / sup.sum(),
               n_queries=len(per_q),
               ndcg_sup=float(np.nanmean(nd_sup)) if nd_sup else float("nan"),
               ndcg=float(np.nanmean(nd)) if nd else float("nan"))

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
    print(f"prec@support {res['overall']:.3f}  chance {res['chance']:.3f}  "
          f"lift {res['overall']/res['chance']:.2f}x  |  "
          f"NDCG@support {res.get('ndcg_sup', float('nan')):.3f} "
          f"(random floor 0.150)  n={res['n_queries']}{ci_s}")
    print(f"{'group':22s} {'prec':>6s} {'chance':>7s} {'lift':>6s} {'sup':>5s}")
    for k, v in sorted(res[per].items(), key=lambda x: -x[1]["prec"]):
        flag = "  <- 0.70" if v["prec"] >= 0.70 else ""
        print(f"{k:22s} {v['prec']:6.3f} {v['chance']:7.3f} "
              f"{v['prec']/v['chance']:5.2f}x {v['support']:5d}{flag}")
