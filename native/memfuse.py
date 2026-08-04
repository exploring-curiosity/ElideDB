"""Fusion/aggregation sweep for the memory index. ONE score pass, many
combination strategies - so the choice is measured, not guessed.

The shipped system's weakness is NEEDLE queries (support 8-18 score
0.00-0.28 while support 165-247 score 0.66-0.90). For a robot memory
that is the important case: "find the time this happened" has a
handful of instances, not hundreds. Two aggregation choices dominate
that regime and neither was ever measured here:

  seed aggregation  MAX over seed examples fires on one lucky seed;
                    MEAN demands consensus - with few positives, the
                    consensus estimate is far less noisy.
  channel combining all-channel RRF vs z-sum vs seed-selected top-k.

Everything is query-time only: no truth touches scoring, and no
channel knows what corpus it is in.

    python native/memfuse.py --store lake/fresh_bench
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from mem import (load_channels, build_reps, batch_dtw,          # noqa: E402
                 arg)


def per_seed_scores(reps, eps, seeds):
    """{channel_variant -> (n_seeds, n_eps) score matrix}. Computed
    once per (query, seed-group); every strategy reuses it."""
    pos = {e: i for i, e in enumerate(eps)}
    out = {}
    for name, R in reps.items():
        P = R["pool"]
        if P:
            dim = len(next(iter(P.values())))
            M = np.stack([P.get(e, np.zeros(dim, np.float32))
                          for e in eps])
            have = np.array([e in P for e in eps])
            rows = []
            for s in seeds:
                q = P.get(s)
                rows.append(np.where(have, M @ q, -1e9)
                            if q is not None
                            else np.full(len(eps), -1e9, np.float32))
            out[f"{name}.pool"] = np.stack(rows)
        for var in ("seq", "dseq"):
            S = R[var]
            keys = [e for e in eps if e in S]
            if len(keys) < 10:
                continue
            B = np.stack([S[e] for e in keys])
            rows, mx_rows, ch_rows = [], [], []
            for s in seeds:
                q = S.get(s)
                v = np.full(len(eps), -1e9, np.float32)
                vmx = np.full(len(eps), -1e9, np.float32)
                vch = np.full(len(eps), -1e9, np.float32)
                if q is not None:
                    d = batch_dtw(q, B)
                    # MOMENT matching: whole-sequence alignment
                    # averages away the one second that defines a
                    # needle query. mx = best single moment match,
                    # ch = every query moment must appear somewhere
                    # (chamfer). Order-free by construction.
                    C = np.einsum("id,njd->nij", q, B)
                    dmx = C.max(axis=(1, 2))
                    dch = C.max(axis=2).mean(axis=1)
                    for e, x, y, z2 in zip(keys, d, dmx, dch):
                        v[pos[e]] = x
                        vmx[pos[e]] = y
                        vch[pos[e]] = z2
                rows.append(v)
                mx_rows.append(vmx)
                ch_rows.append(vch)
            out[f"{name}.{var}"] = np.stack(rows)
            out[f"{name}.{var}mx"] = np.stack(mx_rows)
            out[f"{name}.{var}ch"] = np.stack(ch_rows)
    return out


def _rank(v):
    r = np.empty(len(v), np.float64)
    r[np.argsort(-v)] = np.arange(len(v))
    return r


def _z(v):
    m = v > -1e8
    if m.sum() < 3:
        return np.zeros_like(v, np.float64)
    mu, sd = float(v[m].mean()), float(v[m].std()) + 1e-8
    out = (v - mu) / sd
    out[~m] = -1e9
    return out


def combine(S, seeds, eps, agg, mode, topk=3):
    """S: {chan -> (n_seeds, n_eps)} -> one fused score vector."""
    per = {k: (M.max(0) if agg == "max" else M.mean(0))
           for k, M in S.items()}
    names = sorted(per)
    if mode.startswith("top"):
        pos = {e: i for i, e in enumerate(eps)}
        qual = {}
        for k, M in S.items():
            sc = []
            for hi, h in enumerate(seeds):
                rest = [i for i in range(len(seeds)) if i != hi]
                if not rest:
                    continue
                v = (M[rest].max(0) if agg == "max"
                     else M[rest].mean(0)).copy()
                for i in rest:
                    if seeds[i] in pos:
                        v[pos[seeds[i]]] = -1e9
                j = pos.get(h)
                if j is None:
                    continue
                sc.append(1.0 / (1.0 + int((v > v[j]).sum())))
            qual[k] = float(np.mean(sc)) if sc else 0.0
        names = sorted(qual, key=lambda k: -qual[k])[:topk]
    tot = np.zeros(len(eps), np.float64)
    for k in names:
        v = per[k]
        tot += (1.0 / (60.0 + _rank(v)) if mode.endswith("rrf")
                else _z(v))
    return tot


STRATS = [("max", "all_rrf"), ("mean", "all_rrf"),
          ("max", "all_z"), ("mean", "all_z"),
          ("max", "top_rrf"), ("mean", "top_rrf"),
          ("mean", "top_z")]


def main():
    from elidedb import Store
    import pyarrow.parquet as pq
    store = ROOT / arg("--store", "lake/fresh_bench")
    db = Store.open(str(store))
    t0 = time.time()
    reps = build_reps(load_channels(db, store.name))
    eps = sorted({e for R in reps.values() for e in R["pool"]})
    print(f"{len(eps)} episodes, channels {sorted(reps)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    ep = db.table("episodes").scan()
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet") \
        .to_pydict()
    G, sup = {}, {}
    have = set(eidx)
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        ei = int(ei)
        G[(int(q), ei)] = int(v)
        if ei in have:
            sup[int(q)] = sup.get(int(q), 0) + int(v)

    res = {s: {} for s in STRATS}
    for qi in range(11):
        s = sup.get(qi, 0)
        if not s:
            continue
        truths = [e for e in eidx if G.get((qi, e)) == 1]
        ns = min(5, len(truths))
        if ns < 2 or len(truths) - ns < 1:
            continue                      # degenerate (q09/q10)
        k_max = int(np.ceil(s * 1.5))
        rs = np.random.RandomState(0)
        acc = {st: [] for st in STRATS}
        for _ in range(5):
            sd = sorted(set(int(x) for x in
                            rs.choice(truths, ns, replace=False)))
            S = per_seed_scores(reps, eps, sd)
            pos = {e: i for i, e in enumerate(eps)}
            for st in STRATS:
                tot = combine(S, sd, eps, st[0], st[1])
                for x in sd:
                    if x in pos:
                        tot[pos[x]] = -1e9
                got = [eps[i] for i in np.argsort(-tot)[:k_max]]
                tr = sum(1 for e in got if G.get((qi, e)) == 1)
                acc[st].append((tr / s, tr / max(len(got), 1)))
        for st in STRATS:
            res[st][qi] = (float(np.mean([a[0] for a in acc[st]])),
                           float(np.mean([a[1] for a in acc[st]])))
        best = max(STRATS, key=lambda st: res[st][qi][0])
        print(f"  q{qi:02d} sup {s:<4} " + " ".join(
            f"{st[0][:2]}/{st[1][:5]}={res[st][qi][0]:.2f}"
            for st in STRATS) + f"  BEST {best[0]}/{best[1]}",
            flush=True)
    print("\nSTRATEGY MEANS (over real queries)")
    for st in STRATS:
        ys = [v[0] for v in res[st].values()]
        ps = [v[1] for v in res[st].values()]
        print(f"  {st[0]:<5} {st[1]:<8} yield {np.mean(ys):.3f}  "
              f"prec {np.mean(ps):.3f}")


if __name__ == "__main__":
    main()
