"""THE MEMORY INDEX: multi-encoder temporal-sequence retrieval.

Domain-blind by construction. Every signal is a frozen pretrained
encoder's output over TIME; every episode is a sequence; every score is
either a pooled cosine or a temporal alignment of two sequences. There
is no object model, no event grammar, no per-corpus vocabulary and no
trained component anywhere - the identical code runs on tabletop sim,
kitchen demos, or any corpus a customer uploads, and adds whatever
encoders that store happens to carry.

Two measured facts drive the design:
  * temporal ALIGNMENT roughly doubles pooled scoring on manipulation
    queries for the same encoder (V-JEPA q04 0.37 -> 0.61, q05 0.38 ->
    0.72) - sequence shape is signal that pooling destroys;
  * no single channel is best everywhere, and which one wins is a
    property of the QUERY, so the choice is made per query from the
    seed examples alone (leave-one-seed-out), never from truth.

Speed: sequences are resampled to a fixed length so DTW runs BATCHED
in numpy over all candidates at once (one DP over an (N,L,L) tensor),
which is what makes a full multi-channel benchmark a 2-minute job.

    python native/mem.py --store lake/fresh_bench --all
    python native/mem.py --store lake/sim_chains --sim
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

L = 12                      # fixed sequence length (resampled)
BAND = 0.35                 # DTW band as a fraction of L
SKIP = {"frame_vectors"}    # trained student: retired by directive


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def norm(X):
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True),
                          1e-8)


def resample(V, n=L):
    """Sequence -> fixed length by nearest-sample interpolation."""
    if len(V) == 0:
        return None
    idx = np.linspace(0, len(V) - 1, n).round().astype(int)
    return V[idx]


def load_channels(db, store_name):
    """{channel -> {episode -> (T,d) sequence}} from every encoder
    table in the store, plus the standalone V-JEPA window cache."""
    import bisect
    ep = db.table("episodes").scan().to_pydict()
    spans = sorted((int(a), int(b), int(e)) for a, b, e in
                   zip(ep["ts"], ep["t1"], ep["episode_index"]))
    starts = [s for s, _, _ in spans]
    out = {}
    for t in sorted(db.tables()):
        if t in SKIP or not (t.endswith("_vectors")):
            continue
        meta = db.table(t).state().meta or {}
        if not meta.get("model"):
            continue
        d = db.table(t).scan().to_pydict()
        if "vector" not in d:
            continue
        rows = {}
        for i in range(len(d["ts"])):
            tt = int(d["ts"][i])
            j = bisect.bisect_right(starts, tt) - 1
            if j < 0 or tt > spans[j][1]:
                continue
            rows.setdefault(spans[j][2], []).append(
                (tt, d["vector"][i]))
        seqs = {}
        for e, rs in rows.items():
            rs.sort(key=lambda r: r[0])
            seqs[e] = norm(np.array([v for _, v in rs], np.float32))
        if seqs:
            out[t.replace("_vectors", "")] = seqs
    # the standalone V-JEPA 2s/1s window cache (native/seq.py)
    try:
        from seq import cache_dir
        cdir = cache_dir(store_name)
        seqs = {}
        for p in sorted(cdir.glob("*.npz")):
            seqs[int(p.stem)] = norm(
                np.load(p)["V"].astype(np.float32))
        if seqs:
            out["vjseq"] = seqs
    except Exception:
        pass
    return out


def build_reps(chans):
    """Per channel: pooled vectors + fixed-length sequences + their
    delta (change-profile) sequences."""
    reps = {}
    for name, seqs in chans.items():
        pool, seq, dseq = {}, {}, {}
        for e, V in seqs.items():
            if len(V) == 0:
                continue
            pool[e] = norm(V.mean(0))
            if len(V) >= 3:
                R = resample(V)
                seq[e] = R
                dseq[e] = norm(R[1:] - R[:-1])
        reps[name] = dict(pool=pool, seq=seq, dseq=dseq)
    return reps


def batch_dtw(A, B):
    """Banded DTW similarity of one (L,d) sequence against a stack of
    (N,L,d) sequences - the DP runs over the N axis in numpy, so the
    whole corpus is scored in one pass."""
    n, la, lb = len(B), len(A), B.shape[1]
    C = 1.0 - np.einsum("id,njd->nij", A, B)      # N,la,lb
    r = max(int(BAND * max(la, lb)), 1)
    INF = 1e9
    acc = np.full((n, la + 1, lb + 1), INF, np.float32)
    acc[:, 0, 0] = 0.0
    for i in range(1, la + 1):
        j0 = max(1, int(i * lb / la) - r)
        j1 = min(lb, int(i * lb / la) + r)
        for j in range(j0, j1 + 1):
            acc[:, i, j] = C[:, i - 1, j - 1] + np.minimum(
                np.minimum(acc[:, i - 1, j - 1], acc[:, i - 1, j]),
                acc[:, i, j - 1])
    return 1.0 - acc[:, la, lb] / (la + lb)


def channel_scores(reps, eps, seed_eps):
    """{channel_variant -> score vector over eps} for one seed set."""
    out = {}
    for name, R in reps.items():
        # pooled cosine
        P = R["pool"]
        if P:
            M = np.stack([P.get(e, np.zeros_like(next(iter(
                P.values())))) for e in eps])
            have = np.array([e in P for e in eps])
            sc = np.full(len(eps), -1e9, np.float32)
            qs = [P[s] for s in seed_eps if s in P]
            if qs:
                s_ = np.max(np.stack([M @ q for q in qs]), 0)
                sc = np.where(have, s_, -1e9)
                out[f"{name}.pool"] = sc
        # temporal alignment on the delta (change-profile) sequence
        for var in ("seq", "dseq"):
            S = R[var]
            qs = [S[s] for s in seed_eps if s in S]
            if not qs:
                continue
            keys = [e for e in eps if e in S]
            if len(keys) < 10:
                continue
            B = np.stack([S[e] for e in keys])
            best = None
            for q in qs:
                v = batch_dtw(q, B)
                best = v if best is None else np.maximum(best, v)
            sc = np.full(len(eps), -1e9, np.float32)
            pos = {e: i for i, e in enumerate(eps)}
            for e, v in zip(keys, best):
                sc[pos[e]] = v
            out[f"{name}.{var}"] = sc
    return out


def loo_quality(reps, eps, seeds):
    """Per-channel leave-one-seed-out quality - QUERY TIME ONLY."""
    q = {}
    for h in seeds:
        rest = [s for s in seeds if s != h]
        if not rest:
            continue
        sc = channel_scores(reps, eps, rest)
        pos = {e: i for i, e in enumerate(eps)}
        for k, v in sc.items():
            v2 = v.copy()
            for s in rest:
                if s in pos:
                    v2[pos[s]] = -1e9
            hi = pos.get(h)
            if hi is None:
                continue
            rank = int((v2 > v2[hi]).sum())
            q.setdefault(k, []).append(1.0 / (1.0 + rank))
    return {k: float(np.mean(v)) for k, v in q.items()}


def fused_rank(reps, eps, seeds, topk=3):
    """Seed-selected top channels, combined by reciprocal rank."""
    qual = loo_quality(reps, eps, seeds)
    if not qual:
        return None, []
    best = sorted(qual, key=lambda k: -qual[k])[:topk]
    sc = channel_scores(reps, eps, seeds)
    tot = np.zeros(len(eps), np.float64)
    for k in best:
        v = sc.get(k)
        if v is None:
            continue
        r = np.empty(len(v))
        r[np.argsort(-v)] = np.arange(len(v))
        tot += 1.0 / (60.0 + r)
    pos = {e: i for i, e in enumerate(eps)}
    for s in seeds:
        if s in pos:
            tot[pos[s]] = -1e9
    return tot, best


def bench_kitchen(db, reps, eps, want, seeds_n=5):
    import pyarrow.parquet as pq
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
    ys, ps, rows = [], [], []
    for qi in want:
        s = sup.get(qi, 0)
        if not s:
            continue
        truths = [e for e in eidx if G.get((qi, e)) == 1]
        ns = min(seeds_n, len(truths))
        if ns < 2 or len(truths) - ns < 1:
            rows.append((qi, s, None, None, "degenerate"))
            continue
        k_max = int(np.ceil(s * 1.5))
        rs = np.random.RandomState(0)
        runs, picks = [], []
        for _ in range(5):
            sd = sorted(set(int(x) for x in
                            rs.choice(truths, ns, replace=False)))
            tot, best = fused_rank(reps, eps, sd)
            if tot is None:
                continue
            got = [eps[i] for i in np.argsort(-tot)[:k_max]]
            tr = sum(1 for e in got if G.get((qi, e)) == 1)
            runs.append((tr / s, tr / max(len(got), 1)))
            picks.append(",".join(b.split(".")[0] for b in best))
        if not runs:
            continue
        my = float(np.mean([r[0] for r in runs]))
        mp = float(np.mean([r[1] for r in runs]))
        ys.append(my)
        ps.append(mp)
        rows.append((qi, s, my, mp, picks[0]))
        print(f"  q{qi:02d} sup {s:<4} yield {my:.2f}  prec {mp:.2f}"
              f"   [{picks[0]}]", flush=True)
    if ys:
        print(f"  MEAN yield {np.mean(ys):.3f}  prec {np.mean(ps):.3f}"
              f"  (over {len(ys)} queries)", flush=True)
    return rows


def bench_sim(reps, eps):
    import pyarrow.parquet as pq
    import math
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    for label, targets in (("DEV", ("swap", "precarious",
                                    "push_then_build",
                                    "build_unstack_move")),
                           ("HOLDOUT", ("relocate_build",
                                        "two_sites_merge"))):
        rs = np.random.RandomState(0)
        ys, ps = [], []
        for tg in targets:
            pool = sorted(e for e, tm in tmpl.items()
                          if tm == tg and e in set(eps))
            sd = sorted(int(x) for x in rs.choice(pool, 5,
                                                  replace=False))
            support = len(pool) - len(sd)
            k = math.ceil(1.5 * support)
            tot, best = fused_rank(reps, eps, sd)
            got = [eps[i] for i in np.argsort(-tot)[:k]]
            tr = sum(1 for e in got if tmpl.get(e) == tg)
            ys.append(tr / support)
            ps.append(tr / len(got))
            print(f"   {tg:<20} yield {tr/support:.2f} "
                  f"prec {tr/len(got):.2f}  [{','.join(b.split('.')[0] for b in best)}]")
        print(f"  {label} MEAN yield {np.mean(ys):.3f}  "
              f"prec {np.mean(ps):.3f}", flush=True)


def main():
    from elidedb import Store
    store = ROOT / arg("--store", "lake/fresh_bench")
    db = Store.open(str(store))
    t0 = time.time()
    chans = load_channels(db, store.name)
    print(f"channels: {sorted(chans)}  "
          f"({time.time()-t0:.0f}s load)", flush=True)
    reps = build_reps(chans)
    eps = sorted({e for R in reps.values() for e in R["pool"]})
    print(f"{len(eps)} episodes indexed", flush=True)
    if "--sim" in sys.argv:
        bench_sim(reps, eps)
    else:
        want = list(range(11)) if "--all" in sys.argv else [3, 4, 5]
        bench_kitchen(db, reps, eps, want)


if __name__ == "__main__":
    main()
