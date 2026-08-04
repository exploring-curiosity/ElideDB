"""Chain QbE the way fresh_bench won: MANY channels + seed selection.

The kitchen corpus reached 0.90+ (q04 0.94 / q05 0.90) not through one
perfect symbolic pipeline but through channel PLURALITY: episodes
scored by similarity in several continuous spaces at once, the winning
channel picked PER QUERY by leave-one-out over the 5 seed examples
(query-time computation - no truth in any path), z-fusion as fallback.
A symbolic token route multiplies stage recalls (0.88-good stages
compounded to 0.30, measured); continuous channels fail soft.

Channels, all vision-native, all already in the store:
    iv2       InternVideo2 episode embedding (1/episode)
    sig2      SigLIP2 window embeddings (8/episode) - image tower only
    fdnnv     FDNN-V frame series -> pooled + DTW
    scene     DINOv3 frame series -> pooled + DTW + novelty profile
              (consecutive-frame distance rhythm - layout-invariant)
    motion    motion-encoder event sequences -> NW alignment
    tokens    chain_delta manipulation tokens (route 3)
    evstream  chain_delta raw EVENT stream, no pairing (ablates the
              brittle pairing stage away)

    python scripts/chain_channels.py [--store lake/sim_chains]
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
import chain_qbe                                               # noqa: E402

DEV = ("swap", "precarious", "push_then_build", "build_unstack_move")
HOLD = ("relocate_build", "two_sites_merge")


def norm(X):
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True),
                          1e-8)


def episode_spans(db):
    ep = db.table("episodes").scan().to_pydict()
    return sorted((int(t), int(t1), int(e)) for t, t1, e in
                  zip(ep["ts"], ep["t1"], ep["episode_index"]))


def by_episode(db, table, spans, stream=None, stride=1):
    """{episode -> (ts-sorted vectors)} for a vector table."""
    sc = db.table(table).scan()
    d = sc.to_pydict()
    starts = [s for s, _, _ in spans]
    import bisect
    out = defaultdict(list)
    for i in range(0, len(d["ts"]), stride):
        if stream is not None and str(d["stream"][i]) != stream:
            continue
        t = int(d["ts"][i])
        j = bisect.bisect_right(starts, t) - 1
        if j < 0 or t > spans[j][1]:
            continue
        out[spans[j][2]].append((t, d["vector"][i]))
    return {e: norm([v for _, v in sorted(rows, key=lambda r: r[0])])
            for e, rows in out.items()}


def dtw_sim(A, B, band=0.2):
    """Normalised DTW similarity over cosine distances."""
    n, m = len(A), len(B)
    D = 1.0 - A @ B.T
    r = max(int(band * max(n, m)), 2)
    INF = 1e9
    acc = np.full((n + 1, m + 1), INF, np.float32)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, int(i * m / n) - r)
        j1 = min(m, int(i * m / n) + r)
        for j in range(j0, j1 + 1):
            acc[i, j] = D[i - 1, j - 1] + min(acc[i - 1, j - 1],
                                              acc[i - 1, j],
                                              acc[i, j - 1])
    return 1.0 - float(acc[n, m]) / (n + m)


def S_from_seqs(seqs, eps, kind="dtw"):
    n = len(eps)
    S = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = seqs.get(eps[i]), seqs.get(eps[j])
            if a is None or b is None or not len(a) or not len(b):
                continue
            if kind == "dtw":
                S[i, j] = S[j, i] = dtw_sim(a, b)
            else:                       # pooled cosine
                S[i, j] = S[j, i] = float(
                    norm(a.mean(0)) @ norm(b.mean(0)))
    return S


def S_tokens(seqs, eps):
    n = len(eps)
    S = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = chain_qbe.align(seqs[eps[i]],
                                                seqs[eps[j]])
    return S


def bench_S(S, eps, tmpl, targets, label):
    pos = {e: i for i, e in enumerate(eps)}
    rs = np.random.RandomState(0)
    ys, ps = [], []
    for target in targets:
        pool = sorted(e for e, tm in tmpl.items()
                      if tm == target and e in pos)
        seeds = sorted(int(x) for x in rs.choice(pool, 5,
                                                 replace=False))
        support = len(pool) - len(seeds)
        k = math.ceil(1.5 * support)
        si = [pos[e] for e in seeds]
        sc = S[si].max(0)
        for x_ in si:
            sc[x_] = -1e9
        got = [eps[i] for i in np.argsort(-sc)[:k]]
        true = sum(1 for e in got if tmpl.get(e) == target)
        ys.append(true / support)
        ps.append(true / len(got))
    print(f"   {label:<26} yield {np.mean(ys):.3f}  "
          f"prec {np.mean(ps):.3f}  "
          f"({' '.join(f'{y:.2f}' for y in ys)})")
    return float(np.mean(ys))


def loo_select(mats, eps, tmpl, targets, label, topk=2):
    """Per-query channel choice by leave-one-out over the 5 seeds
    (query-time only), z-fused over the top channels."""
    pos = {e: i for i, e in enumerate(eps)}
    rs = np.random.RandomState(0)
    ys, ps, picks = [], [], []
    names = sorted(mats)
    for target in targets:
        pool = sorted(e for e, tm in tmpl.items()
                      if tm == target and e in pos)
        seeds = sorted(int(x) for x in rs.choice(pool, 5,
                                                 replace=False))
        support = len(pool) - len(seeds)
        k = math.ceil(1.5 * support)
        si = [pos[e] for e in seeds]
        # LOO: how highly does each channel rank a held-out seed when
        # queried with the other four?
        scores = {}
        for nm in names:
            S = mats[nm]
            rr = []
            for h in si:
                rest = [x for x in si if x != h]
                sc = S[rest].max(0)
                sc[rest] = -1e9
                rank = int((sc > sc[h]).sum())
                rr.append(1.0 / (1.0 + rank))
            scores[nm] = float(np.mean(rr))
        best = sorted(names, key=lambda nm: -scores[nm])[:topk]
        picks.append(",".join(best))
        z = np.zeros(len(eps), np.float32)
        for nm in best:
            sc = mats[nm][si].max(0).copy()
            for x_ in si:
                sc[x_] = -1e9
            mu, sd = float(sc[sc > -1e8].mean()), \
                float(sc[sc > -1e8].std()) + 1e-8
            z += (sc - mu) / sd
        for x_ in si:
            z[x_] = -1e9
        got = [eps[i] for i in np.argsort(-z)[:k]]
        true = sum(1 for e in got if tmpl.get(e) == target)
        ys.append(true / support)
        ps.append(true / len(got))
    print(f"   {label:<26} yield {np.mean(ys):.3f}  "
          f"prec {np.mean(ps):.3f}  "
          f"({' '.join(f'{y:.2f}' for y in ys)})")
    for t, p in zip(targets, picks):
        print(f"      {t:<20} -> {p}")
    return float(np.mean(ys))


def delta_channels(eps):
    """Token + raw-event-stream channels from the chain_delta cache."""
    sp = Path(os.environ.get(
        "ELIDEDB_DELTA_CACHE",
        "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
        "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
        "scratchpad/delta_events.npz"))
    out = {}
    if not sp.exists():
        return out
    import chain_delta as cd
    z = np.load(sp, allow_pickle=True)
    events = [list(e) for e in z["events"]]
    dirs, junk = cd.classify_events(events)
    med_r = 0.9 * cd.DS * math.sqrt(np.median([e[5] for e in events]))
    mans = cd.manipulations(events, dirs, junk, med_r)
    out["tokens"] = S_tokens(cd.tokenise(mans), eps)
    # raw event stream: no pairing to get wrong
    mags = np.array([e[5] for e in events])
    mq = np.percentile(mags, [33, 66])
    ev_by_ep = defaultdict(list)
    for ev, dr, jk in zip(events, dirs, junk):
        if jk:
            continue
        ev_by_ep[int(ev[0])].append((int(ev[2]), int(dr),
                                     int(np.searchsorted(mq, ev[5]))))
    seqs = {}
    for e, rows in ev_by_ep.items():
        rows.sort()
        toks, prev = [], None
        for t, dr, q in rows:
            if prev is not None:
                gap = max((t - prev) / 1e9, 0.0)
                toks.append((("G", int(min(gap / 4.0, 2)), 0),
                             -1, 0.0, None))
            toks.append((("E", dr, q), -1, 0.0, None))
            prev = t
        seqs[e] = toks
    for e in eps:
        seqs.setdefault(e, [])
    out["evstream"] = S_tokens(seqs, eps)
    return out


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    spans = episode_spans(db)
    eps = [e for _, _, e in spans]

    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}

    mats = {}
    print("building channels...", flush=True)
    iv2 = by_episode(db, "iv2_vectors", spans)
    mats["iv2"] = S_from_seqs(iv2, eps, "pool")
    sig2 = by_episode(db, "sig2_vectors", spans)
    mats["sig2_pool"] = S_from_seqs(sig2, eps, "pool")
    mats["sig2_dtw"] = S_from_seqs(sig2, eps, "dtw")
    mo = by_episode(db, "motion_vectors", spans)
    mats["motion"] = S_from_seqs(mo, eps, "dtw")
    fv = by_episode(db, "frame_vectors", spans, stream="simA",
                    stride=1)
    fv5 = {e: v[::5] for e, v in fv.items()}
    mats["fdnnv_pool"] = S_from_seqs(fv, eps, "pool")
    mats["fdnnv_dtw"] = S_from_seqs(fv5, eps, "dtw")
    sv = by_episode(db, "scene_vectors", spans, stream="simA",
                    stride=1)
    sv5 = {e: v[::5] for e, v in sv.items()}
    mats["scene_pool"] = S_from_seqs(sv, eps, "pool")
    mats["scene_dtw"] = S_from_seqs(sv5, eps, "dtw")
    # novelty profile: rhythm of change, layout-invariant
    nov = {}
    for e, v in sv.items():
        if len(v) < 8:
            continue
        d = 1.0 - (v[1:] * v[:-1]).sum(1)
        d = np.convolve(d, np.ones(5) / 5, mode="valid")
        nov[e] = norm(np.stack([d, np.gradient(d)], 1))
    mats["novelty"] = S_from_seqs(nov, eps, "dtw")
    mats.update(delta_channels(eps))
    print(f"channels: {sorted(mats)}", flush=True)

    print("-- single channels, DEV")
    for nm in sorted(mats):
        bench_S(mats[nm], eps, tmpl, DEV, nm)
    print("-- single channels, HOLDOUT")
    for nm in sorted(mats):
        bench_S(mats[nm], eps, tmpl, HOLD, nm)
    print("-- seed-LOO selection")
    loo_select(mats, eps, tmpl, DEV, "DEV select top2")
    loo_select(mats, eps, tmpl, HOLD, "HOLDOUT select top2")
    loo_select(mats, eps, tmpl, DEV, "DEV select top1", topk=1)
    loo_select(mats, eps, tmpl, HOLD, "HOLDOUT select top1", topk=1)

    np.savez(Path(os.environ.get(
        "ELIDEDB_CHANNELS_OUT",
        "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
        "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
        "scratchpad/chain_channels.npz")),
        eps=np.array(eps), **{f"S_{k}": v for k, v in mats.items()})


if __name__ == "__main__":
    main()
