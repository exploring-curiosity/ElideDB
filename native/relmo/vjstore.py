"""The serve-time read path: STORE-SCOPED, prefiltered, banded.

STORES. A store is one robot's / one deployment's memory. Owner ruling
2026-08-17: rcasa is ONE store - the rcasa/rcasa_eval/rcasa_atomic_full/
rcasa_composite_full split is an EXPERIMENT-side concept (train pools and
eval seals), not a product concept. Five stores exist. A query addresses
exactly one; cross-store search is a deliberate loop over stores, never a
default. Every statistic the read path uses is computed inside the store
(z-scoring is per-recording; the prefilter matrix is per-store), so nothing
leaks between deployments.

Duplicates: rcasa and rcasa_atomic_full share 266 demonstrations (same demo,
different bitrate). Within a store the first-listed dataset wins, same rule
as training.

SPEED. The full-scan read cost is quadratic in sequence length and linear in
corpus size - fine at rcasa scale (36-step clips, 0.75 s), pathological on
composite-length episodes (208 steps, ~12 s/query). Nothing regressed; the
data got longer. Two standard, exact-enough cuts:

  1. PREFILTER: pooled-cosine over the store -> top-M candidates. Costs one
     matvec. DTW then runs on M recordings instead of the whole store.
  2. BAND: Sakoe-Chiba half-width as a fraction of reference length
     (vjmatch.dtw's `band`, present since v6 and never switched on). Exact
     for any alignment warping less than the band allows.

Both knobs are measured, not assumed: vjbench reports latency AND fidelity
against the full scan on the same queries, because a fast path that quietly
changes the answers is a regression wearing a speedup's clothes.

    python3 -m relmo.vjstore --bench --store rcasa --queries 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrank2  # noqa: E402
from relmo.vjeval import l2  # noqa: E402
from relmo.vjmatch import dtw  # noqa: E402
from relmo.vjzeval import PAD_COST, _pad  # noqa: E402

STORES = {
    "rcasa": ("rcasa", "rcasa_eval", "rcasa_atomic_full",
              "rcasa_composite_full"),
    "bridge": ("bridge_wide",),
    "kitti": ("kitti_seq",),
    "oxford": ("oxford_seq",),
    "drone": ("drone_fpv",),
}


def zs(X):
    """Per-recording standardisation - fitted on nothing but the recording."""
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)


class Store:
    """One memory. Load once, query many."""

    def __init__(self, name, rep="fixsig"):
        if name not in STORES:
            raise KeyError(f"unknown store {name!r}; stores: {list(STORES)}")
        self.name = name
        D, seen = {}, set()
        for ds in STORES[name]:
            loaded = vjrank2.load_corpus(datasets=(ds,))
            for k, v in loaded.items():
                if k in seen:                 # cross-dataset duplicate demo
                    continue
                seen.add(k)
                D[k] = v
        if not D:
            raise SystemExit(f"store {name!r} has no records on disk")
        self.ids = sorted(D)
        self.Z = {i: np.concatenate(
            [zs(D[i]["fix"].astype(np.float32)),
             zs(D[i]["sig"].astype(np.float32))], -1) for i in self.ids}
        # padded DTW bank, store-scoped
        self.P, self.ok = _pad([l2(self.Z[i]) for i in self.ids])
        self.L = np.array([len(self.Z[i]) for i in self.ids])
        # PREFILTER RUNS ON RAW POOLED CHANNELS. The DTW representation is
        # standardized PER RECORDING, so its per-recording mean is the zero
        # vector BY CONSTRUCTION - a prefilter pooled from it is numerical
        # noise and selects random candidates. Caught by the fidelity gate:
        # P@10 0.765 -> 0.007, overlap 0.002. Raw channel means, L2'd per
        # channel, carry the actual content.
        # WHITENED per store: raw pooled sig sits in a cone (mean|cos| 0.86
        # measured), where un-whitened cosine barely discriminates - the
        # first fidelity bench read P@10 0.182 for exactly this reason. The
        # miner, where pooled prefiltering demonstrably worked (98.9%
        # same-task pairs), whitened first. Label-free, fitted on this store.
        from relmo.vjreps import apply_w, fit_whiten
        def rawpool(key):
            X = np.stack([D[i][key].astype(np.float32).mean(0)
                          for i in self.ids])
            w = fit_whiten(X, min(256, len(X) - 1))
            Y = apply_w(X, w)
            return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9), w
        self.pf, self._wf = rawpool("fix")
        self.ps, self._ws = rawpool("sig")
        self.raw = {i: (D[i]["fix"].astype(np.float32),
                        D[i]["sig"].astype(np.float32)) for i in self.ids}

    def query(self, fix, sig, top_k=10, prefilter_m=50, band=0.1,
              exclude=None):
        """Raw channel traces (T,1024),(T,768) -> [(id, score)].
        Standardization for DTW happens HERE; the prefilter uses the raw
        pooled means. exclude: ids to bar (e.g. the query's own rollout)."""
        fix = np.asarray(fix, np.float32)
        sig = np.asarray(sig, np.float32)
        from relmo.vjreps import apply_w
        q = l2(np.concatenate([zs(fix), zs(sig)], -1))
        qf = apply_w(fix.mean(0)[None], self._wf)[0]
        qf /= (np.linalg.norm(qf) + 1e-9)
        qs_ = apply_w(sig.mean(0)[None], self._ws)[0]
        qs_ /= (np.linalg.norm(qs_) + 1e-9)
        sims = 0.5 * (self.pf @ qf + self.ps @ qs_)
        if exclude:
            bar = np.fromiter((i in exclude for i in self.ids), bool,
                              len(self.ids))
            sims = np.where(bar, -np.inf, sims)
        m = min(prefilter_m, int(np.isfinite(sims).sum()))
        cand = np.argpartition(-sims, m - 1)[:m]
        C = 1.0 - np.einsum("sd,nkd->nsk", q, self.P[cand])
        C = np.where(self.ok[cand][:, None, :], C, PAD_COST)
        s = -dtw(C, False, self.L[cand], band)
        o = np.argsort(-s)[:top_k]
        return [(self.ids[cand[i]], float(s[i])) for i in o]

    def query_full(self, fix, sig, top_k=10, exclude=None):
        """The exact full scan - the fidelity reference, not the product."""
        return self.query(fix, sig, top_k, prefilter_m=len(self.ids),
                          band=0.0, exclude=exclude)


def bench(store_name, n_q, seed=0):
    from relmo import vjrel
    from relmo.vjrel import parse
    st = Store(store_name)
    print(f"store {store_name!r}: {len(st.ids)} recordings, "
          f"median {int(np.median(st.L))} steps", flush=True)
    AM = vjrel.all_meta(tuple(STORES[store_name]))
    graded = [i for i in st.ids if i in AM]
    rng = np.random.default_rng(seed)
    qs = [graded[k] for k in rng.choice(len(graded), min(n_q, len(graded)),
                                        replace=False)]
    ev = {i: parse(i)["task"] for i in graded} if graded else {}

    def hits(res, q):
        return sum(ev.get(r, "?") == ev.get(q, "!") for r, _ in res) / len(res)

    # legs isolate the two knobs: prefilter recall (M) vs band exactness
    LEGS = [("M=50 band=0", dict(prefilter_m=50, band=0.0)),
            ("M=50 band=.25", dict(prefilter_m=50, band=0.25)),
            ("M=100 band=.25", dict(prefilter_m=100, band=0.25)),
            ("M=200 band=0", dict(prefilter_m=200, band=0.0))]
    res = {n: dict(lat=[], p10=[], ov=[]) for n, _ in LEGS}
    lat_full, p10F, recall50, recall200 = [], [], [], []
    for q in qs:
        excl = {i for i in st.ids
                if AM.get(i, {}).get("rollout") == AM[q]["rollout"]}
        fq, sq = st.raw[q]
        t0 = time.time()
        rF = st.query_full(fq, sq, 10, exclude=excl)
        lat_full.append(time.time() - t0)
        p10F.append(hits(rF, q))
        fullset = {r for r, _ in rF}
        # prefilter containment of the full-scan top-10
        from relmo.vjreps import apply_w
        qf = apply_w(fq.mean(0)[None], st._wf)[0]; qf /= np.linalg.norm(qf) + 1e-9
        qs_ = apply_w(sq.mean(0)[None], st._ws)[0]; qs_ /= np.linalg.norm(qs_) + 1e-9
        sims = 0.5 * (st.pf @ qf + st.ps @ qs_)
        bar = np.fromiter((i in excl for i in st.ids), bool, len(st.ids))
        sims = np.where(bar, -np.inf, sims)
        order = np.argsort(-sims)
        c50 = {st.ids[j] for j in order[:50]}
        c200 = {st.ids[j] for j in order[:200]}
        recall50.append(len(fullset & c50) / 10)
        recall200.append(len(fullset & c200) / 10)
        for name, kw in LEGS:
            t0 = time.time()
            r = st.query(fq, sq, 10, exclude=excl, **kw)
            res[name]["lat"].append(time.time() - t0)
            res[name]["p10"].append(hits(r, q))
            res[name]["ov"].append(len({x for x, _ in r} & fullset) / 10)
    ms = lambda v, p: float(np.percentile(np.array(v) * 1000, p))  # noqa: E731
    print(f"\n{'':16s} {'p50 ms':>8s} {'p99 ms':>8s} {'P@10':>7s} {'ovl':>6s}")
    print("-" * 50)
    print(f"{'FULL scan':16s} {ms(lat_full,50):8.0f} {ms(lat_full,99):8.0f} "
          f"{np.mean(p10F):7.3f} {'1.00':>6s}")
    for name, _ in LEGS:
        r = res[name]
        print(f"{name:16s} {ms(r['lat'],50):8.0f} {ms(r['lat'],99):8.0f} "
              f"{np.mean(r['p10']):7.3f} {np.mean(r['ov']):6.2f}")
    print(f"\nprefilter recall of full top-10:  @50 {np.mean(recall50):.3f}"
          f"   @200 {np.mean(recall200):.3f}")
    R.log("vjstore_bench", store=store_name, n=len(qs),
          p10_full=round(float(np.mean(p10F)), 4),
          recall50=round(float(np.mean(recall50)), 4),
          recall200=round(float(np.mean(recall200)), 4),
          **{n.replace("=", "").replace(" ", "_").replace(".", ""):
             round(float(np.mean(r["p10"])), 4) for n, r in res.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--store", default="rcasa")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.bench:
        bench(a.store, a.queries, a.seed)
