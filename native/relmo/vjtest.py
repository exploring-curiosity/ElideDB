"""Test a PINNED snapshot on a corpus it was never trained on. No training.

This is the read path exactly as shipped - the trained head from the snapshot,
weights not updated - scored against the same ground truth and the same three
metrics as rcasa, so the numbers are comparable to 0.855 test / 0.772 ood
rather than being a new scale.

TWO THINGS THIS FILE REFUSES TO DO SILENTLY.

  1. It re-hashes every head checkpoint against the snapshot before scoring.
     A snapshot that names weights it did not actually run is worse than no
     snapshot, and "I loaded the tag" is not evidence the file is unchanged.

  2. It reports EVERY seed and their spread, never a single number. A single
     checkpoint of this head has read 0.678 where its own config averages
     0.603 +/- 0.049; one seed is an anecdote.

WHY A QUERY SAMPLE IS HONEST BUT A POOL SAMPLE IS NOT. Precision depends on
how much of the corpus can outrank a true match, so the POOL must always be
the whole corpus - shrinking it inflates every number. The query set only
decides the standard error, so it may be sampled; --queries reports the count
and the 95% CI it buys. An 80-episode pool once flattered a result by 10x
here, which is why the two are not treated alike.

Cross-view pairs (same rollout, different camera) stay barred, as everywhere.

    python -m relmo.vjtest --snapshot v9 --dataset rcasa_atomic_full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrel, vjrank2  # noqa: E402
from relmo.vjeval import l2  # noqa: E402
from relmo.vjmatch import dtw  # noqa: E402
from relmo.vjsnap import SNAP, sha  # noqa: E402
from relmo.vjzeval import PAD_COST, _pad  # noqa: E402

MIN_SUPPORT = 5


def score(Z, qids, pool_ids, AM, rng=None, n_q=0):
    """-> dict of prec@support, NDCG@support, wAUC, with counts."""
    pool = sorted(pool_ids)
    q = sorted(qids)
    if n_q and n_q < len(q):
        q = [q[i] for i in sorted(rng.choice(len(q), n_q, replace=False))]
    uni = list(dict.fromkeys(q + pool))
    up = {i: k for k, i in enumerate(uni)}
    REL, _ = vjrel.relevance(uni, AM)
    P, ok = _pad([l2(Z[i]) for i in pool])
    L = np.array([len(Z[i]) for i in pool])
    pe = np.array([AM[i]["rollout"] for i in pool])
    pg = np.array([AM[i]["event"] for i in pool])
    parr = np.array(pool)

    prec, NS, WA, CH, sup_tot, used = [], [], [], [], 0, 0
    per_query = {}
    for qq in q:
        keep = pe != AM[qq]["rollout"]           # cross-view stays barred
        if keep.sum() < MIN_SUPPORT:
            continue
        sv = pg[keep] == AM[qq]["event"]
        k = int(sv.sum())
        if k < MIN_SUPPORT:
            continue
        # a random ranking gets support/pool of the top-k right. Without it a
        # precision is unreadable across corpora, because a bigger pool lowers
        # the floor: 0.355 over 2250 and 0.900 over 447 are not the same
        # distance above chance.
        CH.append(k / int(keep.sum()))
        C = 1.0 - np.einsum("sd,nkd->nsk", l2(Z[qq]), P[keep])
        C = np.where(ok[keep][:, None, :], C, PAD_COST)
        s = -dtw(C, False, L[keep]).astype(np.float64)
        o = np.argsort(-s)
        prec.append(sv[o[:k]].sum() / k)
        per_query[qq] = (prec[-1], CH[-1])
        sup_tot += k
        used += 1
        rel = REL[up[qq]][[up[i] for i in parr[keep]]]
        if rel.max() > 0:
            N = len(s)
            rk = np.empty(N)
            rk[o] = np.arange(N)
            dd = 1 / np.log2(np.arange(2, k + 2))
            NS.append((rel[o[:k]] * dd).sum()
                      / max((np.sort(rel)[::-1][:k] * dd).sum(), 1e-9))
            pct = 1.0 - rk / (N - 1)
            ws = np.sort(rel)[::-1]
            ps = np.sort(pct)[::-1]
            WA.append(((rel * pct).sum() - (ws * ps[::-1]).sum())
                      / max((ws * ps).sum() - (ws * ps[::-1]).sum(), 1e-9))
    pa = np.array(prec)
    ch = float(np.mean(CH)) if CH else float("nan")
    return dict(prec=float(pa.mean()) if len(pa) else float("nan"),
                ndcg=float(np.nanmean(NS)) if NS else float("nan"),
                wauc=float(np.nanmean(WA)) if WA else float("nan"),
                chance=ch,
                lift=float(pa.mean()) / ch if CH and ch > 0 else float("nan"),
                ci95=float(1.96 * pa.std(ddof=1) / np.sqrt(len(pa)))
                if len(pa) > 1 else float("nan"),
                queries=used, pool=len(pool),
                mean_support=sup_tot / max(used, 1),
                per_query=per_query)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="v9")
    ap.add_argument("--dataset", action="append", required=True,
                    help="repeatable; each is scored against ITSELF as pool")
    ap.add_argument("--queries", type=int, default=0,
                    help="0 = every recording is a query")
    ap.add_argument("--seed", type=int, default=0, help="query-sample seed")
    ap.add_argument("--breakdown", default="",
                    help="dataset whose task set defines 'seen in training'; "
                         "queries are then reported split by it. DIAGNOSTIC "
                         "ONLY - the task name is read from the TRAINING "
                         "manifest to label a result, never to retrieve one, "
                         "and nothing here reaches the read path.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sf = SNAP / f"{a.snapshot}.json"
    if not sf.exists():
        raise SystemExit(f"no snapshot at {sf}")
    snap = json.loads(sf.read_text())
    from relmo.vjrank import CKPT
    for tag, rec in snap["head"].items():
        got = sha(CKPT / f"{tag}.pt")
        if got != rec["sha256"]:
            raise SystemExit(f"{tag}.pt is {got}, snapshot {a.snapshot} pins "
                             f"{rec['sha256']} - refusing to score a head the "
                             f"snapshot does not describe")
    tags = sorted(snap["head"])
    print(f"snapshot {a.snapshot} @ {snap['git_commit'][:8]} | heads verified: "
          f"{', '.join(tags)}", flush=True)

    # the meta cache is filled on FIRST call - prime it with every corpus in
    # play or the new ones silently read as empty
    AM = vjrel.all_meta(tuple(["rcasa", "rcasa_eval"] + a.dataset))
    rng = np.random.default_rng(a.seed)
    from relmo.vjrel import parse                              # noqa: F401
    seen_tasks = seen_scenes = None
    if a.breakdown:
        beps = R.read_manifest(a.breakdown)["episodes"]
        seen_tasks = {parse(e["id"])["task"] for e in beps}
        seen_scenes = {str(e.get("scene", "?")) for e in beps}
        print(f"breakdown vs {a.breakdown}: {len(seen_tasks)} tasks and "
              f"{len(seen_scenes)} scenes seen in training", flush=True)
    results = {}
    for ds in a.dataset:
        t0 = time.time()
        D = vjrank2.load_corpus(datasets=(ds,))
        have = sorted(set(D) & set(AM))
        print(f"\n{ds}: {len(D)} records loaded, {len(have)} with ground "
              f"truth, median {int(np.median([len(D[i]['tok']) for i in D]))} "
              f"steps", flush=True)
        if len(have) < 50:
            print(f"  SKIPPED - too few graded records")
            continue
        per_seed = []
        for tag in tags:
            m, _ = vjrank2.load_ckpt(tag)
            Z = vjrank2.encode_all(m, D, ids=have)
            r = score(Z, have, have, AM, rng, a.queries)
            per_seed.append(r)
            print(f"  {tag:16s} prec {r['prec']:.3f} +/-{r['ci95']:.3f}  "
                  f"(chance {r['chance']:.3f}, {r['lift']:.2f}x)  "
                  f"NDCG {r['ndcg']:.3f}  wAUC {r['wauc']:.3f}  "
                  f"({r['queries']} queries, pool {r['pool']}, "
                  f"mean support {r['mean_support']:.1f})", flush=True)
            if seen_tasks is not None:
                # 2x2: a familiar TASK in an unfamiliar SCENE is a different
                # failure from a new task, and only one of them is fixed by
                # more task variety. Reported separately or the two are
                # indistinguishable in the combined number.
                for tl, tw in (("task seen", True), ("task UNSEEN", False)):
                    for sl, sw in (("scene seen", True),
                                   ("scene UNSEEN", False)):
                        v = [(p, c) for i, (p, c) in r["per_query"].items()
                             if (parse(i)["task"] in seen_tasks) is tw
                             and (AM[i]["scene"] in seen_scenes) is sw]
                        if len(v) < 10:
                            continue
                        pp = np.array([x[0] for x in v])
                        cc = np.array([x[1] for x in v])
                        print(f"    {tl:12s} {sl:13s} prec {pp.mean():.3f}  "
                              f"chance {cc.mean():.3f}  "
                              f"{pp.mean()/cc.mean():.2f}x  ({len(v)} q)",
                              flush=True)
        agg = {k: (float(np.mean([s[k] for s in per_seed])),
                   float(np.std([s[k] for s in per_seed], ddof=1))
                   if len(per_seed) > 1 else 0.0)
               for k in ("prec", "ndcg", "wauc", "chance", "lift")}
        print(f"  {'SEED MEAN':16s} prec {agg['prec'][0]:.3f}+/-{agg['prec'][1]:.3f}"
              f"  (chance {agg['chance'][0]:.3f}, {agg['lift'][0]:.2f}x)  "
              f"NDCG {agg['ndcg'][0]:.3f}+/-{agg['ndcg'][1]:.3f}  "
              f"wAUC {agg['wauc'][0]:.3f}+/-{agg['wauc'][1]:.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        results[ds] = dict(
            seeds={t: {k: v for k, v in s.items() if k != "per_query"}
                   for t, s in zip(tags, per_seed)},
            mean={k: v[0] for k, v in agg.items()},
            sd={k: v[1] for k, v in agg.items()},
            n_records=len(have))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
        print(f"\nVERIFIED: wrote {a.out}")
    R.log("vjtest", snapshot=a.snapshot,
          **{ds: round(r["mean"]["prec"], 4) for ds, r in results.items()})


if __name__ == "__main__":
    main()
