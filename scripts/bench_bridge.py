"""Contextual retrieval on BridgeData2, graded against human task labels.

WHY THIS BENCHMARK IS DIFFERENT
-------------------------------
Every earlier number in this repo was judged by a VLM, which shares a model
family with the captioner and can only ever be a proxy. Bridge ships a
human-written task string per episode ("put the pot in the sink"). Those
strings are NOT in the database — `scripts/bridge_ingest.py` writes them to
`eval/bridge_truth.parquet`, outside the store, and nothing on the query path
can read it. So this measures the real thing: given a description of an
activity, does the database find the clip of it, having only ever seen pixels?

PROTOCOL
--------
A query is a task string. An episode is relevant if its ground-truth task
equals that string. The ranked segments a query returns are mapped to the
episode each one overlaps most, deduplicated in rank order, and scored with
standard IR metrics. Most tasks in this corpus occur exactly once among ~1300
episodes, so Recall@5 is "did it find the one right clip".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb import context as C                             # noqa: E402


def load_truth(store):
    """Episodes that the context index can actually reach.

    The store may hold episodes from video files that were never embedded
    (frame vectors and captions are built per stream). Those can never be
    returned by any query, so scoring against them would measure ingest
    coverage, not retrieval quality. Restrict to episodes on streams the
    context table covers, and require the episode to overlap at least one
    indexed window.
    """
    t = pq.read_table("eval/bridge_truth.parquet").to_pydict()
    ep = store.table("episodes").scan()
    ep_stream = dict(zip(ep.column("episode_index").to_pylist(),
                         ep.column("stream").to_pylist()))
    ctx = store.table("context").scan()
    covered = {}
    for s, a, b in zip(ctx.column("stream").to_pylist(),
                       ctx.column("ts").to_pylist(),
                       ctx.column("t1").to_pylist()):
        lo, hi = covered.get(s, (a, b))
        covered[s] = (min(lo, a), max(hi, b))
    rows = []
    for i, a, b, task in zip(t["episode_index"], t["ts"], t["t1"], t["task"]):
        i = int(i)
        s = ep_stream.get(i)
        if not task or s not in covered:
            continue
        lo, hi = covered[s]
        if b >= lo and a <= hi:
            rows.append({"ep": i, "t0": int(a), "t1": int(b), "task": task,
                         "stream": s})
    return rows


def segments_to_episodes(hits, eps):
    """Rank-ordered, de-duplicated episode ids for a hit list.

    Matching requires the STREAM to agree, not just the time range. Each
    packed video file gets its own slot on the synthetic timeline, and those
    slots were briefly allowed to overlap (a 5042 s file in a 3000 s slot), so
    time alone could map a segment from one file onto an episode from
    another — scoring a hit as correct because two unrelated clips happened to
    share a timestamp.
    """
    by_stream = {}
    for e in eps:
        by_stream.setdefault(e["stream"], []).append(e)
    out = []
    for h in hits:
        best, bestov = None, 0
        for e in by_stream.get(h["stream"], ()):
            ov = min(h["t1"], e["t1"]) - max(h["t0"], e["t0"])
            if ov > bestov:
                best, bestov = e["ep"], ov
        if best is not None and best not in out:
            out.append(best)
    return out


def metrics(ranked, relevant, ks=(1, 5, 10)):
    rel = set(relevant)
    out = {}
    for k in ks:
        out[f"R@{k}"] = float(len(set(ranked[:k]) & rel) > 0)
    rr = 0.0
    for i, e in enumerate(ranked):
        if e in rel:
            rr = 1.0 / (i + 1)
            break
    out["MRR"] = rr
    dcg = sum((1.0 / np.log2(i + 2)) for i, e in enumerate(ranked[:10])
              if e in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), 10)))
    out["nDCG@10"] = float(dcg / idcg) if idcg else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="lake/bridge")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="also try ranker pairs and weight tilts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    db = Store.open(args.store)
    eps = load_truth(db)
    by_task = {}
    for e in eps:
        by_task.setdefault(e["task"], []).append(e["ep"])
    print(f"{len(eps)} episodes, {len(by_task)} distinct tasks in this store")

    rng = np.random.default_rng(args.seed)
    tasks = sorted(by_task)
    pick = rng.choice(len(tasks), size=min(args.queries, len(tasks)),
                      replace=False)
    queries = [tasks[i] for i in pick]

    configs = {
        # each ranker alone, to see what it actually contributes
        "appearance": {"appearance": 1.0, "context": 0.0, "lexical": 0.0},
        "context":    {"appearance": 0.0, "context": 1.0, "lexical": 0.0},
        "lexical":    {"appearance": 0.0, "context": 0.0, "lexical": 1.0},
        "rrf_all":    {"appearance": 1.0, "context": 1.0, "lexical": 1.0},
    }
    if args.sweep:
        # Weights are the only free parameter in the fused ranker, so they get
        # measured rather than guessed. Pairs first (does the third ranker pay
        # for itself?), then tilts of the full fusion.
        configs.update({
            "app+ctx":     {"appearance": 1.0, "context": 1.0, "lexical": 0.0},
            "app+lex":     {"appearance": 1.0, "context": 0.0, "lexical": 1.0},
            "ctx+lex":     {"appearance": 0.0, "context": 1.0, "lexical": 1.0},
            "lex_heavy":   {"appearance": 1.0, "context": 1.0, "lexical": 2.0},
            "ctx_heavy":   {"appearance": 1.0, "context": 2.0, "lexical": 1.0},
            "app_light":   {"appearance": 0.5, "context": 1.0, "lexical": 1.0},
        })
    results = {n: {"m": [], "ms": []} for n in configs}
    if args.rerank:
        results["rrf_rerank"] = {"m": [], "ms": []}

    for qi, q in enumerate(queries):
        rel = by_task[q]
        for name, w in configs.items():
            t = time.perf_counter()
            hits, _ = C.search(db, q, k=args.k, weights=w)
            ms = (time.perf_counter() - t) * 1e3
            ranked = segments_to_episodes(hits, eps)
            results[name]["m"].append(metrics(ranked, rel))
            results[name]["ms"].append(ms)
        if args.rerank:
            t = time.perf_counter()
            hits, _ = C.search(db, q, k=args.k, rerank=True, rerank_top=8)
            ms = (time.perf_counter() - t) * 1e3
            ranked = segments_to_episodes(hits, eps)
            results["rrf_rerank"]["m"].append(metrics(ranked, rel))
            results["rrf_rerank"]["ms"].append(ms)
        if (qi + 1) % 10 == 0:
            print(f"  {qi + 1}/{len(queries)} queries", flush=True)

    print(f"\n{'method':13s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} "
          f"{'MRR':>6s} {'nDCG':>6s} {'ms':>8s}")
    print("-" * 56)
    summary = {}
    for name, r in results.items():
        agg = {k: float(np.mean([m[k] for m in r["m"]]))
               for k in r["m"][0]}
        agg["median_ms"] = float(np.median(r["ms"]))
        summary[name] = agg
        print(f"{name:13s} {agg['R@1']:6.3f} {agg['R@5']:6.3f} "
              f"{agg['R@10']:6.3f} {agg['MRR']:6.3f} {agg['nDCG@10']:6.3f} "
              f"{agg['median_ms']:8.1f}")

    Path("bench") / "bench_bridge.json".write_text(json.dumps(
        {"summary": summary, "queries": len(queries),
         "episodes": len(eps), "distinct_tasks": len(by_task)}, indent=2))
    print("\nwrote bench_bridge.json")


if __name__ == "__main__":
    main()
