"""The 90% question: index recall + BATCH judge = curated precision@10?

Interactive search is index-only (measured zero-shot ceiling ~29%
precision@10 across every composition). Dataset CURATION is a batch
export: latency-free, so the 7B judge (pretrained, pixels-only, no
metadata) is allowed. Pipeline per query: fused index top-30 -> 7B
swap-contrast/clip judgment -> keep top-10 by margin. Graded like
regress10. This is the off-the-shelf number a fresh customer gets on
day one from `export --curated`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elidedb import Store                                    # noqa: E402
import elidedb.verified as V                                 # noqa: E402
from regress10 import QUERIES                                # noqa: E402

V._verdict_map = lambda s: {}                                # cold: no cache


def main():
    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = [{"t0": int(a), "t1": int(b), "task": (k or "").lower(),
            "stream": stream_of.get(int(i))}
           for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                                 t["task"]) if k]
    by_s = {}
    for e in eps:
        by_s.setdefault(e["stream"], []).append(e)

    def label(s, a, b):
        best = ("?", 0)
        for e in by_s.get(s, []):
            ov = min(b, e["t1"]) - max(a, e["t0"])
            if ov > best[1]:
                best = (e["task"], ov)
        return best[0]

    db.search_context("warmup", k=1)
    total = 0
    t_all = time.time()
    for q, pred in QUERIES:
        t0 = time.time()
        # index recall: top-30 candidates, then the judge decides
        hits, _ = db.search_context(q, k=30, pool=40, deep=30,
                                    verify="sync")
        el = time.time() - t0
        top = hits[:10]
        labs = [label(h["stream"], h["t0"], h["t1"]) for h in top]
        n = sum(pred(l) for l in labs)
        total += n
        print(f"{n:2d}/10  ({el:5.0f}s)  {q}", flush=True)
    print(f"\n== CURATED batch export: {total}/140 "
          f"({(time.time() - t_all) / 60:.0f} min total) ==")


if __name__ == "__main__":
    main()
