"""READ BENCHMARK: what every query class costs, in ms and in bytes.

Latency alone hides the thing that matters. A query can be fast because
the corpus is small, because the page cache is warm, or because it
genuinely read almost nothing - and only the third one keeps being fast
as the corpus grows. So every row reports BOTH: wall-clock, and the
bytes the reader had to touch against the bytes in the store.

Query classes, in the order a planner should reach for them:

  metadata      episode by index, verb, label, object - all answered
                from statistics over clustered columns
  join          query-by-example: objects in a clip, then every other
                episode holding one of them
  window        the frame index for a time range - which packets, where
  materialize   the pixels: pread those byte ranges and decode
  scan          the vector table, as the denominator every other row is
                measured against

Warm-cache numbers. macOS `purge` needs a password, so cold reads are
not measured here rather than being guessed at; the byte counts are
cache-independent and carry the argument.

  python scripts/read_bench.py [--store lake/bench] [--reps 50]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402
from elidedb.plan import (label_lookup, like_this_clip,        # noqa: E402
                          object_lookup)
from elidedb.store import QueryStats                           # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402


def timeit(fn, reps):
    """Median and p99 of `reps` runs, plus whatever the call returns."""
    fn()                                       # warm the path, not the disk
    ts, out = [], None
    for _ in range(reps):
        a = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - a) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 99)), out


def main():
    argv = sys.argv
    db = Store.open(argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/bench")
    reps = int(argv[argv.index("--reps") + 1] if "--reps" in argv else 50)

    corpus = sum(db.table(t).state().bytes for t in db.tables())
    ep = db.table("episodes").scan().to_pydict()
    n_ep = len(ep["ts"])
    picks = np.linspace(0, n_ep - 1, 12).round().astype(int)
    oids = sorted({int(v) for v in
                   db.table("instances").scan().column("object_id").to_pylist()})

    rows = []

    def add(name, ms50, ms99, nbytes, detail=""):
        rows.append({"query": name, "p50_ms": round(ms50, 3),
                     "p99_ms": round(ms99, 3), "bytes": int(nbytes),
                     "pct_of_store": round(100 * nbytes / corpus, 5),
                     "elided_pct": round(100 * (1 - nbytes / corpus), 3),
                     "detail": detail})

    # ---- 1. episode by index: a point lookup on a clustered column
    def q_episode():
        st = QueryStats()
        tb = db.table("episodes").scan(columns=["ts", "t1", "episode_index",
                                                "stream"], stats=st)
        i = int(picks[len(rows) % len(picks)])
        tb.filter(pc.equal(tb.column("episode_index"),
                           int(ep["episode_index"][i])))
        return st.bytes_touched
    m, p, b = timeit(q_episode, reps)
    add("episode by index", m, p, b, f"{n_ep} episodes")

    # ---- 2. verb: events clustered by kind
    kinds = sorted({str(k) for k in
                    db.table("events").scan().column("kind").to_pylist()})

    def q_verb():
        tb, st = db.table("events").scan_values(
            "kind", ["open"], columns=["stream", "ts", "kind"])
        return st.bytes_touched, len(tb)
    m, p, (b, n) = timeit(q_verb, reps)
    add("verb = open", m, p, b, f"{n} events of {len(kinds)} kinds")

    # ---- 3. label: the inverted index
    vals = sorted({str(v) for v in
                   db.table("labels").scan().column("value").to_pylist()})

    def q_label():
        got, st = label_lookup(db, [vals[0]])
        return st.bytes_touched, len(got or ())
    m, p, (b, n) = timeit(q_label, reps)
    add(f"label = {vals[0]}", m, p, b, f"{n} episodes, {len(vals)} values")

    # ---- 4. object: instances clustered by object_id
    def q_object():
        got, st = object_lookup(db, [oids[len(oids) // 2]])
        return st.bytes_touched, len(got or ())
    m, p, (b, n) = timeit(q_object, reps)
    add("object by id", m, p, b, f"{n} episodes, {len(oids)} objects")

    # ---- 5. query by example: two index lookups, no vectors
    i = int(picks[6])

    def q_example():
        r = like_this_clip(db, str(ep["stream"][i]), int(ep["ts"][i]),
                           int(ep["t1"][i]))
        return r["bytes_touched"], len(r["episodes"] or ()), len(r["objects"])
    m, p, (b, n, no) = timeit(q_example, reps)
    add("like this clip", m, p, b, f"{no} objects -> {n} episodes")

    # ---- 6. window: the frame index for a 2 s range
    ft = db.table("frames")

    def q_window():
        st = QueryStats()
        a = int(ep["ts"][i])
        tb = ft.scan(t0=a, t1=a + 2_000_000_000, stats=st)
        return st.bytes_touched, len(tb)
    m, p, (b, n) = timeit(q_window, reps)
    add("2 s frame index", m, p, b, f"{n} frames")

    # ---- 7. materialize: pread the byte ranges and decode
    def q_decode():
        st = QueryStats()
        a = int(ep["ts"][i])
        tb = ft.scan(t0=a, t1=a + 2_000_000_000, stats=st)
        fs = FrameSet(db, "frames", tb)
        got = fs.decode()
        return st.bytes_touched + fs.last_bytes_read, len(got)
    m, p, (b, n) = timeit(q_decode, max(reps // 5, 5))
    add("2 s decode (pixels)", m, p, b, f"{n} frames decoded")

    # ---- 8. the denominator: a full vector scan
    fv = "frame_vectors"

    def q_scan():
        st = QueryStats()
        tb = db.table(fv).scan(columns=["ts", "vector"], stats=st)
        return st.bytes_touched, len(tb)
    m, p, (b, n) = timeit(q_scan, max(reps // 10, 3))
    add("full vector scan", m, p, b, f"{n} vectors")

    w = max(len(r["query"]) for r in rows)
    print(f"\n{'query':<{w}} {'p50 ms':>9} {'p99 ms':>9} {'bytes':>13} "
          f"{'% store':>9} {'elided':>9}   detail")
    for r in rows:
        print(f"{r['query']:<{w}} {r['p50_ms']:>9.3f} {r['p99_ms']:>9.3f} "
              f"{r['bytes']:>13,} {r['pct_of_store']:>8.4f}% "
              f"{r['elided_pct']:>8.3f}%   {r['detail']}")
    print(f"\nstore {corpus:,} B over {len(db.tables())} tables, "
          f"{n_ep} episodes")
    (ROOT / "bench_read.json").write_text(json.dumps(
        {"corpus_bytes": corpus, "episodes": n_ep, "reps": reps,
         "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
