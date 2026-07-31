"""Lay the frame index out for the query unit: the clip.

The frame index is the table every video read goes through, and it was
the worst-laid-out table in the store: 39,026 rows in ONE row group of
981 KB, so a query for a single 6-second episode decompressed every
frame in the corpus and layer-2 pruning measured 0.45%.

Two changes, both about layout rather than content:

  1  an `episode_index` column, joined from the episodes table by ts
     range. The planner already narrows to a set of EPISODES; without
     this column it then has to translate that back into timestamps to
     touch the frames. The join key belongs in the table.

  2  sorted by (stream, episode_index, ts) with row groups sized by the
     same U-curve build_index measured - sorting is what makes the
     statistics prunable, group size is the independent dial.

MEASURED, and the optimum is NOT the one labels wanted:

  rows/grp  groups   file      footer   per-episode read   elided
      64      475   1,352 KB   39.9%        541 KB          60.0%
     256      143     563 KB   29.1%        166 KB          70.4%
    1024       38     304 KB   14.6%         51 KB          83.1%
    4096       10     226 KB    5.5%         33 KB          85.3%   <-
   65536        1     202 KB    1.1%        201 KB           0.5%

labels floored at 256 rows/group and frames floors at 4096, because the
optimum is not a property of the workload alone - it is the ratio of
metadata cost to DATA cost. The whole frames table is ~200 KB of column
data, so per-group footer overhead swamps everything and the right move
is FEW groups; a wide vector table inverts that. A single global
row-group policy is wrong for a multimodal store, which is the argument
the Lance paper makes about Parquet and the reason this had to be
measured per table rather than set once in the writer.

  python scripts/regroup_frames.py [--store lake/bench] [--group-rows N]
"""
from __future__ import annotations

import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.store import QueryStats                         # noqa: E402


def measure(db, n=8):
    """Bytes a real single-episode window read costs, averaged."""
    ep = db.table("episodes").scan(columns=["ts", "t1", "stream"])
    a = ep.column("ts").to_pylist()
    b = ep.column("t1").to_pylist()
    idx = np.linspace(0, len(a) - 1, n).round().astype(int)
    tot_b = tot_c = 0
    rows = 0
    for i in idx:
        qs = QueryStats()
        t = db.table("frames").scan(int(a[i]), int(b[i]), stats=qs)
        tot_b += qs.bytes_touched
        tot_c += qs.corpus_bytes
        rows += len(t)
    return {"bytes_touched": tot_b // n, "corpus_bytes": tot_c // n,
            "elided_pct": round(100 * (1 - tot_b / max(tot_c, 1)), 2),
            "rows_per_read": rows // n}


def main():
    argv = sys.argv
    db = Store.open(argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/bench")
    GROUP = int(argv[argv.index("--group-rows") + 1]
                if "--group-rows" in argv else 4096)
    before = measure(db)

    t0 = time.time()
    fr = db.table("frames").scan()
    ep = db.table("episodes").scan(columns=["ts", "t1", "episode_index",
                                            "stream"])
    # episode lookup per stream: sorted starts + ends, bisect per frame
    by = {}
    for s, a, b, e in zip(ep.column("stream").to_pylist(),
                          ep.column("ts").to_pylist(),
                          ep.column("t1").to_pylist(),
                          ep.column("episode_index").to_pylist()):
        by.setdefault(str(s), []).append((int(a), int(b), int(e)))
    for s in by:
        by[s].sort()
    starts = {s: [x[0] for x in v] for s, v in by.items()}

    fs = fr.column("stream").to_pylist()
    fts = fr.column("ts").to_pylist()
    epi = np.full(len(fts), -1, np.int32)
    for i in range(len(fts)):
        v = by.get(str(fs[i]))
        if not v:
            continue
        j = bisect.bisect_right(starts[str(fs[i])], int(fts[i])) - 1
        if j >= 0 and v[j][0] <= int(fts[i]) <= v[j][1]:
            epi[i] = v[j][2]

    if "episode_index" in fr.column_names:
        fr = fr.drop(["episode_index"])
    fr = fr.append_column("episode_index", pa.array(epi, pa.int32()))
    db.table("frames").append_grouped(
        fr, "episode_index", kind="frame_index", replace=True,
        evolve=True, sort_by=["stream", "episode_index", "ts"],
        min_group_rows=GROUP,
        meta={"builder": "regroup_frames", "min_group_rows": GROUP})
    dt = time.time() - t0

    after = measure(db)
    st = db.table("frames").state()
    print(json.dumps({
        "rows": fr.num_rows, "unmatched_frames": int((epi < 0).sum()),
        "row_groups": st.meta.get("row_groups"),
        "seconds": round(dt, 2),
        "before": before, "after": after}, indent=1))


if __name__ == "__main__":
    main()
