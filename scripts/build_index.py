"""Turn extracted structure into DATABASE STATISTICS.

The elements were already being computed and then thrown into a vector
space, which meant a query for "a yellow object" had to encode text and
multiply against every episode in the corpus. That is a filesystem with
a similarity function on top. A database keeps statistics and answers a
value predicate from them.

Two artifacts, both derived from tables the write path already produces:

  labels     the inverted index: one row per (episode, kind, value),
             SORTED BY (kind, value) so a lookup is a contiguous range
             and the zone maps drop everything else. Row groups are
             aligned to `value`, so the group for 'yellow object' is the
             only group a lookup for it decompresses.

  episodes.has_<kind>   a boolean per transition kind on the episode row.
             min/max statistics on a bool ARE set membership, so a row
             group whose max is false cannot contain the action and is
             skipped without being read. This only works because the
             kind set is closed and small - it would be wrong for names.

  python scripts/build_index.py [--store lake/bench]
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.plan import FLAG_KINDS                          # noqa: E402


def main():
    argv = sys.argv
    db = Store.open(argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/bench")
    t0 = time.time()
    ev_name = "events" if "events" in db.tables() else "events_s"
    ev = db.table(ev_name).scan().to_pydict()
    ep = db.table("episodes").scan()

    # ---- labels: one row per (episode, kind, value)
    rows = defaultdict(set)
    for i in range(len(ev["ts"])):
        key = (str(ev["stream"][i]), int(ev["ts"][i]), int(ev["t1"][i]))
        k = ev["kind"][i]
        if k in FLAG_KINDS or k == "agent":
            rows[key].add(("action", k))
        nm = (ev["name"][i] or "").strip().lower()
        if nm:
            rows[key].add(("object", nm))

    L = {"ts": [], "t1": [], "stream": [], "kind": [], "value": [],
         "ep_ts": []}
    for (s, a, b), pairs in rows.items():
        for kind, val in pairs:
            # `ts` carries the episode start so the schema law holds and
            # time pruning still works on this table; ep_ts is the join
            # key back to the episode, kept explicit rather than implied.
            L["ts"].append(a); L["t1"].append(b); L["stream"].append(s)
            L["kind"].append(kind); L["value"].append(val)
            L["ep_ts"].append(a)
    lab = pa.table({
        "ts": pa.array(L["ts"], pa.int64()),
        "t1": pa.array(L["t1"], pa.int64()),
        "stream": pa.array(L["stream"]),
        "kind": pa.array(L["kind"]),
        "value": pa.array(L["value"]),
        "ep_ts": pa.array(L["ep_ts"], pa.int64()),
    })
    # GROUPED BY VALUE: the lookup unit is the value, so that is the row
    # group boundary. A probe for one name decompresses one group.
    # MEASURED, not guessed. Sweeping rows-per-group on this table:
    #   1 -> 406 groups, footer 40.3% of a 711 KB file, 40.4% read
    #  64 ->  29 groups, footer 11.5% of 188 KB,        12.3% read
    # 256 ->  14 groups, footer  6.6% of 163 KB,         9.8% read  <-
    # 1024 ->  6 groups, footer  3.9% of 130 KB,        12.8% read
    # 2048 ->  4 groups, footer  3.3% of 110 KB,        42.1% read
    # A U-curve with two different failure modes: too small and the
    # footer dominates (it is read on EVERY query, so the index costs
    # more to consult than the table costs to scan); too large and a
    # group spans many values so a single-value probe drags in data it
    # cannot use. 256 is the floor of that curve here.
    GROUP_ROWS = int(argv[argv.index("--group-rows") + 1]
                     if "--group-rows" in argv else 256)
    db.table("labels").append_grouped(
        lab, "value", kind="index", replace=True,
        sort_by=["kind", "value", "ts"], min_group_rows=GROUP_ROWS,
        meta={"builder": "build_index", "source": ev_name,
              "min_group_rows": GROUP_ROWS})

    # ---- episode flags
    have = defaultdict(set)
    for i in range(len(ev["ts"])):
        have[(str(ev["stream"][i]), int(ev["ts"][i]))].add(ev["kind"][i])
    # APPEND columns to the existing Arrow table rather than rebuilding
    # from pydict: a round trip re-infers types and silently widened
    # n_frames from int32 to int64, which the store's validator refused
    # outright. It is right to refuse - a type change that slips through
    # a rebuild is exactly the kind of drift a schema law exists to stop.
    # IDEMPOTENT. Appending derived columns without dropping the previous
    # ones made the second run produce a table with duplicate field
    # names, and Parquet cannot even OPEN that - `episodes` became
    # unreadable until it was restored from version 1 through the log.
    # A derived-column builder has to be re-runnable; the base columns
    # are whatever this builder did not add.
    derived = {f"has_{k}" for k in FLAG_KINDS} | {"n_labels"}
    base = [c for c in ep.column_names if c not in derived]
    ep = ep.select(base)
    d = ep.to_pydict()
    n = len(d["ts"])
    ep2 = ep
    for k in FLAG_KINDS:
        ep2 = ep2.append_column(
            f"has_{k}", pa.array(
                [k in have.get((str(d["stream"][i]), int(d["ts"][i])), ())
                 for i in range(n)], pa.bool_()))
    ep2 = ep2.append_column("n_labels", pa.array(
        [len(rows.get((str(d["stream"][i]), int(d["ts"][i]),
                       int(d["t1"][i])), ())) for i in range(n)],
        pa.int32()))
    ep2 = ep2.take(pc.sort_indices(ep2.column("ts")))
    db.table("episodes").replace(ep2, kind="timeseries", evolve=True,
                                 meta={"builder": "build_index"})

    st = db.table("labels").state()
    import json
    print(json.dumps({
        "labels_rows": lab.num_rows,
        "distinct_values": len(set(L["value"])),
        "row_groups": st.meta.get("row_groups"),
        "labels_bytes": st.bytes,
        "episodes_flagged": n,
        "seconds": round(time.time() - t0, 2)}, indent=1))


if __name__ == "__main__":
    main()
