"""METADATA-FIRST PLANNING. Vectors are the last resort, not the first.

The defect this fixes: asking for "a yellow object" ran a text encoder
and a matmul over every episode vector in the corpus, because the only
thing the store indexed was embeddings. That is a filesystem with a
similarity function bolted on, not a database. A database answers a
value predicate from STATISTICS and touches the payload only for rows
that survive.

Three layers, cheapest first, each strictly narrowing the candidate set:

  1  LABELS       an inverted index sorted by (kind, value, ts). A lookup
                  for value='yellow object' is a contiguous RANGE, so the
                  zone maps in the commit log drop whole files and the
                  Parquet row-group statistics drop whole groups before
                  anything decompresses. Exact, not approximate.
  2  FLAGS        has_open / has_close / ... booleans on the episode row.
                  min/max statistics on a bool are free set membership:
                  max=false means the group cannot contain the action, so
                  it is skipped without being read.
  3  VECTORS      only over what layers 1-2 left standing.

The order matters more than any one layer. Layer 1 is exact and costs
kilobytes; layer 3 is fuzzy and costs the corpus. Running 3 first, as
the old path did, means paying the most for the least selective filter.
"""
from __future__ import annotations

import numpy as np
import pyarrow.compute as pc

from .store import QueryStats

def flag_kinds(store):
    """The transition kinds THIS corpus produced, read from the store.

    Was a literal tuple of eight English verbs - open, close, put_into,
    put_on, take_out, adjust, contact, release - which is a table-top
    manipulation prior. A driving log has none of them and a warehouse
    camera has no drawers, so the system could not recognise a domain it
    had not been told about.

    Now: whatever `has_*` columns the write path created, which are
    whatever transition types the corpus turned out to have. Low
    cardinality is still the requirement - a bool column's min/max IS
    the set-membership test - and discovery enforces that by clustering
    rather than by someone keeping the list short.
    """
    if "episodes" not in store.tables():
        return ()
    return tuple(sorted(c[4:] for c in store.table("episodes").scan().column_names
                        if c.startswith("has_")))


def label_lookup(store, values, kinds=None, stats: QueryStats | None = None):
    """Episodes carrying ANY of `values`, from the inverted index alone.

    Returns {(stream, ts)} plus the bytes it cost. No vectors are touched
    and no video is opened; this is a range scan over a sorted string
    column with statistics doing the pruning.
    """
    if "labels" not in store.tables():
        return None, stats
    t = store.table("labels")
    stats = stats or QueryStats()
    vals = [v.lower() for v in values if v]
    if not vals:
        return set(), stats
    # PUSHED DOWN, not filtered afterwards. The table is sorted by
    # (kind, value) with one row group per value, so the reader skips
    # every group whose min/max cannot contain a requested value.
    tb, stats = t.scan_values(
        "value", vals, columns=["stream", "value", "kind", "ep_ts"],
        stats=stats)
    if kinds and len(tb):
        tb = tb.filter(pc.field("kind").isin(list(kinds)))
    out = set(zip([str(s) for s in tb.column("stream").to_pylist()],
                  [int(a) for a in tb.column("ep_ts").to_pylist()]))
    return out, stats


def flag_lookup(store, kinds, stats: QueryStats | None = None):
    """Episodes whose has_<kind> boolean is true, for any kind asked."""
    if "episodes" not in store.tables():
        return None, stats
    t = store.table("episodes")
    stats = stats or QueryStats()
    cols = [f"has_{k}" for k in kinds if f"has_{k}"
            in t.scan().column_names]
    if not cols:
        return None, stats
    tb = t.scan(columns=["stream", *cols], stats=stats)
    m = None
    for c in cols:
        col = np.asarray(tb.column(c).to_pylist(), bool)
        m = col if m is None else (m | col)
    ts = np.asarray(tb.column("ts").to_pylist(), np.int64)
    ss = tb.column("stream").to_pylist()
    return {(str(ss[i]), int(ts[i])) for i in np.where(m)[0]}, stats


def objects_in(store, stream, t0, t1, stats: QueryStats | None = None):
    """Which object ids were visible in [t0, t1] of `stream`.

    A time-range scan of `instances`, which is a timeseries table, so
    the commit-log zone map on ts drops files and the row-group
    statistics drop groups. Returns ids, not names - there are no names.
    """
    if "instances" not in store.tables():
        return set(), stats
    stats = stats or QueryStats()
    tb = store.table("instances").scan(
        t0=t0, t1=t1, columns=["stream", "object_id", "n_frames"],
        stats=stats)
    if not len(tb):
        return set(), stats
    m = pc.equal(tb.column("stream"), stream)
    tb = tb.filter(m)
    return {int(v) for v in tb.column("object_id").to_pylist()}, stats


def object_lookup(store, object_ids, stats: QueryStats | None = None):
    """Episodes containing ANY of these physical objects.

    The point of the whole object store, expressed as a database
    operation: an equality predicate on a CLUSTERED column, answered by
    statistics. `instances` is sorted and grouped by object_id, so the
    lookup is a contiguous range - the commit log drops files, the
    footer drops row groups, the page index drops pages, and only then
    does anything decompress.

    No embedding is touched and no video is opened. This is layer 1 of
    the planner for identity, exactly as `labels` is for vocabulary.
    """
    if "instances" not in store.tables() or not object_ids:
        return None, stats
    stats = stats or QueryStats()
    tb, stats = store.table("instances").scan_values(
        "object_id", [int(o) for o in object_ids],
        columns=["stream", "ep_ts", "object_id"], stats=stats)
    return set(zip([str(s) for s in tb.column("stream").to_pylist()],
                   [int(a) for a in tb.column("ep_ts").to_pylist()])), stats


def like_this_clip(store, stream, t0, t1):
    """Query by example, metadata only: same physical objects, elsewhere.

    Identify the objects in the given window, then find every other
    episode holding one of them. Two index lookups, no vector scan, no
    text anywhere in the path - which is what having a stable id per
    physical object buys.
    """
    stats = QueryStats()
    ids, stats = objects_in(store, stream, t0, t1, stats)
    before = stats.bytes_touched
    eps, stats = object_lookup(store, ids, stats)
    return {"objects": sorted(ids), "episodes": eps,
            "steps": [{"stage": "objects_in", "kept": len(ids),
                       "bytes": before},
                      {"stage": "object_lookup",
                       "kept": len(eps or ()),
                       "bytes": stats.bytes_touched - before}],
            "bytes_touched": stats.bytes_touched,
            "corpus_bytes": stats.corpus_bytes}


def plan(store, text, nouns=None, kinds=None, object_ids=None):
    """Narrow by metadata, report what each layer cost.

    The return value is deliberately auditable: a caller can see how many
    episodes survived each stage and how many bytes it took, which is the
    only way to tell a real prune from a decorative one.
    """
    steps = []
    stats = QueryStats()
    cand = None

    if nouns:
        got, stats = label_lookup(store, nouns, stats=stats)
        if got is not None:
            steps.append({"stage": "labels", "asked": sorted(nouns),
                          "kept": len(got),
                          "bytes": stats.bytes_touched})
            cand = got if cand is None else (cand & got)

    if kinds:
        before = stats.bytes_touched
        got, stats = flag_lookup(store, kinds, stats=stats)
        if got is not None:
            steps.append({"stage": "flags", "asked": sorted(kinds),
                          "kept": len(got),
                          "bytes": stats.bytes_touched - before})
            cand = got if cand is None else (cand & got)

    if object_ids:
        before = stats.bytes_touched
        got, stats = object_lookup(store, object_ids, stats=stats)
        if got is not None:
            steps.append({"stage": "objects",
                          "asked": sorted(int(o) for o in object_ids),
                          "kept": len(got),
                          "bytes": stats.bytes_touched - before})
            cand = got if cand is None else (cand & got)

    return {"candidates": cand, "steps": steps,
            "bytes_touched": stats.bytes_touched,
            "corpus_bytes": stats.corpus_bytes}
