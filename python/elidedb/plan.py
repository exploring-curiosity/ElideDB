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

# the transition kinds that get a boolean column on the episode row.
# Low cardinality is the requirement: a bool column's min/max IS the
# set-membership test, so this only works for a closed, small set.
FLAG_KINDS = ("open", "close", "put_into", "put_on", "take_out",
              "adjust", "contact", "release")


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


def plan(store, text, nouns=None, kinds=None):
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

    return {"candidates": cand, "steps": steps,
            "bytes_touched": stats.bytes_touched,
            "corpus_bytes": stats.corpus_bytes}
