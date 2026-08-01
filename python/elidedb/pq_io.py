"""Parquet I/O and the no-cache read policy. One concern, one module.

Extracted from store.py, which had reached 1,501 lines doing five
unrelated jobs: byte-level file access, cache policy, read accounting,
table semantics and store semantics. Everything here is LEAF - it knows
about Parquet files and nothing about Tables, Stores or queries - which
is what makes it a safe seam.

THE DATABASE DOES NOT CACHE, deliberately. A buffer pool is real work
with real invalidation rules and it is the LAST thing to build, after
the layout and the pruning are right. Until then a repeat read must cost
what a first read costs, or a benchmark quietly borrows from the OS.
Store files open with F_NOCACHE - scoped to OUR files, so the system
cache and every other process are untouched, which `sudo purge` can
never say. ELIDEDB_CACHE=1 opts back in.

OWNERSHIP, which cost an hour: pyarrow does NOT take ownership of a file
object passed to ParquetFile or read_table, and keeps a reference for
lazy access - so `with _uncached(p) as fh: pq.read_table(fh)` closes
nothing. A leaked Python file object hangs the interpreter at SHUTDOWN
rather than at the leak, which presents as a script that printed all its
output and then sat at 0% CPU forever. _PF owns its handle and closes it.
"""
from __future__ import annotations

import fcntl
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .log import fsync_file

ROW_GROUP_ROWS = 64 * 1024  # the amortize-vs-overfetch dial (Parquet's
                            # row-group size == SDX's chunk_target_rows)
ROW_GROUP_TARGET_BYTES = 8 * 1024 * 1024

_NOCACHE = os.environ.get("ELIDEDB_CACHE", "0") not in ("1", "true")
_F_NOCACHE = 48                      # <sys/fcntl.h>, Darwin


def _row_width(schema: pa.Schema) -> int:
    """Approximate uncompressed bytes per row, for row-group sizing."""
    w = 0
    for f in schema:
        t = f.type
        try:
            if pa.types.is_fixed_size_list(t):
                w += t.list_size * (t.value_type.bit_width // 8)
            else:
                w += t.bit_width // 8
        except (ValueError, AttributeError):
            w += 32                       # strings/lists: a guess is fine
    return max(w, 1)


def _uncached(path):
    """Open a store file so its pages are not retained by the kernel.

    Returns a plain binary file object; pyarrow accepts any file-like,
    so this drops in wherever a path was passed. Falls back to a normal
    open anywhere the fcntl is unavailable (Linux, odd filesystems) -
    losing the no-cache property is worth strictly less than crashing,
    and the caller is told by ELIDEDB_CACHE what it asked for.
    """
    f = open(path, "rb", buffering=0)
    try:
        fcntl.fcntl(f.fileno(), _F_NOCACHE, 1)
    except Exception:
        pass
    return f


class _PF:
    """pq.ParquetFile that OWNS its file object and closes it.

    The first version handed `_uncached(path)` straight to ParquetFile
    and returned. ParquetFile does not take ownership, so every footer
    read leaked one open file object - and a leaked Python file object
    with a live buffer makes the interpreter hang at SHUTDOWN, not at
    the point of the leak. That is why an inspection script printed all
    of its output and then sat at 0% CPU for 27 minutes holding nothing
    visible: it was stuck tearing down, and `lsof` showed zero because
    the fd table was already gone.

    Every read path since the no-cache change was leaking. Closing is
    the fix; being a context manager as well means callers can be
    explicit where it matters.
    """

    __slots__ = ("_fh", "pf")

    def __init__(self, path):
        self._fh = _uncached(path) if _NOCACHE else None
        self.pf = pq.ParquetFile(self._fh if self._fh is not None
                                 else str(path))

    def __getattr__(self, k):
        return getattr(self.pf, k)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        try:
            self.pf.close()
        except Exception:
            pass
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def __del__(self):
        self.close()


def _pf(path):
    """pq.ParquetFile honouring the no-cache policy, and closing."""
    return _PF(path)


def _read(path, **kw):
    """pq.read_table honouring the no-cache policy.

    Reads through a _PF and COPIES the result, so the file object cannot
    outlive this call. `with _uncached(path) as fh: pq.read_table(fh)`
    looks safe and is not - pyarrow keeps a reference to the handle for
    lazy access, so the `with` closes nothing and the leaked handle hangs
    the interpreter at shutdown. That is the same failure the _pf fix
    addressed, surviving in the other reader: a two-table scan would sit
    at 0% CPU forever while a single-table one exited fine.
    """
    if not _NOCACHE:
        return pq.read_table(str(path), **kw)
    f = _PF(path)
    try:
        cols = kw.get("columns")
        filters = kw.get("filters")
        t = f.pf.read(columns=cols)
        if filters is not None:
            t = t.filter(filters)
        return t.combine_chunks()      # materialise before the handle dies
    finally:
        f.close()


def _top(md, c: int) -> str:
    """Top-level column name for row-group column index `c`.

    Parquet flattens nested types to LEAVES, and md.schema.names is the
    leaf list: a fixed-size-list column named `vector` appears there as
    `element`, with path_in_schema `vector.list.element`. Matching a
    projection against the leaf name therefore never matched a vector
    column, so its bytes were never charged - 8.7 MB per row group on
    frame_vectors, silently absent from every elision number in a store
    that is 87% vectors. Metrics that undercount are worse than no
    metrics: they make the headline claim look better than it is.
    """
    return md.row_group(0).column(c).path_in_schema.split(".")[0]


def _col_index(md, name: str):
    """Row-group column index for a TOP-LEVEL column name, or None."""
    for c in range(md.num_columns):
        if _top(md, c) == name:
            return c
    return None


def _max_end(table: pa.Table) -> int:
    """Latest interval END in this table: max(t1), falling back to
    max(ts) for a table that has no t1 (point rows end where they
    start)."""
    col = "t1" if "t1" in table.column_names else "ts"
    try:
        v = pc.max(table.column(col)).as_py()
    except Exception:
        return 0
    return int(v or 0)


def _zone(table: pa.Table, columns) -> dict:
    """File-level min/max for the columns a file is clustered on.

    Recorded in the commit log, so a value predicate can drop whole
    files from JSON already in memory - before any Parquet footer is
    opened. Strings are kept as strings and numbers as numbers; the
    comparison at read time is the caller's, and it must compare like
    with like or it will prune wrongly rather than merely badly.
    """
    z = {}
    for c in columns or ():
        if c not in table.column_names or c == "ts":
            continue      # ts already has its own dedicated pair
        col = table.column(c)
        try:
            lo, hi = pc.min(col).as_py(), pc.max(col).as_py()
        except pa.ArrowNotImplementedError:
            continue      # lists, structs: no order, no zone map
        if lo is not None and hi is not None:
            z[c] = [lo, hi]
    return z


def write_parquet(table: pa.Table, path):
    """One writer for every file in the store. `ts` gets
    DELTA_BINARY_PACKED — timestamps are near-arithmetic, so delta encoding
    beats generic zstd ~3x on that column (the Gorilla/TSDB observation);
    string columns keep dictionary encoding; everything rides zstd.

    Row groups are sized by ROW WIDTH to a byte target, not a fixed row
    count: at 64k rows a 1152-d float32 vector table packed ~295 MB into
    ONE group, so time-range pruning inside a file could skip nothing —
    the elision law applied to layout. A narrow sensor table still gets
    tens of thousands of rows per group; a vector table gets ~1.8k, and a
    2 s window read touches one group instead of the whole file."""
    rows = min(ROW_GROUP_ROWS,
               max(4096, ROW_GROUP_TARGET_BYTES // _row_width(table.schema)))
    dict_cols = [f.name for f in table.schema
                 if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    pq.write_table(table, path, row_group_size=rows,
                   compression="zstd",
                   use_dictionary=dict_cols,
                   # The PAGE INDEX (per-page min/max + offsets) is what
                   # lets a reader skip pages INSIDE a surviving row group
                   # — the layer that makes a columnar file behave like an
                   # index for selective reads. pyarrow omits it by
                   # default, so every file written before 2026-07-28 can
                   # only prune to row-group granularity. It costs a small
                   # constant in the footer and is read only when a query
                   # has a predicate that can use it.
                   write_page_index=True,
                   column_encoding={"ts": "DELTA_BINARY_PACKED"})
    fsync_file(path)  # durability: data reaches disk BEFORE the commit that

