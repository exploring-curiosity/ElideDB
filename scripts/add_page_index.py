"""Rewrite a store's Parquet files so they carry the PAGE INDEX.

Files written before 2026-07-28 have row-group statistics but no page
index, so a reader can only prune to whole row groups. The page index
(per-page min/max plus an offset index) is what lets a scan skip pages
INSIDE a surviving group — the difference between "read 8192 rows to
answer a 40-row question" and reading the pages that hold those rows.

Content is untouched: same rows, same order, same encodings, new
footer. It lands as a normal log commit (op=reindex_pages) so the
change is auditable and the previous version stays readable.

Usage: python scripts/add_page_index.py <store> [table ...]
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb.log import FileEntry  # noqa: E402
from elidedb.store import Store  # noqa: E402


def has_page_index(path: Path) -> bool:
    md = pq.ParquetFile(path).metadata
    if md.num_row_groups == 0:
        return True
    c = md.row_group(0).column(0)
    # both halves are needed: the offset index locates pages, the column
    # index carries their min/max
    return bool(c.has_offset_index and c.has_column_index)


def rewrite(store: Store, name: str) -> dict | None:
    t = store.table(name)
    st = t.state()
    if not st.files:
        return None
    if all(has_page_index(t.dir / f.path) for f in st.files):
        return None
    add, remove = [], []
    before = after = 0
    for fe in st.files:
        src = t.dir / fe.path
        pf = pq.ParquetFile(src)
        tbl = pf.read()
        md = pf.metadata
        rows_per_group = max(md.row_group(0).num_rows, 1)
        schema = tbl.schema
        dict_cols = [f.name for f in schema
                     if pa.types.is_string(f.type)
                     or pa.types.is_large_string(f.type)]
        bss = [f"{f.name}.list.element" for f in schema
               if pa.types.is_fixed_size_list(f.type)
               and pa.types.is_float16(f.type.value_type)]
        new = f"part-{uuid.uuid4().hex[:12]}.parquet"
        kw = dict(row_group_size=rows_per_group, compression="zstd",
                  write_statistics=True, write_page_index=True)
        if bss:
            kw.update(use_dictionary=False, use_byte_stream_split=bss)
        elif dict_cols:
            kw.update(use_dictionary=dict_cols)
        pq.write_table(tbl, t.dir / new, **kw)
        before += fe.bytes
        after += (t.dir / new).stat().st_size
        ts = tbl.column("ts")
        add.append(FileEntry(new, len(tbl), (t.dir / new).stat().st_size,
                             pa.compute.min(ts).as_py(),
                             pa.compute.max(ts).as_py()))
        remove.append(fe.path)
    version = t.log.commit(
        op="reindex_pages", kind=st.kind, schema=st.schema, add=add,
        remove=remove, meta={**st.meta, "page_index": True})
    return {"table": name, "version": version, "before": before,
            "after": after}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    store = Store.open(sys.argv[1])
    names = sys.argv[2:] or store.tables()
    tb = ta = 0
    for name in names:
        r = rewrite(store, name)
        if r is None:
            print(f"{name:20s} already indexed")
            continue
        print(f"{r['table']:20s} v{r['version']}: {r['before']:>12,} -> "
              f"{r['after']:>12,} bytes "
              f"({100 * (r['after'] - r['before']) / max(r['before'], 1):+.2f}%)")
        tb += r["before"]
        ta += r["after"]
    if tb:
        print(f"{'TOTAL':20s}     {tb:>12,} -> {ta:>12,} "
              f"({100 * (ta - tb) / tb:+.2f}% for the index)")


if __name__ == "__main__":
    main()
