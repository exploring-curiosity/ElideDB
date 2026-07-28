"""R3 slice 1: rewrite fp32 vector columns to fp16 + BYTE_STREAM_SPLIT.

Measured before this existed (docs/ENGINE.md): zstd over fp32 vectors is
1.08x (mantissas are entropy); fp16+BSS+zstd is 2.39x with cosine top-10
overlap 0.9990. The rewrite is a normal log commit (op=compress): the old
files stay as the previous version (time travel intact) until vacuum.

Gates before any real store is migrated (run on a COPY):
  ELIDEDB_BENCH_STORE=<copy> python scripts/bench_truth.py   (frozen bench)
  elide vindex/vselftest on the rewritten tables (recall >= 0.98)

Usage: python scripts/compress_vectors.py <store> [table ...]
       (default: every table whose schema has fixed_size_list<float>)
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

ROW_GROUP = 4096  # pruning granularity for the rewritten files


def fp16_schema(schema: pa.Schema) -> pa.Schema | None:
    """The same schema with every fixed_size_list<float> demoted to
    halffloat; None if the table has no fp32 vector column."""
    fields, hit = [], False
    for f in schema:
        t = f.type
        if pa.types.is_fixed_size_list(t) and pa.types.is_float32(t.value_type):
            fields.append(pa.field(f.name, pa.list_(
                pa.field(t.value_field.name, pa.float16()), t.list_size)))
            hit = True
        else:
            fields.append(f)
    return pa.schema(fields) if hit else None


def rewrite(store: Store, name: str) -> dict | None:
    t = store.table(name)
    st = t.state()
    if not st.files:
        return None
    first = pq.read_schema(t.dir / st.files[0].path)
    target = fp16_schema(first)
    if target is None:
        return None
    bss_cols = [f"{f.name}.list.element" for f in target
                if pa.types.is_fixed_size_list(f.type)
                and pa.types.is_float16(f.type.value_type)]
    old_bytes = st.bytes
    add, remove = [], []
    for fe in st.files:
        tbl = pq.read_table(t.dir / fe.path).cast(target)
        new = f"part-{uuid.uuid4().hex[:12]}.parquet"
        pq.write_table(
            tbl, t.dir / new, row_group_size=ROW_GROUP, compression="zstd",
            use_dictionary=False, use_byte_stream_split=bss_cols,
            write_statistics=True, write_page_index=True)
        ts = tbl.column("ts")
        add.append(FileEntry(new, len(tbl), (t.dir / new).stat().st_size,
                             pa.compute.min(ts).as_py(),
                             pa.compute.max(ts).as_py()))
        remove.append(fe.path)
    schema_str = "\n".join(f"{f.name}: {f.type}" for f in target)
    version = t.log.commit(
        op="compress", kind=st.kind, schema=schema_str, add=add,
        remove=remove,
        meta={**st.meta, "encoding": "fp16-bss",
              "compress_note": "fp32->fp16+byte_stream_split, "
                               "measured 0.9990 top-10 overlap"})
    new_bytes = sum(a.bytes for a in add)
    return {"table": name, "version": version, "old_bytes": old_bytes,
            "new_bytes": new_bytes, "ratio": old_bytes / max(new_bytes, 1)}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    store = Store.open(sys.argv[1])
    names = sys.argv[2:] or store.tables()
    total_old = total_new = 0
    for name in names:
        r = rewrite(store, name)
        if r is None:
            continue
        print(f"{r['table']:20s} v{r['version']}: "
              f"{r['old_bytes']:>12,} -> {r['new_bytes']:>12,} "
              f"({r['ratio']:.2f}x)")
        total_old += r["old_bytes"]
        total_new += r["new_bytes"]
    if total_old:
        print(f"{'TOTAL':20s}     {total_old:>12,} -> {total_new:>12,} "
              f"({total_old / max(total_new, 1):.2f}x)  "
              f"(old versions remain until vacuum)")


if __name__ == "__main__":
    main()
