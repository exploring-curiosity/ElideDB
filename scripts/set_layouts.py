"""Declare each element table's cluster key, then rebuild on it.

The five elements were all WRITTEN as columns and only one of them was
INDEXED. A column is metadata a header can prune on only when the file
is clustered on it; otherwise its min/max spans the table and every
lookup opens every row group. Measured before this ran:

    element        column        table      lookup opened
    events         kind          events        100% of RG   <- the verb
    participants   role          events        100%
    answer         verb_seq      answers2      100%
    agent          agent_span    answers2      100%
    scene          scene_vec     answers2      vector, unprunable ever

Only `labels` (14%) and `frames` (10%) were real indexes, and both got
there through append_grouped. Nothing was wrong with the data; the
layout simply never expressed the predicate.

This declares the cluster key on the TABLE, so every future writer
inherits it, then rewrites each table once so existing files match the
declaration. Declaring an index does not reorganise a table - the
rewrite is what does that, and it is explicit here.

  python scripts/set_layouts.py           report only
  python scripts/set_layouts.py --apply   declare + rebuild
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402

# cluster key per table, with the reason it is that column and not
# another. A table can be clustered ONE way; this is the predicate the
# retrieval path actually issues.
LAYOUTS = {
    # the verb, and the role that says agent vs participant vs
    # container. Both are the events table's reason to exist.
    "events": ("kind", ["kind", "role", "ts"], 512),
    "events_s": ("kind", ["kind", "role", "ts"], 512),
    # the ordered rollup. Clustering on it makes "demos whose answer
    # starts with open" a range, since verb_seq is a > -joined string.
    "answers2": ("verb_seq", ["verb_seq", "ts"], 128),
    # the per-transition record: verb first, then the named site.
    "answers": ("verb", ["verb", "kind", "ts"], 256),
}


def audit(db, name, col):
    st = db.table(name).state()
    if not st.files:
        return None
    p = db.table(name).dir / st.files[-1].path
    md = pq.ParquetFile(p).metadata
    if col not in md.schema.names:
        return None
    ci = md.schema.names.index(col)
    rngs, data = [], 0
    for g in range(md.num_row_groups):
        s = md.row_group(g).column(ci).statistics
        b = sum(md.row_group(g).column(c).total_compressed_size
                for c in range(md.num_columns))
        data += b
        if s is None or s.min is None:
            return dict(rg=md.num_row_groups, opens=1.0, bytes=p.stat().st_size)
        rngs.append((s.min, s.max, b))
    foot = p.stat().st_size - data
    vals = sorted({v for lo, hi, _ in rngs for v in (lo, hi)})
    opens = np.mean([sum(1 for lo, hi, _ in rngs if lo <= v <= hi)
                     for v in vals]) / len(rngs)
    cost = np.mean([foot + sum(b for lo, hi, b in rngs if lo <= v <= hi)
                    for v in vals])
    return dict(rg=md.num_row_groups, opens=float(opens), lookup=float(cost),
                bytes=p.stat().st_size)


def main():
    db = Store.open(sys.argv[sys.argv.index("--store") + 1]
                    if "--store" in sys.argv else "lake/bench")
    apply = "--apply" in sys.argv

    print(f"{'table':<12}{'cluster on':<12}{'RG':>4}{'opens':>8}"
          f"{'lookup B':>11}   (before)")
    before = {}
    for name, (col, _, _) in LAYOUTS.items():
        if name not in db.tables():
            continue
        a = audit(db, name, col)
        before[name] = a
        if a:
            print(f"{name:<12}{col:<12}{a['rg']:>4}{a['opens'] * 100:>7.0f}%"
                  f"{a.get('lookup', a['bytes']):>11,.0f}")

    if not apply:
        print("\n--apply to declare and rebuild")
        return

    print()
    for name, (col, sort_by, mgr) in LAYOUTS.items():
        if name not in db.tables():
            continue
        t = db.table(name)
        tb = t.scan()
        if col not in tb.column_names:
            print(f"{name}: no column {col}, skipped")
            continue
        v = t.set_layout(col, sort_by=sort_by, min_group_rows=mgr)
        # rebuild so the FILES match the declaration; replace() now
        # routes through the grouped writer because the layout exists
        t.replace(tb, kind=t.state().kind,
                  meta={"rebuilt_for": "layout", "declared_at": v})

    print(f"{'table':<12}{'cluster on':<12}{'RG':>4}{'opens':>8}"
          f"{'lookup B':>11}   (after)")
    for name, (col, _, _) in LAYOUTS.items():
        if name not in db.tables():
            continue
        a = audit(db, name, col)
        b = before.get(name)
        gain = ("" if not (a and b and b.get("lookup"))
                else f"   {100 * (1 - a['lookup'] / b['lookup']):>4.0f}% fewer bytes")
        if a:
            print(f"{name:<12}{col:<12}{a['rg']:>4}{a['opens'] * 100:>7.0f}%"
                  f"{a.get('lookup', a['bytes']):>11,.0f}{gain}")


if __name__ == "__main__":
    main()
