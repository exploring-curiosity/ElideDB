"""Backfill events.role, which the one-pass writer hardcodes empty.

THE DEFECT. `role` is one of the five elements - it says WHICH PART an
event's entity plays, and without it `agent`, `participants` and
`container` are indistinguishable rows in one table. build_teacher.py
assigns it correctly:

    kind agent            -> role agent
    kind contact/release  -> role participant
    kind open/close       -> role container

but scripts/write_once.py, which is what the fast one-pass write actually
runs, contains

    "role": pa.array(["" for _ in ev_rows]),

i.e. it hardcodes the column empty. Two writers for one table, one of
them silently dropping an element. Measured on lake/fresh_bench: 15,175
event rows, 100% of role empty.

WHY THIS IS A FIX AND NOT A REBUILD. role is a pure function of kind
under the definition above - no pixels, no models, no decode. The mapping
is read off build_teacher rather than invented here, so the two writers
agree afterwards instead of drifting further.

This store's transition kinds are the corpus's own discovered type ids
(t0..t14) rather than build_teacher's open/close, because the one-pass
path types transitions by clustering instead of by a cavity threshold.
They are relocations OF A PARTICIPANT, so they take role participant -
stated explicitly because it is the one mapping not literally present in
build_teacher.

    python scripts/fix_event_roles.py [--store lake/fresh_bench] [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                     # noqa: E402

TYPE_ID = re.compile(r"^t\d+$")


def role_of(kind):
    k = (kind or "").strip()
    if k == "agent":
        return "agent"
    if k in ("open", "close"):
        return "container"
    if k in ("contact", "release") or TYPE_ID.match(k):
        return "participant"
    return ""


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    dry = "--dry-run" in argv
    db = Store.open(str(store))
    tb = db.table("events").scan()
    d = tb.to_pydict()
    before = sum(1 for r in d["role"] if r)
    roles = [role_of(k) for k in d["kind"]]
    import collections
    print(f"{store.name}: {len(tb)} events, role populated "
          f"{before} -> {sum(1 for r in roles if r)}")
    for r, n in collections.Counter(roles).most_common():
        print(f"    {r or '(unmapped)':<14}{n:>8}")
    unmapped = sorted({k for k, r in zip(d["kind"], roles) if not r})
    if unmapped:
        print(f"  UNMAPPED KINDS (left empty, deliberately): {unmapped}")
    if dry:
        print("--dry-run: nothing written")
        return
    d["role"] = roles
    out = pa.table({c: pa.array(d[c], tb.schema.field(c).type)
                    for c in tb.column_names})
    # replace, not append: this corrects rows that already exist
    db.table("events").replace(out, kind="events",
                              meta={"fix": "role backfilled from kind "
                                           "(write_once hardcoded empty)"})
    n = sum(1 for r in db.table("events").scan().to_pydict()["role"] if r)
    print(f"  verified after write: {n} rows carry a role")


if __name__ == "__main__":
    main()
