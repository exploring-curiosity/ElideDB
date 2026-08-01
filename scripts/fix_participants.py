"""Build the missing `participants` element from the store's OWN media.

THE GAP. Of the five elements, `participants` is absent and `answer` is
half-built. Both trace to one cause: the identity stage died with the
killed write, so nothing ever assigned a stable id per physical object.
`stage_index` then builds the answer's object half from `events.name`,
which is empty and MUST stay empty - naming at write is banned by the
no-text-identity rule.

    scene         frame_vectors      ok
    agent         events role=agent  ok (fixed: role was hardcoded empty)
    participants  instances/objects  MISSING  <- this script
    events        events kind=...    ok
    answer        labels             action terms only, no participants

THE FIX IS AN ID, NOT A NAME. A participant is identified by which
physical instance it is - vision-level, stable across episodes - not by a
word. That is what the rule prescribes and what `presence.object_id`
already is. So the answer's participant half becomes `object:<id>`, which
is corpus-derived, language-free, and works on a corpus of forklifts.

NOT A REBUILD. This decodes the store's own 2,097 media segments through
the same byte-range path a query uses. frames, frame_vectors, events and
every channel table are untouched.

MEMORY. stage_presence held every crop until the end of the run - the
same O(corpus x tracks x pixels) that took the machine to 0.2 GB free
pages and 43 of 44 GB of swap. It now closes each track as it ends
(close_track), so peak is bounded by one batch of frames.

    python scripts/fix_participants.py [--store lake/fresh_bench]
    python scripts/fix_participants.py --limit 200      # smoke test first
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                     # noqa: E402


def rebuild_labels(db):
    """events + presence -> labels, with participants as object ids.

    The action half is unchanged. The object half used events.name and
    got nothing; it now comes from presence: an object whose interval
    overlaps an episode was PRESENT in it, and that is the participant
    fact, stated as an id.
    """
    import pyarrow.compute as pc
    ev = db.table("events").scan().to_pydict()
    ep = db.table("episodes").scan().to_pydict()
    pres = db.table("presence").scan().to_pydict()
    rows = {}
    for i in range(len(ev["ts"])):
        k = (str(ev["stream"][i]), int(ev["ts"][i]), int(ev["t1"][i]))
        if ev["kind"][i]:
            rows.setdefault(k, set()).add(("action", ev["kind"][i]))
        if ev["role"][i]:
            rows.setdefault(k, set()).add(("role", ev["role"][i]))
    # participants: presence intervals overlapping the episode
    by_stream = {}
    for s, a, b, oid in zip(pres["stream"], pres["ts"], pres["t1"],
                            pres["object_id"]):
        by_stream.setdefault(str(s), []).append((int(a), int(b), int(oid)))
    n_part = 0
    for s, a, b in zip(ep["stream"], ep["ts"], ep["t1"]):
        k = (str(s), int(a), int(b))
        for pa_, pb, oid in by_stream.get(str(s), ()):
            if pa_ <= int(b) and int(a) <= pb:          # interval overlap
                rows.setdefault(k, set()).add(("object", f"object:{oid}"))
                n_part += 1
    L = {c: [] for c in ("ts", "t1", "stream", "kind", "value", "ep_ts")}
    for (s, a, b), pairs in rows.items():
        for kind, val in sorted(pairs):
            L["ts"].append(a); L["t1"].append(b); L["stream"].append(s)
            L["kind"].append(kind); L["value"].append(val)
            L["ep_ts"].append(a)
    tbl = pa.table({
        "ts": pa.array(L["ts"], pa.int64()),
        "t1": pa.array(L["t1"], pa.int64()),
        "stream": pa.array(L["stream"]),
        "kind": pa.array(L["kind"]),
        "value": pa.array(L["value"]),
        "ep_ts": pa.array(L["ep_ts"], pa.int64()),
    })
    db.table("labels").set_layout("value", sort_by=["kind", "value", "ts"],
                                  min_group_rows=256)
    db.table("labels").replace(tbl, kind="index",
                               meta={"builder": "fix_participants",
                                     "participants": "object_id, not name"})
    return len(tbl), n_part


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    db = Store.open(str(store))
    have = db.tables()
    for need in ("frames", "events", "episodes"):
        if need not in have:
            raise SystemExit(f"{store} has no {need}; nothing to fix from")
    n_seg = len(list((store / "media").glob("*.h264")))
    print(f"{store.name}: {len(db.table('frames').scan()):,} frames over "
          f"{n_seg:,} media segments")
    if "presence" in have and len(db.table("presence").scan()):
        print("  presence already populated; nothing to do "
              "(delete the table to rebuild)")
    else:
        from full_write import stage_presence
        t0 = time.time()
        n_rows, n_obj, cost = stage_presence(db)
        print(json.dumps({"presence_rows": n_rows, "objects": n_obj,
                          "seconds": round(time.time() - t0, 1),
                          "cost": {k: round(v, 1) for k, v in cost.items()}},
                         indent=1))
        # THE ARTIFACT IS THE TEST
        got = len(db.table("presence").scan()) if "presence" in db.tables() \
            else 0
        if not got:
            raise SystemExit("presence wrote no rows - participants not built")
        print(f"  verified: {got:,} presence intervals, {n_obj} objects")

    n_lab, n_part = rebuild_labels(db)
    print(f"  labels rebuilt: {n_lab:,} rows, {n_part:,} participant facts")
    import collections
    d = db.table("labels").scan().to_pydict()
    print("  label kinds now: "
          + str(collections.Counter(d["kind"]).most_common()))


if __name__ == "__main__":
    main()
