"""Re-key the graded truthset on episode_index, which survives a rewrite.

THE FAILURE THIS FIXES
----------------------
`eval/truthsets/bridge4h.parquet` keyed each graded clip on (stream, t0).
A timestamp is not an identity: the demo-separation write shifts every
demo after the first by a cumulative gap, so t0 is a property of the
WRITE, not of the clip. After the next full write the graded file joined
to 4 of 1122 rows against `bridge4h` and 2 of 1122 against `fresh_bench`
- and it joined SILENTLY, producing a support of 0 rather than an error.
A benchmark run at that moment would have reported a number, and the
number would have been meaningless.

Worse, the gap is a running sum per stream (`ingest.gapped`: each demo
starts at the previous demo's end plus GAP_S). One demo added, dropped or
re-timed anywhere in a stream moves every t0 after it. So a timestamp key
is invalidated by any change to the corpus, not only by a re-gap.

THE RECOVERY
------------
The verdicts were never lost, only the file built from them. Three
artifacts survive from grading time:

  eval/truthsets/verdicts.jsonl   1237 verdicts {q, stream, t0, v}
  eval/truthsets/candidates.parquet   the clips that were shown
  eval/truthsets/sheets/manifest.json the contact sheets that were graded

All three key on the SOURCE timeline (ungapped), because grading happened
before the split. `eval/bridge_full_truth.parquet` carries
(episode_index, ts) for all 50,415 source episodes, so source ts resolves
to episode_index, and episode_index is what the store carries. That chain
resolves 1122 of 1122 graded clips, and reproduces the original per-query
supports exactly (8/17/14/247/165/196/12/18/2/2), which is the check that
says the mapping is right and not merely plausible.

WHY episode_index IS THE RIGHT KEY
----------------------------------
It comes from the source dataset, not from our write. It is stable across
re-writes, re-gaps, subsetting and compression. A store that holds a demo
holds it under the same episode_index no matter how its clock was laid
out. Nothing about eval should depend on a storage decision.

    python scripts/rekey_truthset.py            # writes eval/truthsets/graded.parquet
    python scripts/rekey_truthset.py --check    # verify against every store, write nothing
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                     # noqa: E402

VERDICTS = ROOT / "eval/truthsets/verdicts.jsonl"
SOURCE_TS = ROOT / "eval/bridge_full_truth.parquet"
OUT = ROOT / "eval/truthsets/graded.parquet"

# The supports the grading run produced. Reproducing these exactly is the
# evidence that the ts -> episode_index chain is the right one; a mapping
# that merely resolved every row could still be resolving it wrongly.
EXPECT = {0: 8, 1: 17, 2: 14, 3: 247, 4: 165, 5: 196,
          7: 12, 8: 18, 9: 2, 10: 2}


def load():
    """Graded verdicts re-keyed on episode_index."""
    v = [json.loads(l) for l in VERDICTS.read_text().splitlines() if l.strip()]
    src = pq.read_table(SOURCE_TS).to_pydict()
    ts2idx = {int(a): int(i)
              for i, a in zip(src["episode_index"], src["ts"])}
    rows, miss = [], 0
    for r in v:
        i = ts2idx.get(int(r["t0"]))
        if i is None:
            miss += 1
            continue
        rows.append((int(r["q"]), i, int(bool(r["v"])), r["stream"]))
    return rows, miss, len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    rows, miss, total = load()
    print(f"{total} verdicts -> {len(rows)} re-keyed, {miss} unresolved")

    sup = collections.Counter(q for q, _, t, _ in rows if t)
    bad = {q: (sup.get(q, 0), n) for q, n in EXPECT.items()
           if sup.get(q, 0) != n}
    if bad:
        print("SUPPORT MISMATCH - the mapping is wrong, refusing to write:")
        for q, (got, want) in sorted(bad.items()):
            print(f"   q{q:02d}: got {got}, grading produced {want}")
        sys.exit(1)
    print("supports reproduce the grading run exactly: "
          + " ".join(f"q{q:02d}={sup.get(q, 0)}" for q in sorted(EXPECT)))

    # What each store can actually measure, per query.
    print(f"\n{'store':<14}{'graded':>8}{'queries':>9}   per-query support "
          "(0 = unmeasurable there)")
    for p in sorted((ROOT / "lake").iterdir()):
        if not (p / "_store.json").exists():
            continue
        try:
            db = Store.open(str(p))
            if "episodes" not in db.tables():
                continue
            ep = db.table("episodes").scan()
            if "episode_index" not in ep.column_names:
                continue
        except Exception:
            continue
        have = set(int(i) for i in ep.column("episode_index").to_pylist())
        s = collections.Counter(q for q, i, t, _ in rows if t and i in have)
        n = sum(1 for _, i, _, _ in rows if i in have)
        live = sum(1 for q in EXPECT if s.get(q, 0))
        print(f"{p.name:<14}{n:>8}{live:>6}/{len(EXPECT)}   "
              + " ".join(f"q{q:02d}={s.get(q, 0):<4}" for q in sorted(EXPECT)))

    if a.check:
        print("\n--check: nothing written")
        return
    tb = pa.table({
        "query_id": pa.array([r[0] for r in rows], pa.int32()),
        "episode_index": pa.array([r[1] for r in rows], pa.int64()),
        "true": pa.array([r[2] for r in rows], pa.int8()),
        "stream": pa.array([r[3] for r in rows]),
    })
    pq.write_table(tb, OUT)
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(tb)} rows, "
          "keyed on episode_index)")


if __name__ == "__main__":
    main()
