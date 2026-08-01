"""Build the channel teachers, with progress you can watch and trust.

Run it yourself:

    python scripts/build_teachers.py --store lake/fresh_bench
    python scripts/build_teachers.py --store lake/fresh_bench --only iv2,xclip
    python scripts/build_teachers.py --store lake/fresh_bench --force

WHY THIS EXISTS RATHER THAN A SHELL LOOP
----------------------------------------
A shell loop reports that a stage finished. It does not report that the
stage WORKED, and the difference cost an hour: `pe_ingest` printed
"DONE in 434s" and wrote no table, `sig2_ingest` "DONE in 16s" and wrote
no table, and the loop marched on to the third channel because bash does
not stop on error. Reading the loop's position as evidence of success
was wrong.

So this driver verifies the ARTIFACT, never the exit:

  * before each channel it counts the rows already in the target table
  * after it, it counts again
  * a channel that added no rows is FAILED, loudly, regardless of what
    its exit code said

It is also resumable. A channel whose table already has rows is skipped
unless --force, so a run interrupted at channel 4 does not redo 1-3.
Nothing here writes outside the store's own transaction log, so an
interrupted channel leaves the store consistent: the write path commits
atomically or not at all.

PROGRESS BELONGS TO THE INGEST, not to this driver. Each ingest carries
tqdm on its own per-recording loop and writes straight to your terminal,
so the bar advances once per recording and its ETA is real. The earlier
version piped stdout and drove a bar from the scripts' "every 200th"
prints - three movements over a 600-item run, which says nothing about
the minutes between them. This driver only sequences and verifies.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# channel -> (script, table it must fill)
# `motion` is first because it costs seconds and needs no model: it is
# the normalised delta of appearance across each event span, so it rides
# on frame_vectors the write already wrote. Ordering it first means a run
# that gets interrupted still leaves the direction channel in place.
CHANNELS = {
    "motion": ("motion_ingest.py", "motion_vectors"),
    "pe": ("pe_ingest.py", "pe_vectors"),
    "sig2": ("sig2_ingest.py", "sig2_vectors"),
    "iv2": ("iv2_ingest.py", "iv2_vectors"),
    "xclip": ("xclip_ingest.py", "xclip_vectors"),
    "vjepa": ("vjepa_channel.py", "vjepa_vectors"),
    "act": ("action_ingest.py", "action_probs"),
}


def rows(store_path, table):
    """Rows in a table right now, 0 if it does not exist. Opened fresh
    each time: the point is to see what the child just committed."""
    from elidedb import Store
    try:
        db = Store.open(store_path)
        return len(db.table(table).scan()) if table in db.tables() else 0
    except Exception:
        return 0


def run_channel(name, script, table, store_path):
    """Run one ingest, INHERITING the terminal.

    The bar belongs to the ingest, not to this driver. An earlier
    version piped the child's stdout and drove a bar here from its
    "N/M" prints - but those fire every 100-200 recordings, so the bar
    moved three times over a 600-item run and said nothing about the
    minutes in between. The ingests now carry tqdm on their own loop and
    write straight to the terminal; this only sequences them and checks
    what they wrote.
    """
    before = rows(store_path, table)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), str(store_path)])
    after = rows(store_path, table)
    return {"channel": name, "table": table,
            "seconds": round(time.time() - t0, 1),
            "rows_before": before, "rows_after": after,
            "ok": after > before, "exit": proc.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="lake/fresh_bench")
    ap.add_argument("--only", default="", help="comma list, e.g. iv2,xclip")
    ap.add_argument("--force", action="store_true",
                    help="rebuild channels that already have rows")
    a = ap.parse_args()
    store = Path(a.store)
    if not (store / "_store.json").exists():
        raise SystemExit(f"{store} is not a store")
    want = ([c.strip() for c in a.only.split(",") if c.strip()]
            or list(CHANNELS))
    bad = [c for c in want if c not in CHANNELS]
    if bad:
        raise SystemExit(f"unknown channel(s) {bad}; have {list(CHANNELS)}")

    print(f"store {store}")
    todo = []
    for c in want:
        script, tbl = CHANNELS[c]
        have = rows(store, tbl)
        if have and not a.force:
            print(f"  {c:<6} SKIP   {tbl} already has {have:,} rows "
                  f"(--force to rebuild)")
        else:
            todo.append(c)
    if not todo:
        print("nothing to do")
        return
    print(f"building {len(todo)}: {', '.join(todo)}\n")

    results = []
    try:
        for i, c in enumerate(todo, 1):
            script, tbl = CHANNELS[c]
            print(f"\n[{i}/{len(todo)}] {c}  ->  {tbl}", flush=True)
            r = run_channel(c, script, tbl, store)
            results.append(r)
            # THE ARTIFACT IS THE TEST. A channel can exit 0 and write
            # nothing: pe_ingest printed "DONE in 434s" and left an
            # empty table, and reading the loop's position as success
            # cost an hour.
            print(f"  {c}: {'ok' if r['ok'] else 'FAILED'} - "
                  f"{r['rows_after']:,} rows in {tbl} "
                  f"({r['seconds']}s, exit {r['exit']})", flush=True)
            if not r["ok"]:
                print(f"  scroll up for {c}'s own error output",
                      flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted - the store is consistent; "
              "rerun to resume (finished channels are skipped)")

    print(f"\n{'channel':<8}{'table':<16}{'rows':>10}{'seconds':>10}  status")
    for r in results:
        print(f"{r['channel']:<8}{r['table']:<16}{r['rows_after']:>10,}"
              f"{r['seconds']:>10.1f}  {'ok' if r['ok'] else 'FAILED'}")
    n_bad = sum(1 for r in results if not r["ok"])
    if n_bad:
        print(f"\n{n_bad} channel(s) FAILED - they wrote no rows. "
              "A non-zero exit is not required to fail; the artifact is "
              "the test.")
        sys.exit(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
