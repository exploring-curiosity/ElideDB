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
  * a channel that added no rows is FAILED, loudly, with the captured
    error tail - regardless of what its exit code said

It is also resumable. A channel whose table already has rows is skipped
unless --force, so a run interrupted at channel 4 does not redo 1-3.
Nothing here writes outside the store's own transaction log, so an
interrupted channel leaves the store consistent: the write path commits
atomically or not at all.

PROGRESS is real, not decorative. Every ingest prints "  N/M ..." as it
walks recordings; that line drives the bar. A channel that stops
emitting them has stalled, and you will see the bar stop rather than a
spinner that means nothing.
"""
from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# channel -> (script, table it must fill)
CHANNELS = {
    "pe": ("pe_ingest.py", "pe_vectors"),
    "sig2": ("sig2_ingest.py", "sig2_vectors"),
    "iv2": ("iv2_ingest.py", "iv2_vectors"),
    "xclip": ("xclip_ingest.py", "xclip_vectors"),
    "vjepa": ("vjepa_channel.py", "vjepa_vectors"),
    "act": ("action_ingest.py", "action_probs"),
}

PROG = re.compile(r"^\s*(\d+)\s*/\s*(\d+)")


def rows(store_path, table):
    """Rows in a table right now, 0 if it does not exist. Opened fresh
    each time: the whole point is to see what another process just
    committed."""
    from elidedb import Store
    try:
        db = Store.open(store_path)
        return len(db.table(table).scan()) if table in db.tables() else 0
    except Exception:
        return 0


def run_channel(name, script, table, store_path, tqdm):
    before = rows(store_path, table)
    bar = tqdm(total=None, desc=f"{name:<6}", unit="rec",
               bar_format="{desc} {n_fmt}/{total_fmt} {bar} {elapsed}<{remaining}")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / script), str(store_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1)
    tail, t0 = [], time.time()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-40]
            m = PROG.match(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                if bar.total != total:
                    bar.total = total
                    bar.refresh()
                bar.n = done
                bar.refresh()
    except KeyboardInterrupt:
        proc.terminate()
        bar.close()
        raise
    proc.wait()
    bar.close()

    after = rows(store_path, table)
    took = time.time() - t0
    ok = after > before
    return {"channel": name, "table": table, "seconds": round(took, 1),
            "rows_before": before, "rows_after": after, "ok": ok,
            "exit": proc.returncode, "tail": tail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="lake/fresh_bench")
    ap.add_argument("--only", default="", help="comma list, e.g. iv2,xclip")
    ap.add_argument("--force", action="store_true",
                    help="rebuild channels that already have rows")
    a = ap.parse_args()
    try:
        from tqdm import tqdm
    except ImportError:
        raise SystemExit("pip install tqdm")

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
        outer = tqdm(todo, desc="channels", unit="ch", position=0,
                     leave=True)
        for c in outer:
            script, tbl = CHANNELS[c]
            r = run_channel(c, script, tbl, store, tqdm)
            results.append(r)
            if not r["ok"]:
                outer.write(f"  {c}: FAILED - wrote 0 rows to {tbl} "
                            f"(exit {r['exit']}, {r['seconds']}s)")
                for ln in r["tail"][-6:]:
                    outer.write(f"      {ln[:160]}")
            else:
                outer.write(f"  {c}: ok - {r['rows_after']:,} rows in "
                            f"{tbl} ({r['seconds']}s)")
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
