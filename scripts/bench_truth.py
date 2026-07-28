"""THE LEDGER BENCH — automatic, self-grading, runs after every change.

lake/bench contains only truthset-adjudicated episodes, so returns are
graded by table lookup: precision of returned (strict: a return graded
for a DIFFERENT query counts false — unverifiable is not deliverable),
yield = true_returned / min(k, support), support printed. Every run
appends a ledger row to BENCHMARKS.md with the git commit — claims are
diffs in a ledger, not narratives.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_product import QUERIES                            # noqa: E402
from elidedb import Store                                    # noqa: E402
from elidedb.scenario import search_set                      # noqa: E402

K = 10


def main():
    # ELIDEDB_BENCH_STORE points the SAME measurement at a store copy for
    # migration gates (e.g. the fp16 rewrite); the ledger append is skipped
    # then so BENCHMARKS.md rows always describe lake/bench itself.
    import os
    store_path = os.environ.get("ELIDEDB_BENCH_STORE", "lake/bench")
    db = Store.open(store_path)
    t = pq.read_table("eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {}
    support = {}
    for q, s, t0, v in zip(t["query_id"], t["stream"], t["t0"],
                           t["true"]):
        truth[(int(q), s, int(t0))] = int(v)
        support[int(q)] = support.get(int(q), 0) + int(v)
    covered = sorted(support)
    rows = []
    for qi, q in enumerate(QUERIES):
        if qi not in covered and qi != 6:
            continue
        r = search_set(db, q, purity="fast", k_max=K)
        clips = r["clips"]
        if qi == 6:                       # fold: zero support, gate test
            ok = r.get("no_match", False) or not clips
            rows.append((qi, q, len(clips), 0, 0, 0,
                         "PASS" if ok else "FAIL", r["ms"]))
            print(f"q{qi:02d} no-match gate "
                  f"{'PASS' if ok else 'FAIL'}  {q}")
            continue
        tru = ung = 0
        for c in clips:
            v = truth.get((qi, c["stream"], int(c["t0"])))
            if v is None:
                ung += 1
            elif v == 1:
                tru += 1
        sup = support.get(qi, 0)
        prec = tru / len(clips) if clips else None
        yld = tru / min(K, sup) if sup else None
        rows.append((qi, q, len(clips), tru, ung, sup, prec, yld,
                     r["ms"]))
        print(f"q{qi:02d} ret {len(clips):2d} true {tru:2d} "
              f"ungraded {ung}  sup {sup:3d}  "
              f"prec {prec if prec is None else f'{prec:.2f}'}  "
              f"yield {yld if yld is None else f'{yld:.2f}'}  "
              f"{r['ms']:5.0f}ms  {q}", flush=True)

    graded = [r for r in rows if len(r) == 9]
    n = sum(r[2] for r in graded)
    tp = sum(r[3] for r in graded)
    mp = (sum(r[6] for r in graded if r[6] is not None)
          / max(1, len([r for r in graded if r[6] is not None])))
    my = (sum(r[7] for r in graded if r[7] is not None)
          / max(1, len([r for r in graded if r[7] is not None])))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True,
                            text=True).stdout.strip()
    stamp = datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%d %H:%M")
    line = (f"| {stamp} | {commit} | {tp}/{n} returned true | "
            f"mean prec {mp:.2f} | mean yield {my:.2f} | "
            + " ".join(f"q{r[0]:02d}:{r[3]}/{r[2]}" for r in graded)
            + " |\n")
    if store_path != "lake/bench":
        print(f"\n== mean precision {mp:.2f}, mean yield {my:.2f}, "
              f"{tp}/{n} returned true — gate run on {store_path}, "
              f"ledger NOT appended ==")
        return
    led = Path("BENCHMARKS.md")
    txt = led.read_text() if led.exists() else "# Benchmarks\n"
    if "## Truthset ledger" not in txt:
        txt += ("\n## Truthset ledger (lake/bench, k=10, strict: "
                "ungraded=false)\n\n| when | commit | true/returned | "
                "mean prec | mean yield | per-query |\n"
                "|---|---|---|---|---|---|\n")
    led.write_text(txt + line)
    print(f"\n== mean precision {mp:.2f}, mean yield {my:.2f}, "
          f"{tp}/{n} returned true — ledger row appended ==")


if __name__ == "__main__":
    main()
