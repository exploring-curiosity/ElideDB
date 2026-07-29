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

# k SCALES WITH THE QUERY, it is not a global constant: a query with
# 247 true episodes and one with 2 are not the same retrieval problem,
# and a fixed k makes the metric mostly a statement about support. The
# operating point is 1.5x support - enough headroom to return
# everything true plus half again - and k stays a CEILING inside it:
# the confidence cut decides how many actually come back.
#
#   yield = true / support          did it find everything that exists
#   prec  = true / returned         is what came back worth reading
#
# ELIDEDB_BENCH_K still forces a fixed k for k-sweep diagnostics.
K_MULT = float(__import__("os").environ.get("ELIDEDB_BENCH_KMULT", "1.5"))
K_FIXED = __import__("os").environ.get("ELIDEDB_BENCH_K")


def main():
    # ELIDEDB_BENCH_STORE points the SAME measurement at a store copy for
    # migration gates (e.g. the fp16 rewrite); the ledger append is skipped
    # then so BENCHMARKS.md rows always describe lake/bench itself.
    import os
    store_path = os.environ.get("ELIDEDB_BENCH_STORE", "lake/bench")
    db = Store.open(store_path)
    # HONEST MODE: score every query with weights fitted WITHOUT it
    # (leave-one-query-out). The shipped artifact was fitted on all ten
    # truthset queries, so the default run is in-sample - useful as a
    # diagnostic, not as a generalization claim. Research consensus on
    # small query sets is nested CV: tune in an inner loop, report from
    # an outer loop that never saw the tuning.
    import json as _json
    folds = None
    if os.environ.get("ELIDEDB_BENCH_HONEST") == "1":
        fp = Path(store_path) / "_set_weights.loqo.json"
        if not fp.exists():
            raise SystemExit(
                f"honest mode needs {fp} - run scripts/fit_set_weights.py "
                f"to produce the per-fold configs")
        folds = _json.loads(fp.read_text())
        print(f"HONEST (leave-one-query-out) mode: {len(folds)} folds")
    # ELIDEDB_TRUTHSET: the demo-separation migration re-keys episodes
    # (timeline gains gaps), so its gate run needs the remapped copy —
    # the original truthset file is never modified.
    t = pq.read_table(os.environ.get(
        "ELIDEDB_TRUTHSET", "eval/truthsets/bridge4h.parquet")).to_pydict()
    truth = {}
    support = {}
    for q, s, t0, v in zip(t["query_id"], t["stream"], t["t0"],
                           t["true"]):
        truth[(int(q), s, int(t0))] = int(v)
        support[int(q)] = support.get(int(q), 0) + int(v)
    covered = sorted(support)
    rows = []
    degraded = {}   # channel -> error, union over all queries
    for qi, q in enumerate(QUERIES):
        if qi not in covered and qi != 6:
            continue
        sup_q = support.get(qi, 0)
        K = (int(K_FIXED) if K_FIXED
             else max(1, int(-(-sup_q * K_MULT // 1))) if sup_q else 10)
        r = search_set(db, q, purity="fast", k_max=K,
                       cfg_override=(folds or {}).get(q))
        # A dead channel silently cost 0.38 -> 0.13 once (2026-07-28,
        # transformers 5 vs the IV2 port). The ledger is a record of
        # the SYSTEM, so a run missing a fitted channel must never be
        # written into it as if it were a model result.
        for c in r.get("degraded", ()):
            degraded[c] = r["channels_failed"][c]
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
        sup = sup_q
        prec = tru / len(clips) if clips else None
        yld = tru / sup if sup else None
        rows.append((qi, q, len(clips), tru, ung, sup, prec, yld,
                     r["ms"]))
        print(f"q{qi:02d} k {K:3d} ret {len(clips):3d} true {tru:3d} "
              f"sup {sup:3d}  "
              f"yield {yld if yld is None else f'{yld:.2f}'}  "
              f"prec {prec if prec is None else f'{prec:.2f}'}  "
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
    line = (f"| {stamp} | {commit} | k=1.5xsup | {tp}/{n} returned true | "
            f"mean yield {my:.2f} | mean prec {mp:.2f} | "
            + " ".join(f"q{r[0]:02d}:{r[3]}/{r[2]}" for r in graded)
            + " |\n")
    if degraded:
        print(f"\n== mean YIELD {my:.2f} (true/support), mean prec {mp:.2f}, "
              f"{tp}/{n} returned true — DEGRADED RUN, ledger NOT "
              f"appended ==")
        for c, err in sorted(degraded.items()):
            print(f"   channel {c} failed: {err}")
        print("   fix the channel (or its environment: see "
              "requirements-local.txt) and rerun.")
        raise SystemExit(2)
    if store_path != "lake/bench":
        print(f"\n== mean YIELD {my:.2f} (true/support), mean prec {mp:.2f}, "
              f"{tp}/{n} returned true — gate run on {store_path}, "
              f"ledger NOT appended ==")
        return
    led = Path("BENCHMARKS.md")
    txt = led.read_text() if led.exists() else "# Benchmarks\n"
    if "## Truthset ledger" not in txt:
        txt += ("\n## Truthset ledger (lake/bench, k=1.5x support, "
                "strict: ungraded=false)\n\n| when | commit | k | "
                "true/returned | mean yield | mean prec | per-query |\n"
                "|---|---|---|---|---|---|---|\n")
    led.write_text(txt + line)
    print(f"\n== mean YIELD {my:.2f} (true/support), mean prec {mp:.2f}, "
          f"{tp}/{n} returned true — ledger row appended ==")


if __name__ == "__main__":
    main()
