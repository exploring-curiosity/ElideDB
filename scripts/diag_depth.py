"""DEPTH DIAGNOSTIC: is the ranking wrong, or is the cut too tight?

Those are the two failures a low yield can mean, and they have opposite
fixes. Returning 9 clips for a query with 196 true episodes caps yield at
0.046 no matter how good the ranking is - so before touching the ranking,
measure what the ranking WOULD deliver if the cut allowed it.

For each query this prints precision and yield as a function of depth
over the FULL fused ordering, plus where the live cut actually landed.
Read it as:

  prec@sup high, live cut tiny   -> the cut is the problem, ranking fine
  prec@sup low at every depth    -> the ranking is the problem

The truthset is keyed on episode_index (see rekey_truthset.py); returned
clips are graded through the store's own (stream, ts) -> episode_index
map, so grading never depends on how a write laid out its clock.

    python scripts/diag_depth.py                 # q03,q04,q05
    python scripts/diag_depth.py --all
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                     # noqa: E402
from elidedb.scenario import search_set                       # noqa: E402
from _common import queries                                   # noqa: E402

DEPTHS = (5, 10, 25, 50, 100, 150, 200, 300, 500)


def main():
    store = ROOT / (sys.argv[sys.argv.index("--store") + 1]
                    if "--store" in sys.argv else "lake/fresh_bench")
    want = ([int(x) for x in sys.argv[sys.argv.index("--q") + 1].split(",")]
            if "--q" in sys.argv else
            list(range(11)) if "--all" in sys.argv else [3, 4, 5])
    db = Store.open(str(store))
    QS = queries()

    ept = db.table("episodes").scan()
    idx_of = {(s, int(a)): int(i) for s, a, i in
              zip(ept.column("stream").to_pylist(),
                  ept.column("ts").to_pylist(),
                  ept.column("episode_index").to_pylist())}

    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    truth, support = {}, {}
    in_store = set(idx_of.values())
    for q, ei, v in zip(t["query_id"], t["episode_index"], t["true"]):
        ei = int(ei)
        truth[(int(q), ei)] = int(v)
        if ei in in_store:
            support[int(q)] = support.get(int(q), 0) + int(v)

    from tqdm import tqdm
    out = []
    for qi in tqdm(want, desc="depth", unit="q", dynamic_ncols=True):
        if qi >= len(QS):
            continue
        sup = support.get(qi, 0)
        if not sup:
            continue
        k = int(np.ceil(sup * 1.5))
        a = time.perf_counter()
        r = search_set(db, QS[qi], purity="fast", k_max=k,
                       return_ranking=True)
        ms = (time.perf_counter() - a) * 1000
        rank = r.get("ranking") or []
        # grade the full ordering once
        lab = []
        for s_, a_, _sc in rank:
            ei = idx_of.get((s_, int(a_)))
            lab.append(0 if ei is None else truth.get((qi, ei), 0))
        lab = np.asarray(lab, np.int8)
        cum = np.cumsum(lab)
        live = len(r.get("clips") or ())
        row = {"q": qi, "support": sup, "k": k, "live_returned": live,
               "live_true": int(cum[live - 1]) if live and len(cum) else 0,
               "ranked": len(lab), "ms": round(ms, 1), "depth": {}}
        for d in DEPTHS:
            if d > len(lab):
                continue
            tp = int(cum[d - 1])
            row["depth"][d] = {"prec": round(tp / d, 3),
                               "yield": round(tp / sup, 3)}
        # the ceiling: best yield reachable while precision holds >= 0.8
        best = None
        for d in range(1, len(lab) + 1):
            tp = int(cum[d - 1])
            if tp / d >= 0.8:
                best = {"depth": d, "prec": round(tp / d, 3),
                        "yield": round(tp / sup, 3)}
        row["best_at_prec80"] = best
        out.append(row)

    w = 78
    for r in out:
        print("\n" + "=" * w)
        print(f"q{r['q']:02d}  support {r['support']}  k=1.5xsup {r['k']}  "
              f"ranked {r['ranked']}  {r['ms']:.0f}ms")
        print(f"  LIVE: returned {r['live_returned']}, true {r['live_true']}"
              f"  -> yield {r['live_true']/r['support']:.3f}"
              f"  prec {r['live_true']/max(r['live_returned'],1):.3f}")
        print(f"  {'depth':>7} {'prec':>7} {'yield':>7}")
        for d, v in r["depth"].items():
            mark = "  <- k" if d >= r["k"] and d - 25 < r["k"] else ""
            print(f"  {d:>7} {v['prec']:>7.3f} {v['yield']:>7.3f}{mark}")
        b = r["best_at_prec80"]
        print(f"  CEILING at prec>=0.80: "
              + (f"depth {b['depth']}, prec {b['prec']}, yield {b['yield']}"
                 if b else "never reaches 0.80 at any depth"))
    (ROOT / "bench" / "diag_depth.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote bench/diag_depth.json")


if __name__ == "__main__":
    main()
