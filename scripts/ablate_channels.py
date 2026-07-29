"""What is each channel actually worth? Leave-one-out on the truthset.

Eight channels is not a feature, it is a cost: every one of them is a
model that must run at ingest. This drops each channel in turn, through
the PRODUCTION path (scenario.search_set reads ELIDEDB_DROP_CHANNELS),
and reports what the frozen truthset says happened.

A channel that can be removed without moving true/returned is not
contributing. Everything loads once, so all runs share one process.

  python scripts/ablate_channels.py [store]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb.store import Store  # noqa: E402

QUERIES = [
    "pick up a green object from table and put it into the drawer",
    "pick up a yellow object from table and put it into the drawer",
    "pick up a red object from the drawer and put it on the table",
    "pick up a vessel and put it on the stove",
    "robot arm holds the handle and closes the drawer",
    "robot arm opens the drawer",
    "fold a piece of towel",
    "put the lid on the pot",
    "place the spoon on top of the cloth",
    "put the eggplant into the drawer",
    "put the banana on top of the drawer",
]
CHANNELS = ["pe", "act", "sig2", "iv2", "vid", "obj", "mot", "prf", "conj"]
K = 10


def evaluate(db, truth, support, covered):
    from elidedb.scenario import search_set
    ret = tru = 0
    per = {}
    for qi, q in enumerate(QUERIES):
        if qi not in covered:
            continue
        r = search_set(db, q, purity="fast", k_max=K)
        clips = r["clips"]
        t = sum(1 for c in clips
                if truth.get((qi, c["stream"], int(c["t0"]))) == 1)
        ret += len(clips)
        tru += t
        per[qi] = (t, len(clips))
    return tru, ret, per


def main():
    store = sys.argv[1] if len(sys.argv) > 1 else "lake/_bench_recovered"
    db = Store.open(store)
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth, support = {}, {}
    for q, s, t0, v in zip(t["query_id"], t["stream"], t["t0"], t["true"]):
        truth[(int(q), s, int(t0))] = int(v)
        support[int(q)] = support.get(int(q), 0) + int(v)
    covered = set(support)

    print(f"store: {store}   queries: {len(covered)}   k={K}\n")
    os.environ.pop("ELIDEDB_DROP_CHANNELS", None)
    t0 = time.perf_counter()
    base_t, base_r, base_per = evaluate(db, truth, support, covered)
    print(f"{'dropped':10s} {'true':>5} {'ret':>5} {'prec':>6}  "
          f"{'delta true':>10}  verdict     ({time.perf_counter()-t0:.0f}s "
          f"for the baseline)")
    bp = base_t / max(base_r, 1)
    print(f"{'(none)':10s} {base_t:>5} {base_r:>5} {bp:>6.3f}  "
          f"{'-':>10}  baseline")

    rows = []
    for c in CHANNELS:
        os.environ["ELIDEDB_DROP_CHANNELS"] = c
        tr, rr, _ = evaluate(db, truth, support, covered)
        d = tr - base_t
        p = tr / max(rr, 1)
        verdict = ("CARRIES IT" if d < -2 else
                   "helps" if d < 0 else
                   "DEAD WEIGHT" if d == 0 else "HURTS")
        rows.append((c, tr, rr, p, d, verdict))
        print(f"{c:10s} {tr:>5} {rr:>5} {p:>6.3f}  {d:>+10}  {verdict}")
    os.environ.pop("ELIDEDB_DROP_CHANNELS", None)

    print("\nRemovable without losing a single true result:")
    dead = [r[0] for r in rows if r[4] >= 0]
    print("  " + (", ".join(dead) if dead else "(none)"))
    print("Channels the result depends on:")
    keep = [r[0] for r in rows if r[4] < 0]
    print("  " + (", ".join(keep) if keep else "(none)"))


if __name__ == "__main__":
    main()
