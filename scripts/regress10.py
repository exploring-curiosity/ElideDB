"""The pilot's bar: 10 realistic queries, top-10, index-only, cold.

Queries are PARAPHRASES of what the corpus actually contains (users
never type the label verbatim); relevance = the hit's overlapped
episode label satisfies the query's predicate. Every miss is attributed:
which channels ranked it there. Target: 100/100. Latency reported —
quality gains that cost query time are rejected by design.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
import elidedb.verified as V                                 # noqa: E402

V._verdict_map = lambda s: {}
V._verify_segments = lambda *a, **k: {}

QUERIES = [
    ("the robot closes the drawer",
     lambda L: "close" in L and "drawer" in L),
    ("opening the drawer",
     lambda L: "open" in L and "drawer" in L and "put" not in L),
    ("take an object out of the drawer",
     lambda L: "drawer" in L and any(w in L for w in
                                     ("take", "remove", "get", "out"))),
    ("put an object into the drawer",
     lambda L: "drawer" in L and any(w in L for w in
                                     ("put", "place", "move", "in"))),
    ("moving the silver pot",
     lambda L: ("silver" in L or "pot" in L) and
               any(w in L for w in ("move", "put", "place"))),
    ("pick up a yellow object and place it somewhere",
     lambda L: "yellow" in L),
    ("putting a red object down on the table",
     lambda L: "red" in L and any(w in L for w in
                                  ("put", "place", "move"))),
    ("the robot moves a green object",
     lambda L: "green" in L),
    ("picking something up from the table",
     lambda L: "pick" in L or "take" in L or "grab" in L),
    ("moving the pot to the burner",
     lambda L: ("burner" in L or "stove" in L or "pot" in L)),
    # relational / binding shapes (user-reported failing): the OBJECT and
    # the DESTINATION must both be right, not just present
    ("put the lid on a vessel",
     lambda L: "lid" in L),
    ("put the green object in the drawer",
     lambda L: "green" in L and "drawer" in L),
    ("place a toy on top of the towel",
     lambda L: ("towel" in L or "cloth" in L) and
               any(w in L for w in ("put", "place", "move", "top"))),
    ("take something out of the pot",
     lambda L: "pot" in L and any(w in L for w in
                                  ("take", "out", "remove", "get"))),
]


def main():
    store = sys.argv[1] if len(sys.argv) > 1 else "bridge4h"
    db = Store.open(f"lake/{store}")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = [{"t0": int(a), "t1": int(b), "task": (k or "").lower(),
            "stream": stream_of.get(int(i))}
           for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                                 t["task"]) if k]
    by_s = {}
    for e in eps:
        by_s.setdefault(e["stream"], []).append(e)

    def label(s, a, b):
        best = ("?", 0)
        for e in by_s.get(s, []):
            ov = min(b, e["t1"]) - max(a, e["t0"])
            if ov > best[1]:
                best = (e["task"], ov)
        return best[0]

    db.search_context("warmup", k=1)
    total = 0
    lat = []
    for q, pred in QUERIES:
        t0 = time.perf_counter()
        hits, _ = db.search_context(q, k=10)
        lat.append((time.perf_counter() - t0) * 1e3)
        labs = [label(h["stream"], h["t0"], h["t1"]) for h in hits[:10]]
        ok = [pred(l) for l in labs]
        n = sum(ok)
        total += n
        print(f"{n:2d}/10  {q}")
        # attribute every miss: which channels liked it
        attr = getattr(V, "_LAST_ATTR", None)
        for i, (l, o) in enumerate(zip(labs, ok)):
            if o:
                continue
            why = ""
            if attr:
                seg = (hits[i]["stream"], hits[i]["t0"], hits[i]["t1"])
                if seg in attr["segs"]:
                    j = attr["segs"].index(seg)
                    rks = {}
                    for c, vals in attr["channels"].items():
                        v = np.array(vals, float)
                        if not np.isnan(v[j]):
                            rks[c] = int((np.nan_to_num(v, nan=-9) >
                                          v[j]).sum()) + 1
                    why = " ← " + ",".join(f"{c}#{r}" for c, r in
                                           sorted(rks.items(),
                                                  key=lambda kv: kv[1])[:3])
            print(f"      miss@{i + 1}: {l[:44]}{why}")
    print(f"\n== {total}/100 @top10 · p50 {sorted(lat)[5]:.0f}ms · "
          f"store {store} ==")


if __name__ == "__main__":
    main()
