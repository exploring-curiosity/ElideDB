"""Re-bind events.object_id by motion CONCENTRATION, from trajectories.

The first binding picked the participant with the largest raw in-span
displacement. That confuses "moved during the event" with "moves all
the time": a flickering shadow blob or a jittering detection outranks
the drawer that moved five pixels exactly when the event says
something moved. Concentration fixes it with the same geometry:

    concentration = displacement inside the event span
                    / displacement over the track's whole life

The true participant of a transition moves THEN and rests otherwise -
concentration near 1. Perpetual movers score near their span fraction.
Ties (and the degenerate all-still track) break by in-span magnitude,
so the old criterion survives as the tiebreak rather than the rule.

Reads the trajectories table - no decode, no models, seconds.

    python scripts/rebind_events.py [--store lake/fresh_bench] [--dry-run]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    dry = "--dry-run" in argv
    db = Store.open(str(store))

    tr = db.table("trajectories").scan().to_pydict()
    tracks = defaultdict(list)
    for i in range(len(tr["ts"])):
        if tr["is_agent"][i]:
            continue
        k = (str(tr["stream"][i]), int(tr["track_ts"][i]),
             int(tr["t1"][i]), int(tr["object_id"][i]))
        tracks[k].append((int(tr["ts"][i]),
                          float(tr["px"][i]), float(tr["py"][i])))
    for v in tracks.values():
        v.sort()
    by_stream = defaultdict(list)
    total = {}
    for k, pts in tracks.items():
        p = np.asarray([(x, y) for _, x, y in pts])
        t = np.asarray([a for a, _, _ in pts])
        d = np.hypot(*np.diff(p, axis=0).T) if len(p) > 1 else np.zeros(0)
        total[k] = float(d.sum())
        by_stream[k[0]].append((k, t, d))

    ev = db.table("events").scan()
    d = ev.to_pydict()
    new, changed, bound = [], 0, 0
    for i in range(len(d["ts"])):
        s = str(d["stream"][i])
        e0 = int(d["ev_t0"][i] or d["ts"][i])
        e1 = int(d["ev_t1"][i] or d["t1"][i])
        best, bc, bm = -1, -1.0, -1.0
        for k, t, disp in by_stream.get(s, ()):
            if k[1] > e1 or e0 > k[2] or not len(disp):
                continue
            m = (t[1:] >= e0) & (t[1:] <= e1)
            if m.sum() < 1:
                continue
            span_d = float(disp[m].sum())
            conc = span_d / total[k] if total[k] > 1e-6 else 0.0
            if (conc, span_d) > (bc, bm):
                best, bc, bm = k[3], conc, span_d
        new.append(int(best))
        bound += best >= 0
        changed += int(best) != int(d["object_id"][i])
    n = len(new)
    print(f"{n:,} events: bound {bound:,} ({100.0*bound/n:.0f}%), "
          f"changed {changed:,} ({100.0*changed/n:.0f}%)")
    if dry:
        print("--dry-run: nothing written")
        return
    d["object_id"] = new
    out = pa.table({c: pa.array(d[c], ev.schema.field(c).type)
                    for c in ev.column_names})
    db.table("events").replace(out, kind="events",
                               meta={"join": "object_id = highest motion "
                                             "CONCENTRATION in ev span "
                                             "(in-span/lifetime disp), "
                                             "magnitude tiebreak"})
    got = db.table("events").scan().to_pydict()["object_id"]
    assert sum(1 for x in got if int(x) >= 0) == bound
    print("verified after write")


if __name__ == "__main__":
    main()
