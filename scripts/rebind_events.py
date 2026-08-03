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
    # AGENT-APPEARANCE VETO. is_agent flags one track, but the proposer
    # shatters the arm into many, and arm fragments have the most
    # contact-gated motion of anything in an event span - measured:
    # motion criteria bound the true mover's colour 17% (concentration)
    # and 5% (naive contact gate). By the store's own identity standard
    # a track whose descriptor matches the agent's gallery at the
    # calibrated instance cut IS the agent; it cannot be the thing the
    # agent is manipulating.
    OV = db.table("object_vectors").scan().to_pydict()
    cut = float((db.table("object_vectors").state().meta
                 or {}).get("match_cut", 0.86))
    dim = len(OV["vector"][0])
    desc = {}
    for s_, a_, b_, o_, v_ in zip(OV["stream"], OV["ts"], OV["t1"],
                                  OV["object_id"], OV["vector"]):
        v_ = np.asarray(v_, np.float32)
        desc[(str(s_), int(a_), int(b_), int(o_))] = \
            v_ / max(float(np.linalg.norm(v_)), 1e-8)
    agent_keys = {(str(tr["stream"][i]), int(tr["track_ts"][i]),
                   int(tr["t1"][i]), int(tr["object_id"][i]))
                  for i in range(len(tr["ts"])) if tr["is_agent"][i]}
    A = np.stack([desc[k] for k in sorted(agent_keys) if k in desc]) \
        if agent_keys else np.zeros((0, dim), np.float32)
    if len(A) > 500:
        A = A[np.random.RandomState(0).choice(len(A), 500, replace=False)]

    def agent_like(k):
        v = desc.get(k)
        if v is None or not len(A):
            return False
        return float((A @ v).max()) >= cut

    tracks = defaultdict(list)
    for i in range(len(tr["ts"])):
        if tr["is_agent"][i]:
            continue
        k = (str(tr["stream"][i]), int(tr["track_ts"][i]),
             int(tr["t1"][i]), int(tr["object_id"][i]))
        tracks[k].append((int(tr["ts"][i]),
                          float(tr["px"][i]), float(tr["py"][i]),
                          bool(tr["contact"][i])))
    vetoed = {k for k in tracks if agent_like(k)}
    for k in vetoed:
        del tracks[k]
    print(f"agent-appearance veto: {len(vetoed):,} of "
          f"{len(vetoed) + len(tracks):,} tracks (cut {cut})")
    for v in tracks.values():
        v.sort()
    by_stream = defaultdict(list)
    total = {}
    for k, pts in tracks.items():
        p = np.asarray([(x, y) for _, x, y, _ in pts])
        t = np.asarray([a for a, _, _, _ in pts])
        c = np.asarray([cc for _, _, _, cc in pts], bool)
        d = np.hypot(*np.diff(p, axis=0).T) if len(p) > 1 else np.zeros(0)
        # a displacement step counts as IN CONTACT if either endpoint is
        cstep = (c[1:] | c[:-1]) if len(c) > 1 else np.zeros(0, bool)
        total[k] = float(d.sum())
        by_stream[k[0]].append((k, t, d, cstep))

    ev = db.table("events").scan()
    d = ev.to_pydict()
    new, changed, bound = [], 0, 0
    for i in range(len(d["ts"])):
        s = str(d["stream"][i])
        e0 = int(d["ev_t0"][i] or d["ts"][i])
        e1 = int(d["ev_t1"][i] or d["t1"][i])
        # CONTACT-GATED motion first: a manipulated object moves WHILE
        # the agent holds it, and the things concentration alone kept
        # picking - arm shadows, table patches, blur transients - move
        # exactly during the event but are never held. Measured before
        # this gate: the bound object matched the true mover's colour
        # 17% of the time. Contactless motion (a toppled block) falls
        # back to the old (concentration, magnitude) order.
        best, key = -1, (-1.0, -1.0, -1.0)
        for k, t, disp, cstep in by_stream.get(s, ()):
            if k[1] > e1 or e0 > k[2] or not len(disp):
                continue
            m = (t[1:] >= e0) & (t[1:] <= e1)
            if m.sum() < 1:
                continue
            span_d = float(disp[m].sum())
            c_d = float(disp[m & cstep].sum())
            conc = span_d / total[k] if total[k] > 1e-6 else 0.0
            if (c_d, conc, span_d) > key:
                best, key = k[3], (c_d, conc, span_d)
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
                               meta={"join": "object_id = most CONTACT-"
                                             "GATED motion in ev span; "
                                             "concentration+magnitude "
                                             "fallback for contactless "
                                             "movers"})
    got = db.table("events").scan().to_pydict()["object_id"]
    assert sum(1 for x in got if int(x) >= 0) == bound
    print("verified after write")


if __name__ == "__main__":
    main()
