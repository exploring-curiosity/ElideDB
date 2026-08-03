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

    # agent box per timestamp + agent scale: the two RELATIONAL gates
    # the film demanded (drawn boxes showed table-sized proposals and
    # arm joints winning every previous criterion) - a thing the agent
    # can hold is at most agent-sized, and while held it travels
    # INSIDE the agent's own box
    agent_box = {}
    agent_areas = []
    for i in range(len(tr["ts"])):
        if not tr["is_agent"][i]:
            continue
        bx = (int(tr["x0"][i]), int(tr["y0"][i]),
              int(tr["x1"][i]), int(tr["y1"][i]))
        agent_box[(str(tr["stream"][i]), int(tr["ts"][i]))] = bx
        agent_areas.append(max(bx[2] - bx[0], 1) * max(bx[3] - bx[1], 1))
    max_area = float(np.median(agent_areas)) if agent_areas else 1e18

    def inside_agent(s_, ts_, x_, y_, pad=0.2):
        bx = agent_box.get((s_, ts_))
        if bx is None:
            return False
        w_, h_ = bx[2] - bx[0], bx[3] - bx[1]
        return (bx[0] - pad * w_ <= x_ <= bx[2] + pad * w_
                and bx[1] - pad * h_ <= y_ <= bx[3] + pad * h_)

    tracks = defaultdict(list)
    areas = defaultdict(list)
    for i in range(len(tr["ts"])):
        if tr["is_agent"][i]:
            continue
        k = (str(tr["stream"][i]), int(tr["track_ts"][i]),
             int(tr["t1"][i]), int(tr["object_id"][i]))
        ts_ = int(tr["ts"][i])
        x_, y_ = float(tr["px"][i]), float(tr["py"][i])
        held = bool(tr["contact"][i]) and inside_agent(k[0], ts_, x_, y_)
        tracks[k].append((ts_, x_, y_, held))
        areas[k].append(max(int(tr["x1"][i]) - int(tr["x0"][i]), 1)
                        * max(int(tr["y1"][i]) - int(tr["y0"][i]), 1))
    vetoed = {k for k in tracks if agent_like(k)}
    oversize = {k for k in tracks
                if k not in vetoed and np.median(areas[k]) > max_area}
    for k in vetoed | oversize:
        del tracks[k]
    print(f"agent-appearance veto: {len(vetoed):,}, oversize veto: "
          f"{len(oversize):,} of {len(vetoed) + len(oversize) + len(tracks):,}"
          f" tracks (cut {cut}, agent med area {max_area:.0f}px)")
    for v in tracks.values():
        v.sort()
    # candidates are OBJECTS, not tracks: with identity consolidated,
    # a block's evidence spans its tracks - motion during the event
    # from one, existence at rest from another
    by_obj = defaultdict(list)
    total_obj = defaultdict(float)
    for k, pts in tracks.items():
        p = np.asarray([(x, y) for _, x, y, _ in pts])
        t = np.asarray([a for a, _, _, _ in pts])
        c = np.asarray([cc for _, _, _, cc in pts], bool)
        d = np.hypot(*np.diff(p, axis=0).T) if len(p) > 1 else np.zeros(0)
        # a displacement step counts as IN CONTACT if either endpoint is
        cstep = (c[1:] | c[:-1]) if len(c) > 1 else np.zeros(0, bool)
        ko = (k[0], k[3])
        total_obj[ko] += float(d.sum())
        by_obj[ko].append((t, d, cstep, c))

    def at_rest_outside(ko, e0, e1, pad_ns=int(0.3e9)):
        """Does this object EXIST AT REST outside the event span? The
        true mover is a persistent thing - it sits still before the
        pick or after the place - while the junk that motion criteria
        kept electing (arm shadows, blur transients) exists only while
        something moves. >=5 contact-free samples outside the span
        whose accumulated drift stays under 8px."""
        for t, d, cstep, c in by_obj[ko]:
            out = ((t < e0 - pad_ns) | (t > e1 + pad_ns)) & ~c
            if out.sum() < 5:
                continue
            so = (t[1:] < e0 - pad_ns) | (t[1:] > e1 + pad_ns)
            if len(d) and float(d[so & ~cstep].sum()) < 8.0:
                return True
        return False

    ev = db.table("events").scan()
    d = ev.to_pydict()
    new, changed, bound = [], 0, 0
    for i in range(len(d["ts"])):
        s = str(d["stream"][i])
        e0 = int(d["ev_t0"][i] or d["ts"][i])
        e1 = int(d["ev_t1"][i] or d["t1"][i])
        # Selection order: REST-BACKED objects first (persistent things
        # that sit still outside the event), then contact-gated motion
        # (a manipulated object moves while held), then concentration
        # and magnitude. Each layer was added against a measured
        # failure: concentration alone bound arm shadows (17% true-
        # mover colour), contact alone elected arm fragments (5%), and
        # rest-existence is what none of that junk has.
        best, key = -1, (-1, -1.0, -1.0, -1.0)
        for ko in list(by_obj):
            if ko[0] != s:
                continue
            c_d = span_d = 0.0
            hit = False
            for t, disp, cstep, c in by_obj[ko]:
                if not len(disp) or t[0] > e1 or t[-1] < e0:
                    continue
                m = (t[1:] >= e0) & (t[1:] <= e1)
                if m.sum() < 1:
                    continue
                hit = True
                span_d += float(disp[m].sum())
                c_d += float(disp[m & cstep].sum())
            if not hit:
                continue
            conc = (span_d / total_obj[ko]
                    if total_obj[ko] > 1e-6 else 0.0)
            rest = int(at_rest_outside(ko, e0, e1))
            if (rest, c_d, conc, span_d) > key:
                best, key = ko[1], (rest, c_d, conc, span_d)
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
                               meta={"join": "object_id = rest-backed "
                                             "object with most contact-"
                                             "gated motion in ev span "
                                             "(agent-appearance veto; "
                                             "conc+magnitude tiebreak)"})
    got = db.table("events").scan().to_pydict()["object_id"]
    assert sum(1 for x in got if int(x) >= 0) == bound
    print("verified after write")


if __name__ == "__main__":
    main()
