"""TRAJECTORIES + the joins the architecture was missing.

The spec (docs/ELEMENTS.md, user 2026-08-01): mot is DELETED - a delta
of appearance is not a trajectory. A trajectory is the PATH of a
participant: per participant AND agent, its position over time, where
the position is the AGENT-CONTACT POINT when agent and participant
boxes overlap and the centroid otherwise (v1: no pose). z waits on the
Depth Pro verdict, which is the user's call - this table is 2D by
construction and gains a z column later rather than being rebuilt.

One decode+propose+track pass (~28 min measured: decode 6.8 + propose
19.2 + track 1.2), persisting what close_track used to throw away - the
per-frame (ts, box) series. Three products:

    trajectories   one row per (interval, sample): position, box,
                   trajectory point, contact flag, is_agent flag
    agent join     per episode, the AGENT's object_id - the self-moving
                   thing, chosen by path-length x persistence, geometry
                   only, no dataset knowledge
    events join    events gains object_id (the participant that MOVED
                   most during the event's span) + agent_object_id,
                   via schema-evolving replace

object_id comes from matching each track to the presence interval with
the same (stream, ts, t1, box0): the pass is a re-run of the same
deterministic pipeline that built presence, and the match rate is
REPORTED, not assumed. Unmatched tracks keep object_id -1 and stay in
the table - a trajectory is real even when its id failed to attach.

    python scripts/build_trajectories.py [--store lake/fresh_bench]
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402
from elidedb.identity import Stream, propose                   # noqa: E402
from build_dinov3 import segments                              # noqa: E402

CKPT_EVERY = 300


def collect(db):
    """The pass: every closed track with its full (ts, box) series."""
    ft, segs = segments(db)
    ck = db.dir / "_cache" / "build_traj.npz"
    tracks, done = [], 0
    if ck.exists():
        d = np.load(ck, allow_pickle=True)
        tracks = list(d["tracks"])
        done = int(d["done"])
        print(f"resuming at segment {done} ({len(tracks):,} tracks held)")
    print("loading FastSAM...", flush=True)
    _ = propose([np.zeros((64, 64, 3), np.uint8)])
    cost = defaultdict(float)
    for seg_no in tqdm(range(done, len(segs)), initial=done,
                       total=len(segs), desc="segments"):
        sname, i, n_b = segs[seg_no]
        a = time.time()
        chunk = FrameSet(db, "frames", ft.slice(i, n_b)).decode()
        cost["decode"] += time.time() - a
        if chunk:
            chunk = sorted(chunk)
            ims = [c[1] for c in chunk]
            a = time.time()
            dets = propose(ims)
            cost["propose"] += time.time() - a
            a = time.time()
            st = Stream(views=0)          # no crops: identity is done
            closed = []
            for (ts, im), b in zip(chunk, dets):
                b = np.asarray(b, np.int32).reshape(-1, 4)
                closed += st.update(int(ts),
                                    (b, np.ones(len(b), np.float32),
                                     np.ones(len(b), np.float32)))
            closed += st.flush()
            cost["track"] += time.time() - a
            for _, t in closed:
                tracks.append((sname, seg_no, t["ts"], t["t1"],
                               np.asarray(t["t"], np.int64),
                               np.asarray(t["box"], np.int32)))
        if (seg_no + 1) % CKPT_EVERY == 0:
            np.savez(ck, tracks=np.array(tracks, object), done=seg_no + 1)
    print(f"pass cost: "
          f"{json.dumps({k: round(v/60, 1) for k, v in cost.items()})} min",
          flush=True)
    return tracks, ck


def attach_ids(db, tracks):
    """track -> presence.object_id via (stream, ts, t1, box0)."""
    P = db.table("presence").scan().to_pydict()
    key = {}
    for s, a, b, bx, oid in zip(P["stream"], P["ts"], P["t1"],
                                P["box"], P["object_id"]):
        key[(str(s), int(a), int(b), tuple(int(x) for x in bx))] = int(oid)
    ids, hit = [], 0
    for s, _, a, b, _, box in tracks:
        k = (s, int(a), int(b), tuple(int(x) for x in box[0]))
        oid = key.get(k, -1)
        hit += oid >= 0
        ids.append(oid)
    print(f"id attach: {hit:,}/{len(tracks):,} "
          f"({100.0*hit/max(len(tracks),1):.1f}%) matched presence")
    return np.asarray(ids, np.int32)


def pick_agents(tracks, ids):
    """Per segment, the self-moving thing: path length x persistence.

    Geometry only. The agent moves farther, for longer, than anything
    else in a manipulation scene; a corpus where that heuristic fails
    (a static arm, a driving camera) re-fits at its own ingest, and the
    column records the choice so it can be audited.
    """
    by_seg = defaultdict(list)
    for k, (s, seg, a, b, ts, box) in enumerate(tracks):
        by_seg[seg].append(k)
    agent_of = {}
    for seg, ks in by_seg.items():
        best, bs = -1, -1.0
        span = max(max(tracks[k][3] for k in ks)
                   - min(tracks[k][2] for k in ks), 1)
        for k in ks:
            box = tracks[k][5].astype(np.float32)
            if len(box) < 2:
                continue
            cx = (box[:, 0] + box[:, 2]) / 2
            cy = (box[:, 1] + box[:, 3]) / 2
            path = float(np.hypot(np.diff(cx), np.diff(cy)).sum())
            persist = (tracks[k][3] - tracks[k][2]) / span
            s_ = path * persist
            if s_ > bs:
                best, bs = k, s_
        if best >= 0:
            agent_of[seg] = best
    return agent_of


def build_rows(tracks, ids, agent_of):
    """Per-sample rows with the v1 trajectory point."""
    cols = defaultdict(list)
    for k, (s, seg, a, b, ts, box) in enumerate(tqdm(tracks, desc="points")):
        ak = agent_of.get(seg, -1)
        is_agent = (k == ak)
        abox_at = {}
        if ak >= 0 and not is_agent:
            at, ab = tracks[ak][4], tracks[ak][5]
            abox_at = {int(t): bx for t, bx in zip(at, ab)}
        for t, bx in zip(ts, box):
            x0, y0, x1, y1 = (int(v) for v in bx)
            px, py = (x0 + x1) / 2, (y0 + y1) / 2
            contact = False
            ab = abox_at.get(int(t))
            if ab is not None:
                ox0, oy0 = max(x0, int(ab[0])), max(y0, int(ab[1]))
                ox1, oy1 = min(x1, int(ab[2])), min(y1, int(ab[3]))
                if ox1 > ox0 and oy1 > oy0:
                    # the interaction happens where the boxes meet, not
                    # at the object's middle (v1 stand-in for pose)
                    px, py = (ox0 + ox1) / 2, (oy0 + oy1) / 2
                    contact = True
            cols["ts"].append(int(t))
            cols["t1"].append(int(b))
            cols["stream"].append(s)
            cols["object_id"].append(int(ids[k]))
            cols["track_ts"].append(int(a))
            cols["x0"].append(x0); cols["y0"].append(y0)
            cols["x1"].append(x1); cols["y1"].append(y1)
            cols["px"].append(float(px)); cols["py"].append(float(py))
            cols["contact"].append(contact)
            cols["is_agent"].append(is_agent)
    return cols


def bind_events(db, tracks, ids, agent_of):
    """events += object_id (most-moving overlapping participant) and
    agent_object_id, by schema-evolving replace."""
    segs_ep = db.table("episodes").scan().to_pydict()
    # episode -> segment via containment on (stream, span)
    spans = defaultdict(list)
    for k, (s, seg, a, b, ts, box) in enumerate(tracks):
        spans[str(s)].append((int(a), int(b), k))
    for v in spans.values():
        v.sort()
    seg_of_ep = {}
    for s, a, b in zip(segs_ep["stream"], segs_ep["ts"], segs_ep["t1"]):
        for ta, tb, k in spans.get(str(s), ()):
            if ta >= int(a) and ta <= int(b):
                seg_of_ep[(str(s), int(a))] = tracks[k][1]
                break

    ev = db.table("events").scan()
    d = ev.to_pydict()
    obj, aobj = [], []
    by_seg = defaultdict(list)
    for k, (s, seg, *_ ) in enumerate(tracks):
        by_seg[(str(s), seg)].append(k)
    ep_starts = defaultdict(list)
    for s, a, b in zip(segs_ep["stream"], segs_ep["ts"], segs_ep["t1"]):
        ep_starts[str(s)].append((int(a), int(b)))
    for v in ep_starts.values():
        v.sort()
    for i in range(len(d["ts"])):
        s = str(d["stream"][i])
        e0 = int(d["ev_t0"][i] or d["ts"][i])
        e1 = int(d["ev_t1"][i] or d["t1"][i])
        # the event's episode -> its segment -> its agent
        seg = None
        lst = ep_starts.get(s, ())
        j = int(np.searchsorted([a for a, _ in lst], e0, "right")) - 1
        if j >= 0 and e0 <= lst[j][1]:
            seg = seg_of_ep.get((s, lst[j][0]))
        ak = agent_of.get(seg, -1) if seg is not None else -1
        aobj.append(int(ids[ak]) if ak >= 0 else -1)
        best, bs = -1, -1.0
        for k in (by_seg.get((s, seg), ()) if seg is not None else ()):
            if k == ak:
                continue
            ts, box = tracks[k][4], tracks[k][5]
            m = (ts >= e0) & (ts <= e1)
            if m.sum() < 2:
                continue
            bx = box[m].astype(np.float32)
            cx = (bx[:, 0] + bx[:, 2]) / 2
            cy = (bx[:, 1] + bx[:, 3]) / 2
            disp = float(np.hypot(np.diff(cx), np.diff(cy)).sum())
            if disp > bs:
                best, bs = k, disp
        obj.append(int(ids[best]) if best >= 0 else -1)
    d["object_id"] = obj
    d["agent_object_id"] = aobj
    out = pa.table({**{c: pa.array(d[c], ev.schema.field(c).type)
                       for c in ev.column_names},
                    "object_id": pa.array(obj, pa.int32()),
                    "agent_object_id": pa.array(aobj, pa.int32())})
    db.table("events").replace(out, kind="events", evolve=True,
                               meta={"join": "object_id = most-moving "
                                             "participant in ev span; "
                                             "agent from trajectories"})
    n_o = sum(1 for x in obj if x >= 0)
    n_a = sum(1 for x in aobj if x >= 0)
    print(f"events bound: object_id {n_o:,}/{len(obj):,}  "
          f"agent_object_id {n_a:,}/{len(obj):,}")


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    db = Store.open(str(store))
    t0 = time.time()
    tracks, ck = collect(db)
    ids = attach_ids(db, tracks)
    agent_of = pick_agents(tracks, ids)
    print(f"agents picked for {len(agent_of):,} segments")

    cols = build_rows(tracks, ids, agent_of)
    tbl = pa.table({
        "ts": pa.array(cols["ts"], pa.int64()),
        "t1": pa.array(cols["t1"], pa.int64()),
        "stream": pa.array(cols["stream"]),
        "object_id": pa.array(cols["object_id"], pa.int32()),
        "track_ts": pa.array(cols["track_ts"], pa.int64()),
        "x0": pa.array(cols["x0"], pa.int32()),
        "y0": pa.array(cols["y0"], pa.int32()),
        "x1": pa.array(cols["x1"], pa.int32()),
        "y1": pa.array(cols["y1"], pa.int32()),
        "px": pa.array(cols["px"], pa.float32()),
        "py": pa.array(cols["py"], pa.float32()),
        "contact": pa.array(cols["contact"]),
        "is_agent": pa.array(cols["is_agent"]),
    })
    tbl = tbl.take(pc.sort_indices(tbl, sort_keys=[
        ("object_id", "ascending"), ("ts", "ascending")]))
    db.table("trajectories").set_layout("object_id",
                                        sort_by=["object_id", "ts"],
                                        min_group_rows=4096)
    db.table("trajectories").replace(
        tbl, kind="index",
        meta={"unit": "trajectory_sample",
              "point": "agent-contact overlap midpoint else centroid",
              "z": "pending Depth Pro verdict (user gate)"})
    print(f"trajectories: {len(tbl):,} samples, "
          f"{len(set(cols['object_id'])):,} objects, "
          f"contact rate {100.0*np.mean(cols['contact']):.1f}%")

    bind_events(db, tracks, ids, agent_of)

    got = len(db.table("trajectories").scan())
    assert got == len(tbl), (got, len(tbl))
    ck.unlink(missing_ok=True)
    print(f"TOTAL {(time.time()-t0)/60:.1f} min - verified")


if __name__ == "__main__":
    main()
