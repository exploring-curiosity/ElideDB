"""TWO-VIEW chain tokens: cross-view manipulation units. Geometry only.

The single-view ceiling (0.34 mean yield vs oracle 1.00) decomposed
into three measured walls, and every one of them is a projection or
occlusion artifact that a SECOND VIEW resolves:

    blind carries      the block occluded in the gripper in one view
                       is visible in the other; units are the TIME
                       union of associated segments across views
    ON ambiguity       co-location votes from two azimuths - a depth
                       confound is view-specific, so ON = AND(views)
    slot contamination one rest-crop per boundary per view; two
                       samples median out gripper/table pollution

Association across views is FREE: the streams share the episode clock
by construction of ingest (simA/simB timestamps are identical), so
segments that overlap in time are the same manipulation. No
calibration, no models, no text; thresholds are the corpus-fitted
ones from the single-view ladder.

Segmentation keys by (episode, STREAM, object): the identity gallery
can merge one block's A-view and B-view sightings into one id, and
differencing positions across viewpoints would manufacture motion.

    python scripts/chain_2v.py [--store lake/sim_chains]
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
import chain_qbe                                               # noqa: E402
from chain_moves import otsu                                   # noqa: E402

MIN_SEG_S = 0.5
WIN = 0.5


def view_segments(db):
    """Per-view motion segments, the single-view ladder's machinery
    with (episode, stream, object) keys. Returns
    {(e, stream) -> [(a, b, oid, rest_before, rest_after)]} where the
    rests are ((x, y), ts) or None, plus helpers."""
    ep = db.table("episodes").scan().to_pydict()
    spans = sorted((int(t), int(e)) for e, t in
                   zip(ep["episode_index"], ep["ts"]))
    starts = [s for s, _ in spans]
    import bisect

    OV = db.table("object_vectors").scan().to_pydict()
    cut = float((db.table("object_vectors").state().meta
                 or {}).get("match_cut", 0.86))
    desc = {}
    for s_, a_, b_, o_, v_ in zip(OV["stream"], OV["ts"], OV["t1"],
                                  OV["object_id"], OV["vector"]):
        v_ = np.asarray(v_, np.float32)
        desc[(str(s_), int(a_), int(b_), int(o_))] = \
            v_ / max(float(np.linalg.norm(v_)), 1e-8)
    tr = db.table("trajectories").scan().select(
        ["stream", "ts", "px", "py", "object_id", "is_agent",
         "x0", "y0", "x1", "y1", "track_ts", "t1"])
    d = tr.to_pydict()
    agent_keys = {(str(d["stream"][i]), int(d["track_ts"][i]),
                   int(d["t1"][i]), int(d["object_id"][i]))
                  for i in range(len(d["ts"])) if d["is_agent"][i]}
    A = np.stack([desc[k] for k in sorted(agent_keys) if k in desc])
    if len(A) > 500:
        A = A[np.random.RandomState(0).choice(len(A), 500, replace=False)]
    armlike = {k for k, v in desc.items() if float((A @ v).max()) >= cut}
    med_agent_area = np.median([
        max(int(d["x1"][i]) - int(d["x0"][i]), 1)
        * max(int(d["y1"][i]) - int(d["y0"][i]), 1)
        for i in range(len(d["ts"])) if d["is_agent"][i]])

    per_obj = defaultdict(list)
    areas = defaultdict(list)
    for i in range(len(d["ts"])):
        if d["is_agent"][i]:
            continue
        k = (str(d["stream"][i]), int(d["track_ts"][i]),
             int(d["t1"][i]), int(d["object_id"][i]))
        if k in armlike:
            continue
        e = spans[bisect.bisect_right(starts, int(d["ts"][i])) - 1][1]
        ko = (e, str(d["stream"][i]), int(d["object_id"][i]))
        per_obj[ko].append((int(d["ts"][i]), float(d["px"][i]),
                            float(d["py"][i])))
        areas[ko].append(max(int(d["x1"][i]) - int(d["x0"][i]), 1)
                         * max(int(d["y1"][i]) - int(d["y0"][i]), 1))
    per_obj = {k: v for k, v in per_obj.items()
               if np.median(areas[k]) <= med_agent_area}
    med_diag = float(np.sqrt(2 * np.median(
        [np.median(v) for v in areas.values()])))
    thr = 0.5 * med_diag

    series = {}
    for k, rows in per_obj.items():
        rows.sort()
        if len(rows) < 4:
            continue
        t = np.array([r[0] for r in rows], np.int64)
        x = np.array([r[1] for r in rows], np.float32)
        y = np.array([r[2] for r in rows], np.float32)
        nd = np.zeros(len(t), np.float32)
        j0 = 0
        for i in range(len(t)):
            while t[i] - t[j0] > WIN * 1e9:
                j0 += 1
            if np.any(np.diff(t[j0:i + 1]) > 0.15e9):
                continue
            nd[i] = float(np.hypot(x[i] - x[j0], y[i] - y[j0]))
        series[k] = (t, x, y, nd)
    fracs = {k: float((nd >= thr).mean())
             for k, (t, x, y, nd) in series.items() if len(nd) >= 8}
    fbar = otsu(np.array(list(fracs.values())))
    series = {k: series[k] for k, f in fracs.items() if f <= fbar}
    print(f"rest-backed (e,view,obj) series: {len(series):,} "
          f"(bar {fbar:.2f}); thr {thr:.0f}px; diag {med_diag:.0f}px")

    # union runs per (episode, view)
    grid = defaultdict(dict)
    for (e, sv, oid), (t, x, y, nd) in series.items():
        g = grid[(e, sv)]
        for i in range(len(t)):
            if nd[i] >= thr:
                g.setdefault(int(t[i]), {})[oid] = float(nd[i])
    segs = defaultdict(list)
    for (e, sv), g in grid.items():
        tss = sorted(g)
        if not tss:
            continue
        run = [tss[0]]
        for ts_ in tss[1:] + [None]:
            if ts_ is not None and (ts_ - run[-1]) / 1e9 <= 0.45:
                run.append(ts_)
                continue
            a, b = run[0], run[-1]
            if (b - a) / 1e9 >= MIN_SEG_S:
                acc = defaultdict(float)
                for tt in run:
                    for oid, v in g[tt].items():
                        acc[oid] += v
                mover = max(acc, key=acc.get)
                segs[(e, sv)].append((int(a), int(b), int(mover)))
            if ts_ is not None:
                run = [ts_]
    return segs, series, med_diag, thr


def rest_of(series, e, sv, oid, ts, after):
    k = (e, sv, oid)
    if k not in series:
        return None
    t, x, y, nd = series[k]
    import bisect
    if after:
        i = bisect.bisect_right(t, ts)
        rng = range(i, len(t))
    else:
        i = bisect.bisect_left(t, ts) - 1
        rng = range(i, -1, -1)
    for j in rng:
        if nd[j] < 1e9:                      # any sample
            return ((float(x[j]), float(y[j])), int(t[j]))
    return None


def on_at(series, e, sv, mover, pos, ts, med_diag, thr):
    """Co-location with a pre-existing resting participant, one view."""
    if pos is None:
        return 0
    import bisect
    for (e2, sv2, o2), (t2, x2, y2, nd2) in series.items():
        if e2 != e or sv2 != sv or o2 == mover:
            continue
        i2 = bisect.bisect_left(t2, ts)
        for c_ in (i2 - 1, i2):
            if not (0 <= c_ < len(t2)) or abs(int(t2[c_]) - ts) > 0.6e9:
                continue
            if nd2[c_] >= thr:
                continue
            if float(np.hypot(x2[c_] - pos[0],
                              y2[c_] - pos[1])) < 0.9 * med_diag:
                return 1
    return 0


def units(db):
    """Cross-view manipulation units per episode + qualifiers."""
    segs, series, med_diag, thr = view_segments(db)
    out = {}
    eps = sorted({e for e, sv in segs})
    for e in eps:
        allseg = []
        for sv in ("simA", "simB"):
            for a, b, mover in segs.get((e, sv), []):
                allseg.append((a, b, sv, mover))
        allseg.sort()
        # union-find by time overlap (shared clock: overlapping
        # segments across views are one manipulation)
        parent = list(range(len(allseg)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(allseg)):
            for j in range(i + 1, len(allseg)):
                if allseg[j][0] > allseg[i][1] + int(0.3e9):
                    break
                parent[find(i)] = find(j)
        groups = defaultdict(list)
        for i in range(len(allseg)):
            groups[find(i)].append(allseg[i])
        us = []
        for g in groups.values():
            a = min(x[0] for x in g)
            b = max(x[1] for x in g)
            movers = {(sv, m) for _, _, sv, m in g}
            son = eon = trav = 0
            n_on = 0
            labpos = []
            for sv in ("simA", "simB"):
                mv = [m for s2, m in movers if s2 == sv]
                if not mv:
                    continue
                # dominant mover per view
                mvv = mv[0]
                ra = rest_of(series, e, sv, mvv, a, after=False)
                rb = rest_of(series, e, sv, mvv, b, after=True)
                if ra and rb:
                    trav = max(trav, float(np.hypot(
                        rb[0][0] - ra[0][0], rb[0][1] - ra[0][1])))
                if ra:
                    son += on_at(series, e, sv, mvv, ra[0], ra[1],
                                 med_diag, thr)
                if rb:
                    eon += on_at(series, e, sv, mvv, rb[0], rb[1],
                                 med_diag, thr)
                    labpos.append((sv, rb[0], rb[1]))
                n_on += 1
            us.append((a, b, son == max(n_on, 1), eon == max(n_on, 1),
                       trav, labpos))
        us.sort()
        out[e] = us
    return out, med_diag


def colour_slots(db, uu, med_diag):
    import cv2
    import pyarrow.compute as pc
    from elidedb.video import FrameSet
    ft = db.table("frames").scan()
    labs = {}
    from tqdm import tqdm
    for e, us in tqdm(sorted(uu.items()), desc="slots", unit="ep",
                      mininterval=5):
        for ui, (a, b, son, eon, tv, labpos) in enumerate(us):
            vals = []
            for sv, pos, ts_ in labpos:
                sel = ft.filter(pc.and_(
                    pc.equal(ft.column("ts"), int(ts_)),
                    pc.equal(ft.column("stream"), sv)))
                if len(sel) == 0:
                    continue
                dec = FrameSet(db, "frames", sel).decode()
                if not dec:
                    continue
                im = dec[0][1]
                h = int(med_diag * 0.3)
                x0 = max(int(pos[0]) - h, 0)
                y0 = max(int(pos[1]) - h, 0)
                c = im[y0:int(pos[1]) + h, x0:int(pos[0]) + h]
                if c.size < 48:
                    continue
                hh, ww = c.shape[:2]
                core = c[hh // 4:3 * hh // 4 + 1,
                         ww // 4:3 * ww // 4 + 1]
                lab = cv2.cvtColor(core, cv2.COLOR_RGB2LAB) \
                    .reshape(-1, 3)
                vals.append(np.median(lab, 0))
            if vals:
                labs[(e, ui)] = np.mean(vals, 0).astype(np.float32)
    dists = []
    for e in uu:
        ks = [k for k in labs if k[0] == e]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                dists.append(float(np.linalg.norm(labs[ks[i]]
                                                  - labs[ks[j]])))
    cut = otsu(np.array(dists)) if dists else 30.0
    print(f"2v colour-slot cut: {cut:.1f} Lab (otsu, {len(dists)} pairs)")
    slots = {}
    for e, us in uu.items():
        reps = []
        for ui in range(len(us)):
            v = labs.get((e, ui))
            if v is None:
                slots[(e, ui)] = -1
                continue
            hit = None
            for lab0, sid in reps:
                if float(np.linalg.norm(v - lab0)) <= cut:
                    hit = sid
                    break
            if hit is None:
                hit = len(reps)
                reps.append((v, hit))
            slots[(e, ui)] = hit
    return slots


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    uu, med_diag = units(db)
    print(f"episodes {len(uu)}, mean units "
          f"{np.mean([len(v) for v in uu.values()]):.1f}")
    slots = colour_slots(db, uu, med_diag)
    trav = np.array([u[4] for v in uu.values() for u in v])
    tq = np.percentile(trav[trav > 0], [33, 66])
    seqs = {}
    for e, us in uu.items():
        toks = []
        prev = None
        for ui, (a, b, son, eon, tv, _lp) in enumerate(us):
            if prev is not None:
                toks.append((("G", int(min((a - prev) / 4e9, 2)), 0),
                             -1, 0.0, None))
            toks.append((("M", int(son) * 2 + int(eon),
                          int(np.searchsorted(tq, tv))),
                         slots.get((e, ui), -1), (b - a) / 1e9, None))
            prev = b
        seqs[e] = toks
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {}
    for e, tm in zip(t["episode"], t["template"]):
        tmpl[int(e)] = tm
    from chain_moves import bench
    dev = ("swap", "precarious", "push_then_build",
           "build_unstack_move")
    hold = ("relocate_build", "two_sites_merge")
    bench(seqs, tmpl, "DEV 2-view", dev, w_slot=0.6)
    bench(seqs, tmpl, "HOLDOUT 2-view", hold, w_slot=0.6)


if __name__ == "__main__":
    main()
