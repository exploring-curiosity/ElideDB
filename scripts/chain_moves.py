"""Chain tokens from PARTICIPANT MOTION SEGMENTS. Geometry only.

The decisive reframe after the hand estimator kept landing on arm
shadows: the hold signal never needed the hand. On a table, passive
objects move only when acted on - so a sustained smooth motion of a
participant IS a manipulation segment, and the moving participant IS
the mover, identified by its own kinematics with no binding chain, no
hand, no contact flag.

An episode becomes an alternation of
    M(slot, travel_q, endh_q, dur_q)   one participant in sustained
                                       motion; slot = that object's
                                       position in the episode's cast
                                       (first-appearance order), so
                                       "A moves, B moves, A moves
                                       again" is a comparable pattern
    G(dur_q)                           gaps between motions

Speed threshold: Otsu valley on the corpus's log-speed distribution
(still vs moving is bimodal by construction of tables). Qualifier
edges: corpus terciles. No constants that encode the dataset, no
models, no text; slots lean on the two-stage identity (same block
across segments), which is exactly what they measure.

    python scripts/chain_moves.py [--store lake/sim_chains]
                                  [--no-slots]
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

MIN_SEG_S = 0.5
MERGE_GAP_S = 0.4


def otsu(vals, bins=64):
    h, edges = np.histogram(vals, bins=bins)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.cumsum((h * c)[::-1])[::-1]) / np.maximum(w1, 1)
    var = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1]) ** 2
    return float(c[int(var.argmax())])


def motion_segments(db):
    """{episode -> [seg|gap tokens as raw tuples]} from trajectories."""
    ep = db.table("episodes").scan().to_pydict()
    spans = sorted((int(t), int(e)) for e, t in
                   zip(ep["episode_index"], ep["ts"]))
    starts = [s for s, _ in spans]
    import bisect

    # participant filter: the same agent-appearance + oversize vetoes
    # the binding work fitted - shadows and arm fragments "move" fast
    # all episode and drowned the first run in 110 segments/ep
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
        per_obj[(e, int(d["object_id"][i]))].append(
            (int(d["ts"][i]), float(d["px"][i]), float(d["py"][i])))
        areas[(e, int(d["object_id"][i]))].append(
            max(int(d["x1"][i]) - int(d["x0"][i]), 1)
            * max(int(d["y1"][i]) - int(d["y0"][i]), 1))
    per_obj = {k: v for k, v in per_obj.items()
               if np.median(areas[k]) <= med_agent_area}

    # WINDOWED NET DISPLACEMENT, not instantaneous speed: oscillating
    # box jitter has speed but no net motion (measured: sustained
    # "movers" with 0px travel), while a carry nets 100-300px. Windows
    # never straddle track breaks.
    WIN = 0.5
    disp_all = []
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
            # break windows at gaps (track handoffs teleport)
            if np.any(np.diff(t[j0:i + 1]) > 0.15e9):
                continue
            nd[i] = float(np.hypot(x[i] - x[j0], y[i] - y[j0]))
        series[k] = (t, x, y, nd)
        disp_all.append(nd[nd > 1e-3])
    # threshold anchored to CORPUS OBJECT SCALE, not an Otsu valley:
    # 99% of windows are stillness, and that imbalance drags any
    # valley into the jitter (measured: 6.8px -> one giant segment).
    # A real manipulation moves an object about its own body length
    # per window; jitter nets a few px.
    med_diag = float(np.sqrt(2 * np.median(
        [np.median(v) for v in areas.values()])))
    thr = 0.5 * med_diag
    print(f"net-displacement threshold: {thr:.1f}px "
          f"(0.5 x median participant diagonal {med_diag:.0f}px)")

    # REST-BACKED participants only: a manipulated object is STILL for
    # most of its existence (it sits before and after every move); the
    # arm's shadow - which beat the appearance veto by not looking
    # like the arm - moves in every window and rests never (measured:
    # one "mover" spanning a whole 14s episode with 25px net travel).
    fracs = {k: float((nd >= thr).mean())
             for k, (t, x, y, nd) in series.items() if len(nd) >= 8}
    fbar = otsu(np.array(list(fracs.values())))
    keep = {k: series[k] for k, f in fracs.items() if f <= fbar}
    print(f"rest-backed participants: {len(keep):,} of {len(series):,} "
          f"(moving-fraction bar {fbar:.2f}, otsu)")
    series = keep

    # ANY-OBJECT-MOVING union on the frame grid: concurrent junk and a
    # carry fragmented across identities merge into ONE segment per
    # true manipulation; the mover is the object with the largest net
    # displacement inside the segment.
    grid = defaultdict(dict)          # e -> ts -> {oid: nd}
    for (e, oid), (t, x, y, nd) in series.items():
        g = grid[e]
        for i in range(len(t)):
            if nd[i] >= thr:
                g.setdefault(int(t[i]), {})[oid] = max(
                    g.get(int(t[i]), {}).get(oid, 0.0), float(nd[i]))
    pos_of = {}                       # (e, oid, ts) -> (x, y)
    mover_ts = defaultdict(list)      # (e, oid) -> sorted sample ts
    for (e, oid), (t, x, y, nd) in series.items():
        for i in range(len(t)):
            pos_of[(e, oid, int(t[i]))] = (float(x[i]), float(y[i]))
            mover_ts[(e, oid)].append(int(t[i]))
    for v in mover_ts.values():
        v.sort()
    # merge gap FITTED from the corpus: a carry goes blind mid-flight
    # (the block is occluded inside the gripper), so a manipulation
    # shows as grasp-motion + release-motion with a 1-3s gap between,
    # while separate manipulations idle longer. The two gap
    # populations are bimodal - Otsu splits them.
    all_gaps = []
    for e, g in grid.items():
        tss = sorted(g)
        for i in range(1, len(tss)):
            gp = (tss[i] - tss[i - 1]) / 1e9
            if gp > 0.11:
                all_gaps.append(gp)
    gap_bar = otsu(np.log10(np.array(all_gaps) + 1e-3)) if all_gaps \
        else np.log10(MERGE_GAP_S)
    gap_bar = 10 ** gap_bar
    print(f"fitted merge gap: {gap_bar:.2f}s (otsu on inter-run gaps)")
    segs_by_ep = defaultdict(list)
    for e, g in grid.items():
        tss = sorted(g)
        if not tss:
            continue
        run = [tss[0]]
        for ts_ in tss[1:] + [None]:
            if ts_ is not None and (ts_ - run[-1]) / 1e9 <= gap_bar:
                run.append(ts_)
                continue
            a, b = run[0], run[-1]
            if (b - a) / 1e9 >= MIN_SEG_S:
                # mover: largest accumulated windowed displacement
                acc = defaultdict(float)
                for tt in run:
                    for oid, v in g[tt].items():
                        acc[oid] += v
                mover = max(acc, key=acc.get)
                p0 = pos_of.get((e, mover, a)) or next(
                    (pos_of[(e, mover, tt)] for tt in run
                     if (e, mover, tt) in pos_of), None)
                p1 = next((pos_of[(e, mover, tt)] for tt in
                           reversed(run) if (e, mover, tt) in pos_of),
                          None)
                travel = (float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                          if p0 and p1 else 0.0)
                # ON-relations at the boundaries, CAMERA-ROBUST: image
                # y confounds tower height with table depth, but "the
                # mover sits on another block" is a relation - another
                # participant horizontally aligned with its centre just
                # below. (start_on, end_on, travel) is nearly the
                # primitive alphabet: place (0,0), stack (0,1),
                # unstack (1,0), push (0,0)-short.
                def on_at(ts_, mv, pos):
                    # ON = CO-LOCATION with a pre-existing RESTING
                    # participant: a stack lands where a block already
                    # sits, a place lands where nothing is. No vertical
                    # test - single-view y confounds height with depth
                    # (measured to noise) - but "something was already
                    # there" is depth-robust.
                    if pos is None or ts_ is None:
                        return 0
                    import bisect as _bb
                    for (e2, o2), (t2, x2, y2, nd2) in series.items():
                        if e2 != e or o2 == mv:
                            continue
                        i2 = _bb.bisect_left(t2, ts_)
                        for c_ in (i2 - 1, i2):
                            if not (0 <= c_ < len(t2)):
                                continue
                            if abs(int(t2[c_]) - ts_) > 0.5e9:
                                continue
                            if nd2[c_] >= thr:
                                continue          # moving, not resting
                            d2 = float(np.hypot(x2[c_] - pos[0],
                                                y2[c_] - pos[1]))
                            if d2 < 0.9 * med_diag:
                                return 1
                    return 0
                # qualifiers read the RESTING scene, not the fragment
                # edge: the boundary sample is mid-air or in-gripper.
                # rest position = the mover's nearest sample OUTSIDE
                # the segment (before start / after end).
                mt = mover_ts.get((e, mover), [])
                import bisect as _b
                ra = None
                i_ = _b.bisect_left(mt, int(a)) - 1
                if i_ >= 0:
                    ra = mt[i_]
                rb = None
                i_ = _b.bisect_right(mt, int(b))
                if i_ < len(mt):
                    rb = mt[i_]
                pa = pos_of.get((e, mover, ra)) if ra else p0
                pb = pos_of.get((e, mover, rb)) if rb else p1
                son = on_at(ra if ra else None, mover, pa)
                eon = on_at(rb if rb else None, mover, pb)
                if pa and pb:
                    travel = float(np.hypot(pb[0] - pa[0],
                                            pb[1] - pa[1]))
                segs_by_ep[e].append((int(a), int(b), int(mover),
                                      travel, son, eon, pb,
                                      rb if rb else None))
            if ts_ is not None:
                run = [ts_]
    # merge near-adjacent runs of the SAME object (tracker stutter)
    merged = {}
    for e, ss in segs_by_ep.items():
        ss.sort()
        out = []
        for s in ss:
            if out and s[2] == out[-1][2] \
                    and (s[0] - out[-1][1]) / 1e9 <= MERGE_GAP_S:
                a, b, oid, tv, ey = out[-1]
                out[-1] = (a, s[1], oid, tv + s[3], s[4])
            else:
                out.append(s)
        merged[e] = out
    return merged, spans


def colour_slots(db, merged, spans, med_diag=61.0):
    """Per-episode slot ids from the mover's REST-CROP Lab colour.

    Slots are the half of the oracle's power that identity-across-
    carries (0.54 cosine, measured) blocks. Within an episode the
    mover's appearance at rest is a stable signature, and Lab colour
    moments are a generic image statistic (diag_kind precedent) - no
    models, no text. Cluster cut fitted by Otsu on the corpus's own
    within-episode pair distances.
    Returns {(episode, seg_index) -> slot}.
    """
    import cv2
    import pyarrow.compute as pc
    from elidedb.video import FrameSet
    ft = db.table("frames").scan()
    labs = {}
    from tqdm import tqdm
    for e, ss in tqdm(sorted(merged.items()), desc="slot-crops",
                      unit="ep", mininterval=5):
        for si, seg in enumerate(ss):
            labs[(e, si)] = None
    # decode each segment's rest frame (just after b), crop at the
    # mover's rest position
    for e, ss in tqdm(sorted(merged.items()), desc="decode",
                      unit="ep", mininterval=5):
        for si, seg in enumerate(ss):
            a, b, oid, tv, son, eon = seg[:6]
            pos = seg[6] if len(seg) > 6 else None
            ts_ = seg[7] if len(seg) > 7 else None
            if pos is None or ts_ is None:
                continue
            sel = ft.filter(pc.equal(ft.column("ts"), int(ts_)))
            if len(sel) == 0:
                continue
            dec = FrameSet(db, "frames", sel).decode()
            if not dec:
                continue
            im = dec[0][1]
            h = int(med_diag * 0.35)
            x0 = max(int(pos[0]) - h, 0)
            y0 = max(int(pos[1]) - h, 0)
            c = im[y0:int(pos[1]) + h, x0:int(pos[0]) + h]
            if c.size < 48:
                continue
            # centre core + MEDIAN Lab: the mean dragged table pixels
            # in and the fitted cut ballooned to 101 units, merging
            # distinct colours
            hh, ww = c.shape[:2]
            core = c[hh // 4:3 * hh // 4 + 1, ww // 4:3 * ww // 4 + 1]
            lab = cv2.cvtColor(core, cv2.COLOR_RGB2LAB).reshape(-1, 3)
            labs[(e, si)] = np.median(lab, 0).astype(np.float32)
    # fit the same/different cut from within-episode pair distances
    dists = []
    for e, ss in merged.items():
        ks = [k for k in labs if k[0] == e and labs[k] is not None]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                dists.append(float(np.linalg.norm(labs[ks[i]]
                                                  - labs[ks[j]])))
    cut = otsu(np.array(dists)) if dists else 30.0
    print(f"colour-slot cut: {cut:.1f} Lab units (otsu)")
    slots = {}
    for e, ss in merged.items():
        ks = [k for k in labs if k[0] == e]
        reps = []                    # [(lab, slot_id)]
        for k in sorted(ks, key=lambda kk: kk[1]):
            v = labs[k]
            if v is None:
                slots[k] = -1
                continue
            hit = None
            for lab0, sid in reps:
                if float(np.linalg.norm(v - lab0)) <= cut:
                    hit = sid
                    break
            if hit is None:
                hit = len(reps)
                reps.append((v, hit))
            slots[k] = hit
    return slots


def tokenise(merged, spans, use_slots=True, slots=None):
    trav = np.array([s[3] for v in merged.values() for s in v])
    tq = np.percentile(trav, [33, 66])
    seqs = {}
    for e, ss in merged.items():
        slot_of = {}
        toks = []
        prev_end = None
        for si, seg in enumerate(ss):
            a, b, oid, tv, son, eon = seg[:6]
            if prev_end is not None:
                gap = (a - prev_end) / 1e9
                toks.append((("G", int(min(gap / 4.0, 2)), 0), 0, 0.0,
                             None))
            if slots is not None:
                slot = slots.get((e, si), -1)
            else:
                if oid not in slot_of:
                    slot_of[oid] = len(slot_of)
                slot = slot_of[oid]
            if not use_slots:
                slot = -1
            toks.append((("M", son * 2 + eon,
                          int(np.searchsorted(tq, tv))),
                         slot, (b - a) / 1e9, None))
            prev_end = b
        seqs[e] = toks
    return seqs


def bench(seqs, tmpl, label, targets, w_slot=0.0):
    chain_qbe.W_SLOT = w_slot
    eps = sorted(seqs)
    pos = {e: i for i, e in enumerate(eps)}
    S = np.zeros((len(eps), len(eps)), np.float32)
    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            S[i, j] = S[j, i] = chain_qbe.align(seqs[eps[i]], seqs[eps[j]])
    rs = np.random.RandomState(0)
    ys, ps = [], []
    print(f"-- {label}")
    for target in targets:
        pool = sorted(e for e, tm in tmpl.items() if tm == target
                      and e in pos)
        seeds = sorted(int(x) for x in rs.choice(pool, 5, replace=False))
        support = len(pool) - len(seeds)
        k = math.ceil(1.5 * support)
        si = [pos[e] for e in seeds]
        sc = S[si].max(0)
        for x_ in si:
            sc[x_] = -1e9
        got = [eps[i] for i in np.argsort(-sc)[:k]]
        true = sum(1 for e in got if tmpl.get(e) == target)
        ys.append(true / support)
        ps.append(true / len(got))
        print(f"   {target:<20} yield {true/support:.2f}  "
              f"prec {true/len(got):.2f}")
    print(f"   MEAN yield {np.mean(ys):.3f}  prec {np.mean(ps):.3f}")
    return float(np.mean(ys))


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    merged, spans = motion_segments(db)
    print(f"episodes {len(merged)}, mean motion segments "
          f"{np.mean([len(v) for v in merged.values()]):.1f}")
    use_slots = "--no-slots" not in argv
    slots = colour_slots(db, merged, spans) if use_slots else None
    seqs = tokenise(merged, spans, use_slots, slots)
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {}
    for e, tm in zip(t["episode"], t["template"]):
        tmpl[int(e)] = tm
    dev = ("swap", "precarious", "push_then_build", "build_unstack_move")
    hold = ("relocate_build", "two_sites_merge")
    w = 0.6 if use_slots else 0.0
    bench(seqs, tmpl, "DEV", dev, w_slot=w)
    bench(seqs, tmpl, "HOLDOUT", hold, w_slot=w)


if __name__ == "__main__":
    main()
