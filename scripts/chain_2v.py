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

    agents = defaultdict(list)        # (e, sv) -> [(ts, x, y)]
    per_obj = defaultdict(list)
    areas = defaultdict(list)
    for i in range(len(d["ts"])):
        if d["is_agent"][i]:
            e = spans[bisect.bisect_right(starts,
                                          int(d["ts"][i])) - 1][1]
            agents[(e, str(d["stream"][i]))].append(
                (int(d["ts"][i]), int(d["x0"][i]), int(d["y0"][i]),
                 int(d["x1"][i]), int(d["y1"][i])))
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
    for v in agents.values():
        v.sort()
    return segs, series, med_diag, thr, agents


def landing(series, e, sv, ts, thr, after, anchor=None, max_d=None):
    """The rest point a manipulation ENDS at (or starts from) is not
    the mover's own series - identity dies across a carry (0.54
    cosine, measured), so the placed block resumes as a NEW id. The
    placement point is where a resting series is BORN just after the
    unit ends; the pickup point is where one DIES just before it
    starts. Returns ((x, y), ts, oid) or None."""
    best = None
    for (e2, sv2, o2), (t2, x2, y2, nd2) in series.items():
        if e2 != e or sv2 != sv or len(t2) < 3:
            continue
        if after:
            dt = (int(t2[0]) - ts) / 1e9
            if not (-0.5 <= dt <= 2.0):
                continue
            if float(np.median(nd2[:3])) >= thr:
                continue              # born moving: not a landing
            cand = ((float(x2[0]), float(y2[0])), int(t2[0]), o2,
                    abs(dt))
            if anchor is not None and np.hypot(
                    cand[0][0] - anchor[0],
                    cand[0][1] - anchor[1]) > max_d:
                continue
        else:
            dt = (ts - int(t2[-1])) / 1e9
            if not (-0.5 <= dt <= 2.0):
                continue
            if float(np.median(nd2[-3:])) >= thr:
                continue              # died moving: not a departure
            cand = ((float(x2[-1]), float(y2[-1])), int(t2[-1]), o2,
                    abs(dt))
            if anchor is not None and np.hypot(
                    cand[0][0] - anchor[0],
                    cand[0][1] - anchor[1]) > max_d:
                continue
        if best is None or cand[3] < best[3]:
            best = cand
    return best[:3] if best else None


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


def fit_gravity(series, segs):
    """Projected DOWN per (episode, view), fitted from the movers' own
    final descent: a placed block's last approach is along gravity, so
    the median of final-step unit vectors IS the view's down-projection.
    Learned from motion, never assumed - works wherever gravity does.
    Returns {(e, sv) -> unit vec}."""
    import bisect
    vecs = defaultdict(list)
    for (e, sv), ss in segs.items():
        for a, b, mover in ss:
            k = (e, sv, mover)
            if k not in series:
                continue
            t, x, y, nd = series[k]
            i = bisect.bisect_right(t, b) - 1
            j = bisect.bisect_left(t, b - int(0.5e9))
            if 0 <= j < i < len(t):
                v = np.array([x[i] - x[j], y[i] - y[j]], np.float32)
                n = float(np.linalg.norm(v))
                if n > 5.0:
                    vecs[(e, sv)].append(v / n)
    g = {}
    for k, vs in vecs.items():
        m = np.mean(vs, 0)
        n = float(np.linalg.norm(m))
        if n > 0.3:                    # coherent descent direction
            g[k] = m / n
    return g


def on_at(series, e, sv, mover, pos, ts, med_diag, thr, g=None):
    """ON vs BESIDE, one view: another resting participant co-located
    ALONG the fitted gravity projection (below the mover, small
    perpendicular offset). Falls back to isotropic co-location when
    the view has no fitted gravity."""
    if pos is None:
        return 0
    import bisect
    gv = (g or {}).get((e, sv))
    for (e2, sv2, o2), (t2, x2, y2, nd2) in series.items():
        if e2 != e or sv2 != sv or o2 == mover:
            continue
        i2 = bisect.bisect_left(t2, ts)
        for c_ in (i2 - 1, i2):
            if not (0 <= c_ < len(t2)) or abs(int(t2[c_]) - ts) > 0.6e9:
                continue
            if nd2[c_] >= thr:
                continue
            dx = float(x2[c_] - pos[0])
            dy = float(y2[c_] - pos[1])
            if gv is None:
                if np.hypot(dx, dy) < 0.9 * med_diag:
                    return 1
                continue
            along = dx * gv[0] + dy * gv[1]
            perp = abs(-dx * gv[1] + dy * gv[0])
            if 0.15 * med_diag < along < 1.3 * med_diag \
                    and perp < 0.55 * med_diag:
                return 1
    return 0


def units(db):
    """Cross-view manipulation units per episode + qualifiers."""
    segs, series, med_diag, thr, agents = view_segments(db)
    grav = fit_gravity(series, segs)
    print(f"fitted gravity projections: {len(grav)} (episode, view) pairs")
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
        def pos_at(sv, mover, ts_):
            import bisect as _b
            k = (e, sv, mover)
            if k not in series:
                return None
            t2, x2, y2, _ = series[k]
            i2 = min(max(_b.bisect_left(t2, ts_), 0), len(t2) - 1)
            return (float(x2[i2]), float(y2[i2]))

        us = []
        for g in groups.values():
            a = min(x[0] for x in g)
            b = max(x[1] for x in g)
            dep = land = None
            for sv in ("simA", "simB"):
                mem = sorted(x for x in g if x[2] == sv)
                if not mem:
                    continue
                # ANCHORED search: the departure must be near where
                # this unit's motion began, the landing near where it
                # ended - without the anchor any track birth in the
                # scene qualified and dep/land never came up empty,
                # so fragment pairing never fired
                a0 = pos_at(sv, mem[0][3], mem[0][0])
                b0 = pos_at(sv, mem[-1][3], mem[-1][1])
                dep = dep or landing(series, e, sv, a, thr,
                                     after=False, anchor=a0,
                                     max_d=2.5 * med_diag)
                land = land or landing(series, e, sv, b, thr,
                                       after=True, anchor=b0,
                                       max_d=2.5 * med_diag)
            us.append([a, b, dep, land])
        us.sort()
        # FRAGMENT PAIRING: a grasp fragment DEPARTS but never LANDS
        # (the block vanishes into the gripper); a release fragment
        # LANDS but never DEPARTED. Consecutive open+close fragments
        # are one carry - the rest-birth/death structure joins what
        # time, appearance and identity could not (each measured).
        merged = []
        for u in us:
            if merged and merged[-1][3] is None and u[2] is None \
                    and (u[0] - merged[-1][1]) / 1e9 < 6.0:
                merged[-1][1] = u[1]
                merged[-1][3] = u[3]
            else:
                merged.append(u)
        us2 = []
        for a, b, dep, land in merged:
            # EXACT anchors: the unit's own first/last in-motion mover
            # positions, and in-motion sample points for robust colour
            fp, lp_, mids = {}, {}, []
            for sv in ("simA", "simB"):
                mem = sorted(x for x in allseg
                             if x[2] == sv and x[0] >= a and x[1] <= b)
                if not mem:
                    continue
                p0 = pos_at(sv, mem[0][3], mem[0][0])
                p1 = pos_at(sv, mem[-1][3], mem[-1][1])
                if p0:
                    fp[sv] = p0
                if p1:
                    lp_[sv] = p1
                for a2_, b2_, _sv, mv_ in mem:
                    k2 = (e, sv, mv_)
                    if k2 not in series:
                        continue
                    t2, x2, y2, nd2 = series[k2]
                    for i2 in range(len(t2)):
                        if a2_ <= int(t2[i2]) <= b2_ and \
                                float(nd2[i2]) >= thr:
                            mids.append((sv, (float(x2[i2]),
                                              float(y2[i2])),
                                         int(t2[i2])))
            same_id = 1 if (dep and land and dep[2] == land[2]) else 0
            trav = (float(np.hypot(land[0][0] - dep[0][0],
                                   land[0][1] - dep[0][1]))
                    if dep and land else 0.0)
            labpos = []
            for sv in ("simA", "simB"):
                lb = landing(series, e, sv, b, thr, after=True)
                if lb:
                    labpos.append(("land", sv, lb[0], lb[1]))
                dp = landing(series, e, sv, a, thr, after=False)
                if dp:
                    labpos.append(("dep", sv, dp[0], dp[1]))
            if len(mids) > 6:
                mids = mids[:: max(1, len(mids) // 6)][:6]
            us2.append((a, b, same_id, 0, trav, labpos, fp, lp_, mids))
        us = us2
        us.sort()
        out[e] = us
    return out, med_diag, series, thr, agents


class Cropper:
    """Frame-cached Lab crops: the colour machinery is used twice (unit
    colours + gap-rest tests) and often on the same frames."""

    def __init__(self, db, med_diag):
        self.db = db
        self.ft = db.table("frames").scan()
        self.med_diag = med_diag
        self.cache = {}

    def frame(self, sv, ts_):
        import pyarrow.compute as pc
        from elidedb.video import FrameSet
        k = (sv, int(ts_))
        if k in self.cache:
            return self.cache[k]
        sel = self.ft.filter(pc.and_(
            pc.equal(self.ft.column("ts"), int(ts_)),
            pc.equal(self.ft.column("stream"), sv)))
        im = None
        if len(sel):
            dec = FrameSet(self.db, "frames", sel).decode()
            if dec:
                im = dec[0][1]
        self.cache[k] = im
        if len(self.cache) > 3000:
            self.cache.pop(next(iter(self.cache)))
        return im

    def lab(self, sv, pos, ts_):
        import cv2
        im = self.frame(sv, ts_)
        if im is None:
            return None
        h = int(self.med_diag * 0.3)
        x0 = max(int(pos[0]) - h, 0)
        y0 = max(int(pos[1]) - h, 0)
        c = im[y0:int(pos[1]) + h, x0:int(pos[0]) + h]
        if c.size < 48:
            return None
        hh, ww = c.shape[:2]
        core = c[hh // 4:3 * hh // 4 + 1, ww // 4:3 * ww // 4 + 1]
        return np.median(cv2.cvtColor(core, cv2.COLOR_RGB2LAB)
                         .reshape(-1, 3), 0).astype(np.float32)


def unit_colours(db, uu, med_diag):
    """Per-unit colours kept PER SIDE. Averaging departure and landing
    crops was self-poisoning: a grasp fragment's found "landing" is a
    neighbour block, its colour leaked into the unit mean, and every
    same-colour test then matched the neighbour against itself. The
    sides are the semantics: departure = the block as it was picked
    up, landing = the block as it was set down; a unit is CLOSED iff
    they agree."""
    cr = Cropper(db, med_diag)
    labs = {}
    from tqdm import tqdm
    for e, us in tqdm(sorted(uu.items()), desc="colours", unit="ep",
                      mininterval=5):
        for ui, (a, b, son, eon, tv, labpos) in enumerate(us):
            side = {"dep": [], "land": []}
            for sd, sv, pos, ts_ in labpos:
                v = cr.lab(sv, pos, ts_)
                if v is not None:
                    side[sd].append(v)
            labs[(e, ui)] = {
                sd: (np.mean(vs, 0).astype(np.float32) if vs else None)
                for sd, vs in side.items()}
    dists = []
    for e in uu:
        ks = [k for k in labs if k[0] == e]
        vs = [labs[k]["land"] if labs[k]["land"] is not None
              else labs[k]["dep"] for k in ks]
        vs = [v for v in vs if v is not None]
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                dists.append(float(np.linalg.norm(vs[i] - vs[j])))
    cut = otsu(np.array(dists)) if dists else 30.0
    print(f"2v colour cut: {cut:.1f} Lab (otsu, {len(dists)} pairs)")
    return labs, cut, cr


def pair_fragments(uu, agents, med_diag):
    """THE fix, from first principles: the AGENT bridges the blind
    gap. During a carry the arm travels FROM the fragment's end TO the
    resumption point - for every gap moment it stays on the path, so
    d(agent, p1) + d(agent, p2) - d(p1, p2) stays small (the ellipse
    with foci at the two fragments' own motion endpoints). Between two
    SEPARATE manipulations the arm RETREATS - it leaves the ellipse.
    The agent is the one element tracked 100%; anchors are the units'
    own exact endpoints; the tolerance is Otsu-fitted from the
    corpus's own excess distribution. No crops, colours or identity.
    """
    import bisect
    durs = [(u[1] - u[0]) / 1e9 for v in uu.values() for u in v]
    win_s = float(np.percentile(durs, 95)) if durs else 4.0
    print(f"fitted pairing window: {win_s:.1f}s (P95 unit duration)")

    def excess(e, prev, cur):
        vals = []
        for sv in ("simA", "simB"):
            p1 = prev[7].get(sv)
            p2 = cur[6].get(sv)
            ag = agents.get((e, sv))
            if p1 is None or p2 is None or not ag:
                continue
            ts_a = [r[0] for r in ag]
            lo = bisect.bisect_left(ts_a, prev[1])
            hi = bisect.bisect_right(ts_a, cur[0])
            if hi <= lo:
                continue
            base = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

            def dbox(bx, pt):
                # point-to-box distance: the gripper is somewhere in
                # the agent box, and the whole-arm CENTROID mutes the
                # retreat signal (measured: merges at the centroid bar
                # dropped the oracle cap)
                cx = min(max(pt[0], bx[1]), bx[3])
                cy = min(max(pt[1], bx[2]), bx[4])
                return float(np.hypot(pt[0] - cx, pt[1] - cy))

            worst = 0.0
            for i in range(lo, hi):
                bx = ag[i]
                worst = max(worst, dbox(bx, p1) + dbox(bx, p2) - base)
            vals.append(float(worst))
        return min(vals) if vals else None

    # pass 1: candidate excesses, corpus-wide -> fitted bar
    exs = []
    for e, us in uu.items():
        for i in range(1, len(us)):
            if (us[i][0] - us[i - 1][1]) / 1e9 < win_s:
                v = excess(e, us[i - 1], us[i])
                if v is not None:
                    exs.append(v)
    bar = otsu(np.log10(np.array(exs) + 1.0)) if exs else 2.0
    bar = 10 ** bar
    print(f"fitted ellipse-excess bar: {bar:.0f}px "
          f"({len(exs)} candidate gaps)")

    out = {}
    n_merge = 0
    for e, us in uu.items():
        merged = []
        for u in us:
            if merged and (u[0] - merged[-1][1]) / 1e9 < win_s:
                v = excess(e, merged[-1], u)
                if v is not None and v <= bar:
                    n_merge += 1
                    pv = merged[-1]
                    merged[-1] = (pv[0], u[1], pv[2] and u[2], 0,
                                  pv[4] + u[4], pv[5] + u[5],
                                  pv[6], u[7], pv[8] + u[8])
                    continue
            merged.append(u)
        out[e] = merged
    print(f"fragment pairing: {n_merge} merges (agent-bridge ellipse)")
    return out


def assign_slots(uu, colours, cut):
    slots = {}
    for e, us in uu.items():
        reps = []
        for ui in range(len(us)):
            v = colours.get((e, ui))
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
    uu0, med_diag, series, thr, agents = units(db)
    print(f"episodes {len(uu0)}, raw units "
          f"{np.mean([len(v) for v in uu0.values()]):.1f}")
    cr = Cropper(db, med_diag)
    uu = pair_fragments(uu0, agents, med_diag)
    print(f"paired units {np.mean([len(v) for v in uu.values()]):.1f}")
    # slot colours from LANDING-side crops of the merged units (the
    # in-motion variant measured gripper-contaminated)
    colours = {}
    for e, us in uu.items():
        for ui, u in enumerate(us):
            vals = [v for sd, sv, pos, ts_ in u[5] if sd == "land"
                    and (v := cr.lab(sv, pos, ts_)) is not None]
            colours[(e, ui)] = (np.median(np.stack(vals), 0)
                                .astype(np.float32) if vals else None)
    dists = []
    for e in uu:
        vs = [colours[(e, ui)] for ui in range(len(uu[e]))
              if colours.get((e, ui)) is not None]
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                dists.append(float(np.linalg.norm(vs[i] - vs[j])))
    cut = otsu(np.array(dists)) if dists else 30.0
    print(f"colour cut: {cut:.1f} Lab (otsu on in-motion colours)")
    slots = assign_slots(uu, colours, cut)
    trav = np.array([u[4] for v in uu.values() for u in v])
    tq = np.percentile(trav[trav > 0], [33, 66])
    seqs = {}
    for e, us in uu.items():
        toks = []
        prev = None
        for ui, u in enumerate(us):
            a, b, son = u[0], u[1], u[2]
            if prev is not None:
                toks.append((("G", int(min((a - prev) / 4e9, 2)), 0),
                             -1, 0.0, None))
            toks.append((("P" if son else "M", 0, 0),
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
