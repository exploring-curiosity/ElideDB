"""HAND sub-element + hold-alternation chain tokens. Geometry only.

The twice-confirmed wall: the agent element is one arm-sized box, so
"held" (contact = arm-box overlap) fires on half of all participant
samples and carries no alternation, mover binding drowns in
gripper-adjacent soup, and event typing has no hold signal to read.

The hand is DERIVED, not detected: the arm fragments the proposer
produces anyway - identified as agent-like by the store's own identity
standard (descriptor >= calibrated cut vs the agent gallery) - include
the gripper, and interaction happens at table level, so

    hand(t) = the LOWEST agent-like point at t
              (main agent point + agent-like fragment centroids),
              median-smoothed

    held(t) = some participant's centroid within reach of hand(t),
              reach = the corpus's own median participant box diagonal

Everything is geometry + the store's fitted identity cut: no models,
no priors, no text. Tokens are hold-state runs with corpus-tercile
qualifiers (travel, end height within the hand's episode range,
duration); alignment is chain_qbe's.

    python scripts/chain_hand.py --viz          # draw hand on frames
    python scripts/chain_hand.py                # benchmark
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

MIN_SEG_S = 0.4


def load_geometry(db):
    """Per-stream time series: hand candidates + participant tracks."""
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

    tr = db.table("trajectories").scan().to_pydict()
    agent_keys = {(str(tr["stream"][i]), int(tr["track_ts"][i]),
                   int(tr["t1"][i]), int(tr["object_id"][i]))
                  for i in range(len(tr["ts"])) if tr["is_agent"][i]}
    A = np.stack([desc[k] for k in sorted(agent_keys) if k in desc])
    if len(A) > 500:
        A = A[np.random.RandomState(0).choice(len(A), 500, replace=False)]
    is_armlike = {}
    for k, v in desc.items():
        is_armlike[k] = bool(float((A @ v).max()) >= cut)

    # main agent box per timestamp: arm PIECES live inside it, arm
    # SHADOWS are cast outside it - the gate that keeps the hand off
    # the shadow at table level (filmed failure of the first version)
    abox = {}
    for i in range(len(tr["ts"])):
        if tr["is_agent"][i]:
            abox[(str(tr["stream"][i]), int(tr["ts"][i]))] = (
                int(tr["x0"][i]), int(tr["y0"][i]),
                int(tr["x1"][i]), int(tr["y1"][i]))
    hand_pts = defaultdict(list)     # (stream, ts) -> [(y, x)]
    parts = defaultdict(list)        # (stream, oid) -> [(ts, x, y, diag)]
    diags = []
    for i in range(len(tr["ts"])):
        s, ts_ = str(tr["stream"][i]), int(tr["ts"][i])
        x_, y_ = float(tr["px"][i]), float(tr["py"][i])
        k = (s, int(tr["track_ts"][i]), int(tr["t1"][i]),
             int(tr["object_id"][i]))
        if tr["is_agent"][i]:
            hand_pts[(s, ts_)].append((y_, x_))
        elif is_armlike.get(k):
            bx = abox.get((s, ts_))
            if bx and bx[0] <= x_ <= bx[2] and bx[1] <= y_ <= bx[3]:
                hand_pts[(s, ts_)].append((y_, x_))
        else:
            dg = float(np.hypot(int(tr["x1"][i]) - int(tr["x0"][i]),
                                int(tr["y1"][i]) - int(tr["y0"][i])))
            parts[(s, int(tr["object_id"][i]))].append((ts_, x_, y_, dg))
            diags.append(dg)
    reach = 0.8 * float(np.median(diags))
    return hand_pts, parts, reach


def hand_series(db, hand_pts):
    """{episode -> (ts[], hx[], hy[])}, median-smoothed."""
    ep = db.table("episodes").scan().to_pydict()
    spans = sorted((int(t), int(e)) for e, t in
                   zip(ep["episode_index"], ep["ts"]))
    starts = [s for s, _ in spans]
    import bisect
    by_ep = defaultdict(list)
    for (s, ts_), cands in hand_pts.items():
        y, x = max(cands)            # lowest point = max image y
        e = spans[bisect.bisect_right(starts, ts_) - 1][1]
        by_ep[e].append((ts_, x, y))
    out = {}
    for e, rows in by_ep.items():
        rows.sort()
        t = np.array([r[0] for r in rows], np.int64)
        x = np.array([r[1] for r in rows], np.float32)
        y = np.array([r[2] for r in rows], np.float32)
        k = 2
        xs, ys = x.copy(), y.copy()
        for i in range(len(t)):
            lo, hi = max(0, i - k), min(len(t), i + k + 1)
            xs[i], ys[i] = np.median(x[lo:hi]), np.median(y[lo:hi])
        out[e] = (t, xs, ys)
    return out


def hold_signal(hands, parts, reach, db):
    """{episode -> (ts[], held[], hx[], hy[])}: is any participant
    within reach of the hand."""
    ep = db.table("episodes").scan().to_pydict()
    spans = sorted((int(t), int(e)) for e, t in
                   zip(ep["episode_index"], ep["ts"]))
    starts = [s for s, _ in spans]
    import bisect
    # participant positions indexed by (episode, ts)
    at = defaultdict(list)
    for (s, oid), rows in parts.items():
        for ts_, x_, y_, dg in rows:
            e = spans[bisect.bisect_right(starts, ts_) - 1][1]
            at[(e, ts_)].append((x_, y_))
    out = {}
    for e, (t, hx, hy) in hands.items():
        held = np.zeros(len(t), bool)
        for i in range(len(t)):
            for x_, y_ in at.get((e, int(t[i])), ()):
                if np.hypot(x_ - hx[i], y_ - hy[i]) <= reach:
                    held[i] = True
                    break
        # median-smooth flicker
        sm = held.copy()
        for i in range(2, len(held) - 2):
            sm[i] = np.median(held[i - 2:i + 3]) > 0.5
        out[e] = (t, sm, hx, hy)
    return out


def tokens(holds):
    """Hold-state runs -> tokens with corpus-tercile qualifiers."""
    raw = {}
    trav_all, dur_all = [], []
    for e, (t, held, hx, hy) in holds.items():
        segs = []
        i0 = 0
        ylo, yhi = float(hy.min()), float(hy.max())
        for i in range(1, len(held) + 1):
            if i == len(held) or held[i] != held[i0]:
                dur = (t[i - 1] - t[i0]) / 1e9
                if dur >= MIN_SEG_S:
                    travel = float(np.abs(np.diff(hx[i0:i])).sum())
                    endh = (float(hy[i - 1]) - ylo) / max(yhi - ylo, 1e-6)
                    segs.append((bool(held[i0]), travel, endh, dur))
                    trav_all.append(travel)
                    dur_all.append(dur)
                i0 = i
        raw[e] = segs
    tq = np.percentile(trav_all, [33, 66])
    seqs = {}
    for e, segs in raw.items():
        seqs[e] = [(("H" if h else "E",
                     int(np.searchsorted(tq, tr_)),
                     int(min(eh * 3, 2))), 0, 0.0, None)
                   for h, tr_, eh, du in segs]
    return seqs, raw


def bench(seqs, tmpl, label, targets):
    eps = sorted(seqs)
    pos = {e: i for i, e in enumerate(eps)}
    S = np.zeros((len(eps), len(eps)), np.float32)
    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            S[i, j] = S[j, i] = chain_qbe.align(seqs[eps[i]], seqs[eps[j]])
    rs = np.random.RandomState(0)
    ys = []
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
        print(f"   {target:<20} yield {true/support:.2f}  "
              f"prec {true/len(got):.2f}")
    print(f"   mean yield {np.mean(ys):.3f}")
    return float(np.mean(ys))


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    hand_pts, parts, reach = load_geometry(db)
    hands = hand_series(db, hand_pts)
    holds = hold_signal(hands, parts, reach, db)
    nseg = [sum(1 for _ in range(1)) for _ in ()]  # placeholder
    counts = []
    for e, (t, held, hx, hy) in holds.items():
        c = 1 + int(np.sum(held[1:] != held[:-1]))
        counts.append(c)
    print(f"episodes {len(holds)}, reach {reach:.0f}px, "
          f"mean hold-alternations {np.mean(counts):.1f}")

    if "--viz" in argv:
        viz(db, hands, holds)
        return

    seqs, raw = tokens(holds)
    print(f"mean tokens/episode "
          f"{np.mean([len(v) for v in seqs.values()]):.1f}")
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {}
    for e, tm in zip(t["episode"], t["template"]):
        tmpl[int(e)] = tm
    bench(seqs, tmpl, "DEV templates",
          ("swap", "precarious", "push_then_build", "build_unstack_move"))
    bench(seqs, tmpl, "HOLDOUT templates",
          ("relocate_build", "two_sites_merge"))


def viz(db, hands, holds, n=8):
    import cv2
    import pyarrow.compute as pc
    from elidedb.video import FrameSet
    S = Path("/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
             "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3"
             "/scratchpad/chk")
    ft = db.table("frames").scan()
    rng = np.random.default_rng(0)
    eps = sorted(holds)
    made = 0
    for e in rng.choice(eps, 4, replace=False):
        t, held, hx, hy = holds[int(e)]
        for frac in (0.3, 0.6):
            i = int(frac * (len(t) - 1))
            sel = ft.filter(pc.equal(ft.column("ts"), int(t[i])))
            if len(sel) == 0:
                continue
            dec = FrameSet(db, "frames", sel).decode()
            if not dec:
                continue
            im = dec[0][1].copy()
            cv2.circle(im, (int(hx[i]), int(hy[i])), 10,
                       (0, 255, 0) if held[i] else (0, 0, 255), 3)
            cv2.putText(im, f"ep{int(e)} held={bool(held[i])}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(str(S / f"hand{made}.png"), im[:, :, ::-1])
            made += 1
    print(f"wrote {made} viz frames")


if __name__ == "__main__":
    main()
