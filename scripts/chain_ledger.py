"""REST-LEDGER chains: object permanence as bookkeeping. Route 2.

Every lab-side route died on the same primitive - identifying the
mover WHILE IT MOVES (14-40% reliable everywhere measured). The
measured asymmetry points the other way: AT REST, blocks are reliable
(positions solid, colours 0.78 from one crop, better from medians).
So this never looks at motion. It tracks the WORLD'S REST STATE:

    entry      a block resting at a spot: (position, colour, span).
               Entries persist across track breaks - same place +
               same colour = the same entry continuing - so tracker
               churn becomes a no-op instead of a fragment.
    itinerary  per colour, the time-ordered sequence of its entries.
               A manipulation IS a transition: entry k ends (pickup),
               entry k+1 begins elsewhere (set-down). The carry
               between them is never observed and never needs to be -
               the blind gap that defeated seven pairing signals is
               dissolved by object permanence.

Chain tokens = all transitions ordered by departure time:
M(colour-slot, travel_q) with G gaps. Everything fitted from the
corpus (motion/rest thresholds from chain_2v, colour cut by Otsu over
within-episode entry pairs); colours are pixels; no models, no text,
no truth. Cross-view: entries merged by colour + time overlap.

    python scripts/chain_ledger.py [--store lake/sim_chains]
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
from chain_moves import bench, otsu                            # noqa: E402
from chain_2v import Cropper, view_segments                    # noqa: E402

MIN_REST_S = 0.8


def rest_intervals(series, thr):
    """Maximal resting stretches per (e, sv, oid) series."""
    out = []
    for (e, sv, oid), (t, x, y, nd) in series.items():
        i0 = None
        for i in range(len(t) + 1):
            resting = i < len(t) and nd[i] < thr
            if resting and i0 is None:
                i0 = i
            elif not resting and i0 is not None:
                if (t[i - 1] - t[i0]) / 1e9 >= MIN_REST_S:
                    out.append((e, sv,
                                float(np.median(x[i0:i])),
                                float(np.median(y[i0:i])),
                                int(t[i0]), int(t[i - 1])))
                i0 = None
    return out


def spot_chain(iv, med_diag):
    """Chain rest intervals into SPOT runs per (episode, view): same
    place = the same sitting block continuing across track breaks.
    Colour is checked at the ENTRY level afterwards; position alone
    chains here, and a different-coloured newcomer at the same spot is
    split later by its colour."""
    by_ev = defaultdict(list)
    for e, sv, x, y, a, b in iv:
        by_ev[(e, sv)].append((a, b, x, y))
    entries = []
    for (e, sv), rows in by_ev.items():
        rows.sort()
        open_ = []                     # [x, y, a, b, members]
        for a, b, x, y in rows:
            hit = None
            for en in open_:
                if np.hypot(x - en[0], y - en[1]) < 0.8 * med_diag \
                        and a - en[3] < int(3.0e9):
                    hit = en
                    break
            if hit is None:
                open_.append([x, y, a, b, [(a, b, x, y)]])
            else:
                n = len(hit[4])
                hit[0] = (hit[0] * n + x) / (n + 1)
                hit[1] = (hit[1] * n + y) / (n + 1)
                hit[3] = max(hit[3], b)
                hit[4].append((a, b, x, y))
        for x, y, a, b, mem in open_:
            entries.append((e, sv, x, y, a, b, mem))
    return entries


def entry_colours(db, entries, med_diag):
    """Median Lab per entry over up to 3 rest crops (rest reads are
    the reliable primitive: 0.78 single-crop, better with medians)."""
    cr = Cropper(db, med_diag)
    from tqdm import tqdm
    out = []
    for e, sv, x, y, a, b, mem in tqdm(entries, desc="entry-crops",
                                       mininterval=5):
        ts_pick = np.linspace(a, b, min(3, max(1, len(mem)))) \
            .astype(np.int64)
        # snap to actual member sample times (frames exist there)
        vals = []
        for tp in ts_pick:
            best = min(mem, key=lambda m: min(abs(m[0] - tp),
                                              abs(m[1] - tp)))
            ts_ = best[0] if abs(best[0] - tp) < abs(best[1] - tp) \
                else best[1]
            v = cr.lab(sv, (x, y), int(ts_))
            if v is not None:
                vals.append(v)
        out.append((e, sv, x, y, int(a), int(b),
                    np.median(np.stack(vals), 0).astype(np.float32)
                    if vals else None))
    return out


def background_colour(cr, entries, n=40):
    """The scene's dominant colour, measured: median Lab over random
    frames (downsampled). Junk ledger entries - shadow halos, table
    patches - sit near it; blocks are saturated against it."""
    import cv2
    rng = np.random.default_rng(0)
    vals = []
    pool = [(sv, a) for e, sv, x, y, a, b, mem in entries]
    while len(vals) < n and pool:
        sv, ts_ = pool[rng.integers(len(pool))]
        im = cr.frame(sv, int(ts_))
        if im is None:
            continue
        lab = cv2.cvtColor(im[::8, ::8], cv2.COLOR_RGB2LAB)             .reshape(-1, 3).astype(np.float32)
        vals.append(np.median(lab, 0))
    bg = np.median(np.stack(vals), 0).astype(np.float32)
    print(f"background colour: Lab {np.round(bg, 0)}")
    return bg


def ledgers(db):
    segs, series, med_diag, thr, agents = view_segments(db)
    iv = rest_intervals(series, thr)
    print(f"rest intervals: {len(iv):,}")
    entries = spot_chain(iv, med_diag)
    print(f"spot entries: {len(entries):,} "
          f"({len(entries)/150:.1f}/episode-view)")
    import os
    cache = Path(os.environ.get("ELIDEDB_LEDGER_CACHE",
                                "/tmp/ledger_ec.npz"))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        ec = list(z["ec"])
        print(f"entry colours from cache ({len(ec)})")
    else:
        ec = entry_colours(db, entries, med_diag)
        ec = [r for r in ec if r[6] is not None]
        np.savez(cache, ec=np.array(ec, object))
        print(f"entry colours cached -> {cache}")
    # JUNK-ENTRY FILTER, measured not assumed: shadow halos and table
    # patches rest as convincingly as blocks (46 entries/ep-view vs ~8
    # real; their colours destroyed the slot fit - cut hit 255 Lab).
    # Drop entries near the measured background or agent colour; both
    # bars Otsu-fitted from the entries' own distance distributions.
    from chain_slotline import agent_colour
    cr2 = Cropper(db, med_diag)
    bg = background_colour(cr2, entries)
    ag = agent_colour(db, agents, cr2)
    d_bg = np.array([float(np.linalg.norm(r[6] - bg)) for r in ec])
    d_ag = np.array([float(np.linalg.norm(r[6] - ag)) for r in ec])
    bar_bg = otsu(d_bg)
    bar_ag = otsu(d_ag)
    keep = (d_bg > bar_bg) & (d_ag > bar_ag)
    print(f"entry filter: {int(keep.sum()):,} of {len(ec):,} kept "
          f"(bg bar {bar_bg:.0f}, agent bar {bar_ag:.0f})")
    ec = [r for r, k in zip(ec, keep) if k]
    # colour cut fitted on within-episode entry pairs
    dists = []
    by_e = defaultdict(list)
    for r in ec:
        by_e[r[0]].append(r)
    for e, rr in by_e.items():
        for i in range(len(rr)):
            for j in range(i + 1, len(rr)):
                dists.append(float(np.linalg.norm(rr[i][6] - rr[j][6])))
    cut = otsu(np.array(dists)) if dists else 30.0
    print(f"entry colour cut: {cut:.1f} Lab (otsu, {len(dists):,} pairs)")
    return by_e, cut, med_diag


def transitions(by_e, cut, med_diag):
    """Per episode: colour-cluster the entries, merge across views by
    colour + time overlap, then per colour the ITINERARY of rest
    spells; consecutive spells = one manipulation each."""
    seqs = {}
    for e, rr in by_e.items():
        # colour clusters (slots by first appearance in time)
        rr = sorted(rr, key=lambda r: r[4])
        reps, cid = [], []
        for r in rr:
            hit = None
            for lab0, sid in reps:
                if float(np.linalg.norm(r[6] - lab0)) <= cut:
                    hit = sid
                    break
            if hit is None:
                hit = len(reps)
                reps.append((r[6], hit))
            cid.append(hit)
        # merge same-colour entries overlapping in time across views
        # (one physical rest spell seen twice)
        spells = {}                    # slot -> [(a, b, x, y)]
        for r, s_ in zip(rr, cid):
            e_, sv, x, y, a, b, _lab = r
            lst = spells.setdefault(s_, [])
            hit = None
            for sp in lst:
                if a <= sp[1] + int(1.0e9) and b >= sp[0] - int(1.0e9):
                    hit = sp
                    break
            if hit is None:
                lst.append([a, b, x, y, 1])
            else:
                hit[0] = min(hit[0], a)
                hit[1] = max(hit[1], b)
        # itineraries -> transitions
        trans = []
        for s_, lst in spells.items():
            lst.sort()
            for i in range(1, len(lst)):
                dep = lst[i - 1][1]
                arr = lst[i][0]
                d = float(np.hypot(lst[i][2] - lst[i - 1][2],
                                   lst[i][3] - lst[i - 1][3]))
                trans.append((dep, arr, s_, d))
        trans.sort()
        seqs[e] = trans
    return seqs


def tokenise(seqs):
    trav = np.array([d for v in seqs.values() for *_ , d in v])
    tq = np.percentile(trav, [33, 66]) if len(trav) else [1, 2]
    out = {}
    for e, trans in seqs.items():
        toks = []
        prev = None
        for dep, arr, s_, d in trans:
            if prev is not None:
                gap = max((dep - prev) / 1e9, 0.0)
                toks.append((("G", int(min(gap / 4.0, 2)), 0),
                             -1, 0.0, None))
            toks.append((("M", int(np.searchsorted(tq, d)), 0),
                         s_, (arr - dep) / 1e9, None))
            prev = arr
        out[e] = toks
    return out


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    by_e, cut, med_diag = ledgers(db)
    seqs_t = transitions(by_e, cut, med_diag)
    print(f"episodes {len(seqs_t)}, mean transitions "
          f"{np.mean([len(v) for v in seqs_t.values()]):.1f}")
    seqs = tokenise(seqs_t)
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {}
    for e, tm in zip(t["episode"], t["template"]):
        tmpl[int(e)] = tm
    chain_qbe.W_KIND = 0.5
    dev = ("swap", "precarious", "push_then_build",
           "build_unstack_move")
    hold = ("relocate_build", "two_sites_merge")
    bench(seqs, tmpl, "DEV ledger", dev, w_slot=1.0)
    bench(seqs, tmpl, "HOLDOUT ledger", hold, w_slot=1.0)


if __name__ == "__main__":
    main()
