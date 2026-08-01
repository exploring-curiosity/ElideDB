"""Re-fit the identity cut and re-assign object ids, from persisted vectors.

THE DEFECT. The match cut is fitted from "free negatives" - pairs proven
to be different objects. free_negatives() proves it with two tests: the
tracks CO-EXIST, and their boxes are DISJOINT. Its docstring is explicit
that the second is load-bearing, because a detector that puts two boxes
on ONE object makes two tracks that co-exist and look identical, and
those land in exactly the high tail a calibration quantile reads.

Both writers - stage_presence and commit_presence - tested co-existence
only. Measured on lake/fresh_bench:

    double detections            0.5% of co-existing pairs
    the quantile in use          q=99.5, i.e. the top 0.5%

so the cut was very nearly a direct readout of the contamination.

    cut on contaminated negatives   0.748
    cut on clean negatives          0.692

WHAT THE CUT IS NOW FITTED ON. Both free samples are SAME-FRAME pairs,
and the decision the gallery actually makes is "is this the object I saw
in a DIFFERENT episode?" - about which no same-frame pair is evidence.
So the cut is now swept against the quantity identity exists to produce:
how many objects recur across episodes. That objective has an interior
maximum (too tight, every sighting is a fresh id recurring nowhere; too
loose, distinct objects collapse into one attractor that is no longer
several objects), so it fits itself. See identity.fit_cut.

NOT A REBUILD. object_vectors persists one descriptor per interval, so
re-assigning is a numpy pass over 73,521 vectors - no decode, no
detector, no ReID. The 54-minute write is not repeated. labels IS
rebuilt, because participants are stated as object:<id> and those ids
change.

    python scripts/refit_identity.py [--store lake/fresh_bench] [--dry-run]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
from elidedb.identity import (Gallery, calibrate,              # noqa: E402
                              fit_cut, interval_pairs)


def load(db):
    """presence joined to its descriptor, in presence row order."""
    P = db.table("presence").scan().to_pydict()
    OV = db.table("object_vectors").scan().to_pydict()
    key = {k: i for i, k in enumerate(zip(OV["stream"], OV["ts"],
                                          OV["t1"], OV["object_id"]))}
    idx = np.array([key[k] for k in zip(P["stream"], P["ts"], P["t1"],
                                        P["object_id"])])
    V = np.asarray(OV["vector"], np.float32).reshape(-1, 512)[idx]
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    rows = [(str(s), {"ts": int(a), "t1": int(b), "box": list(bx)})
            for s, a, b, bx in zip(P["stream"], P["ts"], P["t1"], P["box"])]
    return P, V, rows


def episodes_of(db, rows):
    ep = db.table("episodes").scan().to_pydict()
    by = {}
    for s, a, b in zip(ep["stream"], ep["ts"], ep["t1"]):
        by.setdefault(str(s), []).append((int(a), int(b)))
    for v in by.values():
        v.sort()
    out, nxt = [], {}
    for s, meta in rows:
        k = -1
        for n, (a, b) in enumerate(by.get(s, ())):
            if a <= meta["ts"] <= b:
                k = nxt.setdefault((s, n), len(nxt))
                break
        out.append(k if k >= 0 else nxt.setdefault(("~", meta["ts"]),
                                                   len(nxt)))
    return np.asarray(out, np.int64)


def shape(ids, epi):
    cnt = np.bincount(ids)
    span = int(epi.max()) + 2
    u = np.unique(ids.astype(np.int64) * span + epi)
    per = np.bincount((u // span).astype(np.int64), minlength=len(cnt))
    return (len(cnt), 100.0 * float((cnt == 1).mean()),
            int((per > 1).sum()), int(cnt.max()))


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    dry = "--dry-run" in argv
    db = Store.open(str(store))
    for need in ("presence", "object_vectors", "episodes"):
        if need not in db.tables():
            raise SystemExit(f"{store} has no {need}")

    P, V, rows = load(db)
    epi = episodes_of(db, rows)
    print(f"{store.name}: {len(V):,} presence intervals, "
          f"{len(set(epi.tolist())):,} episodes")

    # ASSIGN IN THE ORDER THE WRITER DOES. Gallery.assign is greedy, so
    # the order tracks arrive in changes the result, and it changes it by
    # more than this whole re-fit does: at a FIXED cut of 0.739, order
    # alone moves cross-episode recurrence from 6,484 to 6,821. The
    # writer assigns as tracks CLOSE; the presence table is then sorted
    # by ts, which interleaves all four cameras and is the single worst
    # order measured. Re-assigning in table order would therefore have
    # scored the new cut against a handicap the writer never had.
    o = np.lexsort((np.asarray(P["t1"], np.int64), np.asarray(P["stream"])))
    back = np.empty(len(o), np.int64)
    back[o] = np.arange(len(o))
    V, epi = V[o], epi[o]
    rows = [rows[i] for i in o]

    t = time.time()
    neg, pos = interval_pairs(rows)
    print(f"  proven-different {len(neg):,}   proven-same {len(pos):,}   "
          f"({time.time() - t:.0f}s)")
    # every co-existing pair, disjoint or not: the sample the cut used to
    # be fitted on, kept here so the contamination is visible, not argued
    allco, _ = interval_pairs(rows, iou_diff=1.01)
    print(f"  cut on CONTAMINATED negatives (co-existence only, "
          f"n={len(allco):,})  {calibrate(V, allco)[0]:.3f}")
    print(f"  cut on CLEAN negatives        (box-disjoint,      "
          f"n={len(neg):,})  {calibrate(V, neg)[0]:.3f}")

    t = time.time()
    cut, report = fit_cut(V, epi, neg, pos)
    print(f"\n  swept against cross-episode recurrence "
          f"({time.time() - t:.0f}s):")
    print(f"    {'cut':>6}{'objects':>9}{'singl%':>8}{'RECUR':>8}"
          f"{'false-merge':>13}{'recovered':>11}")
    for r in report:
        print(f"    {r['cut']:>6.3f}{r['objects']:>9,}"
              f"{r['singleton_pct']:>8.1f}{r['recur']:>8,}"
              f"{r.get('false_merge_pct', 0):>12.3f}%"
              f"{r.get('recovered_pct', 0):>10.1f}%"
              + ("   <- fit" if r["cut"] == cut else ""))

    old = np.asarray(P["object_id"], np.int32)[o]
    ids = Gallery(match=cut).assign(V)
    a, b = shape(old, epi), shape(ids, epi)
    print(f"\n  {'':<10}{'objects':>9}{'singl%':>9}{'recur':>8}{'largest':>9}")
    print(f"  {'before':<10}{a[0]:>9,}{a[1]:>9.1f}{a[2]:>8,}{a[3]:>9,}")
    print(f"  {'after':<10}{b[0]:>9,}{b[1]:>9.1f}{b[2]:>8,}{b[3]:>9,}")
    if dry:
        print("\n--dry-run: nothing written")
        return

    # ids are in close order; put them back in presence row order
    new = ids[back]
    key = {k: n for n, k in enumerate(zip(P["stream"], P["ts"], P["t1"],
                                          P["object_id"]))}
    for name in ("presence", "object_vectors"):
        tb = db.table(name).scan()
        d = tb.to_pydict()
        if name == "presence":
            d["object_id"] = [int(i) for i in new]
        else:                       # same join, back the other way
            d["object_id"] = [int(new[key[k]]) for k in
                              zip(d["stream"], d["ts"], d["t1"],
                                  d["object_id"])]
        out = pa.table({c: pa.array(d[c], tb.schema.field(c).type)
                        for c in tb.column_names})
        db.table(name).replace(out, kind="index",
                               meta={"match_cut": round(float(cut), 3),
                                     "fit": "cross-episode recurrence"})
        print(f"  rewrote {name}: {len(out):,} rows")

    # participants are stated as object:<id>, and the ids just changed
    from fix_participants import rebuild_labels
    n_lab, n_part = rebuild_labels(db)
    print(f"  labels rebuilt: {n_lab:,} rows, {n_part:,} participant facts")

    got = np.asarray(db.table("presence").scan().to_pydict()["object_id"])
    print(f"  verified: {len(set(got.tolist())):,} distinct object ids "
          f"in presence")


if __name__ == "__main__":
    main()
