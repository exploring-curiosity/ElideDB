"""Build the object-identity store from TRACKS, and show what each id got.

A sighting is no longer a frame-crop, it is a track: an unbroken run of
frames over which geometry alone proved "same object". The object store
is consulted once per track, on a descriptor pooled across it - so the
number of questions asked drops by ~95% and each question carries a
whole sighting's worth of evidence instead of one view.

Writes two tables:

  objects    one row per physical object - id, sightings, distinct
             episodes, exemplar feature. No name column, by design.
  instances  one row per TRACK - (ts, t1, stream, ep_ts, object_id,
             n_frames, box, conf). ts..t1 is the interval the object was
             continuously visible, which is what makes this a timeseries
             row rather than a bag of crops. Grouped by object_id, so
             "every sighting of object 7" is a row-group lookup.

  python scripts/build_identity.py --episodes 240 --sheets
  python scripts/build_identity.py --episodes 240 --sweep   (no write)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402
from elidedb import identity as ident                          # noqa: E402
from elidedb.identity import (Gallery, calibrate, descriptors,  # noqa: E402
                              detect, device, free_negatives, link)
from elidedb.video import FrameSet                             # noqa: E402

OUT = ROOT / "scratch_objects"
GROUP = 8          # episodes decoded at once: ~300 frames, ~276 MB


def sheet(crops, path, cols=10, cell=88):
    from PIL import Image
    if not crops:
        return
    rows = (len(crops) + cols - 1) // cols
    cv = Image.new("RGB", (cols * cell, rows * cell), (17, 21, 28))
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        if c.size == 0:
            continue
        p = Image.fromarray(c)
        p.thumbnail((cell - 4, cell - 4))
        cv.paste(p, (k * cell + 2, r * cell + 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.save(path)


def sightings(db, n_ep, want_crops, run=False):
    """Decode consecutive bursts, track them, describe each track once.

    Yields dicts with one entry per TRACK. Episodes are processed in
    groups so detection - which is stateless - batches across episode
    boundaries, while association stays per episode where it belongs.

    `run` takes a CONSECUTIVE run of episodes instead of a spread. The
    spread is the harder retrieval test but the wrong identity test: it
    samples across the whole corpus, where different demos are different
    kitchens, so most objects have no second chance to appear. A write
    path ingests sequentially, and it is within a run of demos of the
    same scene that an object store either recognises the same pot or
    does not.
    """
    ep = db.table("episodes").scan().to_pydict()
    ft = db.table("frames").scan()
    sel = (np.arange(min(n_ep, len(ep["ts"]))) if run else
           np.linspace(0, len(ep["ts"]) - 1, n_ep).round().astype(int))
    cost = defaultdict(float)
    seen = {"frames": 0, "dets": 0, "hours": 0.0, "eps": 0}

    for g0 in range(0, len(sel), GROUP):
        group = []
        t0 = time.perf_counter()
        for i in sel[g0:g0 + GROUP]:
            s, a, b = str(ep["stream"][i]), int(ep["ts"][i]), int(ep["t1"][i])
            sub = ft.filter(pc.and_(
                pc.equal(ft.column("stream"), s),
                pc.and_(pc.greater_equal(ft.column("ts"), a),
                        pc.less_equal(ft.column("ts"), b))))
            if len(sub) < 4:
                continue
            try:
                dec = sorted(FrameSet(db, "frames", sub).decode())
            except Exception:
                continue
            group.append((s, a, [int(x[0]) for x in dec],
                          [x[1] for x in dec]))
            seen["hours"] += (b - a) / 1e9 / 3600
            seen["eps"] += 1
        cost["decode"] += time.perf_counter() - t0
        if not group:
            continue

        flat = [f for e in group for f in e[3]]
        seen["frames"] += len(flat)
        t0 = time.perf_counter()
        dets = detect(flat)
        cost["detect"] += time.perf_counter() - t0
        seen["dets"] += sum(len(d[0]) for d in dets)

        off = 0
        for s, a, tss, frames in group:
            d = dets[off:off + len(frames)]
            off += len(frames)
            t0 = time.perf_counter()
            tracks = link(d, frames)
            cost["associate"] += time.perf_counter() - t0
            if not tracks:
                continue
            t0 = time.perf_counter()
            tids, F = descriptors(frames, tracks)
            cost["reid"] += time.perf_counter() - t0
            # proven-different pairs, in this episode's track-id space;
            # the caller maps them to global row indices
            neg = free_negatives({t: tracks[t] for t in tids})
            yield {"neg": neg, "ep_ts": a}
            for tid, v in zip(tids, F):
                t = tracks[tid]
                best = int(np.argmax(t["conf"]))
                x0, y0, x1, y1 = (int(q) for q in t["box"][best])
                yield {
                    "ts": tss[t["f"][0]], "t1": tss[t["f"][-1]],
                    "stream": s, "ep_ts": a, "track": int(tid),
                    "n_frames": len(t["f"]), "conf": float(max(t["conf"])),
                    "area": float(np.mean(t["area"])),
                    "box": [x0, y0, x1, y1], "vec": v,
                    "crops": ([frames[t["f"][i]][
                        int(t["box"][i][1]):int(t["box"][i][3]),
                        int(t["box"][i][0]):int(t["box"][i][2])]
                        for i in ident._views(t)] if want_crops else []),
                }
        seen["cost"] = dict(cost)
    seen["cost"] = dict(cost)
    yield seen


def collect(db, n_ep, want_crops, run=False):
    """Rows, the proven-different pairs as global row indices, and cost."""
    rows, stats, pending, pairs = [], {}, [], []
    where: dict = {}
    for r in sightings(db, n_ep, want_crops, run):
        if "neg" in r:
            pending.append(r)          # arrives just before its tracks
        elif "vec" in r:
            where[(r["ep_ts"], r["track"])] = len(rows)
            rows.append(r)
        else:
            stats.update(r)
    for p in pending:
        for i, j in p["neg"]:
            a = where.get((p["ep_ts"], i))
            b = where.get((p["ep_ts"], j))
            if a is not None and b is not None:
                pairs.append((a, b))
    return rows, pairs, stats


def sweep(rows, pairs):
    """Show the cut the corpus fits, and what every other cut would cost.

    The calibration signal needs no labels: pairs of tracks holding
    DISJOINT regions of the SAME frame are provably different objects.
    The fraction of those a threshold would merge is a true false-merge
    rate. Cross-episode pairs are the unknown mixture the store has to
    judge, so they are reported but prove nothing on their own.
    """
    V = np.stack([r["vec"] for r in rows])
    ep = np.array([r["ep_ts"] for r in rows])
    neg = np.array([float(V[i] @ V[j]) for i, j in pairs])
    S = V @ V.T
    iu = np.triu_indices(len(V), 1)
    unk = S[iu][ep[iu[0]] != ep[iu[1]]]
    print(f"\n{len(V)} track descriptors  "
          f"({len(neg)} proven-different pairs, {len(unk)} cross-episode)")
    print(f"  proven-different mean {neg.mean():.3f}  p90 "
          f"{np.percentile(neg, 90):.3f}  p99 {np.percentile(neg, 99):.3f}"
          f"  max {neg.max():.3f}")
    print(f"  cross-episode    mean {unk.mean():.3f}  p90 "
          f"{np.percentile(unk, 90):.3f}  p99 {np.percentile(unk, 99):.3f}")
    fit, n_neg = calibrate(V, pairs)
    print(f"  fitted cut {fit:.3f} from {n_neg} negatives")
    print(f"\n{'thr':>6} {'objects':>8} {'multi-ep':>9} {'sight/obj':>10} "
          f"{'false merge':>12} {'cross kept':>11}")
    for thr in (0.45, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, fit):
        g = Gallery(match=thr)
        ids = g.assign(V)
        by = defaultdict(set)
        for i, o in zip(ids, ep):
            by[int(i)].add(int(o))
        multi = sum(1 for k in by if len(by[k]) > 1)
        print(f"{thr:>6.2f} {len(g):>8} {multi:>9} "
              f"{len(V) / max(len(g), 1):>10.2f} "
              f"{(neg >= thr).mean() * 100:>11.2f}% "
              f"{(unk >= thr).mean() * 100:>10.2f}%")


def main():
    argv = sys.argv
    db = Store.open(argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/bench")
    n_ep = int(argv[argv.index("--episodes") + 1]
               if "--episodes" in argv else 120)
    want_sheets = "--sheets" in argv
    only_sweep = "--sweep" in argv

    rows, pairs, stats = collect(db, n_ep, want_sheets,
                                 run="--consecutive" in argv)
    if not rows:
        print("no tracks"); return

    cost = stats["cost"]
    nf = max(stats["frames"], 1)
    hrs = max(stats["hours"], 1e-9)
    L = np.array([r["n_frames"] for r in rows])
    perf = {
        "device": device(), "episodes": stats["eps"], "frames": nf,
        "detections": stats["dets"], "tracks": len(rows),
        "track_len_frames": {"p10": int(np.percentile(L, 10)),
                             "median": int(np.median(L)),
                             "p90": int(np.percentile(L, 90)),
                             "max": int(L.max())},
        "singleton_tracks": int((L == 1).sum()),
        # the number the design exists for
        "uploads_saved_pct": round(
            100 * (1 - len(rows) / max(stats["dets"], 1)), 1),
        "ms_per_frame": {k: round(v / nf * 1000, 2)
                         for k, v in cost.items()},
        "video_hours": round(hrs, 3),
        # decode is ALREADY paid by the write path (frames stream past
        # for the FDNN-V pass), so identity's marginal cost excludes it
        "min_per_hour_marginal": round(
            sum(v for k, v in cost.items() if k != "decode") / 60 / hrs, 2),
        "min_per_hour_with_decode": round(
            sum(cost.values()) / 60 / hrs, 2),
    }
    print(json.dumps(perf, indent=1))

    if only_sweep:
        sweep(rows, pairs)
        return

    V = np.stack([r["vec"] for r in rows])
    fit, n_neg = calibrate(V, pairs)
    gal = Gallery(match=fit)
    ids = gal.assign(V)
    for r, o in zip(rows, ids):
        r["object_id"] = int(o)

    inst = pa.table({
        "ts": pa.array([r["ts"] for r in rows], pa.int64()),
        "t1": pa.array([r["t1"] for r in rows], pa.int64()),
        "stream": pa.array([r["stream"] for r in rows]),
        "ep_ts": pa.array([r["ep_ts"] for r in rows], pa.int64()),
        "object_id": pa.array([r["object_id"] for r in rows], pa.int32()),
        "n_frames": pa.array([r["n_frames"] for r in rows], pa.int32()),
        "conf": pa.array([r["conf"] for r in rows], pa.float32()),
        "mask_px": pa.array([r["area"] for r in rows], pa.float32()),
        "box": pa.array([r["box"] for r in rows], pa.list_(pa.int32(), 4)),
    })
    inst = inst.take(pc.sort_indices(inst.column("ts")))
    # evolve: the unit of a row changed from frame-crop to track, which
    # adds n_frames. The store refuses silent schema drift, so the
    # widening is stated here rather than discovered by a reader later.
    #
    # min_group_rows=256 is the measured knee of this table's U-curve,
    # not a value copied from labels. Average bytes a single-object
    # lookup must touch (footer + surviving groups), over all 132
    # objects: one group 29,833 -> 512 rows 17,665 -> 256 rows 15,912
    # -> 128 rows 16,266 -> one group per object 202,812, where 132
    # groups x 9 columns of footer swamp 111 KB of data. Halving the
    # lookup against a single row group, and the small end is a cliff.
    db.table("instances").append_grouped(
        inst, "object_id", kind="index", replace=True, evolve=True,
        sort_by=["object_id", "ts"], min_group_rows=256,
        meta={"builder": "build_identity", "unit": "track"})

    by_obj, cnt = defaultdict(set), defaultdict(int)
    for r in rows:
        by_obj[r["object_id"]].add(r["ep_ts"])
        cnt[r["object_id"]] += 1
    C = gal.centroids()
    oids = sorted(cnt)
    objs = pa.table({
        "ts": pa.array([min(by_obj[o]) for o in oids], pa.int64()),
        "t1": pa.array([max(by_obj[o]) for o in oids], pa.int64()),
        "object_id": pa.array(oids, pa.int32()),
        "n_instances": pa.array([cnt[o] for o in oids], pa.int32()),
        "n_episodes": pa.array([len(by_obj[o]) for o in oids], pa.int32()),
        "feature": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(
                C[oids].astype(np.float16)).reshape(-1), pa.float16()),
            C.shape[1]),
    })
    objs = objs.take(pc.sort_indices(objs.column("ts")))
    db.table("objects").replace(objs, kind="index",
                                meta={"builder": "build_identity"})

    print(json.dumps({
        "match_cut": round(fit, 3), "fitted_on_negatives": n_neg,
        "objects": len(cnt),
        "sightings_per_object": round(len(rows) / max(len(cnt), 1), 2),
        "multi_episode_objects": sum(1 for o in by_obj if len(by_obj[o]) > 1),
    }, indent=1))

    if want_sheets:
        OUT.mkdir(parents=True, exist_ok=True)
        for p in OUT.glob("obj_*.png"):
            p.unlink()
        crops = defaultdict(list)
        for r in rows:
            crops[r["object_id"]].extend(r["crops"][:4])
        for rank, (oid, n) in enumerate(
                sorted(cnt.items(), key=lambda kv: -kv[1])[:12]):
            sheet(crops[oid][:24],
                  OUT / f"obj_{rank:02d}_id{oid}_n{n}"
                        f"_ep{len(by_obj[oid])}.png")
        print(f"sheets -> {OUT}")


if __name__ == "__main__":
    main()
