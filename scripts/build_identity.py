"""Build the object-identity store and SHOW what each id contains.

Writes two tables:

  objects    one row per physical object - id, how many instances, how
             many distinct episodes it appears in, and its exemplar
             feature. No name column, by design.
  instances  one row per sighting - (ts, stream, object_id, box, conf).
             This is what joins an object to the episodes it is in, so
             "episodes containing object 7" is an index lookup, not a
             vector scan.

  python scripts/build_identity.py --episodes 120 --per-ep 3 --sheets
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

from elidedb import Store                                    # noqa: E402
from elidedb.identity import Gallery, device, features, regions  # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402

OUT = ROOT / "scratch_objects"


def sheet(crops, path, cols=10, cell=88):
    from PIL import Image
    if not crops:
        return
    rows = (len(crops) + cols - 1) // cols
    cv = Image.new("RGB", (cols * cell, rows * cell), (17, 21, 28))
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        p = Image.fromarray(c)
        p.thumbnail((cell - 4, cell - 4))
        cv.paste(p, (k * cell + 2, r * cell + 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.save(path)


def main():
    argv = sys.argv
    db = Store.open(argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/bench")
    n_ep = int(argv[argv.index("--episodes") + 1]
               if "--episodes" in argv else 120)
    per_ep = int(argv[argv.index("--per-ep") + 1]
                 if "--per-ep" in argv else 3)
    want_sheets = "--sheets" in argv

    ep = db.table("episodes").scan().to_pydict()
    ft = db.table("frames").scan()
    sel_idx = np.linspace(0, len(ep["ts"]) - 1, n_ep).round().astype(int)

    gal = Gallery()
    rows, keep_crops = [], defaultdict(list)
    t_dec = t_seg = t_reid = 0.0
    n_frames = 0

    for i in sel_idx:
        s, a, b = str(ep["stream"][i]), int(ep["ts"][i]), int(ep["t1"][i])
        t0 = time.perf_counter()
        sub = ft.filter(pc.and_(
            pc.equal(ft.column("stream"), s),
            pc.and_(pc.greater_equal(ft.column("ts"), a),
                    pc.less_equal(ft.column("ts"), b))))
        if len(sub) < per_ep:
            continue
        pick = np.unique(np.linspace(0, len(sub) - 1, per_ep)
                         .round().astype(int))
        try:
            dec = sorted(FrameSet(db, "frames", sub.take(pick)).decode())
        except Exception:
            continue
        t_dec += time.perf_counter() - t0
        ims = [x[1] for x in dec]
        ts = [int(x[0]) for x in dec]
        n_frames += len(ims)

        t0 = time.perf_counter()
        per = regions(ims)
        t_seg += time.perf_counter() - t0

        for im, t, got in zip(ims, ts, per):
            if not got:
                continue
            boxes = [g[0] for g in got]
            t0 = time.perf_counter()
            F = features(im, boxes)
            t_reid += time.perf_counter() - t0
            ids = gal.assign(F)
            for (box, area, conf), oid in zip(got, ids):
                rows.append({"ts": t, "stream": s, "ep_ts": a,
                             "object_id": int(oid), "box": list(box),
                             "mask_px": float(area), "conf": float(conf)})
                if want_sheets and len(keep_crops[int(oid)]) < 20:
                    x0, y0, x1, y1 = box
                    keep_crops[int(oid)].append(im[y0:y1, x0:x1])

    # ---- tables
    inst = pa.table({
        "ts": pa.array([r["ts"] for r in rows], pa.int64()),
        "t1": pa.array([r["ts"] for r in rows], pa.int64()),
        "stream": pa.array([r["stream"] for r in rows]),
        "ep_ts": pa.array([r["ep_ts"] for r in rows], pa.int64()),
        "object_id": pa.array([r["object_id"] for r in rows], pa.int32()),
        "conf": pa.array([r["conf"] for r in rows], pa.float32()),
        "mask_px": pa.array([r["mask_px"] for r in rows], pa.float32()),
        "box": pa.array([r["box"] for r in rows],
                        pa.list_(pa.int32(), 4)),
    })
    inst = inst.take(pc.sort_indices(inst.column("ts")))
    # grouped by object_id: "every sighting of object 7" is the lookup,
    # so that is the row-group boundary
    db.table("instances").append_grouped(
        inst, "object_id", kind="index", replace=True,
        sort_by=["object_id", "ts"], min_group_rows=512,
        meta={"builder": "build_identity"})

    by_obj = defaultdict(set)
    cnt = defaultdict(int)
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

    if want_sheets:
        OUT.mkdir(parents=True, exist_ok=True)
        for p in OUT.glob("obj_*.png"):
            p.unlink()
        top = sorted(cnt.items(), key=lambda kv: -kv[1])[:10]
        for rank, (oid, n) in enumerate(top):
            sheet(keep_crops[oid],
                  OUT / f"obj_{rank:02d}_id{oid}_n{n}"
                        f"_ep{len(by_obj[oid])}.png")

    # decode is ALREADY paid by the write path (frames stream past for
    # the FDNN-V pass), so the marginal cost of identity is seg + reid
    hours = sum(int(ep["t1"][i]) - int(ep["ts"][i])
                for i in sel_idx) / 1e9 / 3600
    tot = t_dec + t_seg + t_reid
    n_obj = len(cnt)
    print(json.dumps({
        "device": device(), "frames": n_frames,
        "episodes": len(sel_idx), "sightings": len(rows),
        "objects": n_obj,
        "sightings_per_object": round(len(rows) / max(n_obj, 1), 2),
        "multi_episode_objects": sum(1 for o in by_obj
                                     if len(by_obj[o]) > 1),
        "ms_per_frame": {
            "decode": round(t_dec / max(n_frames, 1) * 1000, 1),
            "segment": round(t_seg / max(n_frames, 1) * 1000, 1),
            "reid": round(t_reid / max(n_frames, 1) * 1000, 1),
            "total": round(tot / max(n_frames, 1) * 1000, 1)},
        # hours come from the EPISODE SPANS, not from a frame count and
        # an assumed fps - the sampled frames are 3 per episode however
        # long the episode is, so deriving duration from them is wrong
        "video_hours": round(hours, 3),
        "min_per_hour_video": round(tot / 60 / max(hours, 1e-9), 2),
        "min_per_hour_marginal": round(
            (t_seg + t_reid) / 60 / max(hours, 1e-9), 2),
    }, indent=1))


if __name__ == "__main__":
    main()
