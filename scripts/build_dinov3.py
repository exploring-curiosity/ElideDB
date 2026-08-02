"""FULL BUILD of the DINOv3 substrate: one decode pass, two products.

    scene_vectors    DINOv3 ConvNeXt-Tiny over EVERY frame - the scene
                     series and the appearance space (768-d)
    presence +       identity rebuilt on DINOv3 crop descriptors:
    object_vectors   subset A/B measured AUC 0.9994 vs the nano ReID's
                     0.9047, false merges 6.5x fewer (bench/ab_identity)

One pass because decode dominates and both products consume the same
frames. Everything downstream of decode is sequential on the GPU.

PROJECTED COST, from measured numbers (bench/bench_dinov3.json + the
200-segment A/B scaled x10.5): decode ~6 min, propose ~25 min, track
~1 min, frame embed ~5 min, crop embed ~11 min => ~45-50 min. The
projection prints at start; the actual per-stage cost prints at end.

RESUMABLE: a checkpoint every 200 segments to _cache/, so an interrupt
costs at most 200 segments, not the run.

Identity practice as settled 2026-08-01: descriptors assigned in
TRACK-CLOSE order (order beats the cut); the cut fitted by sweeping
cross-episode recurrence (fit_cut); pairs proven by box geometry
(interval_pairs), both sides.

    python scripts/build_dinov3.py [--store lake/fresh_bench]
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

from elidedb import Store, dinov3                              # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402
from elidedb.identity import (Gallery, Stream, fit_cut,        # noqa: E402
                              interval_pairs, propose)

CKPT_EVERY = 200
VIEWS = 4


def segments(db):
    ft = db.table("frames").scan()
    ft = ft.take(pc.sort_indices(ft, sort_keys=[("stream", "ascending"),
                                                ("ts", "ascending")]))
    src = ft.column("source").to_pylist()
    stream = ft.column("stream").to_pylist()
    out, start = [], 0
    for i in range(1, len(src) + 1):
        if i == len(src) or src[i] != src[start]:
            out.append((stream[start], start, i - start))
            start = i
    return ft, out


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/fresh_bench")
    db = Store.open(str(store))
    ft, segs = segments(db)
    print(f"{store.name}: {len(ft):,} frames over {len(segs):,} segments")
    print("projection: ~45-50 min total (decode 6, propose 25, track 1, "
          "frame-embed 5, crop-embed 11)", flush=True)

    ck = db.dir / "_cache" / "build_dinov3.npz"
    ck.parent.mkdir(exist_ok=True)
    scene, rows, done = [], [], 0
    if ck.exists():
        d = np.load(ck, allow_pickle=True)
        scene = list(d["scene"])
        rows = list(d["rows"])
        done = int(d["done"])
        print(f"resuming at segment {done} "
              f"({len(scene):,} frame rows, {len(rows):,} tracks held)")

    print("loading models (DINOv3 ConvNeXt-Tiny + FastSAM)...", flush=True)
    dinov3._load()
    _ = propose([np.zeros((64, 64, 3), np.uint8)])   # load before the bar

    cost = defaultdict(float)
    t_start = time.time()
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
            F = dinov3.embed(ims)
            cost["frame_embed"] += time.time() - a
            for (ts, _), v in zip(chunk, F):
                scene.append((int(ts), sname, v.astype(np.float16)))
            a = time.time()
            dets = propose(ims)
            cost["propose"] += time.time() - a
            a = time.time()
            st = Stream(views=VIEWS)
            closed = []
            for (ts, im), b in zip(chunk, dets):
                b = np.asarray(b, np.int32).reshape(-1, 4)
                closed += st.update(int(ts),
                                    (b, np.ones(len(b), np.float32),
                                     np.ones(len(b), np.float32)), im)
            closed += st.flush()
            cost["track"] += time.time() - a
            # descriptors at close, in close order, pixels dropped here
            a = time.time()
            crops, owner = [], []
            metas = []
            for _, t in closed:
                if not t["crops"]:
                    continue
                metas.append((sname, seg_no,
                              {"ts": t["ts"], "t1": t["t1"], "n": t["n"],
                               "conf": float(max(t["conf"])) if t["conf"]
                               else 0.0,
                               "box": [int(x) for x in t["box"][0]]}))
                for c in t["crops"]:
                    crops.append(c[1])
                    owner.append(len(metas) - 1)
            if crops:
                E = dinov3.embed(crops)
                V = np.zeros((len(metas), E.shape[1]), np.float32)
                n = np.zeros(len(metas), np.int32)
                for e, k in zip(E, owner):
                    V[k] += e
                    n[k] += 1
                V /= np.maximum(n[:, None], 1)
                V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True),
                                1e-8)
                for m, v in zip(metas, V):
                    rows.append((*m, v.astype(np.float16)))
            cost["crop_embed"] += time.time() - a
        if (seg_no + 1) % CKPT_EVERY == 0:
            np.savez(ck, scene=np.array(scene, object),
                     rows=np.array(rows, object), done=seg_no + 1)
    print(f"pass done in {(time.time()-t_start)/60:.1f} min  cost="
          f"{json.dumps({k: round(v/60, 1) for k, v in cost.items()})} min",
          flush=True)

    # ---- identity: pairs -> fitted cut -> ids, in close order --------
    geo = [(s, m) for s, _, m, _ in rows]
    neg, pos = interval_pairs(geo)
    V = np.stack([v for *_, v in rows]).astype(np.float32)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
    epi = np.array([seg for _, seg, _, _ in rows])
    print(f"{len(rows):,} tracks  proven-diff {len(neg):,}  "
          f"proven-same {len(pos):,}", flush=True)
    a = time.time()
    cut, report = fit_cut(V, epi, neg, pos or None)
    for r in report:
        print("  " + json.dumps(r) + ("   <- fit" if r["cut"] == cut
                                      else ""), flush=True)
    ids = Gallery(match=cut).assign(V)
    print(f"cut {cut}  ({time.time()-a:.0f}s)  objects "
          f"{int(ids.max())+1:,}", flush=True)

    # ---- writes ------------------------------------------------------
    mid = dinov3.MID.split("/")[-1]
    sc = pa.table({
        "ts": pa.array([t for t, _, _ in scene], pa.int64()),
        "t1": pa.array([t for t, _, _ in scene], pa.int64()),
        "stream": pa.array([s for _, s, _ in scene]),
        "vector": pa.array([v.astype(np.float32) for _, _, v in scene],
                           pa.list_(pa.float32(), V.shape[1])),
    })
    sc = sc.take(pc.sort_indices(sc.column("ts")))
    db.table("scene_vectors").set_layout("stream",
                                         sort_by=["stream", "ts"],
                                         min_group_rows=4096)
    db.table("scene_vectors").replace(sc, kind="vectors",
                                      meta={"model": mid, "dim": V.shape[1],
                                            "unit": "frame",
                                            "role": "scene series + "
                                                    "appearance"})
    print(f"scene_vectors: {len(sc):,} rows", flush=True)

    pres = pa.table({
        "ts": pa.array([m["ts"] for _, _, m, _ in rows], pa.int64()),
        "t1": pa.array([m["t1"] for _, _, m, _ in rows], pa.int64()),
        "stream": pa.array([s for s, _, _, _ in rows]),
        "object_id": pa.array([int(i) for i in ids], pa.int32()),
        "n_frames": pa.array([m["n"] for _, _, m, _ in rows], pa.int32()),
        "conf": pa.array([m["conf"] for _, _, m, _ in rows], pa.float32()),
        "box": pa.array([m["box"] for _, _, m, _ in rows],
                        pa.list_(pa.int32(), 4)),
    })
    order = pc.sort_indices(pres.column("ts"))
    db.table("presence").set_layout("object_id",
                                    sort_by=["object_id", "ts"],
                                    min_group_rows=256)
    db.table("presence").replace(pres.take(order), kind="index",
                                 meta={"unit": "presence_interval",
                                       "model": mid,
                                       "match_cut": round(float(cut), 3),
                                       "fit": "cross-episode recurrence"})
    ov = pa.table({
        "ts": pres.column("ts"), "t1": pres.column("t1"),
        "stream": pres.column("stream"),
        "object_id": pres.column("object_id"),
        "vector": pa.array([v.astype(np.float32) for *_, v in rows],
                           pa.list_(pa.float32(), V.shape[1])),
    })
    db.table("object_vectors").set_layout("object_id",
                                          sort_by=["object_id", "ts"],
                                          min_group_rows=256)
    db.table("object_vectors").replace(ov.take(order), kind="index",
                                       meta={"match_cut": round(float(cut), 3),
                                             "unit": "track_descriptor",
                                             "encoder": mid})
    print(f"presence/object_vectors: {len(pres):,} rows", flush=True)

    from fix_participants import rebuild_labels
    n_lab, n_part = rebuild_labels(db)
    print(f"labels: {n_lab:,} rows, {n_part:,} participant facts")

    # THE ARTIFACT IS THE TEST
    for t, want in (("scene_vectors", len(sc)), ("presence", len(pres)),
                    ("object_vectors", len(pres))):
        got = len(db.table(t).scan())
        assert got == want, (t, got, want)
    ck.unlink(missing_ok=True)
    print(f"TOTAL {(time.time()-t_start)/60:.1f} min - verified all tables")


if __name__ == "__main__":
    main()
