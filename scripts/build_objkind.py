"""KIND descriptors for event-bound objects, corpus-wide.

The kind-gap diagnostic (scripts/diag_kind.py) measured the wall and
the fix on the same bank: the shipped instance descriptors separate
object KINDS at mean AUC 0.704, DINOv3 ViT-S fp32 at 0.750, ViT-S +30%
context at 0.759 - roughly double the raw gaps (q00 0.096 -> 0.212,
q07 0.165 -> 0.340). This builds the winning variant over every
event-bound track in the corpus.

    objkind_vectors   one row per (interval, object): DINOv3 ViT
                      fp32 over NVIEW native-resolution crops with
                      +30% context, mean-pooled, unit-norm

meta declares "encoder", NOT "model": this is an INDEX for the
consensus join ("what do the seeds share that the corpus does not"),
not an episode retrieval channel - the same distinction that keeps
object_vectors out of spaces(). And per the Goodhart finding (a channel
whose construction mirrors the selection statistic games it), anything
built on these descriptors is evaluated STANDALONE, never enrolled in
pairwise selection.

Boxes come from trajectories; no detector, no tracker runs here.

    ELIDEDB_KIND_MID=...vits16... python scripts/build_objkind.py
"""
from __future__ import annotations

import json
import os
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

from elidedb import Store                                      # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402
from diag_kind import _crop                                    # noqa: E402

MID = os.environ.get("ELIDEDB_KIND_MID",
                     "facebook/dinov3-vits16-pretrain-lvd1689m")
PAD = 0.30
NVIEW = 4
CKPT_EVERY = 300


def main():
    db = Store.open(str(ROOT / (sys.argv[sys.argv.index("--store") + 1]
                                if "--store" in sys.argv
                                else "lake/fresh_bench")))
    ev = db.table("events").scan().to_pydict()
    bound = {(str(s), int(o)) for s, o in zip(ev["stream"], ev["object_id"])
             if int(o) >= 0}
    tr = db.table("trajectories").scan().to_pydict()
    tracks = defaultdict(list)
    for j in range(len(tr["ts"])):
        s, oid = str(tr["stream"][j]), int(tr["object_id"][j])
        if tr["is_agent"][j] or (s, oid) not in bound:
            continue
        tracks[(s, int(tr["track_ts"][j]), int(tr["t1"][j]), oid)].append(
            (int(tr["ts"][j]), (int(tr["x0"][j]), int(tr["y0"][j]),
                                int(tr["x1"][j]), int(tr["y1"][j]))))
    for v in tracks.values():
        v.sort()
    print(f"{len(tracks):,} event-bound tracks "
          f"(~{len(tracks) * NVIEW:,} crops) with {MID.split('/')[-1]}",
          flush=True)

    # group needed frames by segment via the frames table
    ft = db.table("frames").scan()
    ft = ft.take(pc.sort_indices(ft, sort_keys=[("stream", "ascending"),
                                                ("ts", "ascending")]))
    src = ft.column("source").to_pylist()
    stream_col = ft.column("stream").to_pylist()
    ts_col = [int(v) for v in ft.column("ts").to_pylist()]
    seg_of, bounds_list, start = {}, [], 0
    for i in range(1, len(src) + 1):
        if i == len(src) or src[i] != src[start]:
            bounds_list.append((start, i - start))
            for j in range(start, i):
                seg_of[(stream_col[j], ts_col[j])] = len(bounds_list) - 1
            start = i
    by_seg = defaultdict(list)
    for k, pts in tracks.items():
        pick = np.unique(np.linspace(0, len(pts) - 1, NVIEW)
                         .round().astype(int))
        want = [pts[i] for i in pick]
        sg = seg_of.get((k[0], want[0][0]))
        if sg is not None:
            by_seg[sg].append((k, want))

    from elidedb import dinov3
    print("loading DINOv3 (fp32 for ViT - the fp16 NaN lesson)...",
          flush=True)
    dinov3._load(MID)

    ck = db.dir / "_cache" / "objkind.npz"
    rows, done = [], set()
    if ck.exists():
        d = np.load(ck, allow_pickle=True)
        rows = list(d["rows"])
        done = set(int(x) for x in d["done"])
        print(f"resuming: {len(rows):,} rows, {len(done)} segments")

    t0 = time.time()
    todo = sorted(by_seg)
    for n_done, sg in enumerate(tqdm(todo, desc="objkind", unit="seg")):
        if sg in done:
            continue
        i0, n_b = bounds_list[sg]
        chunk = FrameSet(db, "frames", ft.slice(i0, n_b)).decode()
        if not chunk:
            done.add(sg)
            continue
        frame_at = {int(t): im for t, im in chunk}
        crops, owner, metas = [], [], []
        for k, want in by_seg[sg]:
            got = []
            for t, box in want:
                im = frame_at.get(t)
                if im is None:
                    continue
                c = _crop(im, box, PAD)
                if c is not None:
                    got.append(c)
            if got:
                metas.append(k)
                for c in got:
                    crops.append(c)
                    owner.append(len(metas) - 1)
        if crops:
            E = dinov3.embed(crops, mid=MID)
            V = np.zeros((len(metas), E.shape[1]), np.float32)
            n = np.zeros(len(metas), np.int32)
            for e, o in zip(E, owner):
                V[o] += e
                n[o] += 1
            V /= np.maximum(n[:, None], 1)
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
            for k, v in zip(metas, V):
                rows.append((*k, v.astype(np.float16)))
        done.add(sg)
        if (n_done + 1) % CKPT_EVERY == 0:
            np.savez(ck, rows=np.array(rows, object),
                     done=np.array(sorted(done)))

    dim = len(rows[0][4])
    tbl = pa.table({
        "ts": pa.array([a for _, a, _, _, _ in rows], pa.int64()),
        "t1": pa.array([b for _, _, b, _, _ in rows], pa.int64()),
        "stream": pa.array([s for s, *_ in rows]),
        "object_id": pa.array([o for _, _, _, o, _ in rows], pa.int32()),
        "vector": pa.array([v.astype(np.float32) for *_, v in rows],
                           pa.list_(pa.float32(), dim)),
    })
    tbl = tbl.take(pc.sort_indices(tbl, sort_keys=[
        ("object_id", "ascending"), ("ts", "ascending")]))
    db.table("objkind_vectors").set_layout(
        "object_id", sort_by=["object_id", "ts"], min_group_rows=256)
    db.table("objkind_vectors").replace(
        tbl, kind="index",
        meta={"encoder": MID, "dim": dim, "unit": "event_bound_track",
              "crop": f"native+{int(PAD*100)}% context, {NVIEW} views",
              "why": "kind separation AUC 0.759 vs instance 0.704 "
                     "(diag_kind 2026-08-02)"})
    got = len(db.table("objkind_vectors").scan())
    assert got == len(tbl)
    ck.unlink(missing_ok=True)
    print(json.dumps({"rows": got, "dim": dim,
                      "minutes": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    main()
