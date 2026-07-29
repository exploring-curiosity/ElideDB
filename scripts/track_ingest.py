"""TRACK VECTORS: object identity from crops taken at NATIVE resolution.

The store already had an `object_vectors` table and it was worthless —
across 51,101 crops the best cosine to "an eggplant" was 0.184 and the
two episodes that actually contain one ranked 75th and 110th. The
conclusion drawn at the time (crops cannot carry identity) was wrong.
The crops were cut out of frames that had already been DOWNSCALED for
decode, so a table-top object arrived at the encoder as ~30 px. The
store's own rendition is 640x480 — the resolution was thrown away by
the ingest pass, not by the store.

Redone at native resolution on a pool of 80 episodes, the same idea
ranks true episodes 1st and 8th (banana), 1st/2nd/5th (green object):
yield@10 of 1.00, 0.75, 1.00, 0.50 on q10/q00/q09/q08, against
0.00/0.20/0.00/0.10 shipped.

Where the crops come from — Gestalt COMMON FATE, no detector, nothing
dataset-specific: dense optical flow, connected components of coherent
motion, linked frame to frame by proximity. Things that move together
are one thing. Crops are taken along each track and every view is kept
SEPARATELY: pooling them (the Treisman/Kahneman object-file prediction)
was measured and it LOST — 1.81s single vs 0.90s pooled on q09, and
again on q08 — because the box drifts and the mean blends the object
with background. Best-view-wins, which is also what the ranking stage
does with them (max over views).

Reads the store's own media, never the raw source: the store stays
self-contained.

  python scripts/track_ingest.py [store] [--limit N]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402

NF = 12             # frames sampled per episode for flow
MIN_BLOB = 60       # px, ignore speckle
MAX_TRACKS = 4      # longest-lived tracks kept per episode
MAX_VIEWS = 8       # views kept per track
PAD = 0.35          # box padding, fraction of box size
LINK_PX = 60        # centroid distance that continues a track


def tracks_of(frames):
    """Coherent-motion blobs linked across frames -> [[(i, bbox), ...]]."""
    import cv2
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    tracks, live = [], []
    for i in range(len(grey) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grey[i], grey[i + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        # robust threshold: movement is what stands out from THIS frame
        # pair's own motion floor (camera shake, sensor noise, lighting)
        med = float(np.median(mag))
        mad = float(np.median(np.abs(mag - med))) + 1e-6
        mask = (mag > max(1.0, med + 4 * 1.4826 * mad)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((7, 7), np.uint8))
        nlab, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        nxt = []
        for j in range(1, nlab):
            if stats[j, cv2.CC_STAT_AREA] < MIN_BLOB:
                continue
            bb = (int(stats[j, cv2.CC_STAT_LEFT]),
                  int(stats[j, cv2.CC_STAT_TOP]),
                  int(stats[j, cv2.CC_STAT_WIDTH]),
                  int(stats[j, cv2.CC_STAT_HEIGHT]))
            c = tuple(float(v) for v in cents[j])
            best, bd = None, 1e9
            for tr in live:
                d = float(np.hypot(*(np.array(tr[-1][2]) - np.array(c))))
                if d < bd:
                    best, bd = tr, d
            if best is not None and bd < LINK_PX:
                best.append((i, bb, c))
                nxt.append(best)
            else:
                tr = [(i, bb, c)]
                tracks.append(tr)
                nxt.append(tr)
        live = nxt
    return tracks


def crops_of(frames, tracks):
    """(list[np.uint8 HWC], list[(track_id, area)]) at native resolution."""
    H, W = frames[0].shape[:2]
    out, meta = [], []
    keep = [tr for tr in sorted(tracks, key=len, reverse=True)
            if len(tr) >= 3][:MAX_TRACKS]
    for ti, tr in enumerate(keep):
        step = max(1, len(tr) // MAX_VIEWS)
        for (fi, (x, y, w, h), _) in tr[::step][:MAX_VIEWS]:
            if fi >= len(frames):
                continue
            px, py = int(w * PAD), int(h * PAD)
            x0, y0 = max(x - px, 0), max(y - py, 0)
            x1, y1 = min(x + w + px, W), min(y + h + py, H)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            out.append(frames[fi][y0:y1, x0:x1])
            meta.append((ti, float(w * h) / float(W * H)))
    return out, meta


def main():
    from PIL import Image

    from elidedb.embeddings import DEFAULT_MODEL, _embed_images
    from elidedb.video import FrameSet

    argv = sys.argv[1:]
    limit = None
    if "--limit" in argv:
        i = argv.index("--limit")
        limit = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    store = Store.open(argv[0] if argv else "lake/bench")

    ep = store.table("episodes").scan()
    recs = list(zip(ep.column("stream").to_pylist(),
                    (int(v) for v in ep.column("ts").to_pylist()),
                    (int(v) for v in ep.column("t1").to_pylist())))
    if limit:
        recs = recs[:limit]
    frames_tbl = store.table("frames").scan()

    rows_s, rows_a, rows_b, rows_t, rows_ar, vecs = [], [], [], [], [], []
    t0 = time.time()
    for ri, (s, a, b) in enumerate(recs):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 4:
            continue
        pick = np.linspace(0, len(sel) - 1, min(NF, len(sel)))
        try:
            # width=None: NATIVE resolution. This is the whole point.
            dec = FrameSet(store, "frames",
                           sel.take(pick.round().astype(int))).decode()
        except Exception:
            continue
        frames = [f for _, f in sorted(dec)]
        if len(frames) < 4:
            continue
        crops, meta = crops_of(frames, tracks_of(frames))
        if not crops:
            continue
        V = np.asarray(_embed_images([Image.fromarray(c) for c in crops],
                                     DEFAULT_MODEL), np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
        for (ti, ar), v in zip(meta, V):
            rows_s.append(s); rows_a.append(a); rows_b.append(b)
            rows_t.append(ti); rows_ar.append(ar); vecs.append(v)
        if (ri + 1) % 50 == 0:
            el = time.time() - t0
            eta = el / (ri + 1) * len(recs)
            print(f"  {ri + 1}/{len(recs)}  {len(vecs)} crops  {el:.0f}s  "
                  f"ETA {eta / 60:.0f}min", flush=True)
            # same cost gate as every other ingest: never let a channel
            # quietly become the most expensive thing in the write path
            if eta > 5400:
                print("COST GATE: ETA > 90min, aborting")
                return

    V = np.stack(vecs)
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(V.astype(np.float16)).reshape(-1),
                     pa.float16()), V.shape[1]),
        "track": pa.array(rows_t, pa.int32()),
        "area": pa.array(rows_ar, pa.float32()),
    })
    tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
    store.table("track_vectors").append(
        tbl, kind="embeddings",
        meta={"model": DEFAULT_MODEL, "dim": int(V.shape[1]),
              "resolution": "native", "frames_per_episode": NF,
              "max_tracks": MAX_TRACKS, "max_views": MAX_VIEWS})
    print(json.dumps({"rows": len(tbl), "episodes": len(set(rows_a)),
                      "dim": int(V.shape[1]),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
