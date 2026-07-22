"""FDNN-V2 training cache for bridge4h: EVERY frame + L1 targets.

Frames as raw .npy (memmap-able — 5.8 GB of uint8 should never transit
through pickle), L1 predictive targets = the store's own frame_vectors
(V1 appearance embeddings, one per frame, already computed at load time).
The predictive objective needs no teacher and no labels: the future is its
own supervision.
"""
import sys, time
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb import Store
from elidedb.video import FrameSet

OUT = Path("data/cache/b4h")
OUT.mkdir(parents=True, exist_ok=True)
db = Store.open("lake/bridge4h")
fv = db.table("frame_vectors").scan()
frames = db.table("frames").scan()
streams = sorted(set(fv.column("stream").to_pylist()))

n_total = len(fv)
px = np.lib.format.open_memmap(OUT / "px.npy", mode="w+",
                               dtype=np.uint8, shape=(n_total, 144, 192, 3))
vec = np.zeros((n_total, 1152), np.float32)
sid = np.zeros(n_total, np.int32)
ts = np.zeros(n_total, np.int64)
row = 0
t0 = time.time()
for si, s in enumerate(streams):
    sub = fv.filter(pc.equal(fv.column("stream"), s))
    tt = np.sort(sub.column("ts").to_numpy())
    vmap = {int(a): np.asarray(v, np.float32) for a, v in
            zip(sub.column("ts").to_pylist(), sub.column("vector").to_pylist())}
    rows = frames.filter(pc.equal(frames.column("stream"), s))
    rows = rows.take(pc.sort_indices(rows.column("ts")))
    for i in range(0, len(rows), 512):
        dec = FrameSet(db, "frames", rows.slice(i, 512)).decode(width=192)
        for t, img in dec:
            if int(t) not in vmap:
                continue
            px[row] = img
            vec[row] = vmap[int(t)]
            sid[row] = si
            ts[row] = int(t)
            row += 1
    print(f"  {s[-8:]}: total {row} ({time.time()-t0:.0f}s)", flush=True)
px.flush()
np.save(OUT / "vec.npy", vec[:row])
np.save(OUT / "sid.npy", sid[:row])
np.save(OUT / "ts.npy", ts[:row])
np.save(OUT / "streams.npy", np.array(streams))
print(f"cache: {row:,} frames, {time.time()-t0:.0f}s -> {OUT}")
