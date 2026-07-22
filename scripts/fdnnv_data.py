"""Build the FDNN-V distillation cache: (frame pixels, teacher embedding).

The pairs are FREE — SigLIP embeddings for these frames already sit in the
`frame_vectors` table from the context-index build. No labels, no leakage:
teacher and student both see only pixels.

Decode is CHUNKED. A dense selection over a whole packed file merges into one
byte run, and one run means one ffmpeg whose rawvideo stdout for 20k frames is
~19 GB buffered in RAM. Chunking the *selection* keeps each decode's buffer
under ~1 GB while still hitting the fast batched path inside each chunk.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402

OUT = Path("data/cache/fdnnv_train.npz")
WIDTH = 192          # -> 192x144, the student's input resolution


def main():
    db = Store.open("lake/bridge")
    fv = db.table("frame_vectors").scan()
    frames = db.table("frames").scan()

    all_px, all_vec, all_ts, all_stream = [], [], [], []
    t0 = time.time()
    for stream in sorted(set(fv.column("stream").to_pylist())):
        sub = fv.filter(pc.equal(fv.column("stream"), stream))
        ts = np.sort(sub.column("ts").to_numpy())
        vecs = {t: v for t, v in zip(sub.column("ts").to_pylist(),
                                     sub.column("vector").to_pylist())}
        rows = frames.filter(pc.equal(frames.column("stream"), stream))
        rts = rows.column("ts").to_numpy()
        want_pos = np.searchsorted(rts, ts)

        for i in range(0, len(want_pos), 512):          # chunked decode
            block = rows.take(want_pos[i:i + 512])
            dec = FrameSet(db, "frames", block).decode(width=WIDTH)
            for t, img in dec:
                v = vecs.get(int(t))
                if v is None:
                    continue
                all_px.append(img)                       # (144, 192, 3) u8
                all_vec.append(np.asarray(v, np.float16))
                all_ts.append(int(t))
                all_stream.append(stream)
            if (i // 512) % 8 == 0:
                print(f"  {stream[-8:]}: {i + len(dec)}/{len(want_pos)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    px = np.stack(all_px)
    vec = np.stack(all_vec)
    streams = sorted(set(all_stream))
    sid = np.array([streams.index(s) for s in all_stream], np.int32)
    ts = np.array(all_ts, np.int64)
    order = np.lexsort((ts, sid))                        # per-stream time order
    px, vec, sid, ts = px[order], vec[order], sid[order], ts[order]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, px=px, vec=vec, sid=sid, ts=ts,
             streams=np.array(streams))
    print(f"cache: {len(px):,} pairs, {px.nbytes / 1e9:.2f} GB pixels, "
          f"{time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
