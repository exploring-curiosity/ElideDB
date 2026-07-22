"""Build `lake/bridge4h`: ~4 h of BridgeData2, fully loaded AND fully
embedded (every frame), with a hard wall-clock budget of 5 minutes.

WHY THIS IS FAST — the algorithmic changes, not tuning
------------------------------------------------------
1. Embedding does NOT wait for the store's read path. The write path used to
   be: transcode -> index -> byte-range decode of the transcoded copy ->
   embed. But embedding-at-load consumes every frame IN ORDER, and sequential
   consumption needs no random access: the SOURCE file pipes straight through
   ffmpeg into the encoder at 2,185 fps (measured), 4.5x faster than chunked
   byte-range reads, with ffmpeg doing the resize. Transcode (which only the
   CLIP-PLAYBACK path needs) runs in PARALLEL, not in front.
2. Every stage that can overlap does: 4 transcodes, 4 decode->embed streams
   (GPU serialised by a lock — FDNN-V needs 19 s of GPU for 70k frames, so
   the GPU is never the constraint), and the sensor/episode parquet work, all
   concurrent. The wall clock is max(tracks), not sum(stages).
3. One transaction per table at the end — the multi-gigabyte load is a
   single atomic append, per the store's own write discipline.

Task labels are NOT ingested (eval/bridge4h_truth.parquet, outside the
store, as always).
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import mlx.core as mx                                        # noqa: E402

from elidedb import Store                                    # noqa: E402
from elidedb.embeddings import pool_windows                  # noqa: E402
from elidedb.fdnnvideo import load_encoder                   # noqa: E402
from elidedb.fftools import find                             # noqa: E402

ROOT = Path("data/bridge")
CAM = "observation.images.image_0"
FILES = [119, 120, 129, 132]                 # 3.91 h, most task-diverse
EPOCH_NS = 1_704_067_200_000_000_000
FILE_STRIDE_NS = 20_000_000_000_000
FPS = 5.0
BUDGET_S = 300.0

GPU = threading.Lock()                       # one GPU; serialise MLX calls


def src_of(f):
    return ROOT / f"videos/{CAM}/chunk-000/file-{f:03d}.mp4"


# ---------------------------------------------------------------------------
# Track A: transcode (for the byte-range READ path; embedding never waits)
# ---------------------------------------------------------------------------
def transcode(db, f):
    src = src_of(f)
    dest = db._media_dest(src).with_suffix(".h264")
    if not dest.exists():
        subprocess.run(
            [find("ffmpeg"), "-v", "error", "-y", "-i", str(src),
             "-c:v", "libx264", "-preset", "fast", "-crf", "26",
             "-g", "5", "-keyint_min", "5", "-an", "-f", "h264", str(dest)],
            check=True)
    return f


# ---------------------------------------------------------------------------
# Track B: source -> ffmpeg pipe -> FDNN-V, streaming, state carried
# ---------------------------------------------------------------------------
def embed_source(model, f, width=192, height=144, chunk=256):
    fb = width * height * 3
    proc = subprocess.Popen(
        [find("ffmpeg"), "-v", "error", "-i", str(src_of(f)),
         "-vf", f"scale={width}:{height}", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE)
    base = EPOCH_NS + f * FILE_STRIDE_NS
    h = model.init_state(1)
    n_done, vecs = 0, []
    buf = b""
    while True:
        b = proc.stdout.read(fb * chunk - len(buf))
        if b:
            buf += b
        n = len(buf) // fb
        if n == 0 and not b:
            break
        if n == 0:
            continue
        frames = np.frombuffer(buf[:n * fb], np.uint8).reshape(n, height,
                                                               width, 3)
        buf = buf[n * fb:]
        with GPU:
            x = mx.array(frames.astype(np.float32) / 127.5 - 1.0)[None]
            e, h = model(x, h0=h)
            vecs.append(np.array(e[0], dtype=np.float32))
        n_done += n
        if not b and not buf:
            break
    proc.wait()
    ts = base + np.round(np.arange(n_done) * 1e9 / FPS).astype(np.int64)
    return f, ts, np.concatenate(vecs) if vecs else np.zeros((0, 1152))


# ---------------------------------------------------------------------------
# Track C: episodes + robot state (CPU parquet work) + truth (outside store)
# ---------------------------------------------------------------------------
def tables_track():
    key = f"videos/{CAM}"
    meta = pq.read_table(
        ROOT / "meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "tasks", "length", f"{key}/file_index",
                 f"{key}/from_timestamp", f"{key}/to_timestamp"]).to_pydict()
    fi = np.array(meta[f"{key}/file_index"])
    keep = np.where(np.isin(fi, FILES))[0]

    ep = {"ts": [], "t1": [], "episode_index": [], "stream": [],
          "n_frames": [], "file_index": []}
    truth = {"episode_index": [], "ts": [], "t1": [], "task": []}
    chosen_eps = set()
    for j in keep:
        f = int(fi[j])
        base = EPOCH_NS + f * FILE_STRIDE_NS
        a = base + int(round(meta[f"{key}/from_timestamp"][j] * 1e9))
        b = base + int(round(meta[f"{key}/to_timestamp"][j] * 1e9)) - 1
        e = int(meta["episode_index"][j])
        chosen_eps.add(e)
        ep["ts"].append(a)
        ep["t1"].append(b)
        ep["episode_index"].append(e)
        ep["stream"].append(f"{CAM}/file-{f:03d}")
        ep["n_frames"].append(int(meta["length"][j]))
        ep["file_index"].append(f)
        t = meta["tasks"][j] or [""]
        truth["episode_index"].append(e)
        truth["ts"].append(a)
        truth["t1"].append(b)
        truth["task"].append(t[0])

    order = np.argsort(ep["ts"])
    ep_tbl = pa.table({
        "ts": pa.array([ep["ts"][i] for i in order], pa.int64()),
        "t1": pa.array([ep["t1"][i] for i in order], pa.int64()),
        "episode_index": pa.array([ep["episode_index"][i] for i in order],
                                  pa.int64()),
        "stream": pa.array([ep["stream"][i] for i in order]),
        "n_frames": pa.array([ep["n_frames"][i] for i in order], pa.int32()),
        "file_index": pa.array([ep["file_index"][i] for i in order],
                               pa.int32()),
    })

    ep_to = {int(meta["episode_index"][j]):
             (int(fi[j]), float(meta[f"{key}/from_timestamp"][j]))
             for j in keep}
    parts = []
    for pf in sorted((ROOT / "data/chunk-000").glob("file-*.parquet")):
        t = pq.read_table(pf, columns=["observation.state", "action",
                                       "timestamp", "episode_index"])
        m = pc.is_in(t.column("episode_index"),
                     value_set=pa.array(sorted(chosen_eps)))
        parts.append(t.filter(m))
    fr = pa.concat_tables(parts).to_pydict()
    ts_c, st_c, ac_c = [], [], []
    for s, a, tsec, ei in zip(fr["observation.state"], fr["action"],
                              fr["timestamp"], fr["episode_index"]):
        f, from_s = ep_to[int(ei)]
        base = EPOCH_NS + f * FILE_STRIDE_NS
        ts_c.append(base + int(round((from_s + float(tsec)) * 1e9)))
        st_c.append(s)
        ac_c.append(a)
    sd, ad = len(st_c[0]), len(ac_c[0])
    robot = pa.table({
        "ts": pa.array(ts_c, pa.int64()),
        **{f"state_{i}": pa.array([r[i] for r in st_c], pa.float32())
           for i in range(sd)},
        **{f"action_{i}": pa.array([r[i] for r in ac_c], pa.float32())
           for i in range(ad)},
    })
    robot = robot.take(pc.sort_indices(robot.column("ts")))

    out = Path("eval/bridge4h_truth.parquet")
    out.parent.mkdir(exist_ok=True)
    pq.write_table(pa.table({k: pa.array(v) for k, v in truth.items()}), out)
    return ep_tbl, robot, len(truth["task"]), len(set(truth["task"]))


def main():
    t0 = time.time()
    stamp = {}
    path = Path("lake/bridge4h")
    if path.exists():
        import shutil
        shutil.rmtree(path)
    db = Store.create(path, "bridge-4h")
    model, _ = load_encoder("lake/bridge/models/fdnnv")
    # warm the GPU graph once so the first chunk is not paying compilation
    with GPU:
        e, _h = model(mx.array(np.zeros((1, 8, 144, 192, 3), np.float32)))
        mx.eval(e)
    stamp["setup"] = time.time() - t0

    with ThreadPoolExecutor(max_workers=10) as ex:
        f_trans = [ex.submit(transcode, db, f) for f in FILES]
        f_embed = [ex.submit(embed_source, model, f) for f in FILES]
        f_tab = ex.submit(tables_track)

        emb = {}
        for fut in f_embed:
            f, ts, vecs = fut.result()
            emb[f] = (ts, vecs)
        stamp["embed_done"] = time.time() - t0

        for fut in f_trans:
            fut.result()
        stamp["transcode_done"] = time.time() - t0

        # index the transcoded copies (dest exists -> ffprobe scan + append)
        for f in FILES:
            base = EPOCH_NS + f * FILE_STRIDE_NS
            n = len(emb[f][0])
            ts = [base + int(round(i * 1e9 / FPS)) for i in range(n)]
            db.ingest_video("frames", src_of(f), timestamps_ns=ts,
                            stream=f"{CAM}/file-{f:03d}", transcode="h264",
                            gop_s=1.0, crf=26,
                            meta={"dataset": "bridgedata2-4h",
                                  "file_index": f, "fps": FPS})
        stamp["index_done"] = time.time() - t0

        ep_tbl, robot, n_eps, n_tasks = f_tab.result()
        db.table("episodes").append(ep_tbl, meta={
            "note": "episode boundaries only; task text is NOT in this store"})
        db.table("robot").append(robot, meta={"dataset": "bridgedata2-4h"})
        stamp["tables_done"] = time.time() - t0

    rows_ts, rows_stream, rows_vec = [], [], []
    for f in FILES:
        ts, vecs = emb[f]
        rows_ts.extend(int(t) for t in ts)
        rows_stream.extend([f"{CAM}/file-{f:03d}"] * len(ts))
        rows_vec.append(vecs)
    allv = np.concatenate(rows_vec)
    fvt = pa.table({
        "ts": pa.array(rows_ts, pa.int64()),
        "stream": pa.array(rows_stream),
        "vector": pa.array([v.tolist() for v in allv],
                           pa.list_(pa.float32(), allv.shape[1])),
    })
    db.table("frame_vectors").append(fvt, kind="embeddings", meta={
        "model": "fdnnv", "every_frame": True, "dim": int(allv.shape[1]),
        "teacher": "mlx-community/siglip-so400m-patch14-384"})
    stamp["frame_vectors"] = time.time() - t0

    pool_windows(db, window_s=4.0, stride_s=2.0)
    stamp["windows"] = time.time() - t0

    total = time.time() - t0
    n_frames = len(fvt)
    print(f"\nstore lake/bridge4h: {n_frames:,} frames "
          f"({n_frames / FPS / 3600:.2f} h), {n_eps} episodes, "
          f"{n_tasks} distinct tasks (labels OUTSIDE the store)")
    last = 0.0
    for k, v in stamp.items():
        print(f"  {k:16s} +{v - last:6.1f}s  (t={v:6.1f}s)")
        last = v
    verdict = "PASS" if total < BUDGET_S else "FAIL"
    print(f"\nTOTAL {total:.1f}s of {BUDGET_S:.0f}s budget -> {verdict} "
          f"({n_frames / total:,.0f} frames/s, "
          f"{n_frames / FPS / total:.0f}x real time)")
    Path("bench_bridge4h.json").write_text(json.dumps(
        {"frames": n_frames, "hours": round(n_frames / FPS / 3600, 2),
         "total_s": round(total, 1), "budget_s": BUDGET_S,
         "verdict": verdict, "stages": {k: round(v, 1)
                                        for k, v in stamp.items()}}, indent=2))


if __name__ == "__main__":
    main()
