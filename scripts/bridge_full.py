"""Build `lake/bridge-full`: ALL of BridgeData2 (~90 h, 152 files), fully
loaded and fully embedded — the scale test the pilot needs answered.

Same pipeline as bridge4h.py (source-pipe embedding parallel to transcode,
one commit per table) with the changes scale forces:
  - per-file frame_vector batches consumed AS EMBEDS COMPLETE and written
    via append_batches (one file per batch, ONE commit) — holding 1.6M
    vectors as Python lists was the 4h loader's hidden 50 GB assumption
  - separate transcode/embed pools so 152 queued transcodes cannot starve
    the embed track
  - per-file ingest_video as each transcode lands (needs only the frame
    count, which the embed result carries)
Truth labels go to eval/bridge_full_truth.parquet, OUTSIDE the store.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import mlx.core as mx                                        # noqa: E402

from elidedb import Store                                    # noqa: E402
from elidedb.embeddings import pool_windows                  # noqa: E402
from elidedb.fdnnvideo import fdnnv_dir, load_encoder                   # noqa: E402
from elidedb.fftools import find                             # noqa: E402

ROOT = Path("data/bridge")
CAM = "observation.images.image_0"
EPOCH_NS = 1_704_067_200_000_000_000
FILE_STRIDE_NS = 20_000_000_000_000
FPS = 5.0
OUT = Path("lake/bridge-full")

GPU = threading.Lock()


def all_files():
    return sorted(int(p.stem.split("-")[1]) for p in
                  (ROOT / f"videos/{CAM}/chunk-000").glob("file-*.mp4"))


def src_of(f):
    return ROOT / f"videos/{CAM}/chunk-000/file-{f:03d}.mp4"


def transcode(db, f):
    dest = db._media_dest(src_of(f)).with_suffix(".h264")
    if not dest.exists():
        subprocess.run(
            [find("ffmpeg"), "-v", "error", "-y", "-i", str(src_of(f)),
             "-c:v", "libx264", "-preset", "fast", "-crf", "26",
             "-g", "5", "-keyint_min", "5", "-an", "-f", "h264", str(dest)],
            check=True)
    return f


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
    v = np.concatenate(vecs) if vecs else np.zeros((0, 1152), np.float32)
    return f, ts, v


def fv_batch(f, ts, vecs):
    return pa.table({
        "ts": pa.array([int(t) for t in ts], pa.int64()),
        "stream": pa.array([f"{CAM}/file-{f:03d}"] * len(ts)),
        "vector": pa.array([v.tolist() for v in vecs],
                           pa.list_(pa.float32(), vecs.shape[1])),
    })


def tables_track(files):
    key = f"videos/{CAM}"
    meta = pq.read_table(
        ROOT / "meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "tasks", "length", f"{key}/file_index",
                 f"{key}/from_timestamp", f"{key}/to_timestamp"]).to_pydict()
    fi = np.array(meta[f"{key}/file_index"])
    keep = np.where(np.isin(fi, files))[0]

    ep = {"ts": [], "t1": [], "episode_index": [], "stream": [],
          "n_frames": [], "file_index": []}
    truth = {"episode_index": [], "ts": [], "t1": [], "task": []}
    for j in keep:
        f = int(fi[j])
        base = EPOCH_NS + f * FILE_STRIDE_NS
        a = base + int(round(meta[f"{key}/from_timestamp"][j] * 1e9))
        b = base + int(round(meta[f"{key}/to_timestamp"][j] * 1e9)) - 1
        ep["ts"].append(a)
        ep["t1"].append(b)
        ep["episode_index"].append(int(meta["episode_index"][j]))
        ep["stream"].append(f"{CAM}/file-{f:03d}")
        ep["n_frames"].append(int(meta["length"][j]))
        ep["file_index"].append(f)
        t = meta["tasks"][j] or [""]
        truth["episode_index"].append(int(meta["episode_index"][j]))
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
    out = Path("eval/bridge_full_truth.parquet")
    out.parent.mkdir(exist_ok=True)
    pq.write_table(pa.table({k: pa.array(v) for k, v in truth.items()}), out)
    return ep_tbl, len(truth["task"]), len(set(truth["task"]))


def main():
    t0 = time.time()
    stamp = {}
    files = all_files()
    print(f"{len(files)} source files", flush=True)
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    db = Store.create(OUT, "bridge-full")
    model, _ = load_encoder(fdnnv_dir())
    with GPU:
        e, _h = model(mx.array(np.zeros((1, 8, 144, 192, 3), np.float32)))
        mx.eval(e)
    stamp["setup"] = time.time() - t0

    counts = {}
    batches = []
    with ThreadPoolExecutor(max_workers=4) as ex_t, \
         ThreadPoolExecutor(max_workers=4) as ex_e, \
         ThreadPoolExecutor(max_workers=1) as ex_m:
        f_tab = ex_m.submit(tables_track, files)
        f_trans = {ex_t.submit(transcode, db, f): f for f in files}
        f_embed = [ex_e.submit(embed_source, model, f) for f in files]

        for fut in as_completed(f_embed):
            f, ts, vecs = fut.result()
            counts[f] = len(ts)
            batches.append(fv_batch(f, ts, vecs))
            del ts, vecs
            if len(counts) % 20 == 0:
                print(f"  embedded {len(counts)}/{len(files)} files "
                      f"(t={time.time() - t0:.0f}s)", flush=True)
        stamp["embed_done"] = time.time() - t0

        db.table("frame_vectors").append_batches(
            batches, kind="embeddings",
            meta={"model": "fdnnv", "every_frame": True, "dim": 1152,
                  "teacher": "mlx-community/siglip-so400m-patch14-384"})
        batches.clear()
        stamp["frame_vectors"] = time.time() - t0

        for fut in as_completed(f_trans):
            f = f_trans[fut]
            fut.result()
            base = EPOCH_NS + f * FILE_STRIDE_NS
            ts = [base + int(round(i * 1e9 / FPS))
                  for i in range(counts[f])]
            db.ingest_video("frames", src_of(f), timestamps_ns=ts,
                            stream=f"{CAM}/file-{f:03d}", transcode="h264",
                            gop_s=1.0, crf=26,
                            meta={"dataset": "bridgedata2-full",
                                  "file_index": f, "fps": FPS})
        stamp["transcode_index_done"] = time.time() - t0

        ep_tbl, n_eps, n_tasks = f_tab.result()
        db.table("episodes").append(ep_tbl, meta={
            "note": "episode boundaries only; task text is NOT in this store"})
        stamp["tables_done"] = time.time() - t0

    pool_windows(db, window_s=4.0, stride_s=2.0)
    stamp["windows"] = time.time() - t0

    total = time.time() - t0
    n_frames = int(sum(counts.values()))
    print(f"\nstore {OUT}: {n_frames:,} frames "
          f"({n_frames / FPS / 3600:.2f} h), {n_eps} episodes, "
          f"{n_tasks} distinct tasks (labels OUTSIDE the store)")
    last = 0.0
    for k, v in stamp.items():
        print(f"  {k:22s} +{v - last:7.1f}s  (t={v:7.1f}s)")
        last = v
    print(f"\nTOTAL {total:.1f}s "
          f"({n_frames / total:,.0f} frames/s, "
          f"{n_frames / FPS / total:.0f}x real time)")
    Path("bench") / "bench_bridge_full.json".write_text(json.dumps(
        {"frames": n_frames, "hours": round(n_frames / FPS / 3600, 2),
         "files": len(files), "total_s": round(total, 1),
         "stages": {k: round(v, 1) for k, v in stamp.items()}}, indent=2))


if __name__ == "__main__":
    main()
