"""Ingest NVIDIA BridgeData2 (LeRobot v3) into an ElideDB store.

WHAT GOES IN THE DATABASE
-------------------------
Pixels, robot state, robot action, and time. That is all.

WHAT DOES NOT
-------------
The task strings (`meta/tasks.parquet`, and the `tasks` column of the episode
metadata). Those describe what each clip *is about*, which is precisely the
thing the database is supposed to work out from the pixels. Ingesting them
would make every later retrieval number meaningless — you would be measuring a
join against a label column, not contextual retrieval.

They are written to `eval/bridge_truth.parquet`, OUTSIDE the store directory,
and nothing on the query path can read it. It exists only so
`scripts/bench_bridge.py` can grade answers against ground truth.

TIME
----
Bridge has no wall-clock timestamps: each episode's `timestamp` restarts at 0.
But the videos pack ~321 episodes back to back per file, and the episode
metadata gives each episode's offset *within its file*. So the file's own
timeline is real, and the only synthetic part is where each file starts. Files
are laid out end to end from a fixed epoch with a gap between them, which
makes `ts` globally monotonic while staying exactly aligned to the media.

CODEC
-----
The source is AV1 in MP4. Byte-range decode needs either an intra codec
(every packet standalone) or an elementary stream that can be cut at a GOP
boundary and piped to a decoder — an MP4 byte range is neither, because the
container's index lives in a separate box. So the managed copy is transcoded
to an H.264 elementary stream with a forced keyframe every `gop_s` seconds.
That is the same random-access-vs-compression dial the format already exposes,
and here it is the thing that makes the corpus queryable at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                    # noqa: E402

ROOT = Path("data/bridge")
CAM = "observation.images.image_0"
# A fixed synthetic epoch. Files are laid end to end at this stride, which is
# comfortably longer than any single file (2456 s observed), so timelines never
# collide and each file keeps a visible gap from the next.
EPOCH_NS = 1_704_067_200_000_000_000        # 2024-01-01T00:00:00Z
# Each packed file gets a slot on the synthetic timeline. The slot MUST be
# longer than the longest file or slots overlap and two unrelated clips share
# timestamps — file 120 is 5042 s, which a 3000 s slot silently aliased into
# file 119's range. Streams still disambiguate inside the database, but any
# time-only reasoning (a window query, an evaluation join) would be wrong.
FILE_STRIDE_NS = 20_000_000_000_000         # 20000 s per file slot


def episode_meta():
    key = f"videos/{CAM}"
    cols = ["episode_index", "tasks", "length", f"{key}/file_index",
            f"{key}/from_timestamp", f"{key}/to_timestamp",
            "dataset_from_index", "dataset_to_index"]
    t = pq.read_table(ROOT / "meta/episodes/chunk-000/file-000.parquet",
                      columns=cols)
    return t.to_pydict(), key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=1,
                    help="ingest packed video files 0..N-1")
    ap.add_argument("--file-list", default=None,
                    help="comma-separated file indices instead of --files. "
                         "Files differ enormously in task diversity (file 0 "
                         "has 7 distinct tasks, file 119 has 482), so the "
                         "choice sets what the benchmark can measure.")
    ap.add_argument("--store", default="lake/bridge")
    ap.add_argument("--gop-s", type=float, default=1.0)
    ap.add_argument("--crf", type=int, default=26)
    args = ap.parse_args()

    d, key = episode_meta()
    fidx = np.array(d[f"{key}/file_index"])
    want = ([int(x) for x in args.file_list.split(",")] if args.file_list
            else list(range(args.files)))
    keep = np.where(np.isin(fidx, want))[0]
    tasks_here = {(d["tasks"][j] or [""])[0] for j in keep}
    print(f"{len(keep)} episodes across files {want} "
          f"({len(tasks_here)} distinct tasks)")

    path = Path(args.store)
    db = Store.create(path, "bridgedata2-v3") if not path.exists() \
        else Store.open(path)

    # ---- 1. video: transcode to an elementary stream, then packet-index ----
    for f in want:
        src = ROOT / f"videos/{CAM}/chunk-000/file-{f:03d}.mp4"
        if not src.exists():
            print(f"  missing {src}, stopping")
            break
        # 5 fps, uniform: frame i sits at exactly i/5 s inside the file.
        info = json.loads((ROOT / "meta/info.json").read_text())
        fps = info["fps"]
        import subprocess
        from elidedb.fftools import find
        nb = int(subprocess.run(
            [find("ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(src)],
            capture_output=True, text=True, check=True).stdout.strip())
        base = EPOCH_NS + f * FILE_STRIDE_NS
        ts = [base + int(round(i * 1e9 / fps)) for i in range(nb)]
        print(f"  file-{f:03d}: {nb} frames -> transcode + index", flush=True)
        db.ingest_video("frames", src, timestamps_ns=ts,
                        stream=f"{CAM}/file-{f:03d}", transcode="h264",
                        gop_s=args.gop_s, crf=args.crf,
                        meta={"dataset": "bridgedata2", "file_index": f,
                              "fps": fps})

    # ---- 2. episodes: boundaries only, NO task text -----------------------
    ep_rows = {"ts": [], "t1": [], "episode_index": [], "stream": [],
               "n_frames": [], "file_index": []}
    truth = {"episode_index": [], "ts": [], "t1": [], "task": []}
    for j in keep:
        f = int(fidx[j])
        base = EPOCH_NS + f * FILE_STRIDE_NS
        a = base + int(round(d[f"{key}/from_timestamp"][j] * 1e9))
        b = base + int(round(d[f"{key}/to_timestamp"][j] * 1e9)) - 1
        ep_rows["ts"].append(a)
        ep_rows["t1"].append(b)
        ep_rows["episode_index"].append(int(d["episode_index"][j]))
        ep_rows["stream"].append(f"{CAM}/file-{f:03d}")
        ep_rows["n_frames"].append(int(d["length"][j]))
        ep_rows["file_index"].append(f)
        tasks = d["tasks"][j] or []
        truth["episode_index"].append(int(d["episode_index"][j]))
        truth["ts"].append(a)
        truth["t1"].append(b)
        truth["task"].append(tasks[0] if tasks else "")
    order = np.argsort(ep_rows["ts"])
    db.table("episodes").append(pa.table({
        "ts": pa.array([ep_rows["ts"][i] for i in order], pa.int64()),
        "t1": pa.array([ep_rows["t1"][i] for i in order], pa.int64()),
        "episode_index": pa.array([ep_rows["episode_index"][i] for i in order],
                                  pa.int64()),
        "stream": pa.array([ep_rows["stream"][i] for i in order]),
        "n_frames": pa.array([ep_rows["n_frames"][i] for i in order], pa.int32()),
        "file_index": pa.array([ep_rows["file_index"][i] for i in order],
                               pa.int32()),
    }), meta={"note": "episode boundaries only; task text is NOT in this store"})
    print(f"  episodes table: {len(order)} rows (no task text)")

    # ---- 3. robot state + action as an ordinary timeseries ----------------
    lo, hi = int(min(d["dataset_from_index"][j] for j in keep)), \
        int(max(d["dataset_to_index"][j] for j in keep))
    frames = pq.read_table(ROOT / "data/chunk-000/file-000.parquet",
                           columns=["observation.state", "action", "timestamp",
                                    "episode_index", "frame_index"])
    fr = frames.slice(lo, hi - lo).to_pydict()
    ep_to_file = {int(d["episode_index"][j]): int(fidx[j]) for j in keep}
    ep_to_from = {int(d["episode_index"][j]):
                  float(d[f"{key}/from_timestamp"][j]) for j in keep}
    ts_col, st_col, ac_col = [], [], []
    for s, a, tsec, ei in zip(fr["observation.state"], fr["action"],
                              fr["timestamp"], fr["episode_index"]):
        ei = int(ei)
        if ei not in ep_to_file:
            continue
        base = EPOCH_NS + ep_to_file[ei] * FILE_STRIDE_NS
        ts_col.append(base + int(round((ep_to_from[ei] + float(tsec)) * 1e9)))
        st_col.append(s)
        ac_col.append(a)
    if ts_col:
        sd = len(st_col[0])
        ad = len(ac_col[0])
        tbl = pa.table({
            "ts": pa.array(ts_col, pa.int64()),
            **{f"state_{i}": pa.array([r[i] for r in st_col], pa.float32())
               for i in range(sd)},
            **{f"action_{i}": pa.array([r[i] for r in ac_col], pa.float32())
               for i in range(ad)},
        })
        tbl = tbl.take(pa.compute.sort_indices(tbl.column("ts")))
        db.table("robot").append(tbl, meta={"dataset": "bridgedata2"})
        print(f"  robot table: {len(tbl)} rows, {sd} state + {ad} action dims")

    # ---- 4. ground truth, OUTSIDE the store -------------------------------
    out = Path("eval/bridge_truth.parquet")
    out.parent.mkdir(exist_ok=True)
    new = pa.table({
        "episode_index": pa.array(truth["episode_index"], pa.int64()),
        "ts": pa.array(truth["ts"], pa.int64()),
        "t1": pa.array(truth["t1"], pa.int64()),
        "task": pa.array(truth["task"]),
    })
    if out.exists():                       # accumulate across ingest runs
        old = pq.read_table(out)
        keepmask = ~np.isin(np.array(old.column("episode_index")),
                            np.array(new.column("episode_index")))
        new = pa.concat_tables([old.filter(pa.array(keepmask)), new])
        new = new.take(pa.compute.sort_indices(new.column("ts")))
    pq.write_table(new, out)
    truth = new.to_pydict()
    print(f"  ground truth -> {out} ({len(truth['task'])} episodes, "
          f"{len(set(truth['task']))} distinct tasks) — NOT in the store")

    print("\ntables:", db.tables())


if __name__ == "__main__":
    main()
