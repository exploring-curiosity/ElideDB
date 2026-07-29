"""Separate the stitched demos: per-demo media + a timeline with gaps.

The raw Bridge files are concatenations - the 4 bench source files
hold 2,097 separate demo recordings back-to-back with 0.000s between
them - and the store inherited that lie: consecutive episodes touch
(p50 inter-episode gap 0.00s), so separate recordings read as one
continuous video to anything temporal (smoothing, NMS positions, PRF
neighborhoods, GOP decode reach).

This migration makes the store tell the truth, WITHOUT touching raw
(raw is immutable, project law) and WITHOUT encoding any metadata:

  media    one h264 elementary stream PER DEMO, re-rendered from the
           RAW source (single generation - re-encoding the rendition
           would be a second one), fresh SPS/PPS + one IDR at start
           (byte-slicing the old rendition is impossible anyway:
           episode starts there are mostly P-frames, measured).
           Names are seg-<ordinal>-<contenthash>.h264 - an ordinal
           and a hash carry no information about content.
  timeline every episode after the first in a stream is shifted so a
           uniform GAP_S separates it from its predecessor. Same
           order, same durations, no adjacency.
  tables   every table with a ts column is shifted by its containing
           episode's shift. Values are untouched, so every channel
           score is bit-identical - which is what makes the gate
           below meaningful.
  truthset keys are (stream, t0): a remapped copy is written next to
           the original (which is never modified), plus the mapping.

GATE (run separately): bench_truth on the migrated store with the
remapped truthset must reproduce the current ledger row EXACTLY - the
scores are unchanged, so any difference is a migration bug. Swap only
after the gate passes.

  python scripts/split_demos.py [--limit N] [--dst lake/bench_v2]
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.fftools import find                             # noqa: E402
from elidedb.video import scan_video_packets                 # noqa: E402

SRC_ROOT = ROOT / "data/bridge/videos/observation.images.image_0/chunk-000"
EPOCH_NS = 1_704_067_200_000_000_000
FILE_STRIDE_NS = 20_000_000_000_000
GAP_S = 60.0
FPS = 5.0
CRF = 26


def main():
    argv = sys.argv
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv \
        else None
    dst = Path(argv[argv.index("--dst") + 1]) if "--dst" in argv \
        else ROOT / "lake/bench_v2"

    src_store = Store.open("lake/bench")
    ep = src_store.table("episodes").scan().to_pydict()
    episodes = sorted(zip(ep["stream"],
                          (int(v) for v in ep["ts"]),
                          (int(v) for v in ep["t1"]),
                          (int(v) for v in ep["episode_index"]),
                          (int(v) for v in ep["n_frames"]),
                          (int(v) for v in ep["file_index"])),
                      key=lambda r: (r[0], r[1]))
    if limit:
        episodes = episodes[:limit]

    # ---- the shift map: uniform GAP between consecutive demos --------
    gap_ns = int(GAP_S * 1e9)
    shifts = {}                       # (stream, old_ts) -> shift_ns
    spans = {}                        # stream -> [(old_ts, old_t1)] sorted
    prev_end = {}
    for s, a, b, *_ in episodes:
        if s not in prev_end:
            new_a = a                 # first demo keeps its start
        else:
            new_a = prev_end[s] + gap_ns
        shifts[(s, a)] = new_a - a
        prev_end[s] = new_a + (b - a)
        spans.setdefault(s, []).append((a, b))

    def shift_of(stream, t):
        """Shift for any timestamp = its containing episode's shift.
        Every row in every table was verified to fall inside an
        episode span (0 strays), so a miss is a hard error."""
        lst = spans[stream]
        import bisect
        i = bisect.bisect_right([x[0] for x in lst], t) - 1
        a, b = lst[i]
        assert a <= t <= b, (stream, t)
        return shifts[(stream, a)]

    # ---- fresh store skeleton ---------------------------------------
    if dst.exists():
        shutil.rmtree(dst)
    dst_store = Store.create(str(dst), src_store.meta.get("name", "bench"))
    # store-level config travels unchanged (weights stay valid: every
    # channel score is computed from vectors whose VALUES don't move)
    for f in ("_store.json", "_vocab.json", "_channel_weights.json",
              "_set_weights.json", "_set_weights.prev.json",
              "_set_weights.loqo.json"):
        p = Path("lake/bench") / f
        if p.exists():
            shutil.copy(p, dst / f)

    (dst / "media").mkdir(exist_ok=True)

    # ---- per-demo media from RAW + new frames rows -------------------
    frames_cols = {k: [] for k in ("ts", "byte_offset", "packet_size",
                                   "keyframe", "width", "height",
                                   "codec", "source", "stream")}
    t0 = time.time()
    seg_map = []                       # ordinal bookkeeping only
    for i, (s, a, b, ei, nf, fi) in enumerate(episodes):
        src = SRC_ROOT / f"file-{fi:03d}.mp4"
        t_sec = (a - EPOCH_NS - fi * FILE_STRIDE_NS) / 1e9
        n = nf
        tmp = dst / "media" / f"_tmp{i}.h264"
        # -ss BEFORE -i: fast keyframe seek + forward decode, and
        # MEASURED raw-faithful (first frame MAE 1.57 against the raw
        # frame at f0). The frame-number select variant is equally
        # faithful but decodes from file start - 7.7s/episode vs 1.2.
        # (The old rendition itself sits one frame EARLIER than its own
        # timestamps at ~7 MAE from raw, so it is not the reference;
        # fidelity is judged against RAW only.)
        # Fresh encode from RAW: SPS/PPS + one IDR by construction;
        # -sc_threshold 0 so no surprise keyframes; -bf 0 keeps packet
        # order == presentation order (the frame index depends on it).
        cmd = [find("ffmpeg"), "-v", "error", "-y",
               "-ss", f"{t_sec:.3f}", "-i", str(src),
               "-vf", f"fps={FPS}", "-frames:v", str(n), "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
               "-bf", "0", "-g", "10000", "-keyint_min", "10000",
               "-sc_threshold", "0", "-f", "h264", str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on episode {i}: "
                               f"{r.stderr[-400:]}")
        h = hashlib.sha1(tmp.read_bytes()).hexdigest()[:8]
        seg = dst / "media" / f"seg-{i:05d}-{h}.h264"
        tmp.rename(seg)
        sh = shifts[(s, a)]
        new_a = a + sh
        pk = scan_video_packets(seg)
        npk = len(pk["ts"])
        if npk != n:
            raise RuntimeError(f"episode {i}: {npk} packets != {n} frames")
        for j in range(npk):
            frames_cols["ts"].append(new_a + int(j * 1e9 / FPS))
            frames_cols["byte_offset"].append(int(pk["byte_offset"][j].as_py()))
            frames_cols["packet_size"].append(int(pk["packet_size"][j].as_py()))
            frames_cols["keyframe"].append(bool(pk["keyframe"][j].as_py()))
            frames_cols["width"].append(int(pk["width"][j].as_py()))
            frames_cols["height"].append(int(pk["height"][j].as_py()))
            frames_cols["codec"].append("h264")
            frames_cols["source"].append(f"@media/{seg.name}")
            frames_cols["stream"].append(s)
        seg_map.append({"seg": seg.name, "stream": s,
                        "old_ts": a, "new_ts": new_a})
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(episodes)}  {el:.0f}s  "
                  f"ETA {el / (i + 1) * len(episodes) / 60:.0f}min",
                  flush=True)

    ftbl = pa.table({
        "ts": pa.array(frames_cols["ts"], pa.int64()),
        "byte_offset": pa.array(frames_cols["byte_offset"], pa.int64()),
        "packet_size": pa.array(frames_cols["packet_size"], pa.int32()),
        "keyframe": pa.array(frames_cols["keyframe"]),
        "width": pa.array(frames_cols["width"], pa.int32()),
        "height": pa.array(frames_cols["height"], pa.int32()),
        "codec": pa.array(frames_cols["codec"]),
        "source": pa.array(frames_cols["source"]),
        "stream": pa.array(frames_cols["stream"]),
    })
    ftbl = ftbl.take(pc.sort_indices(ftbl.column("ts")))
    dst_store.table("frames").append(
        ftbl, kind="frame_index",
        meta={"segmented": "per-demo", "gap_s": GAP_S,
              "render": f"h264-crf{CRF}-idr-per-demo"})
    print(f"frames: {ftbl.num_rows} rows over {len(episodes)} segments")

    # ---- every other table: shift ts (and t1) ------------------------
    done_eps = {(s, a) for s, a, *_ in episodes}
    for name in src_store.tables():
        if name == "frames":
            continue
        t = src_store.table(name).scan()
        cols = t.column_names
        d = t.to_pydict()
        keep = []
        for r in range(t.num_rows):
            s = str(d["stream"][r])
            old = int(d["ts"][r])
            lst = spans.get(s, [])
            import bisect
            ix = bisect.bisect_right([x[0] for x in lst], old) - 1
            ok = ix >= 0 and lst[ix][0] <= old <= lst[ix][1] \
                and (s, lst[ix][0]) in done_eps
            keep.append(ok)
            if not ok:
                continue
            sh = shifts[(s, lst[ix][0])]
            d["ts"][r] = old + sh
            if "t1" in cols:
                d["t1"][r] = int(d["t1"][r]) + sh
        if limit:
            d = {k: [v for v, kp in zip(d[k], keep) if kp] for k in cols}
        arrays = []
        for k in cols:
            arrays.append(pa.array(d[k], type=t.schema.field(k).type))
        nt = pa.table(dict(zip(cols, arrays)))
        nt = nt.take(pc.sort_indices(nt.column("ts")))
        st = src_store.table(name).state()
        dst_store.table(name).append(nt, kind=st.kind or "table",
                                     meta=st.meta or {})
        print(f"{name}: {nt.num_rows} rows shifted")

    # ---- truthset remap (original NEVER modified) --------------------
    tt = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet")
    td = tt.to_pydict()
    miss = 0
    for r in range(tt.num_rows):
        s, old = str(td["stream"][r]), int(td["t0"][r])
        if (s, old) in shifts:
            td["t0"][r] = old + shifts[(s, old)]
        else:
            miss += 1
    out = pa.table({k: pa.array(td[k], type=tt.schema.field(k).type)
                    for k in tt.column_names})
    pq.write_table(out, ROOT / "eval/truthsets/bridge4h_v2.parquet")
    (dst / "_split_map.json").write_text(json.dumps(
        {"gap_s": GAP_S, "segments": len(seg_map),
         "truthset_rows_unmapped": miss}))
    print(json.dumps({"segments": len(seg_map),
                      "truthset_unmapped": miss,
                      "seconds": round(time.time() - t0, 1),
                      "dst": str(dst)}))


if __name__ == "__main__":
    main()
