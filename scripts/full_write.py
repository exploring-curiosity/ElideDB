"""FULL WRITE: raw video in, operable database out, one command, timed.

There is no "embed only" and no "index later". A write either produces
a database you can query or it does not, so this runs every stage that
standing between raw mp4 and a queryable store, and reports what each
one cost.

    segment    per-demo H.264 from RAW, one IDR each, plus the packet
               scan that becomes the frame index. This is what makes a
               2-second read a byte range instead of a file decode, and
               it is the only stage that touches the raw files.
    embed      FDNN-V over EVERY frame -> frame_vectors, retaining a
               fixed budget of frames per episode as they stream past
    geometry   agent / participants / transitions from those retained
               frames -> events
    identity   YOLO track + ReID over the segments -> objects, instances
    index      events -> labels inverted index + episode flags
    layout     declare each table's cluster key, so every future writer
               keeps the physical layout the read path depends on

The timeline is GAPPED: demos arrive stitched into 20,000-second files,
and a window query that spans the seam between two demos returns frames
from both. Each demo after the first is shifted so a uniform GAP_S
separates it from its predecessor, which makes "one demo" an actual
time range.

  python scripts/full_write.py --out lake/bench --force
  python scripts/full_write.py --out lake/bench --force --files 119,120
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx                                          # noqa: E402

from elidedb import Store                                      # noqa: E402
from elidedb.fdnnvideo import fdnnv_dir, load_encoder          # noqa: E402
from elidedb.fftools import find                               # noqa: E402
from elidedb.transitions import discover, profile                # noqa: E402
from elidedb.video import FrameSet, scan_video_packets         # noqa: E402
from elidedb.ingest import (CAM, EPOCH_NS, FILE_STRIDE_NS,     # noqa: E402
                            FPS, GAP_S, CRF, NGEOM, gapped,
                            frame_owner, probe_size)
# episode_events / episode_spans / stream_once are still corpus-specific
# extraction living in the sibling script; they move to the package once
# a second corpus needs them and their shape is known to generalise.
from write_once import (DATA, episode_events,                 # noqa: E402
                        episode_spans, stream_once)

SRC = DATA / f"videos/{CAM}/chunk-000"


def stage_segment(db, spans, shift, log_every=100):
    """RAW -> one H.264 segment per demo + the frame index rows.

    -ss BEFORE -i is a keyframe seek then forward decode, measured
    raw-faithful (first-frame MAE 1.57) and 6x cheaper than decoding
    from file start. -bf 0 keeps packet order equal to presentation
    order, which the frame index depends on; -g/-keyint_min large with
    -sc_threshold 0 gives exactly one IDR, so the segment is its own
    random-access unit.
    """
    media = db.dir / "media"
    media.mkdir(exist_ok=True)
    cols = defaultdict(list)
    t0 = time.time()
    for i, s in enumerate(spans):
        src = SRC / f"file-{s['file']:03d}.mp4"
        t_sec = (s["t0"] - EPOCH_NS - s["file"] * FILE_STRIDE_NS) / 1e9
        tmp = media / f"_tmp{i}.h264"
        r = subprocess.run(
            [find("ffmpeg"), "-v", "error", "-y",
             "-ss", f"{t_sec:.3f}", "-i", str(src),
             "-vf", f"fps={FPS}", "-frames:v", str(s["n"]), "-an",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
             "-bf", "0", "-g", "10000", "-keyint_min", "10000",
             "-sc_threshold", "0", "-f", "h264", str(tmp)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed, episode {i}: {r.stderr[-300:]}")
        h = hashlib.sha1(tmp.read_bytes()).hexdigest()[:8]
        seg = media / f"seg-{i:05d}-{h}.h264"
        tmp.rename(seg)
        pk = scan_video_packets(seg)
        n = len(pk["ts"])
        a = s["t0"] + shift[(s["stream"], s["t0"])]
        for j in range(n):
            cols["ts"].append(a + int(j * 1e9 / FPS))
            cols["byte_offset"].append(int(pk["byte_offset"][j].as_py()))
            cols["packet_size"].append(int(pk["packet_size"][j].as_py()))
            cols["keyframe"].append(bool(pk["keyframe"][j].as_py()))
            cols["width"].append(int(pk["width"][j].as_py()))
            cols["height"].append(int(pk["height"][j].as_py()))
            cols["codec"].append("h264")
            cols["source"].append(f"@media/{seg.name}")
            cols["stream"].append(s["stream"])
            cols["episode_index"].append(s["episode"])
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  segment {i + 1}/{len(spans)}  {el:.0f}s  "
                  f"ETA {el / (i + 1) * (len(spans) - i - 1) / 60:.1f} min",
                  flush=True)
    ft = pa.table({
        "ts": pa.array(cols["ts"], pa.int64()),
        "byte_offset": pa.array(cols["byte_offset"], pa.int64()),
        "packet_size": pa.array(cols["packet_size"], pa.int32()),
        "keyframe": pa.array(cols["keyframe"]),
        "width": pa.array(cols["width"], pa.int32()),
        "height": pa.array(cols["height"], pa.int32()),
        "codec": pa.array(cols["codec"]),
        "source": pa.array(cols["source"]),
        "stream": pa.array(cols["stream"]),
        "episode_index": pa.array(cols["episode_index"], pa.int64()),
    })
    ft = ft.take(pc.sort_indices(ft.column("ts")))
    # grouped by episode: the query unit is the clip, so that is the row
    # group boundary. Flat by ts put 39,026 rows in ONE group and a
    # single-episode read decompressed the whole index.
    db.table("frames").set_layout("episode_index",
                                  sort_by=["stream", "episode_index", "ts"],
                                  min_group_rows=4096)
    db.table("frames").append(ft, kind="frame_index",
                              meta={"render": f"h264-crf{CRF}-idr-per-demo",
                                    "gap_s": GAP_S})
    return len(ft)


def stage_embed_geometry(db, spans, shift, model):
    """One decode of each raw file: embed every frame, keep a budget."""
    by_file = defaultdict(list)
    for s in spans:
        by_file[s["file"]].append(s)
    all_ts, all_vec, all_stream, ev_rows = [], [], [], []
    t_stream = t_geom = 0.0
    for f, sp in by_file.items():
        base = EPOCH_NS + f * FILE_STRIDE_NS
        want = {}
        for s in sp:
            i0 = int(round((s["t0"] - base) / 1e9 * FPS))
            i1 = int(round((s["t1"] - base) / 1e9 * FPS))
            if i1 > i0:
                for gi in np.unique(np.linspace(i0, i1, NGEOM).round()
                                    .astype(int)):
                    want[int(gi)] = s["episode"]
        a = time.time()
        ts, vec, keep = stream_once(model, f, sp, want)
        t_stream += time.time() - a

        # the frame vectors carry the SHIFTED timeline, so every table
        # in the store shares one clock
        span_of = {s["episode"]: s for s in sp}
        idx = np.searchsorted(
            np.array([s["t0"] for s in sp]), ts, side="right") - 1
        sh = np.array([shift[(sp[max(k, 0)]["stream"], sp[max(k, 0)]["t0"])]
                       for k in idx], np.int64)
        all_ts.append(ts + sh)
        all_vec.append(vec)
        all_stream += [f"{CAM}/file-{f:03d}"] * len(ts)

        a = time.time()
        for epi, frames in keep.items():
            if len(frames) < 4:
                continue
            s = span_of[epi]
            d = shift[(s["stream"], s["t0"])]
            rows, _ = episode_events(s, frames, s["t0"] + d, s["t1"] + d)
            ev_rows += rows
        t_geom += time.time() - a

    ts = np.concatenate(all_ts)
    vec = np.concatenate(all_vec)
    order = np.argsort(ts, kind="stable")
    db.table("frame_vectors").append(pa.table({
        "ts": pa.array(ts[order], pa.int64()),
        "stream": pa.array([all_stream[i] for i in order]),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(
                vec[order].astype(np.float16)).reshape(-1), pa.float16()),
            vec.shape[1])}), kind="embeddings", meta={"model": "fdnnv"})

    db.table("episodes").append(pa.table({
        "ts": pa.array([s["t0"] + shift[(s["stream"], s["t0"])]
                        for s in spans], pa.int64()),
        "t1": pa.array([s["t1"] + shift[(s["stream"], s["t0"])]
                        for s in spans], pa.int64()),
        "episode_index": pa.array([s["episode"] for s in spans], pa.int64()),
        "stream": pa.array([s["stream"] for s in spans]),
        "n_frames": pa.array([s["n"] for s in spans], pa.int32()),
        "file_index": pa.array([s["file"] for s in spans], pa.int32()),
    }), kind="timeseries")

    # TYPE THE TRANSITIONS FROM THE CORPUS, not from a ladder. Rows
    # carrying a descriptor (index 8) are untyped participant
    # transitions; the corpus decides how many kinds exist and which is
    # which. Rows without one are onset/offset geometry (contact,
    # release) and the agent track, which are measurements rather than
    # categories and need no discovery.
    di = [i for i, r in enumerate(ev_rows) if len(r) > 8]
    tinfo = {"types": 0, "n": 0}
    if di:
        D = np.stack([ev_rows[i][8] for i in di])
        lab, cent, tinfo = discover(D)
        for i, k in zip(di, lab):
            # an integer id, never a name. -1 stays -1: a transition the
            # corpus cannot type is UNTYPED, not swept into a majority
            # bucket the way `adjust` used to be.
            ev_rows[i][3] = f"t{int(k)}" if k >= 0 else ""
        if len(cent):
            db.table("transition_types").append(pa.table({
                "ts": pa.array([spans[0]["t0"]] * len(cent), pa.int64()),
                "t1": pa.array([spans[-1]["t1"]] * len(cent), pa.int64()),
                "type_id": pa.array(list(range(len(cent))), pa.int32()),
                "n_members": pa.array(tinfo["sizes"], pa.int32()),
                "centroid": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.ascontiguousarray(cent).reshape(-1),
                             pa.float32()), cent.shape[1]),
            }), kind="index", meta={"discovered": True,
                                    "profile": json.dumps(
                                        profile(cent, tinfo["mu"],
                                                tinfo["sd"]))})
    E = pa.table({
        "ts": pa.array([r[1] for r in ev_rows], pa.int64()),
        "t1": pa.array([r[2] for r in ev_rows], pa.int64()),
        "stream": pa.array([r[0] for r in ev_rows]),
        "kind": pa.array([r[3] for r in ev_rows]),
        "role": pa.array(["" for _ in ev_rows]),
        "ev_t0": pa.array([r[4] for r in ev_rows], pa.int64()),
        "ev_t1": pa.array([r[5] for r in ev_rows], pa.int64()),
        "name": pa.array([r[7] for r in ev_rows]),
        "conf": pa.array([r[6] for r in ev_rows], pa.float32()),
    })
    E = E.filter(pc.not_equal(E.column("kind"), ""))
    db.table("events").set_layout("kind", sort_by=["kind", "role", "ts"],
                                  min_group_rows=512)
    db.table("events").append(E.take(pc.sort_indices(E.column("ts"))),
                              kind="events", meta={"builder": "full_write"})
    return len(ts), len(ev_rows), t_stream, t_geom


def stage_presence(db, batch=64, proposer="agnostic"):
    """PRESENCE INTERVALS over the continuous stream. Idle included.

    Replaces the per-episode identity stage, and fixes the largest
    violation in the audit (L0): the element path was entirely
    motion-triggered, so an object that sat still produced no track, no
    event and no row. "Every clip where a banana is on the table" was
    not badly answered, it was structurally unanswerable.

    Here every frame is proposed on and tracked, so a thing that never
    moves is still a thing that was THERE, and its presence is one
    interval row however long it lasted.

    Streamed in ts order across the whole stream rather than per
    episode: raw capture is continuous, and episodes are something the
    engine produces, not something it is given.
    """
    from elidedb.identity import (Gallery, Stream, calibrate, detect,
                                  features, propose)
    ft = db.table("frames").scan()
    ft = ft.take(pc.sort_indices(ft, sort_keys=[("stream", "ascending"),
                                                ("ts", "ascending")]))
    streams = sorted(set(ft.column("stream").to_pylist()))
    rows, cost = [], defaultdict(float)
    n_frames = 0
    for sname in streams:
        sub = ft.filter(pc.equal(ft.column("stream"), sname))
        tss = [int(v) for v in sub.column("ts").to_pylist()]
        st = Stream()
        for i in range(0, len(sub), batch):
            a = time.time()
            chunk = FrameSet(db, "frames", sub.slice(i, batch)).decode()
            cost["decode"] += time.time() - a
            if not chunk:
                continue
            chunk = sorted(chunk)
            ims = [c[1] for c in chunk]
            n_frames += len(ims)
            a = time.time()
            dets = (propose(ims) if proposer == "agnostic"
                    else [d[0] for d in detect(ims)])
            cost["propose"] += time.time() - a
            a = time.time()
            for (ts, im), b in zip(chunk, dets):
                b = np.asarray(b, np.int32).reshape(-1, 4)
                d = (b, np.ones(len(b), np.float32),
                     np.ones(len(b), np.float32))
                for tid, t in st.update(int(ts), d, im):
                    rows.append((sname, t))
            cost["track"] += time.time() - a
        for tid, t in st.flush():
            rows.append((sname, t))
    if not rows:
        return 0, 0, dict(cost)

    # one ReID call per CLOSED track, on the views retained while it was
    # open - the same "one question per sighting" that made identity
    # work, now without an episode to tell it when to ask
    a = time.time()
    V = []
    for _, t in rows:
        if not t["crops"]:
            V.append(np.zeros(512, np.float32)); continue
        f = np.stack([features(c[1], [[0, 0, c[1].shape[1],
                                       c[1].shape[0]]])[0]
                      for c in t["crops"]])
        v = f.mean(0)
        V.append(v / (np.linalg.norm(v) + 1e-8))
    V = np.stack(V)
    cost["reid"] = time.time() - a

    # free negatives: two presence intervals that OVERLAP IN TIME on the
    # same stream are two different objects - one thing cannot be in two
    # places. No episode needed to scope it.
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, min(i + 40, len(rows))):
            if rows[i][0] != rows[j][0]:
                continue
            a_, b_ = rows[i][1], rows[j][1]
            if a_["ts"] <= b_["t1"] and b_["ts"] <= a_["t1"]:
                pairs.append((i, j))
    fit, _ = calibrate(V, pairs)
    ids = Gallery(match=fit).assign(V)

    pres = pa.table({
        "ts": pa.array([t["ts"] for _, t in rows], pa.int64()),
        "t1": pa.array([t["t1"] for _, t in rows], pa.int64()),
        "stream": pa.array([s for s, _ in rows]),
        "object_id": pa.array([int(i) for i in ids], pa.int32()),
        "n_frames": pa.array([t["n"] for _, t in rows], pa.int32()),
        "conf": pa.array([float(max(t["conf"])) if t["conf"] else 0.0
                          for _, t in rows], pa.float32()),
        "box": pa.array([[int(v) for v in t["box"][0]] for _, t in rows],
                        pa.list_(pa.int32(), 4)),
    })
    pres = pres.take(pc.sort_indices(pres.column("ts")))
    db.table("presence").set_layout("object_id",
                                    sort_by=["object_id", "ts"],
                                    min_group_rows=256)
    db.table("presence").append(pres, kind="index",
                                meta={"unit": "presence_interval",
                                      "match_cut": round(float(fit), 3),
                                      "proposer": proposer})
    n_obj = len(set(int(i) for i in ids))
    cost["frames"] = n_frames
    return len(rows), n_obj, dict(cost)


def stage_one_pass(db, spans, shift, model, proposer="agnostic",
                   log_every=200):
    """ONE decode of the raw pixels. Everything else rides on it.

    The write decoded the same pixels THREE times:
      segment    ffmpeg -i raw.mp4 ... -c:v libx264   (decode + encode)
      embed      ffmpeg -i raw.mp4 ... rawvideo       (decode again)
      presence   FrameSet.decode over the store's own segments (third)

    Measured, that third decode was 41% of the presence stage, and the
    second was the whole embed stage. The encode is unavoidable - the
    store needs per-demo H.264 with one IDR so a 2 s read is a byte
    range - but it does not need to decode to do it: raw frames go IN on
    the encoder's stdin.

    So: one decoder per FILE at native resolution, and every consumer
    reads the frames as they stream past.
        FDNN-V      resized to its own raster
        geometry    a retained budget per episode
        proposer    native frames, for presence intervals
        encoder     the same frames, piped to libx264
    """
    from elidedb.identity import Gallery, Stream, calibrate, features, propose
    import cv2
    W, H = 192, 144                      # FDNN-V's raster
    media = db.dir / "media"
    media.mkdir(exist_ok=True)
    by_file = defaultdict(list)
    for sp in spans:
        by_file[sp["file"]].append(sp)

    cols = defaultdict(list)
    all_ts, all_vec, all_stream, ev_rows = [], [], [], []
    pres_rows = []          # (stream, meta, descriptor)
    cost = defaultdict(float)
    n_frames = 0

    for f, sp in sorted(by_file.items()):
        sp = sorted(sp, key=lambda r: r["t0"])
        src = SRC / f"file-{f:03d}.mp4"
        probe_r = subprocess.run(
            [find("ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(src)], capture_output=True, text=True)
        fw, fh = (int(x) for x in probe_r.stdout.strip().split(",")[:2])
        fb = fw * fh * 3
        base = EPOCH_NS + f * FILE_STRIDE_NS
        # frame index -> episode, computed once
        # An episode owns EXACTLY n frames, the count the corpus states.
        # Deriving the end from t1 by rounding claims one extra frame at
        # an inclusive bound - verified against the seek-based path,
        # which produced 30 frames for the last episode where this
        # produced 31, with every other frame identical. Trust the
        # declared length, not a rounded timestamp.
        owner = {}
        for e in sp:
            i0 = int(round((e["t0"] - base) / 1e9 * FPS))
            for i in range(i0, i0 + int(e["n"])):
                owner[i] = e["episode"]
        span_of = {e["episode"]: e for e in sp}
        geom_want = {}
        for e in sp:
            i0 = int(round((e["t0"] - base) / 1e9 * FPS))
            i1 = int(round((e["t1"] - base) / 1e9 * FPS))
            if i1 > i0:
                for gi in np.unique(np.linspace(i0, i1, NGEOM).round()
                                    .astype(int)):
                    geom_want[int(gi)] = e["episode"]

        dec = subprocess.Popen(
            [find("ffmpeg"), "-v", "error", "-i", str(src),
             "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "pipe:1"], stdout=subprocess.PIPE, bufsize=fb * 8)
        st_track = Stream()
        h = model.init_state(1)
        cur_ep, enc, seg_path, seg_n = None, None, None, 0
        keep = defaultdict(list)
        buf, idx = b"", 0
        emb_batch, prop_batch, prop_ts = [], [], []
        seg_index = [0]

        def close_seg():
            nonlocal enc, seg_path, seg_n, cur_ep
            if enc is None:
                return
            enc.stdin.close(); enc.wait()
            h_ = hashlib.sha1(seg_path.read_bytes()).hexdigest()[:8]
            final = media / f"seg-{seg_index[0]:05d}-{h_}.h264"
            seg_path.rename(final)
            seg_index[0] += 1
            pk = scan_video_packets(final)
            e = span_of[cur_ep]
            a0 = e["t0"] + shift[(e["stream"], e["t0"])]
            for j in range(len(pk["ts"])):
                cols["ts"].append(a0 + int(j * 1e9 / FPS))
                cols["byte_offset"].append(int(pk["byte_offset"][j].as_py()))
                cols["packet_size"].append(int(pk["packet_size"][j].as_py()))
                cols["keyframe"].append(bool(pk["keyframe"][j].as_py()))
                cols["width"].append(int(pk["width"][j].as_py()))
                cols["height"].append(int(pk["height"][j].as_py()))
                cols["codec"].append("h264")
                cols["source"].append(f"@media/{final.name}")
                cols["stream"].append(e["stream"])
                cols["episode_index"].append(cur_ep)
            enc, seg_path, seg_n = None, None, 0

        def flush_embed():
            nonlocal h, emb_batch
            if not emb_batch:
                return
            a = time.time()
            x = mx.array(np.stack([b[1] for b in emb_batch])
                         .astype(np.float32) / 127.5 - 1.0)[None]
            e_, h = model(x, h0=h)
            all_vec.append(np.array(e_[0], dtype=np.float32))
            all_ts.extend(b[0] for b in emb_batch)
            all_stream.extend(b[2] for b in emb_batch)
            cost["embed"] += time.time() - a
            emb_batch = []

        def flush_prop():
            nonlocal prop_batch, prop_ts
            if not prop_batch:
                return
            a = time.time()
            dets = propose(prop_batch)
            cost["propose"] += time.time() - a
            a = time.time()
            for ts_, im, b in zip(prop_ts, prop_batch, dets):
                b = np.asarray(b, np.int32).reshape(-1, 4)
                for tid, t in st_track.update(
                        int(ts_), (b, np.ones(len(b), np.float32),
                                   np.ones(len(b), np.float32)), im):
                    pres_rows.append(close_track(sp[0]["stream"], t))
            cost["track"] += time.time() - a
            prop_batch, prop_ts = [], []

        while True:
            chunk = dec.stdout.read(fb * 16 - len(buf))
            if chunk:
                buf += chunk
            n = len(buf) // fb
            if n == 0 and not chunk:
                break
            if n == 0:
                continue
            fr = np.frombuffer(buf[:n * fb], np.uint8).reshape(n, fh, fw, 3)
            buf = buf[n * fb:]
            for k in range(n):
                ep_id = owner.get(idx)
                if ep_id != cur_ep:
                    close_seg()
                    flush_prop()
                    cur_ep = ep_id
                    if cur_ep is not None:
                        seg_path = media / f"_tmp{seg_index[0]}.h264"
                        enc = subprocess.Popen(
                            [find("ffmpeg"), "-v", "error", "-y",
                             "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{fw}x{fh}", "-r", str(FPS),
                             "-i", "pipe:0", "-an", "-c:v", "libx264",
                             "-preset", "medium", "-crf", str(CRF),
                             "-bf", "0", "-g", "10000",
                             "-keyint_min", "10000", "-sc_threshold", "0",
                             "-f", "h264", str(seg_path)],
                            stdin=subprocess.PIPE)
                if cur_ep is not None:
                    e = span_of[cur_ep]
                    a0 = e["t0"] + shift[(e["stream"], e["t0"])]
                    ts_ = a0 + int(seg_n * 1e9 / FPS)
                    enc.stdin.write(fr[k].tobytes())
                    seg_n += 1
                    emb_batch.append((ts_, cv2.resize(
                        fr[k], (W, H), interpolation=cv2.INTER_AREA),
                        e["stream"]))
                    prop_batch.append(fr[k].copy()); prop_ts.append(ts_)
                    if idx in geom_want:
                        keep[cur_ep].append(cv2.resize(
                            fr[k], (256, 192), interpolation=cv2.INTER_AREA))
                    if len(emb_batch) >= 256:
                        flush_embed()
                    if len(prop_batch) >= 64:
                        flush_prop()
                    n_frames += 1
                idx += 1
                if log_every and n_frames and n_frames % log_every == 0:
                    print(f"  one-pass {n_frames} frames", flush=True)
        close_seg(); flush_embed(); flush_prop()
        dec.wait()
        for tid, t in st_track.flush():
            pres_rows.append(close_track(sp[0]["stream"], t))

        a = time.time()
        for epi, frames in keep.items():
            if len(frames) < 4:
                continue
            e = span_of[epi]
            d = shift[(e["stream"], e["t0"])]
            rows, _ = episode_events(e, frames, e["t0"] + d, e["t1"] + d)
            ev_rows += rows
        cost["geometry"] += time.time() - a

    cost["frames"] = n_frames
    return cols, all_ts, all_vec, all_stream, ev_rows, pres_rows, dict(cost)


def commit_frames(db, cols):
    ft = pa.table({
        "ts": pa.array(cols["ts"], pa.int64()),
        "byte_offset": pa.array(cols["byte_offset"], pa.int64()),
        "packet_size": pa.array(cols["packet_size"], pa.int32()),
        "keyframe": pa.array(cols["keyframe"]),
        "width": pa.array(cols["width"], pa.int32()),
        "height": pa.array(cols["height"], pa.int32()),
        "codec": pa.array(cols["codec"]),
        "source": pa.array(cols["source"]),
        "stream": pa.array(cols["stream"]),
        "episode_index": pa.array(cols["episode_index"], pa.int64()),
    })
    ft = ft.take(pc.sort_indices(ft.column("ts")))
    db.table("frames").set_layout("episode_index",
                                  sort_by=["stream", "episode_index", "ts"],
                                  min_group_rows=4096)
    db.table("frames").append(ft, kind="frame_index",
                              meta={"render": f"h264-crf{CRF}-idr-per-demo",
                                    "gap_s": GAP_S, "one_pass": True})
    return len(ft)


def commit_vectors(db, ats, avec, astr):
    """FDNN-V output through the TEACHER CONTRACT: coded, so the vector
    table prunes like every other table instead of being the one thing a
    planner cannot refuse."""
    from elidedb.teacher import fit_codebook, table as tt, write as tw
    if not ats:
        return 0
    V = np.concatenate(avec)
    ts = np.asarray(ats, np.int64)
    order = np.argsort(ts, kind="stable")
    V, ts = V[order], ts[order]
    stream = [astr[i] for i in order]
    C = fit_codebook(V)
    tbl, C = tt(ts, ts, stream, V, C)
    tw(db, "frame_vectors", tbl, C, meta={"model": "fdnnv"})
    return len(ts)


def commit_episodes(db, spans, shift):
    db.table("episodes").append(pa.table({
        "ts": pa.array([s["t0"] + shift[(s["stream"], s["t0"])]
                        for s in spans], pa.int64()),
        "t1": pa.array([s["t1"] + shift[(s["stream"], s["t0"])]
                        for s in spans], pa.int64()),
        "episode_index": pa.array([s["episode"] for s in spans], pa.int64()),
        "stream": pa.array([s["stream"] for s in spans]),
        "n_frames": pa.array([s["n"] for s in spans], pa.int32()),
        "file_index": pa.array([s["file"] for s in spans], pa.int32()),
    }), kind="timeseries")


def commit_events(db, ev_rows, spans):
    if not ev_rows:
        return 0
    di = [i for i, r in enumerate(ev_rows) if len(r) > 8]
    tinfo = {"types": 0}
    if di:
        D = np.stack([ev_rows[i][8] for i in di])
        lab, cent, tinfo = discover(D)
        for i, k in zip(di, lab):
            ev_rows[i][3] = f"t{int(k)}" if k >= 0 else ""
        if len(cent):
            db.table("transition_types").append(pa.table({
                "ts": pa.array([spans[0]["t0"]] * len(cent), pa.int64()),
                "t1": pa.array([spans[-1]["t1"]] * len(cent), pa.int64()),
                "type_id": pa.array(list(range(len(cent))), pa.int32()),
                "n_members": pa.array(tinfo["sizes"], pa.int32()),
                "centroid": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.ascontiguousarray(cent).reshape(-1),
                             pa.float32()), cent.shape[1]),
            }), kind="index", meta={"discovered": True,
                                    "profile": json.dumps(
                                        profile(cent, tinfo["mu"],
                                                tinfo["sd"]))})
    E = pa.table({
        "ts": pa.array([r[1] for r in ev_rows], pa.int64()),
        "t1": pa.array([r[2] for r in ev_rows], pa.int64()),
        "stream": pa.array([r[0] for r in ev_rows]),
        "kind": pa.array([r[3] for r in ev_rows]),
        "role": pa.array(["" for _ in ev_rows]),
        "ev_t0": pa.array([r[4] for r in ev_rows], pa.int64()),
        "ev_t1": pa.array([r[5] for r in ev_rows], pa.int64()),
        "name": pa.array([r[7] for r in ev_rows]),
        "conf": pa.array([r[6] for r in ev_rows], pa.float32()),
    })
    E = E.filter(pc.not_equal(E.column("kind"), ""))
    db.table("events").set_layout("kind", sort_by=["kind", "role", "ts"],
                                  min_group_rows=512)
    db.table("events").append(E.take(pc.sort_indices(E.column("ts"))),
                              kind="events", meta={"builder": "one_pass"})
    return len(E)


def close_track(stream, t):
    """A closed track reduced to what presence actually needs.

    THE MEMORY BUG THIS FIXES. `stage_one_pass` used to append the whole
    track - crops included - to `pres_rows`, and `commit_presence` turned
    crops into descriptors only at the very END of the run. So every
    object crop from every episode stayed resident for the whole write:
    O(corpus x tracks x pixels) held to produce O(tracks x 512 floats).

    At 600 episodes it fit. At 2,097 it took the machine to 0.2 GB free
    pages and 26 GB in the compressor, with 43 of 44 GB of swap gone, and
    the run had to be killed 53 minutes in. RSS read only 7 GB throughout,
    which is exactly why it was missed - RSS does not count what the
    compressor is holding.

    The crops exist only to make one 512-d vector. Making it here, at the
    moment the track closes, and dropping the pixels turns the peak into
    a few hundred bytes per track. Same descriptor, same number of ReID
    calls - they just happen as tracks close instead of all at the end.
    """
    from elidedb.identity import features
    if not t["crops"]:
        v = np.zeros(512, np.float32)
    else:
        f = np.stack([features(c[1], [[0, 0, c[1].shape[1],
                                       c[1].shape[0]]])[0]
                      for c in t["crops"]])
        m = f.mean(0)
        v = (m / (np.linalg.norm(m) + 1e-8)).astype(np.float32)
    # `conf` and `box` are lists over the track's life, but presence reads
    # only max(conf) and box[0]. Reduce them here too rather than carrying
    # every frame's copy.
    return (stream,
            {"ts": t["ts"], "t1": t["t1"], "n": t["n"],
             "conf": float(max(t["conf"])) if t["conf"] else 0.0,
             "box": [int(x) for x in t["box"][0]]},
            v)


def commit_presence(db, rows):
    """rows: (stream, meta, descriptor) from close_track."""
    from elidedb.identity import Gallery, calibrate
    if not rows:
        return 0, 0
    V = np.stack([v for _, _, v in rows])
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, min(i + 40, len(rows))):
            if rows[i][0] != rows[j][0]:
                continue
            a_, b_ = rows[i][1], rows[j][1]
            if a_["ts"] <= b_["t1"] and b_["ts"] <= a_["t1"]:
                pairs.append((i, j))
    fit, _ = calibrate(V, pairs)
    ids = Gallery(match=fit).assign(V)
    pres = pa.table({
        "ts": pa.array([t["ts"] for _, t, _ in rows], pa.int64()),
        "t1": pa.array([t["t1"] for _, t, _ in rows], pa.int64()),
        "stream": pa.array([s for s, _, _ in rows]),
        "object_id": pa.array([int(i) for i in ids], pa.int32()),
        "n_frames": pa.array([t["n"] for _, t, _ in rows], pa.int32()),
        "conf": pa.array([t["conf"] for _, t, _ in rows], pa.float32()),
        "box": pa.array([t["box"] for _, t, _ in rows],
                        pa.list_(pa.int32(), 4)),
    })
    pres = pres.take(pc.sort_indices(pres.column("ts")))
    db.table("presence").set_layout("object_id", sort_by=["object_id", "ts"],
                                    min_group_rows=256)
    db.table("presence").append(pres, kind="index",
                                meta={"unit": "presence_interval",
                                      "match_cut": round(float(fit), 3)})
    return len(rows), len(set(int(i) for i in ids))


def stage_index(db):
    """events -> the inverted index and the episode flags."""
    ev = db.table("events").scan().to_pydict()
    ep = db.table("episodes").scan()
    rows = defaultdict(set)
    for i in range(len(ev["ts"])):
        key = (str(ev["stream"][i]), int(ev["ts"][i]), int(ev["t1"][i]))
        k = ev["kind"][i]
        if k:
            rows[key].add(("action", k))
        nm = (ev["name"][i] or "").strip().lower()
        if nm:
            rows[key].add(("object", nm))
    L = defaultdict(list)
    for (s, a, b), pairs in rows.items():
        for kind, val in pairs:
            L["ts"].append(a); L["t1"].append(b); L["stream"].append(s)
            L["kind"].append(kind); L["value"].append(val); L["ep_ts"].append(a)
    db.table("labels").set_layout("value", sort_by=["kind", "value", "ts"],
                                  min_group_rows=256)
    db.table("labels").append(pa.table({
        "ts": pa.array(L["ts"], pa.int64()),
        "t1": pa.array(L["t1"], pa.int64()),
        "stream": pa.array(L["stream"]),
        "kind": pa.array(L["kind"]),
        "value": pa.array(L["value"]),
        "ep_ts": pa.array(L["ep_ts"], pa.int64()),
    }), kind="index", meta={"builder": "full_write"})

    have = defaultdict(set)
    for i in range(len(ev["ts"])):
        have[(str(ev["stream"][i]), int(ev["ts"][i]))].add(ev["kind"][i])
    d = ep.to_pydict()
    n = len(d["ts"])
    # The flag set is WHATEVER THE CORPUS PRODUCED - discovered type ids
    # plus the geometric onsets - not a literal tuple of eight verbs. A
    # driving corpus gets its own columns without a line changing here.
    kinds = sorted({k for v in have.values() for k in v if k})
    for k in kinds:
        ep = ep.append_column(f"has_{k}", pa.array(
            [k in have.get((str(d["stream"][i]), int(d["ts"][i])), ())
             for i in range(n)], pa.bool_()))
    ep = ep.append_column("n_labels", pa.array(
        [len(rows.get((str(d["stream"][i]), int(d["ts"][i]),
                       int(d["t1"][i])), ())) for i in range(n)], pa.int32()))
    db.table("episodes").replace(ep.take(pc.sort_indices(ep.column("ts"))),
                                 kind="timeseries", evolve=True,
                                 meta={"builder": "full_write"})
    return len(L["ts"])


# name -> (script, the table it must fill). The table is not decoration:
# it is how this stage decides whether the channel worked.
# `motion` runs FIRST and costs seconds: it has no model, being the
# normalised difference between mean appearance at the end and the start
# of each event span, so it rides on frame_vectors the write just made.
# It was absent from this list AND had no caller anywhere, so the channel
# the transition anchor is built on never existed in any store - silently,
# because the anchor guards the missing table and contributes nothing.
# Direction is the one thing text cannot express (open/close cosine
# 0.957), so a run without it is not a measurement of this system.
CHANNELS = [("motion", "motion_ingest.py", "motion_vectors"),
            ("pe", "pe_ingest.py", "pe_vectors"),
            ("sig2", "sig2_ingest.py", "sig2_vectors"),
            ("iv2", "iv2_ingest.py", "iv2_vectors"),
            ("xclip", "xclip_ingest.py", "xclip_vectors"),
            ("vjepa", "vjepa_channel.py", "vjepa_vectors"),
            ("act", "action_ingest.py", "action_probs")]


def _channel_rows(out, table):
    """Rows in a channel table right now, 0 if absent. Opened fresh so it
    sees what the child just committed, not what we had cached."""
    try:
        db = Store.open(str(out))
        return len(db.table(table).scan()) if table in db.tables() else 0
    except Exception:
        return 0


def stage_channels(out, only=None):
    """The six retrieval channels. Part of the write, not an extra.

    search_set scores these; a store without them answers metadata and
    object queries and nothing else. Each is a separate pretrained model
    over the whole corpus, and on this hardware `pe` alone measured 32x
    the entire FDNN-V ingest - which is the number that matters far more
    than the total, because it says the retrieval path costs an order of
    magnitude more than the write budget allows.

    Two rules this stage used to break, both of which cost an hour:

    THE CHILD OWNS THE TERMINAL. Every ingest carries a per-recording
    tqdm bar. Capturing its stdout hides that bar, and a hidden bar makes
    a slow channel indistinguishable from a hung one - which is the whole
    reason the bars were added. So no capture_output here.

    THE ARTIFACT IS THE TEST. `pe_ingest` once printed "DONE in 434s",
    exited 0, and wrote an empty table. An exit code reports that a
    process ended, not that it did the work. Count the rows.

    Pass --channels with a name that matches nothing (e.g. `none`) to
    skip the stage entirely and build channels separately with
    scripts/build_teachers.py, which is resumable.
    """
    T = {}
    for name, script, table in CHANNELS:
        if only is not None and name not in only:
            continue
        before = _channel_rows(out, table)
        a = time.time()
        proc = subprocess.run(
            [sys.executable, f"scripts/{script}", str(out)])
        secs = round(time.time() - a, 2)
        after = _channel_rows(out, table)
        if after > before:
            T[name] = secs
            print(f"  channel {name}: ok - {after:,} rows in {table} "
                  f"({secs}s)", flush=True)
        else:
            T[name] = (f"FAILED after {secs}s: {table} still has "
                       f"{after} rows (exit {proc.returncode})")
            print(f"  channel {name}: FAILED - wrote no rows to {table} "
                  f"({secs}s, exit {proc.returncode}); scroll up for its "
                  "own error output", flush=True)
    return T


def main():
    argv = sys.argv
    out = Path(argv[argv.index("--out") + 1] if "--out" in argv
               else "lake/bench")
    files = ([int(x) for x in argv[argv.index("--files") + 1].split(",")]
             if "--files" in argv else [119, 120, 129, 132])
    # start small, then build up: cap the episode count so every stage,
    # channels included, can be timed on a corpus that finishes.
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 0
    only = (set(argv[argv.index("--channels") + 1].split(","))
            if "--channels" in argv else None)
    if out.exists():
        if "--force" not in argv:
            raise SystemExit(f"{out} exists; pass --force")
        shutil.rmtree(out)

    T, N = {}, {}
    t00 = time.time()
    a = time.time()
    db = Store.create(out, out.name)
    model, _ = load_encoder(fdnnv_dir())
    e, _h = model(mx.array(np.zeros((1, 8, 144, 192, 3), np.float32)))
    mx.eval(e)
    spans = episode_spans(files)
    if limit:
        spans = spans[:limit]
    shift = gapped(spans)
    T["setup"] = time.time() - a

    # ONE PASS IS THE DEFAULT. --three-pass keeps the old staged path
    # for comparison; it is not equivalent and should not be trusted for
    # numbers. Verified against it: the frame index is IDENTICAL (4,161
    # = 4,161, every timestamp), while the staged path decoded 32% of
    # frames in presence and wrote 80% ORPHAN frame_vectors - 16,354
    # vectors for frames that are not in the store.
    one_pass = "--three-pass" not in argv
    if one_pass:
        # ONE decode of the raw pixels; every consumer rides on it.
        a = time.time()
        cols, ats, avec, astr, ev_rows, pres_rows, ocost = stage_one_pass(
            db, spans, shift, model,
            proposer=("agnostic" if "--coco" not in argv else "coco"))
        T["one_pass"] = time.time() - a
        T["one_pass_detail"] = {k: (round(v, 2) if isinstance(v, float)
                                    else v) for k, v in ocost.items()}
        N["frames"] = commit_frames(db, cols)
        N["frame_vectors"] = commit_vectors(db, ats, avec, astr)
        commit_episodes(db, spans, shift)
        N["events"] = commit_events(db, ev_rows, spans)
        N["presence"], N["objects"] = commit_presence(db, pres_rows)
        T["segment"] = T["embed"] = T["geometry"] = 0.0
    else:
        a = time.time()
        N["frames"] = stage_segment(db, spans, shift)
        T["segment"] = time.time() - a

        a = time.time()
        nf, nev, t_s, t_g = stage_embed_geometry(db, spans, shift, model)
        T["embed"] = t_s
        T["geometry"] = t_g
        T["embed+geometry"] = time.time() - a
        N["frame_vectors"], N["events"] = nf, nev

    a = time.time()
    if not one_pass:
        N["presence"], N["objects"], pcost = stage_presence(
            db, proposer=("agnostic" if "--coco" not in argv else "coco"))
        T["presence_detail"] = {k: (round(v, 2) if isinstance(v, float)
                                    else v) for k, v in pcost.items()}
    T["identity"] = time.time() - a

    a = time.time()
    N["labels"] = stage_index(db)
    T["index"] = time.time() - a

    a = time.time()
    T["channels"] = stage_channels(out, only)
    T["channels_total"] = time.time() - a

    T["total"] = time.time() - t00
    N["episodes"] = len(spans)
    hours = sum(s["n"] for s in spans) / FPS / 3600
    store_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    raw_bytes = sum((SRC / f"file-{f:03d}.mp4").stat().st_size for f in files)
    out_json = {
        "store": str(out), "hours_of_video": round(hours, 3), "rows": N,
        "seconds": {k: (round(v, 2) if isinstance(v, (int, float)) else v)
                    for k, v in T.items()},
        "min_per_hour_video": {
            k: round(v / 60 / max(hours, 1e-9), 3)
            for k, v in T.items()
            if k not in ("embed+geometry", "channels")
            and isinstance(v, (int, float))},
        "channels_min_per_hour": {
            k: round(v / 60 / max(hours, 1e-9), 3)
            for k, v in T.get("channels", {}).items()
            if isinstance(v, (int, float))},
        "s_per_episode": round(T["total"] / max(len(spans), 1), 3),
        "bytes": {"raw_source": raw_bytes, "store": store_bytes,
                  "store_over_raw": round(store_bytes / raw_bytes, 2)},
    }
    print(json.dumps(out_json, indent=1))
    (ROOT / "bench" / "bench_write.json").write_text(json.dumps(out_json, indent=1))


if __name__ == "__main__":
    main()
