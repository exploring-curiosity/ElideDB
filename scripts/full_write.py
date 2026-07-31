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
from elidedb.plan import FLAG_KINDS                            # noqa: E402
from elidedb.video import scan_video_packets                   # noqa: E402
from write_once import (CAM, DATA, EPOCH_NS, FILE_STRIDE_NS,   # noqa: E402
                        FPS, NGEOM, episode_events,
                        episode_spans, stream_once)

SRC = DATA / f"videos/{CAM}/chunk-000"
GAP_S = 60.0
CRF = 26


def gapped(spans):
    """Uniform gap between consecutive demos of a stream.

    Without it two adjacent demos are contiguous in time and a window
    query cannot express "this demo and not the next one".
    """
    gap = int(GAP_S * 1e9)
    prev, shift = {}, {}
    for s in sorted(spans, key=lambda r: (r["stream"], r["t0"])):
        st = s["stream"]
        new = s["t0"] if st not in prev else prev[st] + gap
        shift[(st, s["t0"])] = new - s["t0"]
        prev[st] = new + (s["t1"] - s["t0"])
    return shift


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
    db.table("events").set_layout("kind", sort_by=["kind", "role", "ts"],
                                  min_group_rows=512)
    db.table("events").append(E.take(pc.sort_indices(E.column("ts"))),
                              kind="events", meta={"builder": "full_write"})
    return len(ts), len(ev_rows), t_stream, t_geom


def stage_identity(db, n_ep):
    """Tracks -> one id per physical object, and where each was seen."""
    from build_identity import collect
    from elidedb.identity import Gallery, calibrate
    rows, pairs, _ = collect(db, n_ep, False, run=True)
    if not rows:
        return 0, 0
    V = np.stack([r["vec"] for r in rows])
    fit, _ = calibrate(V, pairs)
    ids = Gallery(match=fit).assign(V)
    for r, o in zip(rows, ids):
        r["object_id"] = int(o)
    inst = pa.table({
        "ts": pa.array([r["ts"] for r in rows], pa.int64()),
        "t1": pa.array([r["t1"] for r in rows], pa.int64()),
        "stream": pa.array([r["stream"] for r in rows]),
        "ep_ts": pa.array([r["ep_ts"] for r in rows], pa.int64()),
        "object_id": pa.array([r["object_id"] for r in rows], pa.int32()),
        "n_frames": pa.array([r["n_frames"] for r in rows], pa.int32()),
        "conf": pa.array([r["conf"] for r in rows], pa.float32()),
        "box": pa.array([r["box"] for r in rows], pa.list_(pa.int32(), 4)),
    })
    db.table("instances").set_layout("object_id",
                                     sort_by=["object_id", "ts"],
                                     min_group_rows=256)
    db.table("instances").append(inst.take(pc.sort_indices(inst.column("ts"))),
                                 kind="index", meta={"unit": "track"})
    by, cnt = defaultdict(set), defaultdict(int)
    for r in rows:
        by[r["object_id"]].add(r["ep_ts"]); cnt[r["object_id"]] += 1
    oids = sorted(cnt)
    db.table("objects").append(pa.table({
        "ts": pa.array([min(by[o]) for o in oids], pa.int64()),
        "t1": pa.array([max(by[o]) for o in oids], pa.int64()),
        "object_id": pa.array(oids, pa.int32()),
        "n_instances": pa.array([cnt[o] for o in oids], pa.int32()),
        "n_episodes": pa.array([len(by[o]) for o in oids], pa.int32()),
    }), kind="index", meta={"match_cut": round(float(fit), 3)})
    return len(rows), len(oids)


def stage_index(db):
    """events -> the inverted index and the episode flags."""
    ev = db.table("events").scan().to_pydict()
    ep = db.table("episodes").scan()
    rows = defaultdict(set)
    for i in range(len(ev["ts"])):
        key = (str(ev["stream"][i]), int(ev["ts"][i]), int(ev["t1"][i]))
        k = ev["kind"][i]
        if k in FLAG_KINDS or k == "agent":
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
    for k in FLAG_KINDS:
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


CHANNELS = [("pe", "pe_ingest.py"), ("sig2", "sig2_ingest.py"),
            ("iv2", "iv2_ingest.py"), ("xclip", "xclip_ingest.py"),
            ("vjepa", "vjepa_channel.py"), ("act", "action_ingest.py")]


def stage_channels(out, only=None):
    """The six retrieval channels. Part of the write, not an extra.

    search_set scores these; a store without them answers metadata and
    object queries and nothing else. Each is a separate pretrained model
    over the whole corpus, and on this hardware `pe` alone measured 32x
    the entire FDNN-V ingest - which is the number that matters far more
    than the total, because it says the retrieval path costs an order of
    magnitude more than the write budget allows.
    """
    T = {}
    for name, script in CHANNELS:
        if only and name not in only:
            continue
        a = time.time()
        r = subprocess.run([sys.executable, f"scripts/{script}", str(out)],
                           capture_output=True, text=True)
        T[name] = round(time.time() - a, 2)
        if r.returncode != 0:
            T[name] = f"FAILED after {T[name]}s: {r.stderr.strip()[-200:]}"
            print(f"  channel {name}: FAILED\n{r.stderr[-600:]}", flush=True)
        else:
            print(f"  channel {name}: {T[name]}s  {r.stdout.strip()[-160:]}",
                  flush=True)
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
    N["instances"], N["objects"] = stage_identity(db, len(spans))
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
        "seconds": {k: round(v, 2) for k, v in T.items()},
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
    (ROOT / "bench_write.json").write_text(json.dumps(out_json, indent=1))


if __name__ == "__main__":
    main()
