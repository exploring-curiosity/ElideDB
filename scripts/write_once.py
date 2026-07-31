"""ONE PASS: raw video in, operable database out.

The write path was two passes over the same pixels. Ingest streamed
every frame through ffmpeg at 1,109 fps to embed it, and then element
extraction RE-DECODED 12 frames per episode through the store's own
byte-range reader to compute geometry - 62 ms per episode of decode
that had already been paid, plus 310 ms of re-cropping and re-embedding
participants whose frames were in memory moments earlier.

Here the stream is consumed once. As frames go by they are:

    embedded          FDNN-V, the vectors the retrieval path uses
    retained          a fixed budget of frames per episode, chosen up
                      front from the episode's length, so geometry gets
                      what it needs without buffering the whole file
    turned to motion  one GPU pass over the retained frames
    turned to events  agent / participants / transitions, from the
                      frames already in hand

and everything lands with the layouts the read path wants: frames
grouped by episode, labels sorted by value, episode flags for
set-membership pruning.

NAMING IS THE WHOLE COST, measured on 1.14 h / 595 episodes:

    setup            0.03 s
    stream + embed  11.52 s
    geometry         3.76 s
    write            0.58 s
    ------------------------
    subtotal        15.89 s   =  26.8 ms/episode,  0.23 min/hour of video
    naming         238.44 s   = 400.7 ms/episode,  3.49 min/hour

Batching the crops across the whole corpus instead of four at a time
did NOT rescue it - SigLIP on crops is simply the cost, and it is 94%
of the run. So naming is opt-in (--names) and the default write is the
fast one, which is honest about what each buys:

    without names   0.23 min/hour   actions queryable, objects are not
    with names      3.72 min/hour   objects queryable, over a 1 min/hour
                                    budget by 3.7x

The fix is not batching, it is distillation: predict the name VECTOR
from the FDNN-V frame embedding already computed during the stream plus
the participant box, instead of cropping the frame again and running a
second vision model over it. Names are never compared as strings in
this system - matching is cosine in name space - so the vector is the
entire product, and it is a head, not an encoder.

  python scripts/write_once.py --out lake/elide_v2 [--files 119 120]
                               [--names]      slow path, objects indexed
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx                                        # noqa: E402

from elidedb import Store                                    # noqa: E402
from elidedb.fdnnvideo import fdnnv_dir, load_encoder        # noqa: E402
from elidedb.fftools import find                             # noqa: E402

CAM = "observation.images.image_0"
DATA = Path("data/bridge")
EPOCH_NS = 1_704_067_200_000_000_000
FILE_STRIDE_NS = 20_000_000_000_000
FPS = 5.0
NGEOM = 12                       # frames per episode handed to geometry
GEOM_W, GEOM_H = 256, 192        # geometry runs on its own small raster


def episode_spans(files):
    key = f"videos/{CAM}"
    meta = pq.read_table(
        DATA / "meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "length", f"{key}/file_index",
                 f"{key}/from_timestamp", f"{key}/to_timestamp"]).to_pydict()
    fi = np.array(meta[f"{key}/file_index"])
    out = []
    for j in np.where(np.isin(fi, files))[0]:
        f = int(fi[j])
        base = EPOCH_NS + f * FILE_STRIDE_NS
        out.append({
            "file": f, "episode": int(meta["episode_index"][j]),
            "t0": base + int(round(meta[f"{key}/from_timestamp"][j] * 1e9)),
            "t1": base + int(round(meta[f"{key}/to_timestamp"][j] * 1e9)) - 1,
            "n": int(meta["length"][j]),
            "stream": f"{CAM}/file-{f:03d}"})
    out.sort(key=lambda r: (r["file"], r["t0"]))
    return out


def stream_once(model, f, spans, geom_every):
    """Decode once. Embed everything, retain a budget per episode."""
    W, H = 192, 144
    fb = W * H * 3
    proc = subprocess.Popen(
        [find("ffmpeg"), "-v", "error", "-i",
         str(DATA / f"videos/{CAM}/chunk-000/file-{f:03d}.mp4"),
         "-vf", f"scale={W}:{H}", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE)
    base = EPOCH_NS + f * FILE_STRIDE_NS
    h = model.init_state(1)
    vecs, keep = [], {}
    n_done, buf = 0, b""
    while True:
        b = proc.stdout.read(fb * 256 - len(buf))
        if b:
            buf += b
        n = len(buf) // fb
        if n == 0 and not b:
            break
        if n == 0:
            continue
        fr = np.frombuffer(buf[:n * fb], np.uint8).reshape(n, H, W, 3)
        buf = buf[n * fb:]
        x = mx.array(fr.astype(np.float32) / 127.5 - 1.0)[None]
        e, h = model(x, h0=h)
        vecs.append(np.array(e[0], dtype=np.float32))
        # retain the frames geometry asked for, as they pass
        for i in range(n):
            gi = n_done + i
            ep = geom_every.get(gi)
            if ep is not None:
                keep.setdefault(ep, []).append(fr[i].copy())
        n_done += n
        if not b and not buf:
            break
    proc.wait()
    ts = base + np.round(np.arange(n_done) * 1e9 / FPS).astype(np.int64)
    return ts, (np.concatenate(vecs) if vecs else
                np.zeros((0, 1152), np.float32)), keep


def main():
    argv = sys.argv
    out = Path(argv[argv.index("--out") + 1] if "--out" in argv
               else "lake/elide_v2")
    files = ([int(x) for x in argv[argv.index("--files") + 1].split(",")]
             if "--files" in argv else [119, 120, 129, 132])
    if out.exists():
        if "--force" not in argv:
            raise SystemExit(f"{out} exists; pass --force")
        import shutil
        shutil.rmtree(out)

    T = {}
    t00 = time.time()
    db = Store.create(out, out.name)
    model, _ = load_encoder(fdnnv_dir())
    e, _h = model(mx.array(np.zeros((1, 8, 144, 192, 3), np.float32)))
    mx.eval(e)
    T["setup"] = time.time() - t00

    spans = episode_spans(files)
    by_file = {}
    for s in spans:
        by_file.setdefault(s["file"], []).append(s)

    from build_teacher import articulated_box, cavity_series, REL_MIN, CAV_MIN
    from extract_events import (agent_track, causal_participants,
                                motion_mags, _crop)
    # NAMES, BATCHED. The two-pass path cropped and ran SigLIP per demo -
    # ~310 ms of the 535 ms it spent per episode, on batches of four.
    # Here crops accumulate across the whole run and go to the GPU in
    # one sweep at the end, so the model sees batches of hundreds.
    name_crops, name_slot = [], []

    all_ts, all_vec, all_stream = [], [], []
    ev_rows = []
    t_stream = t_geom = 0.0

    for f, sp in by_file.items():
        # which global frame indices to retain, and for which episode
        base = EPOCH_NS + f * FILE_STRIDE_NS
        want = {}
        for s in sp:
            i0 = int(round((s["t0"] - base) / 1e9 * FPS))
            i1 = int(round((s["t1"] - base) / 1e9 * FPS))
            if i1 <= i0:
                continue
            for gi in np.unique(np.linspace(i0, i1, NGEOM).round()
                                .astype(int)):
                want[int(gi)] = s["episode"]

        a = time.time()
        ts, vec, keep = stream_once(model, f, sp, want)
        t_stream += time.time() - a
        all_ts.append(ts)
        all_vec.append(vec)
        all_stream += [f"{CAM}/file-{f:03d}"] * len(ts)

        a = time.time()
        span_of = {s["episode"]: s for s in sp}
        for epi, frames in keep.items():
            if len(frames) < 4:
                continue
            s = span_of[epi]
            times = np.linspace(s["t0"], s["t1"], len(frames)).astype(np.int64)
            mg = motion_mags(frames)
            tr, span, masks, tracks = agent_track(frames, mg)
            abox = articulated_box(frames, masks.any(0), mg)

            def stamp(i):
                return int(times[min(max(i, 0), len(times) - 1)])

            if tr is not None:
                ev_rows.append([s["stream"], s["t0"], s["t1"], "agent",
                                stamp(tr[0][0]), stamp(tr[-1][0] + 1),
                                float(span), ""])
            cav = cavity_series(frames, abox)
            if cav is not None and len(cav) >= 4:
                b0 = float(np.median(cav[:2]))
                run = None
                for i in range(1, len(cav)):
                    d = float(cav[i]) - b0
                    k = ("open" if d >= CAV_MIN else
                         "close" if d <= -CAV_MIN else None)
                    if k and run is None:
                        run = (k, i)
                    elif run and k != run[0]:
                        ev_rows.append([s["stream"], s["t0"], s["t1"],
                                        run[0], stamp(run[1] - 1),
                                        stamp(i), abs(d), ""])
                        run = (k, i) if k else None
                if run:
                    ev_rows.append([s["stream"], s["t0"], s["t1"], run[0],
                                    stamp(run[1] - 1), stamp(len(cav) - 1),
                                    abs(float(cav[-1]) - b0), ""])
            for origin, dest, onset, life, area in causal_participants(
                    tracks, tr, len(frames) - 1):
                c0 = np.array([(origin[0] + origin[2]) / 2,
                               (origin[1] + origin[3]) / 2])
                c1 = np.array([(dest[0] + dest[2]) / 2,
                               (dest[1] + dest[3]) / 2])
                od = float(np.hypot(origin[2] - origin[0],
                                    origin[3] - origin[1]))
                disp = float(np.linalg.norm(c1 - c0)) / max(od, 1.0)

                def inside(c, r):
                    return (r is not None and r[0] <= c[0] <= r[2]
                            and r[1] <= c[1] <= r[3])
                k = ("adjust" if disp < REL_MIN else
                     "take_out" if (inside(c0, abox) and not inside(c1, abox))
                     else "put_into" if inside(c1, abox) else "put_on")
                for kind, i0, i1, box, src in (
                        ("contact", onset - 1, onset, origin, frames[0]),
                        ("release", onset + life - 1, onset + life, dest,
                         frames[-1]),
                        (k, onset, onset + life, dest, frames[0])):
                    ev_rows.append([s["stream"], s["t0"], s["t1"], kind,
                                    stamp(i0), stamp(i1), 1.0, ""])
                    c = _crop(src, box if kind != k else origin)
                    if c is not None and c.size:
                        name_crops.append(c)
                        name_slot.append(len(ev_rows) - 1)
        t_geom += time.time() - a

    T["stream+embed"] = t_stream
    T["geometry"] = t_geom

    # ---- one batched naming sweep for the whole corpus
    a = time.time()
    names = [""] * len(ev_rows)
    NV = None
    if name_crops and "--names" in argv:
        try:
            import torch
            from student_elements import crop_embed
            from elidedb.device import pick
            from elidedb.sig2 import MID
            from transformers import AutoModel, AutoProcessor
            dev, dtype = pick()
            proc_ = AutoProcessor.from_pretrained(MID)
            sig = AutoModel.from_pretrained(
                MID, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
            head = ROOT / "models/student_v1/namer.pt"
            import torch.nn as nn

            class Namer(nn.Module):
                def __init__(self, d=1152):
                    super().__init__()
                    self.f = nn.Sequential(nn.Linear(d, 1024), nn.GELU(),
                                           nn.Linear(1024, d))

                def forward(self, x):
                    y = x + self.f(x)
                    return y / (y.norm(dim=-1, keepdim=True) + 1e-8)
            net = Namer().to(dev)
            net.load_state_dict(torch.load(head, map_location=dev,
                                           weights_only=True))
            net.eval()
            vs = []
            B = 256
            for i in range(0, len(name_crops), B):
                emb = crop_embed(name_crops[i:i + B], sig, proc_, dev)
                with torch.no_grad():
                    vs.append(net(torch.tensor(emb, device=dev,
                                               dtype=torch.float32))
                              .cpu().numpy())
            NV = np.concatenate(vs) if vs else None
        except Exception as ex:
            print(f"  naming skipped: {type(ex).__name__}: {ex}", flush=True)
    T["naming"] = time.time() - a

    # ---- write, with the layouts the read path wants
    a = time.time()
    ts = np.concatenate(all_ts)
    vec = np.concatenate(all_vec)
    order = np.argsort(ts, kind="stable")
    fv = pa.table({
        "ts": pa.array(ts[order], pa.int64()),
        "stream": pa.array([all_stream[i] for i in order]),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(
                vec[order].astype(np.float16)).reshape(-1), pa.float16()),
            vec.shape[1]),
    })
    db.table("frame_vectors").append(fv, kind="embeddings",
                                     meta={"model": "fdnnv"})

    ep = pa.table({
        "ts": pa.array([s["t0"] for s in spans], pa.int64()),
        "t1": pa.array([s["t1"] for s in spans], pa.int64()),
        "episode_index": pa.array([s["episode"] for s in spans], pa.int64()),
        "stream": pa.array([s["stream"] for s in spans]),
        "n_frames": pa.array([s["n"] for s in spans], pa.int32()),
        "file_index": pa.array([s["file"] for s in spans], pa.int32()),
    })
    db.table("episodes").append(ep, kind="timeseries")

    NAMEV = np.zeros((len(ev_rows), 1152), np.float32)
    if NV is not None:
        for j, slot in enumerate(name_slot):
            if j < len(NV):
                NAMEV[slot] = NV[j]
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
        "name_vec": pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(NAMEV.astype(np.float16))
                     .reshape(-1), pa.float16()), NAMEV.shape[1]),
    })
    import pyarrow.compute as pc
    E = E.take(pc.sort_indices(E.column("ts")))
    db.table("events").append_grouped(
        E, "kind", kind="events", sort_by=["kind", "ts"],
        min_group_rows=1024, meta={"builder": "write_once"})
    T["write"] = time.time() - a

    T["total"] = time.time() - t00
    hours = sum(s["n"] for s in spans) / FPS / 3600
    print(json.dumps({
        "store": str(out), "episodes": len(spans), "events": len(ev_rows),
        "frames": int(len(ts)), "hours_of_video": round(hours, 2),
        "seconds": {k: round(v, 2) for k, v in T.items()},
        "s_per_demo": round(T["total"] / max(len(spans), 1), 4),
        "min_per_hour_video": round(T["total"] / 60 / max(hours, 1e-9), 2),
    }, indent=1))


if __name__ == "__main__":
    main()
