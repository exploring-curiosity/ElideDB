"""Sim corpus -> ElideDB store. PIXELS ONLY, one decode per episode.

The corpus layout IS the episode boundary: one mp4 file = one demo.
Nothing is read from the eval sidecars (truth.parquet, meta.json,
signatures.json) - the store meets this corpus exactly as it met the
kitchen corpus: as video files with timestamps. One camera file per
episode directory (the lexically first) becomes the episode's stream;
the second view stays on disk for future multi-view work.

Per episode, one decode feeds three consumers (the one-pass rule):
    segment    re-encode to the store's one-IDR-per-demo H.264 - our
               recordings carry an IDR every 25s, and the read path's
               byte-range discipline requires exactly one, at frame 0
    embed      FDNN-V over every frame -> frame_vectors
    geometry   NGEOM retained frames -> events (transition types are
               DISCOVERED from this corpus's own descriptors, per the
               no-hardwire rule)

Elements (trajectories, identity), channels and the index are the same
store-side builders the kitchen store used - this script only gets the
pixels into the store they operate on:

    python scripts/sim_write.py --out lake/sim_chains --force
    python scripts/build_trajectories.py --store lake/sim_chains
    python scripts/build_dinov3.py --store lake/sim_chains
    python scripts/rebind_events.py --store lake/sim_chains
    python scripts/build_objkind.py --store lake/sim_chains
    python scripts/{motion,iv2,sig2}_ingest.py lake/sim_chains
    python scripts/set_layouts.py --store lake/sim_chains
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2                                                     # noqa: E402
import mlx.core as mx                                          # noqa: E402

from elidedb import Store                                      # noqa: E402
from elidedb.fdnnvideo import fdnnv_dir, load_encoder          # noqa: E402
from elidedb.fftools import find                               # noqa: E402
from elidedb.ingest import CRF, EPOCH_NS, GAP_S, NGEOM         # noqa: E402
from elidedb.video import scan_video_packets                   # noqa: E402

import full_write as fw                                        # noqa: E402
from write_once import episode_events                          # noqa: E402

STREAM = "sim"


def probe(path):
    r = subprocess.run(
        [find("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_packets",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    w, h, rate, n = r.stdout.strip().split(",")[:4]
    num, den = rate.split("/")
    return int(w), int(h), float(num) / float(den), int(n)


def build_spans(src, limit=0):
    """One span per episode directory, from the files alone. The
    timeline is laid out gapped at construction (GAP_S between demos)
    so 'one demo' is an actual time range."""
    eps = sorted(p for p in src.iterdir() if p.is_dir()
                 and p.name.startswith("ep"))
    if limit:
        eps = eps[:limit]
    spans, fps0 = [], None
    t = EPOCH_NS
    for i, d in enumerate(eps):
        f = sorted(d.glob("cam*.mp4"))[0]
        w, h, fps, n = probe(f)
        if fps0 is None:
            fps0 = fps
        assert abs(fps - fps0) < 1e-6, f"{f}: fps {fps} != {fps0}"
        spans.append({"episode": i, "file": i, "stream": STREAM,
                      "path": f, "w": w, "h": h, "n": n,
                      "t0": t, "t1": t + int((n - 1) * 1e9 / fps)})
        t += int(n * 1e9 / fps + GAP_S * 1e9)
    return spans, fps0


def write_episode(sp, fps, media, seg_i, model, cols, ats, avec, astr,
                  ev_rows, cost):
    """One decode -> segment + FDNN-V + geometry, per the one-pass rule."""
    fb = sp["w"] * sp["h"] * 3
    a = time.time()
    raw = subprocess.run(
        [find("ffmpeg"), "-v", "error", "-i", str(sp["path"]),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True).stdout
    n = len(raw) // fb
    fr = np.frombuffer(raw[:n * fb], np.uint8).reshape(n, sp["h"],
                                                       sp["w"], 3)
    cost["decode"] += time.time() - a

    # segment: one IDR, at frame zero - the byte-range contract
    a = time.time()
    tmp = media / "_tmp.h264"
    enc = subprocess.Popen(
        [find("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{sp['w']}x{sp['h']}",
         "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
         "-preset", "medium", "-crf", str(CRF), "-bf", "0",
         "-g", "10000", "-keyint_min", "10000", "-sc_threshold", "0",
         "-f", "h264", str(tmp)], stdin=subprocess.PIPE)
    enc.stdin.write(fr.tobytes())
    enc.stdin.close()
    enc.wait()
    h_ = hashlib.sha1(tmp.read_bytes()).hexdigest()[:8]
    final = media / f"seg-{seg_i:05d}-{h_}.h264"
    tmp.rename(final)
    pk = scan_video_packets(final)
    for j in range(len(pk["ts"])):
        cols["ts"].append(sp["t0"] + int(j * 1e9 / fps))
        cols["byte_offset"].append(int(pk["byte_offset"][j].as_py()))
        cols["packet_size"].append(int(pk["packet_size"][j].as_py()))
        cols["keyframe"].append(bool(pk["keyframe"][j].as_py()))
        cols["width"].append(int(pk["width"][j].as_py()))
        cols["height"].append(int(pk["height"][j].as_py()))
        cols["codec"].append("h264")
        cols["source"].append(f"@media/{final.name}")
        cols["stream"].append(sp["stream"])
        cols["episode_index"].append(sp["episode"])
    cost["segment"] += time.time() - a

    # FDNN-V over every frame; recurrent state per episode - episodes
    # are independent recordings, not a continuous stream
    a = time.time()
    h0 = model.init_state(1)
    for i0 in range(0, n, 256):
        chunk = fr[i0:i0 + 256]
        x = np.stack([cv2.resize(f_, (192, 144),
                                 interpolation=cv2.INTER_AREA)
                      for f_ in chunk])
        e_, h0 = model(mx.array(x.astype(np.float32) / 127.5 - 1.0)[None],
                       h0=h0)
        avec.append(np.array(e_[0], dtype=np.float32))
        ats.extend(sp["t0"] + int((i0 + k) * 1e9 / fps)
                   for k in range(len(chunk)))
        astr.extend([sp["stream"]] * len(chunk))
    cost["embed"] += time.time() - a

    # geometry on the retained budget
    a = time.time()
    keep = [cv2.resize(fr[i], (256, 192), interpolation=cv2.INTER_AREA)
            for i in np.unique(np.linspace(0, n - 1, NGEOM)
                               .round().astype(int))]
    if len(keep) >= 4:
        rows, _ = episode_events(sp, keep, sp["t0"], sp["t1"])
        ev_rows += rows
    cost["geometry"] += time.time() - a
    return n


def main():
    argv = sys.argv
    out = Path(argv[argv.index("--out") + 1] if "--out" in argv
               else "lake/sim_chains")
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 0
    src = ROOT / (argv[argv.index("--src") + 1] if "--src" in argv
                  else "data/sim_chains")
    if out.exists():
        if "--force" not in argv:
            raise SystemExit(f"{out} exists; pass --force")
        shutil.rmtree(out)

    spans, fps = build_spans(src, limit)
    print(f"{len(spans)} episodes from {src} at {fps:g} fps", flush=True)
    db = Store.create(str(out), out.name)
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)

    model, _ = load_encoder(fdnnv_dir())
    model(mx.array(np.zeros((1, 8, 144, 192, 3), np.float32)))  # warm

    cols = defaultdict(list)
    ats, avec, astr, ev_rows = [], [], [], []
    cost = defaultdict(float)
    t0 = time.time()
    from tqdm import tqdm
    for sp in tqdm(spans, desc="sim-write", unit="ep"):
        write_episode(sp, fps, media, sp["episode"], model,
                      cols, ats, avec, astr, ev_rows, cost)

    shift = {(s["stream"], s["t0"]): 0 for s in spans}
    fw.commit_episodes(db, spans, shift)
    nf = fw.commit_frames(db, cols)
    nv = fw.commit_vectors(db, ats, avec, astr)
    ne = fw.commit_events(db, ev_rows, spans)
    print({"episodes": len(spans), "frames": nf, "frame_vectors": nv,
           "events": ne,
           "cost_s": {k: round(v, 1) for k, v in cost.items()},
           "wall_min": round((time.time() - t0) / 60, 1)}, flush=True)
    # the artifact is the test
    assert nf > 0 and nv == nf and len(db.table("episodes").scan()) \
        == len(spans)


if __name__ == "__main__":
    main()
