"""Kubric MOVi import — the SOTA-standard scene tier, consumed not rendered.

WHY (owner, 2026-08-11): "make sure the SIM is the same as the
standards used in training SOTA world models... the least I want is a
world model built on a primitive simulator."

Audited answer: our PHYSICS engine is standard (MuJoCo is DeepMind's
own, and FIGNet - the reference learned rigid-dynamics model - trained
on weaker PyBullet trajectories). Our SCENE tier was not: the
credibility bar for this exact task is Kubric MOVi-C and up, which
uses ~1000 scanned real objects (Google Scanned Objects: hollow,
non-convex, genuinely hard geometry), HDRI dome lighting, and - for
MOVi-E - a MOVING camera. Static-camera primitives are below the bar
for tracker/dynamics training. This is also literally the corpus
TAP-Vid-Kubric / TAPIR / CoTracker were trained on.

We CONSUME the pre-rendered data rather than running Kubric: Kubric
generation is broken on Apple Silicon (linux/amd64 image, AVX-compiled
TF, open issue #288), but consumption needs nothing - no TensorFlow,
no Blender, no Docker. TFRecords are length-prefixed tf.train.Example
protobufs and a ~70-line proto walk reads them.

What each example carries (verified against features.json): RGB, per
-frame instance segmentation, uint16 ray-length depth, forward/backward
flow, normals, per-pixel canonical object_coordinates, per-frame camera
pose + intrinsics, and per-instance positions/quaternions/velocities/
collisions. Everything pixelgt needs, so MOVi episodes land in exactly
the physgen episode layout and the rest of the pipeline is unchanged.

Splits: MOVi's OWN train/validation/test are used, not our hash split.
For MOVi-C/D/E the test split holds out OBJECTS AND BACKGROUNDS, which
is a stronger generalization test than anything we could construct by
hashing episode ids.

    python -m relmo.daemon movi --name movi_e --shards 160
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

GEN_VERSION = 100          # 100+ = imported, not simulated here
BUCKET = "https://storage.googleapis.com/kubric-public/tfds"
FPS = 12                   # MOVi is 24 frames @ 12fps = 2s
SPLIT_SHARDS = dict(train=1024, validation=64, test=256)


# -- TFRecord + protobuf, no tensorflow ------------------------------
def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7


def _fields(b):
    i = 0
    while i < len(b):
        tag, i = _varint(b, i)
        fn, wt = tag >> 3, tag & 7
        if wt == 2:
            ln, i = _varint(b, i)
            yield fn, b[i:i + ln], 2
            i += ln
        elif wt == 0:
            v, i = _varint(b, i)
            yield fn, v, 0
        elif wt == 5:
            yield fn, b[i:i + 4], 5
            i += 4
        elif wt == 1:
            yield fn, b[i:i + 8], 1
            i += 8


def _feature(kv):
    """map<string, Feature>; Feature: bytes=1, float=2, int64=3."""
    k = out = None
    for fn, x, wt in _fields(kv):
        if fn == 1:
            k = x.decode()
        elif fn == 2 and wt == 2:
            for fn2, y, _ in _fields(x):
                if fn2 == 1:
                    out = [z for f3, z, _ in _fields(y) if f3 == 1]
                elif fn2 == 2:
                    fl = []
                    for f3, z, w3 in _fields(y):
                        if f3 == 1:
                            fl.extend(np.frombuffer(z, "<f4") if w3 == 2
                                      else [struct.unpack("<f", z)[0]])
                    out = np.array(fl, np.float32)
                elif fn2 == 3:
                    il = []
                    for f3, z, w3 in _fields(y):
                        if f3 == 1:
                            if w3 == 2:
                                j = 0
                                while j < len(z):
                                    v, j = _varint(z, j)
                                    il.append(v)
                            else:
                                il.append(z)
                    out = np.array(il, np.int64)
    return k, out


def records(path: Path):
    """Yield raw tf.train.Example payloads from a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            ln = struct.unpack("<Q", head)[0]
            f.read(4)                      # length crc
            payload = f.read(ln)
            f.read(4)                      # data crc
            yield payload


def parse(payload):
    ex = {}
    for fn, v, _ in _fields(payload):
        if fn == 1:
            for fn2, kv, _ in _fields(v):
                if fn2 == 1:
                    k, val = _feature(kv)
                    ex[k] = val
    return ex


def _png_seq(blobs):
    import cv2
    return np.stack([cv2.imdecode(np.frombuffer(b, np.uint8),
                                  cv2.IMREAD_UNCHANGED) for b in blobs])


def _f32(ex, key, shape):
    v = ex[key]
    a = np.frombuffer(v[0], "<f4") if isinstance(v, list) else v
    return np.asarray(a, np.float32).reshape(shape)


def episode(ex, out_dir: Path):
    """One MOVi example -> the physgen episode layout."""
    import cv2
    T = int(ex["metadata/num_frames"][0])
    n = int(ex["metadata/num_instances"][0])
    W = int(ex["metadata/width"][0])
    H = int(ex["metadata/height"][0])
    rgb = _png_seq(ex["video"])[..., ::-1]              # cv2 BGR -> RGB
    seg = _png_seq(ex["segmentations"]).reshape(T, H, W).astype(np.uint8)
    dq = _png_seq(ex["depth"]).reshape(T, H, W).astype(np.float32)
    dr = _f32(ex, "metadata/depth_range", (2,))
    # Kubric's own dequantisation (challenges/point_tracking/dataset.py):
    # depth = min + u16*(max-min)/65535, and it is RAY LENGTH from the
    # camera centre, NOT planar z - getting this wrong silently bends
    # every lifted point toward the image edges.
    depth = dr[0] + dq * (dr[1] - dr[0]) / 65535.0
    out_dir.mkdir(parents=True, exist_ok=True)
    _mp4(rgb, out_dir / "frames.mp4", W, H)
    coll = ex.get("events/collisions/frame")
    np.savez_compressed(
        out_dir / "state.npz",
        source="kubric", gen_version=GEN_VERSION, fps=FPS,
        width=W, height=H,
        seg=seg, depth=depth.astype(np.float16),
        cam_positions=_f32(ex, "camera/positions", (T, 3)),
        cam_quaternions=_f32(ex, "camera/quaternions", (T, 4)),
        focal_length=_f32(ex, "camera/focal_length", (1,))[0],
        sensor_width=_f32(ex, "camera/sensor_width", (1,))[0],
        obj_positions=_f32(ex, "instances/positions", (n, T, 3)),
        obj_quaternions=_f32(ex, "instances/quaternions", (n, T, 4)),
        obj_velocities=_f32(ex, "instances/velocities", (n, T, 3)),
        obj_image_positions=_f32(ex, "instances/image_positions", (n, T, 2)),
        collision_frames=(np.asarray(coll) if coll is not None
                          else np.zeros(0, np.int64)),
        video_name=ex["metadata/video_name"][0].decode()
        if isinstance(ex["metadata/video_name"], list) else "")
    dyn = int((np.abs(_f32(ex, "instances/velocities", (n, T, 3)))
               .max(1).max(1) > 0.05).sum())
    return dict(frames=T, n_bodies=n, moving=dyn, width=W, height=H)


def _mp4(frames, path, W, H):
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v",
         "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-bf", "0",
         str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
    p.stdin.close()
    p.wait()


def sync(name="movi_e", variant="256x256", shards=160, val_shards=8,
         test_shards=16, keep_raw=False):
    """Stream shards: download -> convert -> delete. Peak disk stays a
    few hundred MB even though the full variant is ~300 GB, and the
    job is resumable because progress is the episodes on disk."""
    from tqdm import tqdm
    man = R.read_manifest(name)
    man.setdefault("episodes", [])
    man["gen_version"] = GEN_VERSION
    man["config"] = dict(source="kubric", dataset=name, variant=variant,
                         fps=FPS, textured=True, scanned_assets=True,
                         moving_camera=name in ("movi_e", "movi_f"),
                         native_splits=True)
    have = {e["id"] for e in man["episodes"]}
    done_shards = {e["shard"] for e in man["episodes"]}
    raw = R.ROOT / "data" / "relmo" / "movi" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    root = R.dataset_dir(name)
    plan = ([("train", i, shards) for i in range(shards)]
            + [("validation", i, val_shards) for i in range(val_shards)]
            + [("test", i, test_shards) for i in range(test_shards)])
    made = 0
    for split, idx, _ in tqdm(plan, unit="shard", desc=f"movi/{name}"):
        tot = SPLIT_SHARDS[split]
        sid = f"{split}-{idx:05d}"
        if sid in done_shards:
            continue
        fn = f"{name}-{split}.tfrecord-{idx:05d}-of-{tot:05d}"
        url = f"{BUCKET}/{name}/{variant}/1.0.0/{fn}"
        loc = raw / fn
        try:
            if not loc.exists():
                urllib.request.urlretrieve(url, loc)
            for j, payload in enumerate(records(loc)):
                eid = f"{split}_{idx:05d}_{j:03d}"
                if eid in have:
                    continue
                ex = parse(payload)
                d = root / sid / eid
                tmp = root / sid / f".tmp_{eid}"
                st = episode(ex, tmp)
                tmp.rename(d)
                man["episodes"].append(
                    dict(id=eid, shard=sid, split=split, **st))
                made += 1
        except Exception as exc:                     # never die mid-sync
            R.log("movi_error", dataset=name, shard=sid,
                  error=str(exc)[:200])
        finally:
            if loc.exists() and not keep_raw:
                loc.unlink()
        R.write_manifest(name, man)
    man = R.write_manifest(name, man)
    R.log("movi_sync_done", dataset=name, made=made,
          total=man["n_episodes"], fingerprint=man["fingerprint"])
    return man


def split_of(dataset, episode_id):
    """MOVi ships its own splits; for C/D/E the test split holds out
    objects AND backgrounds, so we never re-hash them."""
    return episode_id.split("_")[0].replace("validation", "val")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="movi_e")
    ap.add_argument("--variant", default="256x256")
    ap.add_argument("--shards", type=int, default=160)
    ap.add_argument("--val-shards", type=int, default=8)
    ap.add_argument("--test-shards", type=int, default=16)
    a = ap.parse_args()
    m = sync(a.name, a.variant, a.shards, a.val_shards, a.test_shards)
    print(f"{a.name}: {m['n_episodes']} episodes, "
          f"fingerprint {m['fingerprint']}")
