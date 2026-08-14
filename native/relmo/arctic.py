"""ARCTIC export — exact 3D tracks of articulated objects, no images.

WHY THIS NEEDS NO IMAGE DOWNLOAD: ARCTIC ships two-part meshes
(top.obj / bottom.obj), a per-frame articulation angle, a per-frame 6D
object pose, and per-frame egocentric camera extrinsics + intrinsics.
That is everything needed to place any mesh vertex in space and
project it - the same recipe pixelgt runs on MuJoCo and Kubric. The
494 MB ground-truth payload is genuinely sufficient; the ~600 GB of
images buys appearance we do not consume.

CONVENTION, read from ARCTIC's own common/object_tensors.py (not
guessed): z_axis = [0,0,-1]; the articulation quaternion is
axis_angle_to_quaternion(z_axis * angle) and it rotates the TOP part
ONLY; global rotation and translation then apply to both parts.

VALIDATED before writing a single episode: reconstructing the per-part
rotations and running wm2.screw() on R_bottom^T R_top recovers
ARCTIC's own articulation angle to max|diff| 0.0014 rad over 697
frames, with a screw axis of [-0,-0,-1] at std 7.4e-08. A constant
axis with a varying angle IS the revolute signature, so the joint
head's core statistic provably separates on real human data.

WHAT THIS CONTRIBUTES that no other corpus we hold does: a non-rigid
agent, articulated objects under real manipulation, an egocentric
moving camera, and the grab-vs-use contrast (the SAME object either
carried rigidly or opened) - 239/239 'use' sequences exceed 0.5 rad of
articulation while 'grab' sequences sit at a 0.043 rad median.

    python -m relmo.daemon arctic --limit 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

GEN_VERSION = 300
ROOT = R.ROOT / "vendor_arctic" / "unpack" / "arctic_data"
FPS = 30
NPTS = 384
Z_AXIS = np.array([0.0, 0.0, -1.0])
MM = 1e-3            # mm -> m, see export()
# ARCTIC egocentric frames are 2800x2000; intrinsics ship per sequence
W, H = 2800, 2000


def aa2R(v):
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def load_parts(obj_name):
    import trimesh
    d = ROOT / "meta" / "object_vtemplates" / obj_name
    out = []
    for part in ("bottom", "top"):
        m = trimesh.load(d / f"{part}.obj", process=False)
        out.append(np.asarray(m.vertices, np.float64))
    return out                                   # [bottom, top]


def export(seq_base: Path, out_npz: Path, rng, npts=NPTS):
    obj_name = seq_base.name.split("_")[0]
    o = np.load(f"{seq_base}.object.npy")                    # (T,7)
    cam = np.load(f"{seq_base}.egocam.dist.npy",
                  allow_pickle=True).item()
    bot, top = load_parts(obj_name)
    # ARCTIC mixes units: mesh vertices AND the object translation are in
    # MILLIMETRES, while the egocentric camera translation is in METRES.
    # Measured: composing them raw gives z_med -1111 and in-frame 0.000;
    # scaling object space by 1e-3 gives z_med 0.41 m and in-frame 1.000.
    bot, top = bot * MM, top * MM
    T = len(o)
    Rk = np.asarray(cam["R_k_cam_np"], np.float64)           # (T,3,3)
    Tk = np.asarray(cam["T_k_cam_np"], np.float64)[:, :, 0]  # (T,3)
    K = np.asarray(cam["intrinsics"], np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # sample vertices per part; gbody 0 = bottom, 1 = top
    nb = npts // 2
    ib = rng.choice(len(bot), min(nb, len(bot)), replace=False)
    it = rng.choice(len(top), min(npts - len(ib), len(top)), replace=False)
    Vb, Vt = bot[ib], top[it]
    P = len(ib) + len(it)
    gbody = np.concatenate([np.zeros(len(ib), np.int16),
                            np.ones(len(it), np.int16)])
    gxy = np.zeros((T, P, 2), np.float32)
    gdist = np.zeros((T, P), np.float32)
    gvis = np.zeros((T, P), bool)
    for t in range(T):
        Ra = aa2R(Z_AXIS * o[t, 0])              # articulation: TOP only
        Rg = aa2R(o[t, 1:4])
        tr = o[t, 4:7] * MM
        Xw = np.concatenate([Vb @ Rg.T + tr, (Vt @ Ra.T) @ Rg.T + tr])
        Xc = Xw @ Rk[t].T + Tk[t]                # world -> ego camera
        z = Xc[:, 2]
        ok = z > 1e-6
        u = np.where(ok, fx * Xc[:, 0] / np.where(ok, z, 1) + cx, -1)
        v = np.where(ok, fy * Xc[:, 1] / np.where(ok, z, 1) + cy, -1)
        gxy[t, :, 0], gxy[t, :, 1] = u, v
        gdist[t] = np.where(ok, z, 0)
        gvis[t] = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, ver=GEN_VERSION, source="arctic", tracker="gt_mesh",
        gxy=gxy, gvis=gvis, gdist=gdist, gbody=gbody,
        # gt tracks ARE the input here: there are no images, so there
        # is no tracker output to carry. Flagged so the trainer can
        # apply the measured tracker-noise model as augmentation
        # instead of pretending a tracker ran.
        xy=gxy.copy(), vis=gvis.copy(), synthetic_input=True,
        articulation=o[:, 0].astype(np.float32),
        obj_rot=o[:, 1:4].astype(np.float32),
        obj_trans=o[:, 4:7].astype(np.float32),
        cam_R=Rk.astype(np.float32), cam_T=Tk.astype(np.float32),
        intrinsics=K.astype(np.float32), width=W, height=H,
        object_name=obj_name, fps=FPS)
    art = float(o[:, 0].max() - o[:, 0].min())
    return dict(frames=int(T), points=int(P), articulation_rad=round(art, 3),
                vis_frac=round(float(gvis.mean()), 3), obj=obj_name)


def build(name="arctic_v1", limit=None):
    from tqdm import tqdm
    seqs = sorted(ROOT.glob("raw_seqs/*/*.object.npy"))
    if limit:
        seqs = seqs[:limit]
    man = R.read_manifest(name)
    man.setdefault("episodes", [])
    man["gen_version"] = GEN_VERSION
    man["config"] = dict(source="arctic", fps=FPS, articulated=True,
                         two_part_objects=True, egocentric=True,
                         gt_only=True, npts=NPTS)
    have = {e["id"] for e in man["episodes"]}
    out_dir = R.TRACKS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    made = 0
    for s in tqdm(seqs, unit="seq", desc=f"arctic/{name}"):
        base = Path(str(s)[: -len(".object.npy")])
        eid = f"{base.parent.name}_{base.name}"
        if eid in have:
            continue
        try:
            st = export(base, out_dir / f"{eid}.npz", rng)
            man["episodes"].append(dict(id=eid, shard="raw_seqs",
                                        subject=base.parent.name, **st))
            made += 1
        except Exception as exc:
            R.log("arctic_error", seq=eid, error=str(exc)[:200])
    man = R.write_manifest(name, man)
    R.log("arctic_done", dataset=name, made=made, total=man["n_episodes"],
          fingerprint=man["fingerprint"])
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="arctic_v1")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    m = build(a.name, a.limit)
    print(f"{a.name}: {m['n_episodes']} sequences, fp {m['fingerprint']}")
