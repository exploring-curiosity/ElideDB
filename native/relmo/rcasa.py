"""RoboCasa ingest — replay demos in MuJoCo, emit exact structural GT.

WHY REPLAY AND NOT VIDEO: RoboCasa demos store full MuJoCo state
vectors per timestep plus the scene XML, restored with
`sim.set_state_from_flattened(...)`. So during replay we hold a LIVE
MjModel/MjData and can take anything - including cameras that never
existed in the recording. We are not extracting ground truth from a
dataset; we are re-running the physics.

EXTENDED SCHEMA (the reason this file exists). physgen only ever had
free joints, so `state.npz` had no way to express articulation. A
robot is a KINEMATIC TREE, and the world model is supposed to discover
it, so the answer key has to be recorded:

    body_parentid              the tree itself
    jnt_type/axis/range/bodyid every joint's kind and screw axis
    qpos                       every joint's state per frame
    geom_bodyid                which body each rendered geom belongs to
    contact pairs + force      who is touching whom, per frame
    cam_pos/cam_mat per frame  cameras MOVE here (eye-in-hand), unlike
                               physgen where one static camera was
                               reconstructible from the seed

Measured on a live scene (PickPlaceCounterToCabinet, PandaOmron):
384 bodies, 108 joints (66 revolute / 39 prismatic), 2027 geoms, 267
contacts in a single frame - and the gripper reads out as two
prismatic joints on a SHARED axis with MIRRORED ranges, which is the
pincer signature the world model is meant to discover from motion
alone.

    python -m relmo.daemon rcasa --limit 400
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

GEN_VERSION = 200          # 200+ = replayed from an external corpus
FPS = 20
SEG_DS = 2                 # seg/depth stored at half res, as physgen
MAXC = 64                  # contacts kept per frame


def _find_demos(root: Path):
    return sorted(root.rglob("demo*.hdf5")) + sorted(root.rglob("*.hdf5"))


def _mj(env):
    """robosuite wraps MjModel/MjData; reach the raw structs."""
    m = env.sim.model._model if hasattr(env.sim.model, "_model") else env.sim.model
    d = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    return m, d


def structure(m):
    """The static answer key: tree + joint spec. Written once per
    episode because it cannot change inside one episode."""
    import mujoco
    names = []
    for b in range(m.nbody):
        names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"b{b}")
    return dict(
        body_parentid=np.asarray(m.body_parentid, np.int32),
        body_names=np.array(names),
        jnt_type=np.asarray(m.jnt_type, np.int32),
        jnt_axis=np.asarray(m.jnt_axis, np.float32),
        jnt_range=np.asarray(m.jnt_range, np.float32),
        jnt_bodyid=np.asarray(m.jnt_bodyid, np.int32),
        jnt_qposadr=np.asarray(m.jnt_qposadr, np.int32),
        geom_bodyid=np.asarray(m.geom_bodyid, np.int32),
        body_mass=np.asarray(m.body_mass, np.float32))


def replay(env, states, out_dir: Path, cam="robot0_agentview_left",
           W=320, H=240, stride=1, max_T=200):
    """Restore each stored state, render, and record everything."""
    import mujoco
    m, d = _mj(env)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    if cid < 0:
        cid = 0
    ren = mujoco.Renderer(m, H, W)
    H2, W2 = H // SEG_DS, W // SEG_DS
    idx = list(range(0, len(states), stride))[:max_T]
    T, nb = len(idx), m.nbody
    xpos = np.zeros((T, nb, 3), np.float32)
    xquat = np.zeros((T, nb, 4), np.float32)
    cvel = np.zeros((T, nb, 6), np.float32)
    qpos = np.zeros((T, m.nq), np.float32)
    cam_pos = np.zeros((T, 3), np.float32)
    cam_mat = np.zeros((T, 9), np.float32)
    segs = np.zeros((T, H2, W2), np.uint8)
    deps = np.zeros((T, H2, W2), np.float16)
    con = np.zeros((T, MAXC, 3), np.float32)     # body_a, body_b, force
    ncon = np.zeros(T, np.int32)
    frames = []
    for k, t in enumerate(idx):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        m, d = _mj(env)
        xpos[k], xquat[k] = d.xpos, d.xquat
        cvel[k, :, :3], cvel[k, :, 3:] = d.cvel[:, 3:], d.cvel[:, :3]
        qpos[k] = d.qpos[:m.nq]
        cam_pos[k] = d.cam_xpos[cid]
        cam_mat[k] = d.cam_xmat[cid]
        n = 0
        for c in range(min(d.ncon, MAXC)):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, f)
            con[k, n] = (m.geom_bodyid[g1], m.geom_bodyid[g2],
                         float(np.linalg.norm(f[:3])))
            n += 1
        ncon[k] = n
        ren.disable_depth_rendering()
        ren.disable_segmentation_rendering()
        ren.update_scene(d, camera=cid)
        frames.append(ren.render().copy())
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera=cid)
        s = ren.render()[..., 0]
        sg = np.zeros_like(s, np.uint8)
        ok = (s >= 0) & (s < m.ngeom)
        sg[ok] = np.clip(m.geom_bodyid[s[ok]], 0, 255).astype(np.uint8)
        segs[k] = sg[::SEG_DS, ::SEG_DS]
        ren.disable_segmentation_rendering()
        ren.enable_depth_rendering()
        ren.update_scene(d, camera=cid)
        deps[k] = ren.render()[::SEG_DS, ::SEG_DS].astype(np.float16)
    ren.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    _mp4(frames, out_dir / "frames.mp4", W, H)
    np.savez_compressed(
        out_dir / "state.npz", source="robocasa", gen_version=GEN_VERSION,
        fps=FPS, width=W, height=H, seg_ds=SEG_DS, camera=cam,
        seg=segs, depth=deps, xpos=xpos, xquat=xquat, xvel=cvel,
        qpos=qpos, cam_pos=cam_pos, cam_mat=cam_mat,
        cam_fovy=np.float32(m.cam_fovy[cid]),
        contact_pairs=con, contact_n=ncon, **structure(m))
    mov = int((np.linalg.norm(cvel[:, :, :3], axis=-1).max(0) > 0.02).sum())
    return dict(frames=T, n_bodies=int(nb), n_joints=int(m.njnt),
                moving=mov, contact_frames=int((ncon > 0).sum()))


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


def build(name="rcasa_v1", limit=None, cameras=None, max_T=200,
          stride=2):
    """Replay every demo we hold. Cameras: replaying the SAME demo from
    several viewpoints multiplies episodes for free and destroys any
    viewpoint regularity the model might otherwise latch onto."""
    import h5py
    from tqdm import tqdm
    import robocasa
    cameras = cameras or ["robot0_agentview_left", "robot0_agentview_right",
                          "robot0_eye_in_hand"]
    root = Path(robocasa.__path__[0]) / "models" / "assets" / "datasets"
    files = _find_demos(root)
    if not files:
        R.log("rcasa_no_demos", root=str(root))
        return None
    man = R.read_manifest(name)
    man.setdefault("episodes", [])
    man["gen_version"] = GEN_VERSION
    man["config"] = dict(source="robocasa", fps=FPS, seg_ds=SEG_DS,
                         cameras=cameras, articulated=True,
                         structural_gt=True, max_T=max_T, stride=stride)
    have = {e["id"] for e in man["episodes"]}
    out_root = R.dataset_dir(name)
    made = 0
    todo = files if limit is None else files[:limit]
    for f in tqdm(todo, unit="file", desc=f"rcasa/{name}"):
        try:
            with h5py.File(f, "r") as h:
                keys = sorted(h["data"].keys())[:limit or 10**9]
                for dk in keys:
                    g = h["data"][dk]
                    states = np.array(g["states"])
                    xml = g.attrs.get("model_file")
                    ep_meta = g.attrs.get("ep_meta")
                    for cam in cameras:
                        eid = f"{f.stem}_{dk}_{cam}"
                        if eid in have:
                            continue
                        env = _env_for(xml, ep_meta)
                        if env is None:
                            continue
                        shard = f"shard_{made // 200:04d}"
                        tmp = out_root / shard / f".tmp_{eid}"
                        st = replay(env, states, tmp, cam=cam,
                                    max_T=max_T, stride=stride)
                        tmp.rename(out_root / shard / eid)
                        man["episodes"].append(
                            dict(id=eid, shard=shard, task=f.parent.name,
                                 camera=cam, **st))
                        made += 1
                        env.close()
                        if made % 25 == 0:
                            R.write_manifest(name, man)
        except Exception as exc:
            R.log("rcasa_error", file=f.name, error=str(exc)[:200])
    man = R.write_manifest(name, man)
    R.log("rcasa_done", dataset=name, made=made,
          total=man["n_episodes"], fingerprint=man["fingerprint"])
    return man


_ENV = {}


def _env_for(xml, ep_meta):
    """Rebuild the recorded scene. Cached by xml hash - scene
    construction measured at 6.9s, so rebuilding per demo would
    dominate wall clock."""
    import robosuite
    from robocasa.utils.env_utils import create_env
    try:
        key = hash(xml) if xml is not None else "default"
        if key in _ENV:
            env = _ENV[key]
            if xml is not None:
                env.reset_from_xml_string(env.edit_model_xml(xml))
                env.sim.reset()
            return env
        env = create_env(env_name="PickPlaceCounterToCabinet",
                         robots="PandaOmron", camera_widths=320,
                         camera_heights=240, seed=0)
        env.reset()
        if xml is not None:
            if ep_meta is not None and hasattr(env, "set_ep_meta"):
                env.set_ep_meta(json.loads(ep_meta))
            env.reset_from_xml_string(env.edit_model_xml(xml))
            env.sim.reset()
        _ENV.clear()
        _ENV[key] = env
        return env
    except Exception as exc:
        R.log("rcasa_env_error", error=str(exc)[:200])
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rcasa_v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-t", type=int, default=200)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    m = build(a.name, a.limit, max_T=a.max_t, stride=a.stride)
    print(f"{a.name}: {m['n_episodes'] if m else 0} episodes")
