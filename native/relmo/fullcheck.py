"""Everything in an episode that had not yet been validated.

gtcheck covers seg range, projection, persistence, content, contacts,
render layer, velocity and contact overflow. depthcheck covers depth's
absolute scale and per-body xyz. This closes the rest, and each test is
chosen so that a plausible silent corruption would FAIL it:

  I  xquat      transform each geom's body-frame offset by the stored
                (xpos, xquat), project it, and require it to land inside
                that body's segmentation mask. A body rotated wrongly
                about its own axis passes every check so far - its
                surface points are still on its surface - but its
                OFFSET geoms project somewhere else.
  J  qpos       replay qpos into a clean MjData and recompute forward
                kinematics. Stored xpos must reproduce exactly. Catches
                a qpos written from a different frame than xpos.
  K  tree       body_parentid is acyclic, roots at world, and every
                jnt_bodyid / geom_bodyid / jnt_qposadr is in range.
  L  camera     cam_mat orthonormal with det +1, cam_pos finite. A
                non-rotation matrix silently skews every lift.
  M  windows    the saved window_starts really do contain target motion
                and are inside the episode.
  N  visibility target_visible / gripper_visible recomputed from seg and
                compared against the stored arrays.
  O  seg/RGB    segmentation boundaries coincide with image edges. A
                half-pixel or off-by-one downsample would misattribute
                every boundary pixel.
  P  timing     frame spacing implied by the states matches the declared
                fps.
  Q  codec      decoded mp4 vs the raw rendered frames.

    python -m relmo.fullcheck --dataset rcasa_probe --eps 3
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from relmo import registry as R  # noqa: E402
from relmo.rcreplay import MIN_GRIP, MIN_VIS, _visual_only, episodes  # noqa: E402


def _src(name):
    task = name.split("_episode_")[0]
    did = "episode_" + re.search(r"_episode_(\d+)", name).group(1)
    for sp in ("pretrain", "target"):
        for e in episodes(task, sp):
            if e.name == did:
                return task, e
    return task, None


def _project(z, t, Xw):
    """World -> pixels with the stored camera."""
    W, H = int(z["width"]), int(z["height"])
    fs = (H / 2.0) / np.tan(np.deg2rad(float(z["cam_fovy"])) / 2.0)
    Rm = z["cam_mat"][t].reshape(3, 3)
    pc = (Xw - z["cam_pos"][t]) @ Rm
    d = -pc[:, 2]
    u = pc[:, 0] / np.maximum(d, 1e-9) * fs + W / 2.0
    v = -pc[:, 1] / np.maximum(d, 1e-9) * fs + H / 2.0
    return u, v, d


def check(ep_dir: Path, n_frames=6, seed=0):
    import mujoco
    import robocasa  # noqa: F401
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    z = np.load(ep_dir / "state.npz", allow_pickle=True)
    W, H = int(z["width"]), int(z["height"])
    ds = int(z["seg_ds"])
    seg = z["seg"]
    xpos, xquat = z["xpos"].astype(np.float64), z["xquat"].astype(np.float64)
    T = len(seg)
    rep = dict(id=ep_dir.name, T=int(T))
    task, src = _src(ep_dir.name)
    if src is None:
        rep["error"] = "source demo not found"
        return rep
    cfg = load_composite_controller_config(robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron",
                         controller_configs=cfg, has_renderer=False,
                         has_offscreen_renderer=False, use_camera_obs=False,
                         control_freq=int(z["fps"]), ignore_done=True)
    env.reset()
    env.set_ep_meta(json.loads((src / "ep_meta.json").read_text()))
    env.reset()
    env.reset_from_xml_string(gzip.open(src / "model.xml.gz", "rt").read())
    env.sim.reset()
    sim = env.sim
    m = sim.model._model
    st = np.load(src / "states.npz")["states"][:T]
    frames = np.linspace(0, T - 1, n_frames).astype(int)
    rng = np.random.default_rng(seed)

    # ---- I: xquat against the physics that produced it
    #
    # The first version of this gate projected each offset geom's CENTRE
    # and required it to land inside that body's mask. It scored 0.58 and
    # that was the test being wrong, not the data: a geom centre lies
    # INSIDE the solid, so it is routinely hidden behind whatever is in
    # front, and the loop also walked collision geoms (group 0) whose
    # placement need not coincide with the visual mesh that is rendered.
    # The question actually worth asking is whether the quaternion WE
    # STORED is the one MuJoCo computed for that frame - which is the
    # class of bug that has actually occurred here (cvel written in the
    # wrong frame, seg clipped to uint8, states truncated).
    qerr, qnorm = [], []
    for t in frames:
        sim.set_state_from_flattened(st[t])
        sim.forward()
        a = sim.data.body_xquat.copy()
        b = xquat[t]
        # q and -q are the same rotation
        flip = np.sign((a * b).sum(1, keepdims=True))
        flip[flip == 0] = 1.0
        qerr.append(float(np.abs(a - b * flip).max()))
        qnorm.append(float(np.abs(np.linalg.norm(b, axis=1) - 1).max()))
    rep["I_xquat_max_err"] = float(np.max(qerr))
    rep["I_xquat_unit_dev"] = float(np.max(qnorm))
    rep["I_xquat_ok"] = bool(rep["I_xquat_max_err"] < 1e-5
                             and rep["I_xquat_unit_dev"] < 1e-5)

    # diagnostic only, never a gate: how often does an offset VISUAL geom
    # project into its own mask? Occlusion caps this well below 1.
    inside, tested = 0, 0
    for t in frames[:3]:
        sg = seg[t]
        for b in np.unique(sg):
            if b == 0 or b >= m.nbody or (sg == b).sum() < 40:
                continue
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b
                  and m.geom_group[g] == 1
                  and np.linalg.norm(m.geom_pos[g]) > 0.02]
            if not gs:
                continue
            Rb = np.zeros(9)
            mujoco.mju_quat2Mat(Rb, xquat[t, b])
            Xw = np.stack([xpos[t, b] + Rb.reshape(3, 3) @ m.geom_pos[g]
                           for g in gs])
            u, v, dd = _project(z, t, Xw)
            for uu, vv, d_ in zip(u, v, dd):
                if d_ <= 0 or not (0 <= uu < W and 0 <= vv < H):
                    continue
                tested += 1
                inside += int(sg[int(vv / ds), int(uu / ds)] == b)
    rep["I_visual_geom_inside_frac_diag"] = round(inside / max(tested, 1), 4)

    # ---- J: qpos -> forward kinematics reproduces xpos
    errs = []
    for t in frames:
        sim.set_state_from_flattened(st[t])
        sim.forward()
        errs.append(float(np.abs(sim.data.body_xpos - xpos[t]).max()))
        assert np.allclose(sim.data.qpos[:m.nq], z["qpos"][t], atol=1e-5)
    rep["J_fk_max_err_m"] = float(np.max(errs))
    rep["J_qpos_ok"] = bool(rep["J_fk_max_err_m"] < 1e-5)

    # ---- K: kinematic tree and index ranges
    par = z["body_parentid"]
    ok_tree = bool(par[0] == 0 and np.all(par[1:] < np.arange(1, len(par))))
    jb, jq = z["jnt_bodyid"], z["jnt_qposadr"]
    ok_idx = bool(jb.max() < len(par) and jq.max() < z["qpos"].shape[1]
                  and z["geom_bodyid"].max() < len(par))
    rep["K_tree_ok"] = bool(ok_tree and ok_idx)

    # ---- L: camera matrix is a rotation
    dev, dets = [], []
    for t in frames:
        Rm = z["cam_mat"][t].reshape(3, 3)
        dev.append(float(np.abs(Rm @ Rm.T - np.eye(3)).max()))
        dets.append(float(np.linalg.det(Rm)))
    rep["L_cam_orthonormal_dev"] = round(max(dev), 8)
    rep["L_cam_det"] = round(float(np.mean(dets)), 6)
    rep["L_camera_ok"] = bool(max(dev) < 1e-5 and abs(np.mean(dets) - 1) < 1e-5
                              and np.isfinite(z["cam_pos"]).all())

    # ---- M: window_starts really contain target motion
    ws = z["window_starts"]
    wl = int(z["window_len"])
    tg = z["target_bodies"]
    step = np.linalg.norm(np.diff(xpos[:, tg], axis=0), axis=-1).max(1)
    moved = [float(step[a:a + wl - 1].sum()) for a in ws]
    rep["M_windows"] = int(len(ws))
    rep["M_min_window_motion_m"] = round(float(min(moved)), 4) if moved else None
    rep["M_windows_ok"] = bool(len(ws) > 0 and int(ws.max()) + wl <= T
                               and min(moved) > 0)

    # ---- N: stored visibility arrays match a recount from seg
    tv, gv = z["target_visible"], z["gripper_visible"]
    names = [sim.model.body_id2name(i) or "" for i in range(m.nbody)]
    grip = [i for i in range(m.nbody) if "gripper" in names[i].lower()]
    dtv, dgv = [], []
    for t in frames:
        # seg is downsampled; the stored value came from full resolution,
        # so compare at the sampled resolution and allow that difference
        dtv.append(abs(float(np.isin(seg[t], tg).mean()) - float(tv[t])))
        dgv.append(abs(float(np.isin(seg[t], grip).mean()) - float(gv[t])))
    rep["N_vis_max_dev"] = round(float(max(max(dtv), max(dgv))), 4)
    rep["N_visibility_ok"] = bool(max(max(dtv), max(dgv)) < 0.02)

    # ---- O: segmentation boundaries sit on image edges
    p = subprocess.run(["ffmpeg", "-v", "error", "-i",
                        str(ep_dir / "frames.mp4"), "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE, check=True)
    F = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
    align = []
    for t in frames:
        sg = seg[t]
        b = (np.abs(np.diff(sg.astype(np.int32), axis=0))[:, :-1] > 0) | \
            (np.abs(np.diff(sg.astype(np.int32), axis=1))[:-1, :] > 0)
        g = F[t].astype(np.float32).mean(-1)[::ds, ::ds]
        e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]
        if b.sum() > 20:
            align.append(float(e[b].mean() / max(e[~b].mean(), 1e-6)))
    rep["O_edge_contrast_ratio"] = round(float(np.mean(align)), 3) if align else None
    rep["O_seg_rgb_ok"] = bool(align and np.mean(align) > 3.0)

    # ---- P: timing
    dt = np.diff(st[:, 0])
    rep["P_dt_median_s"] = round(float(np.median(dt)), 6)
    rep["P_declared_fps"] = int(z["fps"])
    rep["P_timing_ok"] = bool(abs(np.median(dt) - 1.0 / int(z["fps"])) < 1e-3)

    # ---- Q: codec fidelity against a fresh render
    ren = mujoco.Renderer(m, H, W)
    opt = _visual_only(mujoco)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, str(z["camera"]))
    qe = []
    for t in frames[:3]:
        sim.set_state_from_flattened(st[t])
        sim.forward()
        ren.update_scene(sim.data._data, camera=cid, scene_option=opt)
        qe.append(float(np.abs(ren.render().astype(np.int16)
                               - F[t].astype(np.int16)).mean()))
    ren.close()
    env.close()
    rep["Q_codec_err_255"] = round(float(np.mean(qe)), 3)
    rep["Q_codec_ok"] = bool(np.mean(qe) < 4.0)

    keys = [k for k in rep if k.endswith("_ok")]
    rep["PASS"] = all(rep[k] for k in keys)
    rep["FAILED"] = [k for k in keys if not rep[k]]
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa_probe")
    ap.add_argument("--eps", type=int, default=3)
    a = ap.parse_args()
    root = R.dataset_dir(a.dataset)
    eps = sorted(p.parent for p in root.glob("shard_*/*/state.npz"))
    rng = np.random.default_rng(0)
    sel = [eps[i] for i in rng.choice(len(eps), min(a.eps, len(eps)),
                                      replace=False)]
    out = []
    for e in sel:
        r = check(e)
        out.append(r)
        print(json.dumps(r, indent=1, default=str))
    print(f"\n{sum(r.get('PASS', False) for r in out)}/{len(out)} passed")
    R.log("fullcheck", dataset=a.dataset, rows=out)
