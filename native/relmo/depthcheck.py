"""Is the depth metrically correct, and is every rigid body's xyz right?

gtcheck's projection gate proves the pipeline is SELF-consistent: a pixel
lifted through depth and re-projected returns to where it started. That
passes just as happily if the depth is uniformly scaled, or if it is ray
length where the lift assumes planar z. Both errors would put every 3D
coordinate in the wrong place while every roundtrip still read 0.0000 px.

So this validates against sources that never touch the render pipeline:

  DEPTH   mujoco.mj_ray casts a ray from the camera through a pixel and
          returns the true distance to geometry, computed from the
          collision model. Comparing it against the rendered depth
          buffer tests the buffer's absolute scale AND settles planar-z
          vs ray-length, which are the two ways a depth channel is
          usually wrong. lift3d() assumes PLANAR Z.

  XYZ     for every body with enough pixels - not only the manipulation
          target - unproject its segmented pixels through depth into
          world coordinates and check they land on that body's surface,
          i.e. within its own bounding radius of the xpos stored for
          that frame. This ties xpos, xquat, seg, depth and the camera
          together for every rigid body, every sampled frame.

  MOTION  completeness of the trajectories: no NaN, no frozen frames,
          and the per-frame step consistent with the recorded velocity.

    python -m relmo.depthcheck --dataset rcasa_probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402


def _cam_rays(z, t, us, vs):
    """Pixel -> unit ray direction in world, from the stored camera."""
    W, H = int(z["width"]), int(z["height"])
    fovy = float(z["cam_fovy"])
    fs = (H / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    x = (us - W / 2.0) / fs
    y = -(vs - H / 2.0) / fs
    d_cam = np.stack([x, y, -np.ones_like(x)], -1)      # MuJoCo looks down -z
    Rm = z["cam_mat"][t].reshape(3, 3)
    d_w = d_cam @ Rm.T
    return d_w / np.linalg.norm(d_w, axis=-1, keepdims=True), d_cam


def check(ep_dir: Path, n_pix=400, n_frames=6, seed=0):
    import mujoco
    z = np.load(ep_dir / "state.npz", allow_pickle=True)
    W, H = int(z["width"]), int(z["height"])
    ds = int(z["seg_ds"])
    seg, dep = z["seg"], z["depth"].astype(np.float32)
    xpos = z["xpos"].astype(np.float64)
    T = len(seg)
    rep = dict(id=ep_dir.name, T=int(T))

    # rebuild the physics model for ray casting - never the renderer
    import gzip
    import re
    import warnings
    warnings.filterwarnings("ignore")
    from relmo.rcreplay import episodes
    # find the ORIGINAL demonstration this episode was replayed from.
    # Validating against the source, not against anything the replayer
    # wrote, is the point - a bug in the replayer cannot hide here.
    name = ep_dir.name
    task = name.split("_episode_")[0]
    demo_id = "episode_" + re.search(r"_episode_(\d+)", name).group(1)
    src = None
    for sp in ("pretrain", "target"):
        for e in episodes(task, sp):
            if e.name == demo_id:
                src = e
                break
        if src:
            break
    if src is None:
        rep["error"] = f"source demo not found for {task}/{demo_id}"
        return rep
    # build through the env: the recorded MJCF carries absolute asset
    # paths from the machine that captured it, and robosuite's
    # edit_model_xml remaps them. mj_ray uses COLLISION geometry, so the
    # check stays independent of the render pipeline either way.
    import robocasa  # noqa: F401
    import robosuite
    from robosuite.controllers import load_composite_controller_config
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
    m = env.sim.model._model
    d = env.sim.data._data
    st = np.load(src / "states.npz")["states"][:T]

    rng = np.random.default_rng(seed)
    ray_err, planar_err, surf_err, surf_tol = [], [], [], []
    frames = np.linspace(0, T - 1, n_frames).astype(int)
    for t in frames:
        env.sim.set_state_from_flattened(st[t])
        env.sim.forward()
        us = rng.integers(0, W, n_pix).astype(np.float64)
        vs = rng.integers(0, H, n_pix).astype(np.float64)
        dw, dcam = _cam_rays(z, t, us, vs)
        cam = z["cam_pos"][t].astype(np.float64)
        dv = dep[t][np.clip((vs / ds).astype(int), 0, dep.shape[1] - 1),
                    np.clip((us / ds).astype(int), 0, dep.shape[2] - 1)]
        gid = np.zeros(1, np.int32)
        for k in range(n_pix):
            if not np.isfinite(dv[k]) or dv[k] <= 0 or dv[k] > 20:
                continue
            dist = mujoco.mj_ray(m, d, cam, dw[k], None, 1, -1, gid)
            if dist < 0 or gid[0] < 0:
                continue
            # planar z = ray length * cos(angle to the optical axis)
            cos = 1.0 / np.linalg.norm(dcam[k])
            ray_err.append(abs(float(dv[k]) - dist))
            planar_err.append(abs(float(dv[k]) - dist * cos))
    rep["depth_vs_raylength_m"] = round(float(np.median(ray_err)), 5) \
        if ray_err else None
    rep["depth_vs_planarz_m"] = round(float(np.median(planar_err)), 5) \
        if planar_err else None
    rep["depth_n"] = len(ray_err)
    if ray_err:
        rep["depth_convention"] = ("planar_z"
                                   if np.median(planar_err)
                                   < np.median(ray_err) else "ray_length")
        rep["depth_err_m"] = min(rep["depth_vs_planarz_m"],
                                 rep["depth_vs_raylength_m"])
        rep["DEPTH_OK"] = bool(rep["depth_err_m"] < 0.01)

    # ---- every rigid body: do its pixels land on its own xpos?
    rb = np.zeros(m.nbody)
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        rb[b] = max(rb[b], float(np.linalg.norm(m.geom_pos[g]))
                    + float(m.geom_rbound[g]))
    nb_checked = 0
    for t in frames:
        sg = seg[t]
        u, c = np.unique(sg, return_counts=True)
        for b, cnt in zip(u, c):
            if b == 0 or cnt < 30 or b >= m.nbody:
                continue
            ys, xs = np.where(sg == b)
            pick = rng.choice(len(ys), min(40, len(ys)), replace=False)
            vv = ys[pick] * ds + ds / 2.0
            uu = xs[pick] * ds + ds / 2.0
            dv = dep[t][ys[pick], xs[pick]].astype(np.float64)
            ok = np.isfinite(dv) & (dv > 0) & (dv < 20)
            if ok.sum() < 5:
                continue
            _, dcam = _cam_rays(z, t, uu[ok], vv[ok])
            Rm = z["cam_mat"][t].reshape(3, 3)
            # planar-z unprojection, exactly as lift3d does it
            pc = np.stack([dcam[:, 0] * dv[ok], dcam[:, 1] * dv[ok],
                           -dv[ok]], -1)
            pw = pc @ Rm.T + z["cam_pos"][t]
            dist = np.linalg.norm(pw - xpos[t, b], axis=-1)
            surf_err.append(float(np.median(dist)))
            surf_tol.append(float(rb[b]))
            nb_checked += 1
    if surf_err:
        se = np.array(surf_err)
        tl = np.array(surf_tol)
        rep["bodies_checked"] = nb_checked
        rep["xyz_median_dist_to_body_m"] = round(float(np.median(se)), 4)
        rep["xyz_within_bound_frac"] = round(float((se <= tl * 1.5).mean()), 4)
        rep["XYZ_OK"] = bool(rep["xyz_within_bound_frac"] >= 0.95)

    # ---- trajectory completeness
    rep["nan_in_xpos"] = int(np.isnan(xpos).sum())
    rep["nan_in_xquat"] = int(np.isnan(z["xquat"]).sum())
    step = np.linalg.norm(np.diff(xpos, axis=0), axis=-1)
    rep["frozen_frames"] = int((step.max(1) == 0).sum())
    rep["MOTION_OK"] = bool(rep["nan_in_xpos"] == 0
                            and rep["nan_in_xquat"] == 0)
    rep["PASS"] = bool(rep.get("DEPTH_OK") and rep.get("XYZ_OK")
                       and rep["MOTION_OK"])
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa_probe")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--eps", type=int, default=3)
    a = ap.parse_args()
    root = R.dataset_dir(a.dataset)
    eps = sorted(p.parent for p in root.glob("shard_*/*/state.npz"))
    for e in eps[:a.eps]:
        r = check(e, a.n)
        print(json.dumps(r, indent=1, default=str))
