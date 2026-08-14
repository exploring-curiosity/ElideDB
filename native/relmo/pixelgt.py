"""Pixel-exact ground truth from the simulator - the owner's "sim
state" made operational: for ANY pixel in ANY stored episode, the
exact 3D point it shows, and where that material point goes in every
other frame.

Nothing is regenerated. physgen2 already stored per-frame body poses
(xpos, xquat), half-res segmentation + metric depth, and the episode
SEED - and the camera was drawn from the same rng stream right after
scene(), so it is exactly reconstructible. From those:

  lift     pixel (u,v) at t0  ->  body id + body-FIXED 3D point
           (unproject through the depth map, then into body frame)
  track    body-fixed point   ->  float32 pixel track over all T,
           camera depth, and visibility by depth-buffer test

This engine has three consumers:
  1. training targets for the world model (clean physics, no tracker
     noise, no fp16 staircase),
  2. the tracker validation harness (trackval.py) - score the FROZEN
     CoTracker checkpoint against exact GT,
  3. tracker finetuning supervision (any query pixel -> GT track).

VALIDATE THE VALIDATOR: selfcheck() must pass before any number from
this file is believed - round-trip lift->project error at t0, and GT
staticness on bodies the sim says are at rest.

    python -m relmo.pixelgt --selfcheck --n 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import physgen2 as PG  # noqa: E402

SEG_DS = PG.SEG_DS
W, H = 640, 480


def _quat_mats(q):
    """(...,4) wxyz -> (...,3,3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R_ = np.empty(q.shape[:-1] + (3, 3), np.float64)
    R_[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R_[..., 0, 1] = 2 * (x * y - w * z)
    R_[..., 0, 2] = 2 * (x * z + w * y)
    R_[..., 1, 0] = 2 * (x * y + w * z)
    R_[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R_[..., 1, 2] = 2 * (y * z - w * x)
    R_[..., 2, 0] = 2 * (x * z - w * y)
    R_[..., 2, 1] = 2 * (y * z + w * x)
    R_[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R_


class EpisodeGT:
    """Camera + poses + depth for one stored episode."""

    def __init__(self, ep_dir: Path):
        import mujoco
        st = np.load(ep_dir / "state.npz")
        self.xpos = st["xpos"].astype(np.float64)
        self.xquat = st["xquat"].astype(np.float64)
        self.depth = st["depth"].astype(np.float32)
        self.seg = st["seg"]
        self.T = self.xpos.shape[0]
        self.Rb = _quat_mats(self.xquat)              # (T,nb,3,3)
        # exact camera reconstruction: same rng stream as run_episode -
        # scene() consumes its draws, then the camera params follow.
        # gen_version picks the scene function (v3 consumes one extra
        # draw for its texture fork - dispatching reproduces it).
        gv = int(st["gen_version"]) if "gen_version" in st.files else 2
        if gv >= 3:
            from relmo import physgen3 as PG3
            scene_fn = PG3.scene
        else:
            scene_fn = PG.scene
        rng = np.random.default_rng(int(st["seed"]))
        xml, _, _, self.driver = scene_fn(rng)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance = float(rng.uniform(0.9, 1.4))
        cam.azimuth = float(rng.uniform(0, 360))
        cam.elevation = float(rng.uniform(-42, -14))
        cam.lookat[:] = [0, 0, 0.18]
        ren = mujoco.Renderer(m, H, W)
        ren.update_scene(d, camera=cam)
        # 3x4 camera matrix, per the official MuJoCo tutorial recipe:
        # average the stereo GL cameras for pos / forward / up
        pos = np.mean([c.pos for c in ren.scene.camera], axis=0)
        z = -np.mean([c.forward for c in ren.scene.camera], axis=0)
        y = np.mean([c.up for c in ren.scene.camera], axis=0)
        rot = np.vstack((np.cross(y, z), y, z))
        fovy = m.vis.global_.fovy
        self.fs = (1.0 / np.tan(np.deg2rad(fovy) / 2)) * H / 2.0
        self.cx, self.cy = (W - 1) / 2.0, (H - 1) / 2.0
        self.rot, self.pos = rot, pos
        ren.close()

    # -- projection ---------------------------------------------------
    def project(self, Xw):
        """world (...,3) -> pixel (...,2), camera z (...,)"""
        Xc = (Xw - self.pos) @ self.rot.T
        zc = Xc[..., 2]
        u = -self.fs * Xc[..., 0] / zc + self.cx
        v = self.fs * Xc[..., 1] / zc + self.cy
        return np.stack([u, v], -1), zc

    def unproject(self, u, v, dist):
        """pixel + metric depth (renderer convention) -> world."""
        z = -dist
        x = (u - self.cx) * dist / self.fs
        y = -(v - self.cy) * dist / self.fs
        return np.stack([x, y, z], -1) @ self.rot + self.pos

    # -- lift / track -------------------------------------------------
    def lift(self, t0, u, v):
        """pixel at t0 -> (body_id, body-frame point) or None."""
        iy = min(max(int(v) // SEG_DS, 0), self.depth.shape[1] - 1)
        ix = min(max(int(u) // SEG_DS, 0), self.depth.shape[2] - 1)
        d = float(self.depth[t0, iy, ix])
        if not np.isfinite(d) or d <= 0 or d > 50:
            return None
        b = int(self.seg[t0, iy, ix])
        Xw = self.unproject(np.float64(u), np.float64(v), d)
        pb = self.Rb[t0, b].T @ (Xw - self.xpos[t0, b])
        # reject lifts through a half-res edge cell: round-trip must
        # land back on the same pixel
        px, _ = self.project(self.Rb[t0, b] @ pb + self.xpos[t0, b])
        if np.linalg.norm(px - [u, v]) > 1.5:
            return None
        return b, pb

    def track(self, b, pb, tol=0.03):
        """body-fixed point -> (T,2) float32 pixels, (T,) vis, (T,) z."""
        Xw = np.einsum("tij,j->ti", self.Rb[:, b], pb) + self.xpos[:, b]
        px, zc = self.project(Xw)
        dist = -zc
        iy = np.clip((px[:, 1] / SEG_DS).astype(int), 0,
                     self.depth.shape[1] - 1)
        ix = np.clip((px[:, 0] / SEG_DS).astype(int), 0,
                     self.depth.shape[2] - 1)
        dm = self.depth[np.arange(self.T), iy, ix]
        inb = ((px[:, 0] >= 0) & (px[:, 0] < W)
               & (px[:, 1] >= 0) & (px[:, 1] < H) & (zc < 0))
        vis = inb & (np.abs(dist - dm) < tol + 0.02 * dist)
        return px.astype(np.float32), vis, dist.astype(np.float32)

    def lift_track(self, t0, u, v):
        lt = self.lift(t0, u, v)
        if lt is None:
            return None
        px, vis, dist = self.track(*lt)
        return dict(body=lt[0], xy=px, vis=vis, dist=dist)


class KubricGT:
    """Same contract as EpisodeGT, for imported Kubric MOVi episodes.

    Conventions are Kubric's, taken from its own point-track code
    (challenges/point_tracking/dataset.py) rather than guessed - two
    of them are silent traps:
      DEPTH IS RAY LENGTH from the camera centre, not planar z. Using
        planar z bends every lifted point outward toward the frame
        edges, and nothing errors.
      NORMALISED INTRINSICS with a flipped y row: f = focal/sensor in
        [0,1] image space, principal point 0.5, camera looks down -Z
        with +Y up, so v grows downward in pixels.
    Segment 0 is BACKGROUND: those points are static in the world and
    still sweep across the image under MOVi-E's moving camera - which
    is exactly the ego-motion signal that a static-camera corpus
    cannot teach.
    """

    def __init__(self, ep_dir: Path):
        st = np.load(ep_dir / "state.npz")
        self.seg = st["seg"]
        self.depth = st["depth"].astype(np.float32)
        self.T = self.seg.shape[0]
        self.H, self.W = int(st["height"]), int(st["width"])
        self.cpos = st["cam_positions"].astype(np.float64)
        self.crot = _quat_mats(st["cam_quaternions"].astype(np.float64))
        f = float(st["focal_length"]) / float(st["sensor_width"])
        self.fx = f * self.W
        self.fy = f * (self.H / self.W) * self.H
        self.opos = st["obj_positions"].astype(np.float64)
        self.orot = _quat_mats(st["obj_quaternions"].astype(np.float64))
        self.driver = "kubric"

    def project(self, Xw, t):
        Xc = (Xw - self.cpos[t]) @ self.crot[t]
        z = Xc[..., 2]
        u = self.W * 0.5 + self.fx * Xc[..., 0] / (-z)
        v = self.H * 0.5 - self.fy * Xc[..., 1] / (-z)
        return np.stack([u, v], -1), z

    def unproject(self, u, v, dist, t):
        """pixel + RAY-LENGTH depth -> world."""
        d = np.stack([(u - self.W * 0.5) / self.fx,
                      -(v - self.H * 0.5) / self.fy,
                      -np.ones_like(np.asarray(u, np.float64))], -1)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return d * dist @ self.crot[t].T + self.cpos[t]

    def lift(self, t0, u, v):
        iy = min(max(int(v), 0), self.H - 1)
        ix = min(max(int(u), 0), self.W - 1)
        d = float(self.depth[t0, iy, ix])
        if not np.isfinite(d) or d <= 0:
            return None
        s = int(self.seg[t0, iy, ix])
        Xw = self.unproject(np.float64(u), np.float64(v), d, t0)
        if s == 0:                                   # world-static point
            return -1, Xw
        i = s - 1                                    # seg 1..N -> index
        if i >= len(self.opos):
            return None
        return i, self.orot[i, t0].T @ (Xw - self.opos[i, t0])

    def track(self, b, pb, tol=0.03):
        if b < 0:
            Xw = np.repeat(np.asarray(pb)[None], self.T, 0)
        else:
            Xw = (np.einsum("tij,j->ti", self.orot[b], pb)
                  + self.opos[b])
        px = np.zeros((self.T, 2), np.float32)
        dist = np.zeros(self.T, np.float32)
        for t in range(self.T):
            p, z = self.project(Xw[t], t)
            px[t] = p
            dist[t] = np.linalg.norm(Xw[t] - self.cpos[t])
            if z >= 0:
                dist[t] = -1.0
        iy = np.clip(px[:, 1].astype(int), 0, self.H - 1)
        ix = np.clip(px[:, 0].astype(int), 0, self.W - 1)
        dm = self.depth[np.arange(self.T), iy, ix]
        sm = self.seg[np.arange(self.T), iy, ix]
        inb = ((px[:, 0] >= 0) & (px[:, 0] < self.W)
               & (px[:, 1] >= 0) & (px[:, 1] < self.H) & (dist > 0))
        # occluded if something nearer is drawn there, or the pixel
        # belongs to a different object (Kubric's depth+segment test)
        vis = inb & (dist <= dm * (1 + tol) + tol) & (sm == (b + 1))
        return px, vis, dist

    def lift_track(self, t0, u, v):
        lt = self.lift(t0, u, v)
        if lt is None:
            return None
        px, vis, dist = self.track(*lt)
        return dict(body=lt[0], xy=px, vis=vis, dist=dist)


class RoboCasaGT:
    """MuJoCo episodes with a MOVING camera and articulated bodies.

    Same contract as EpisodeGT, three differences that matter:
      - the camera is stored PER FRAME (cam_pos/cam_mat), not
        reconstructed from a seed - RoboCasa has eye-in-hand cameras
        that ride the wrist, so there is no single static pose;
      - intrinsics come from cam_fovy;
      - bodies are ARTICULATED, so a point's body-frame coordinate is
        still constant but its parent's pose is not - which is exactly
        the signal the joint-discovery head is meant to find.
    """

    def __init__(self, ep_dir: Path):
        st = np.load(ep_dir / "state.npz")
        self.seg = st["seg"]
        self.depth = st["depth"].astype(np.float32)
        self.T = self.seg.shape[0]
        self.H, self.W = int(st["height"]), int(st["width"])
        self.ds = int(st["seg_ds"]) if "seg_ds" in st.files else 2
        self.cpos = st["cam_pos"].astype(np.float64)
        self.cmat = st["cam_mat"].astype(np.float64).reshape(-1, 3, 3)
        fovy = float(st["cam_fovy"])
        self.fs = (self.H / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.xpos = st["xpos"].astype(np.float64)
        self.xquat = st["xquat"].astype(np.float64)
        self.Rb = _quat_mats(self.xquat)
        self.driver = "robocasa"

    def project(self, Xw, t):
        Xc = (Xw - self.cpos[t]) @ self.cmat[t]      # world -> camera
        z = Xc[..., 2]
        u = self.cx + self.fs * Xc[..., 0] / (-z)
        v = self.cy - self.fs * Xc[..., 1] / (-z)
        return np.stack([u, v], -1), z

    def unproject(self, u, v, dist, t):
        """MuJoCo depth is PLANAR z (distance along -z), matching the
        physgen engine that validated to 0.000px."""
        x = (u - self.cx) * dist / self.fs
        y = -(v - self.cy) * dist / self.fs
        return np.stack([x, y, -dist], -1) @ self.cmat[t].T + self.cpos[t]

    def lift(self, t0, u, v):
        iy = min(max(int(v) // self.ds, 0), self.depth.shape[1] - 1)
        ix = min(max(int(u) // self.ds, 0), self.depth.shape[2] - 1)
        d = float(self.depth[t0, iy, ix])
        if not np.isfinite(d) or d <= 0 or d > 50:
            return None
        b = int(self.seg[t0, iy, ix])
        if b >= len(self.xpos[t0]):
            return None
        Xw = self.unproject(np.float64(u), np.float64(v), d, t0)
        pb = self.Rb[t0, b].T @ (Xw - self.xpos[t0, b])
        px, _ = self.project(self.Rb[t0, b] @ pb + self.xpos[t0, b], t0)
        if np.linalg.norm(px - [u, v]) > 1.5:
            return None
        return b, pb

    def track(self, b, pb, tol=0.03):
        Xw = np.einsum("tij,j->ti", self.Rb[:, b], pb) + self.xpos[:, b]
        px = np.zeros((self.T, 2), np.float32)
        dist = np.zeros(self.T, np.float32)
        for t in range(self.T):
            p, z = self.project(Xw[t], t)
            px[t], dist[t] = p, -z
        iy = np.clip((px[:, 1] / self.ds).astype(int), 0,
                     self.depth.shape[1] - 1)
        ix = np.clip((px[:, 0] / self.ds).astype(int), 0,
                     self.depth.shape[2] - 1)
        dm = self.depth[np.arange(self.T), iy, ix]
        sm = self.seg[np.arange(self.T), iy, ix]
        inb = ((px[:, 0] >= 0) & (px[:, 0] < self.W)
               & (px[:, 1] >= 0) & (px[:, 1] < self.H) & (dist > 0))
        vis = inb & (dist <= dm * (1 + tol) + tol) & (sm == b)
        return px, vis, dist

    def lift_track(self, t0, u, v):
        lt = self.lift(t0, u, v)
        if lt is None:
            return None
        px, vis, dist = self.track(*lt)
        return dict(body=lt[0], xy=px, vis=vis, dist=dist)


def open_gt(ep_dir: Path):
    """One entrance: physgen / Kubric / RoboCasa."""
    st = np.load(ep_dir / "state.npz")
    src = str(st["source"].item()) if "source" in st.files else "mujoco"
    if src == "kubric":
        return KubricGT(ep_dir)
    if src.startswith("robocasa"):
        # startswith, not ==: the replayer writes "robocasa_replay" and an
        # exact match fell through to EpisodeGT, which reconstructs a
        # camera from a generator seed these episodes do not carry. That
        # surfaced as KeyError('seed') rather than as wrong geometry, but
        # a near-miss source string is exactly the kind of thing that
        # silently returns a plausible wrong answer.
        return RoboCasaGT(ep_dir)
    return EpisodeGT(ep_dir)


def body_speed(ep_dir: Path, T: int):
    """(T, n+1) linear speed INDEXED BY SEGMENT VALUE, for either source.

    The two corpora disagree twice and both disagreements are silent:
    MuJoCo stores xvel as (T, nbody, 6) indexed by body id with 0 =
    world, while Kubric stores obj_velocities as (ninstance, T, 3)
    indexed from 0 with background carrying no entry at all. Indexing
    one like the other returns a plausible wrong body rather than an
    error, so every consumer goes through this instead.

    Row 0 is background and is always zero - a world-static point does
    not move even while the CAMERA does, which is exactly the case
    MOVi-E adds and physgen never had.
    """
    st = np.load(ep_dir / "state.npz")
    if "obj_velocities" in st.files:                 # Kubric
        v = np.linalg.norm(st["obj_velocities"][..., :3],
                           axis=-1)                  # (n,T)
        sp = np.zeros((T, v.shape[0] + 1), np.float32)
        sp[:, 1:] = v[:, :T].T
        return sp
    v = np.linalg.norm(st["xvel"][..., :3], axis=-1)  # (T,nb) MuJoCo
    return v[:T].astype(np.float32)


def selfcheck_kubric(dataset="movi_e", n=6, per=200, seed=0):
    """Validate against Kubric's OWN shipped answers.

    Two independent checks, because a self-consistent-but-wrong camera
    would pass the first alone:
      PROJECT  our projection of each instance centre vs the shipped
               instances/image_positions - this is ground truth we did
               not compute.
      LIFT     unproject a pixel then re-project it: must return to
               the same pixel.
    """
    man = R.read_manifest(dataset)
    eps = man.get("episodes", [])
    if not eps:
        return dict(dataset=dataset, passed=False, why="no episodes")
    rng = np.random.default_rng(seed)
    proj, rt, segok = [], [], []
    for e in [eps[i] for i in rng.choice(len(eps), min(n, len(eps)),
                                         replace=False)]:
        d = R.dataset_dir(dataset) / e["shard"] / e["id"]
        gt = KubricGT(d)
        st = np.load(d / "state.npz")
        ip = st["obj_image_positions"].astype(np.float64)
        for i in range(len(gt.opos)):
            for t in range(gt.T):
                p, z = gt.project(gt.opos[i, t], t)
                if z >= 0:
                    continue
                proj.append(float(np.linalg.norm(
                    p - ip[i, t] * [gt.W, gt.H])))
        for _ in range(per):
            u = float(rng.uniform(2, gt.W - 2))
            v = float(rng.uniform(2, gt.H - 2))
            lt = gt.lift(0, u, v)
            if lt is None:
                continue
            b, pb = lt
            Xw = (pb if b < 0 else gt.orot[b, 0] @ pb + gt.opos[b, 0])
            p, _ = gt.project(Xw, 0)
            rt.append(float(np.linalg.norm(p - [u, v])))
            px, vis, _ = gt.track(b, pb)
            if vis.sum() > 3:
                sm = gt.seg[np.arange(gt.T),
                            np.clip(px[:, 1].astype(int), 0, gt.H - 1),
                            np.clip(px[:, 0].astype(int), 0, gt.W - 1)]
                segok.append(float((sm[vis] == b + 1).mean()))
    proj, rt = np.array(proj), np.array(rt)
    rep = dict(dataset=dataset, episodes=min(n, len(eps)),
               project_vs_shipped_px=dict(
                   med=round(float(np.median(proj)), 4),
                   p99=round(float(np.percentile(proj, 99)), 4)),
               lift_roundtrip_px=dict(
                   med=round(float(np.median(rt)), 4),
                   p99=round(float(np.percentile(rt, 99)), 4)),
               seg_agreement=round(float(np.mean(segok)), 4)
               if segok else None,
               passed=bool(np.median(proj) < 0.01
                           and np.median(rt) < 0.6))
    R.log("pixelgt_selfcheck_kubric", **rep)
    return rep


# -- selfcheck: validate the validator --------------------------------
def selfcheck(dataset="physgen_v2", n=6, per=200, seed=0):
    man = R.read_manifest(dataset)
    rng = np.random.default_rng(seed)
    eps = [man["episodes"][i] for i in
           rng.choice(len(man["episodes"]), n, replace=False)]
    rt, drift, lifted, tried = [], [], 0, 0
    for e in eps:
        gt = EpisodeGT(R.dataset_dir(dataset) / e["shard"] / e["id"])
        st = np.load(R.dataset_dir(dataset) / e["shard"] / e["id"]
                     / "state.npz")
        sp = np.linalg.norm(st["xvel"][..., :3], axis=-1)  # (T,nb)
        for _ in range(per):
            u = float(rng.uniform(4, W - 4))
            v = float(rng.uniform(4, H - 4))
            tried += 1
            lt = gt.lift(0, u, v)
            if lt is None:
                continue
            lifted += 1
            b, pb = lt
            px, vis, _ = gt.track(b, pb)
            rt.append(float(np.linalg.norm(px[0] - [u, v])))
            # a body at rest for the whole episode must give a static
            # GT track - any drift here is OUR math, not physics
            if sp[:, b].max() < 0.003:
                stp = np.linalg.norm(np.diff(px, axis=0), axis=-1)
                if vis[:-1].all():
                    drift.append(float(stp.max()))
    rt, drift = np.array(rt), np.array(drift)
    rep = dict(dataset=dataset, episodes=n, tried=tried,
               lift_rate=round(lifted / max(tried, 1), 3),
               roundtrip_px=dict(med=round(float(np.median(rt)), 4),
                                 p99=round(float(np.percentile(rt, 99)), 4)),
               rest_gt_drift_px=dict(
                   med=round(float(np.median(drift)), 5) if len(drift) else None,
                   max=round(float(drift.max()), 5) if len(drift) else None,
                   n=len(drift)),
               passed=bool(np.median(rt) < 0.2
                           and (not len(drift) or drift.max() < 0.05)))
    R.log("pixelgt_selfcheck", **rep)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--dataset", default="physgen_v2")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--kubric", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        import json
        fn = selfcheck_kubric if a.kubric else selfcheck
        print(json.dumps(fn(a.dataset, a.n), indent=1))
