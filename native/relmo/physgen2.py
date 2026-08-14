"""Physics generator v2 — driven manipulation-like motion + depth.

Three changes over v1, each answering a measured or reasoned defect:

1. DEPTH is recorded per frame. Contact and support are 3D facts; the
   image plane cannot express them (measured on the real corpus: box
   gap 0.00 median for BOTH touching and separated pairs). RelMo will
   be trained to lift tracks internally, supervised by this. Note the
   deliberate choice NOT to run a pretrained depth model at inference
   - measured, those carry a rest-on-surface prior that erases the
   lifted state exactly when it matters (true rises read +0.00).

2. ACTUATOR-DRIVEN scenes, with NO manipulator. v1 was ballistic -
   things fell and collided - while manipulation is quasi-static and
   driven. Here a pusher slides, a platform lifts, a paddle sweeps, a
   plunger presses. Same physical alphabet, manipulation-like motion
   statistics, and no arm anywhere (the arm domain is sealed).

   The LIFTER matters most: it carries objects upward while they rest
   on it, which is precisely the situation that collapsed hand-built
   common-fate grouping (a carried thing and its carrier move as
   one). Now it comes with ground-truth grouping attached.

3. Slower, longer episodes so a driven interaction has time to
   develop rather than resolving in three frames.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

GEN_VERSION = 2
FPS = 20
SEG_DS = 2
SHAPES = ("box", "sphere", "cylinder", "capsule")
DRIVERS = ("pusher", "lifter", "paddle", "plunger", "none")


def _obj(name, shape, size, pos, rgba, dens):
    if shape == "box":
        s = f"{size:.3f} {size * 0.9:.3f} {size:.3f}"
    elif shape == "sphere":
        s = f"{size:.3f}"
    elif shape == "cylinder":
        s = f"{size:.3f} {size:.3f}"
    else:
        s = f"{size * 0.7:.3f} {size:.3f}"
    return (f'<body name="{name}" pos="{pos[0]:.3f} {pos[1]:.3f} '
            f'{pos[2]:.3f}"><freejoint/>'
            f'<geom name="g_{name}" type="{shape}" size="{s}" '
            f'rgba="{rgba[0]:.2f} {rgba[1]:.2f} {rgba[2]:.2f} 1" '
            f'density="{dens:.0f}" friction="0.9 0.02 0.001"/></body>')


def scene(rng):
    n = int(rng.integers(2, 5))
    driver = DRIVERS[int(rng.integers(len(DRIVERS)))]
    pz = float(rng.uniform(0.12, 0.24))
    cx, cy = float(rng.uniform(-0.06, 0.06)), float(rng.uniform(-0.06, 0.06))
    parts = [f'<geom name="plat" type="box" pos="0 0 {pz/2:.3f}" '
             f'size="0.45 0.45 {pz/2:.3f}" rgba="0.55 0.45 0.35 1" '
             f'friction="0.9 0.02 0.001"/>']
    act, plan = [], []
    if driver == "pusher":
        # a plate that slides horizontally into the cluster
        ang = float(rng.uniform(0, 6.283))
        parts.append(
            f'<body name="drv" pos="{cx + 0.42*np.cos(ang):.3f} '
            f'{cy + 0.42*np.sin(ang):.3f} {pz + 0.055:.3f}" '
            f'euler="0 0 {ang:.3f}">'
            f'<joint name="dj" type="slide" axis="{-np.cos(ang):.3f} '
            f'{-np.sin(ang):.3f} 0" range="0 0.6"/>'
            f'<geom type="box" size="0.03 0.14 0.055" '
            f'rgba="0.75 0.25 0.2 1" friction="0.9 0.02 0.001"/></body>')
        act = ['<position joint="dj" kp="400" ctrlrange="0 0.6"/>']
        plan = [("ramp", 0.0, float(rng.uniform(0.30, 0.50)))]
    elif driver == "lifter":
        # a platform that rises CARRYING whatever rests on it - the
        # carried-with-carrier case that broke hand-built grouping
        parts.append(
            f'<body name="drv" pos="{cx:.3f} {cy:.3f} {pz + 0.02:.3f}">'
            f'<joint name="dj" type="slide" axis="0 0 1" '
            f'range="0 0.35"/>'
            f'<geom type="box" size="0.13 0.13 0.02" '
            f'rgba="0.2 0.55 0.75 1" friction="1.0 0.02 0.001"/></body>')
        act = ['<position joint="dj" kp="900" ctrlrange="0 0.35"/>']
        plan = [("ramp", 0.0, float(rng.uniform(0.12, 0.30)))]
    elif driver == "paddle":
        ang = float(rng.uniform(0, 6.283))
        parts.append(
            f'<body name="drv" pos="{cx + 0.30*np.cos(ang):.3f} '
            f'{cy + 0.30*np.sin(ang):.3f} {pz:.3f}">'
            f'<joint name="dj" type="hinge" axis="0 0 1" '
            f'range="-1.6 1.6"/>'
            f'<geom type="box" pos="0.09 0 0.06" size="0.09 0.02 0.06" '
            f'rgba="0.8 0.6 0.2 1" friction="0.9 0.02 0.001"/></body>')
        act = ['<position joint="dj" kp="300" ctrlrange="-1.6 1.6"/>']
        plan = [("ramp", float(rng.uniform(-1.2, -0.6)),
                 float(rng.uniform(0.6, 1.2)))]
    elif driver == "plunger":
        parts.append(
            f'<body name="drv" pos="{cx:.3f} {cy:.3f} {pz + 0.42:.3f}">'
            f'<joint name="dj" type="slide" axis="0 0 -1" '
            f'range="0 0.34"/>'
            f'<geom type="cylinder" size="0.05 0.06" '
            f'rgba="0.6 0.3 0.7 1" friction="0.9 0.02 0.001"/></body>')
        act = ['<position joint="dj" kp="600" ctrlrange="0 0.34"/>']
        plan = [("ramp", 0.0, float(rng.uniform(0.18, 0.32)))]
    placed, launch = [], []
    for i in range(n):
        shape = SHAPES[int(rng.integers(len(SHAPES)))]
        size = float(rng.uniform(0.05, 0.09))
        mode = rng.random()
        if placed and mode < 0.35:                # stacked on another
            b = placed[int(rng.integers(len(placed)))]
            x, y = b[0] + rng.normal(0, 0.010), b[1] + rng.normal(0, 0.010)
            z = b[2] + b[3] + size + 0.004
        elif mode < 0.85:                         # resting in reach
            x = cx + float(rng.normal(0, 0.075))
            y = cy + float(rng.normal(0, 0.075))
            z = pz + size + 0.002
        else:                                     # dropped in
            x = cx + float(rng.normal(0, 0.05))
            y = cy + float(rng.normal(0, 0.05))
            z = pz + float(rng.uniform(0.15, 0.35))
        parts.append(_obj(f"o{i}", shape, size, (float(x), float(y), z),
                          rng.uniform(0.15, 0.95, 3),
                          float(rng.uniform(200, 700))))
        placed.append((float(x), float(y), z, size))
        launch.append([0.0] * 6 if driver != "none" else
                      [float(rng.normal(0, 1.2)), float(rng.normal(0, 1.2)),
                       float(rng.uniform(0, 0.4)), 0.0, 0.0, 0.0])
    xml = f"""<mujoco>
  <option timestep="0.004" cone="elliptic" impratio="5"/>
  <visual><global offwidth="640" offheight="480"/>
    <quality shadowsize="2048"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
      rgb1="{rng.uniform(0.15,0.5):.2f} {rng.uniform(0.15,0.5):.2f} 0.55"
      rgb2="0.04 0.04 0.09" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" width="300"
      height="300" rgb1="{rng.uniform(0.25,0.55):.2f} 0.4 0.35"
      rgb2="{rng.uniform(0.4,0.7):.2f} 0.5 0.45"/>
    <material name="gridm" texture="grid" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light pos="{rng.uniform(-1,1):.2f} {rng.uniform(-1,1):.2f} 2.5"
      dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="4 4 .1" material="gridm"/>
    {''.join(parts)}
  </worldbody>
  {'<actuator>' + ''.join(act) + '</actuator>' if act else ''}
</mujoco>"""
    return xml, plan, launch, driver


def run_episode(seed, out_dir, seconds=3.6, scene_fn=None,
                gen_version=None):
    # scene_fn/gen_version let later generator versions (physgen3's
    # textured scenes) reuse this rollout unchanged - the CAMERA draw
    # order after scene() is part of the on-disk contract that pixelgt
    # relies on to reconstruct the camera from the seed alone
    import mujoco
    rng = np.random.default_rng(seed)
    xml, plan, launch, driver = (scene_fn or scene)(rng)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for i, v in enumerate(launch):
        if any(v):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"o{i}")
            adr = m.jnt_dofadr[m.body_jntadr[bid]]
            d.qvel[adr:adr + 6] = v
    ren = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = float(rng.uniform(0.9, 1.4))
    cam.azimuth = float(rng.uniform(0, 360))
    cam.elevation = float(rng.uniform(-42, -14))
    cam.lookat[:] = [0, 0, 0.18]
    nb, T = m.nbody, int(seconds * FPS)
    sp = int(round((1.0 / FPS) / m.opt.timestep))
    H2, W2 = 480 // SEG_DS, 640 // SEG_DS
    xpos = np.zeros((T, nb, 3), np.float32)
    xquat = np.zeros((T, nb, 4), np.float32)
    xvel = np.zeros((T, nb, 6), np.float32)
    con = np.zeros((T, nb, nb), np.float32)
    segs = np.zeros((T, H2, W2), np.uint8)
    deps = np.zeros((T, H2, W2), np.float16)
    frames = []
    for t in range(T):
        f = t / max(T - 1, 1)
        for k, (kind, lo, hi) in enumerate(plan):
            # slow, deliberate drive: manipulation timing, not impact
            d.ctrl[k] = lo + (hi - lo) * min(f / 0.75, 1.0)
        for _ in range(sp):
            mujoco.mj_step(m, d)
        xpos[t], xquat[t] = d.xpos, d.xquat
        xvel[t, :, :3], xvel[t, :, 3:] = d.cvel[:, 3:], d.cvel[:, :3]
        for c in range(d.ncon):
            b1 = m.geom_bodyid[d.contact[c].geom1]
            b2 = m.geom_bodyid[d.contact[c].geom2]
            fc = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, fc)
            v = float(np.linalg.norm(fc[:3]))
            con[t, b1, b2] = max(con[t, b1, b2], v)
            con[t, b2, b1] = con[t, b1, b2]
        ren.disable_depth_rendering()
        ren.disable_segmentation_rendering()
        ren.update_scene(d, camera=cam)
        frames.append(ren.render().copy())
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera=cam)
        s = ren.render()[..., 0]
        sg = np.zeros_like(s, np.uint8)
        ok = (s >= 0) & (s < m.ngeom)
        sg[ok] = m.geom_bodyid[s[ok]].astype(np.uint8)
        segs[t] = sg[::SEG_DS, ::SEG_DS]
        ren.disable_segmentation_rendering()
        ren.enable_depth_rendering()
        ren.update_scene(d, camera=cam)
        deps[t] = ren.render()[::SEG_DS, ::SEG_DS].astype(np.float16)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_mp4(frames, out_dir / "frames.mp4")
    np.savez_compressed(
        out_dir / "state.npz", xpos=xpos, xquat=xquat, xvel=xvel,
        contact=con.astype(np.float16), seg=segs, depth=deps,
        fps=FPS, seed=seed,
        gen_version=gen_version or GEN_VERSION, driver=driver)
    mov = int((np.abs(xvel[:, 1:, :3]).max(0).max(1) > 0.04).sum())
    return dict(seed=int(seed), n_bodies=int(nb - 1), frames=T,
                driver=driver, moving=mov,
                contact_frames=int((con[:, 1:, 1:] > 0).any((1, 2)).sum()))


def _write_mp4(frames, path):
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "-s", "640x480", "-r", str(FPS), "-i", "-", "-c:v",
         "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-bf", "0",
         str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
