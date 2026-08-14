"""Physics data generator for RelMo — generic rigid-body happening.

DESIGN RULE (owner, 2026-08-11): the robot-arm domain is SEALED FOR
EVAL. Nothing here contains a manipulator, a gripper, or a scripted
task. This generator produces *physics*, not tasks: things fall,
collide, slide, topple, come to rest on each other, get knocked by
other things. That is deliberate - if you simulate tasks you learn
task categories, which is the label route that provably does not
transfer. If you simulate physical happening, you learn the alphabet
(contact, support, rigid group, relative motion) that every domain is
written in.

Per episode it emits, versioned and immutable:
    frames.mp4        what a camera saw
    state.npz         xpos/xquat/xvel per body per frame  (privileged)
                      contact (T,B,B) bool + normal force  (privileged)
                      seg (T,H/2,W/2) uint8 body id per pixel (privileged)
Privileged state is TRAINING-ONLY. Nothing downstream of training may
read it, and the eval corpus never comes from here.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

GEN_VERSION = 1          # bump on ANY change to scene distribution
FPS = 20
SEG_DS = 2               # segmentation downsample factor

SHAPES = ("box", "sphere", "cylinder", "capsule")


def _obj(name, shape, size, pos, rgba, mass_scale):
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
            f'density="{300 * mass_scale:.0f}" friction="0.7 0.02 0.001"/>'
            f'</body>')


def scene(rng):
    """One random physical situation. The distribution deliberately
    mixes: resting stacks, free drops, launched projectiles, ramps
    and slides - so contact make/break, support and common motion all
    occur, in varied combinations, with no task structure."""
    n = int(rng.integers(2, 6))
    bodies, plan = [], []
    # a ground platform of random height/extent (some scenes have a
    # table-like support, some are floor-only)
    plat = rng.random() < 0.5
    pz = float(rng.uniform(0.10, 0.30)) if plat else 0.0
    extra = ""
    if plat:
        extra = (f'<geom name="plat" type="box" pos="0 0 {pz/2:.3f}" '
                 f'size="{rng.uniform(0.35,0.6):.3f} '
                 f'{rng.uniform(0.35,0.6):.3f} {pz/2:.3f}" '
                 f'rgba="0.55 0.45 0.35 1" friction="0.8 0.02 0.001"/>')
    if rng.random() < 0.25:      # a ramp: makes sliding/rolling happen
        extra += (f'<geom name="ramp" type="box" pos="'
                  f'{rng.uniform(-0.4,0.4):.3f} {rng.uniform(-0.4,0.4):.3f}'
                  f' {pz + 0.05:.3f}" size="0.25 0.15 0.02" '
                  f'euler="0 {rng.uniform(0.25,0.6):.2f} '
                  f'{rng.uniform(0,6.28):.2f}" rgba="0.5 0.5 0.6 1"/>')
    # objects live in a TIGHT cluster so they actually interact -
    # a scattered scene produces solitary falls and teaches nothing
    # about contact (measured: 1 interacting frame in 48 when spawn
    # radius was 0.28)
    cx, cy = float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.1, 0.1))
    placed = []
    for i in range(n):
        shape = SHAPES[int(rng.integers(len(SHAPES)))]
        size = float(rng.uniform(0.055, 0.105))
        mode = rng.random()
        stack_on = None
        if placed and mode < 0.30:
            # SUPPORT: put it directly on a previously placed object -
            # this is where resting-on / stacked / topple comes from
            stack_on = placed[int(rng.integers(len(placed)))]
            x = stack_on[0] + float(rng.normal(0, 0.012))
            y = stack_on[1] + float(rng.normal(0, 0.012))
            z = stack_on[2] + stack_on[3] + size + 0.004
            vel = [0, 0, 0, 0, 0, 0]
        elif mode < 0.55:                    # resting on the surface
            x = cx + float(rng.normal(0, 0.09))
            y = cy + float(rng.normal(0, 0.09))
            z = pz + size + 0.002
            vel = [0, 0, 0, 0, 0, 0]
        elif mode < 0.78:                    # dropped ONTO the cluster
            x = cx + float(rng.normal(0, 0.06))
            y = cy + float(rng.normal(0, 0.06))
            z = pz + float(rng.uniform(0.18, 0.45))
            vel = [0, 0, 0, 0, 0, 0]
        else:                                # launched AT the cluster
            ang = float(rng.uniform(0, 6.283))
            r0 = float(rng.uniform(0.30, 0.45))
            x, y = cx + r0 * np.cos(ang), cy + r0 * np.sin(ang)
            z = pz + size + float(rng.uniform(0.0, 0.12))
            sp = float(rng.uniform(1.2, 3.0))
            vel = [-np.cos(ang) * sp, -np.sin(ang) * sp,
                   float(rng.uniform(0.0, 0.5)),
                   float(rng.normal(0, 2)), float(rng.normal(0, 2)),
                   float(rng.normal(0, 2))]
        rgba = rng.uniform(0.15, 0.95, 3)
        bodies.append(_obj(f"o{i}", shape, size, (x, y, z), rgba,
                           float(rng.uniform(0.5, 2.0))))
        plan.append(vel)
        placed.append((x, y, z, size))
    az = float(rng.uniform(0, 360))
    el = float(rng.uniform(-45, -15))
    # close enough that a UNIFORM query grid lands on objects
    # (measured: at 1.4-2.4 m only 9 of 576 grid points hit a
    # body, so the grouping target was nearly empty)
    dist = float(rng.uniform(0.85, 1.35))
    xml = f"""<mujoco>
  <option timestep="0.004" cone="elliptic" impratio="5"/>
  <visual><global offwidth="640" offheight="480" azimuth="{az:.1f}"
    elevation="{el:.1f}"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
      rgb1="{rng.uniform(0.2,0.5):.2f} {rng.uniform(0.2,0.5):.2f} 0.6"
      rgb2="0.05 0.05 0.1" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" width="300"
      height="300" rgb1="{rng.uniform(0.3,0.6):.2f} 0.4 0.35"
      rgb2="{rng.uniform(0.4,0.7):.2f} 0.5 0.45"/>
    <material name="gridm" texture="grid" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light pos="{rng.uniform(-1,1):.2f} {rng.uniform(-1,1):.2f} 2.5"
      dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="4 4 .1" material="gridm"/>
    {extra}
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    return xml, plan, dist


def run_episode(seed, out_dir, seconds=2.4):
    """Simulate, render, and record privileged state."""
    import mujoco
    rng = np.random.default_rng(seed)
    xml, plan, dist = scene(rng)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    # launch velocities
    for i, vel in enumerate(plan):
        adr = m.jnt_dofadr[m.body_jntadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"o{i}")]]
        d.qvel[adr:adr + 6] = vel
    ren = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = dist
    cam.azimuth = float(rng.uniform(0, 360))
    cam.elevation = float(rng.uniform(-45, -15))
    cam.lookat[:] = [0, 0, 0.15]
    nb = m.nbody
    T = int(seconds * FPS)
    step_per = int(round((1.0 / FPS) / m.opt.timestep))
    xpos = np.zeros((T, nb, 3), np.float32)
    xquat = np.zeros((T, nb, 4), np.float32)
    xvel = np.zeros((T, nb, 6), np.float32)
    con = np.zeros((T, nb, nb), np.float32)
    segs = np.zeros((T, 480 // SEG_DS, 640 // SEG_DS), np.uint8)
    frames = []
    for t in range(T):
        for _ in range(step_per):
            mujoco.mj_step(m, d)
        xpos[t] = d.xpos
        xquat[t] = d.xquat
        for b in range(nb):
            xvel[t, b, :3] = d.cvel[b, 3:]
            xvel[t, b, 3:] = d.cvel[b, :3]
        for c in range(d.ncon):
            b1 = m.geom_bodyid[d.contact[c].geom1]
            b2 = m.geom_bodyid[d.contact[c].geom2]
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, f)
            mag = float(np.linalg.norm(f[:3]))
            con[t, b1, b2] = max(con[t, b1, b2], mag)
            con[t, b2, b1] = con[t, b1, b2]
        ren.disable_segmentation_rendering()
        ren.update_scene(d, camera=cam)
        frames.append(ren.render().copy())
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera=cam)
        s = ren.render()[..., 0]
        # geom id -> body id, background 0
        sg = np.zeros_like(s, dtype=np.uint8)
        valid = (s >= 0) & (s < m.ngeom)
        sg[valid] = m.geom_bodyid[s[valid]].astype(np.uint8)
        segs[t] = sg[::SEG_DS, ::SEG_DS]
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_mp4(frames, out_dir / "frames.mp4")
    body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                  or f"b{b}" for b in range(nb)]
    np.savez_compressed(
        out_dir / "state.npz", xpos=xpos, xquat=xquat, xvel=xvel,
        contact=con.astype(np.float16), seg=segs,
        body_names=np.array(body_names), fps=FPS, seed=seed,
        gen_version=GEN_VERSION)
    return dict(seed=int(seed), n_bodies=int(nb - 1), frames=T,
                moving=int((np.abs(xvel[:, 1:, :3]).max(0).max(1)
                            > 0.05).sum()),
                contacts=int((con[:, 1:, 1:] > 0).any(0).sum() // 2))


def _write_mp4(frames, path):
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "640x480", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt",
         "yuv420p", "-bf", "0", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
