"""EMBODIMENT REGISTRY: structurally different arms, one adapter.

Cross-embodiment is a training-DATA property, not a labelled axis: the
same tasks executed by visibly different machines force any latent that
predicts well to encode what is common - the world's transitions - with
nobody ever naming "the arm". So the arms here are chosen for REAL
structural variety, not palette swaps:

    panda   7-DoF industrial, twin-tendon parallel gripper (existing)
    xarm7   7-DoF, different proportions, linkage-driven gripper
    vx300s  hobby-class ViperX: exposed servos, thin links, short jaw
    z1      Unitree Z1: 6-DoF metal arm, single-jaw claw

Everything that CAN be derived from the model is derived, because name
conventions differ per vendor and silent mismatches waste nights:
  - arm actuators are found via the joint each actuator drives
  - gripper open/close direction is MEASURED (settle at both ctrl
    extremes, larger finger spread = open)
  - fingertip offset is measured from finger geoms at home
Config only states what cannot be derived: model file, end-effector
body, finger bodies, a home pose, and which actuator is the gripper.

Keyframes are stripped from a working copy of each vendor XML: several
menagerie models ship keyframes sized to the bare robot, and adding
free blocks makes nq mismatch a compile error.

    python scripts/sim_arms.py --smoke            # all arms
    python scripts/sim_arms.py --smoke --arm z1
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEN = ROOT / "mujoco_menagerie"

ARMS = {
    "panda": dict(
        straddle=0.4, dir="franka_emika_panda", xml="panda.xml", ee="hand",
        fingers=("left_finger", "right_finger"), grip_act="actuator8",
        home={"joint1": 0, "joint2": -0.4, "joint3": 0, "joint4": -2.2,
              "joint5": 0, "joint6": 1.9, "joint7": 0.785}),
    "xarm7": dict(
        straddle=1.0, dir="ufactory_xarm7", xml="xarm7.xml", ee="link7",
        fingers=("left_finger", "right_finger"), grip_act="gripper",
        home={"joint1": 0, "joint2": -0.55, "joint3": 0, "joint4": 0.9,
              "joint5": 0, "joint6": 1.45, "joint7": 0}),
    "vx300s": dict(
        straddle=0.8, dir="trossen_vx300s", xml="vx300s.xml", ee="gripper_link",
        fingers=("left_finger_link", "right_finger_link"),
        grip_act="gripper",
        home={"waist": 0, "shoulder": -0.5, "elbow": 0.6,
              "forearm_roll": 0, "wrist_angle": 1.45, "wrist_rotate": 0}),
    "z1": dict(
        straddle=0.8, dir="unitree_z1", xml="z1_gripper.xml", ee="link06",
        fingers=("gripperMover", "link06"), grip_act="motorGripper",
        home={"joint1": 0, "joint2": 1.1, "joint3": -0.9,
              "joint4": 0.35, "joint5": 0, "joint6": 0}),
}


def stripped_model(name):
    """Vendor XML minus <keyframe> blocks, written beside the original
    so relative asset paths keep resolving."""
    a = ARMS[name]
    src = MEN / a["dir"] / a["xml"]
    dst = MEN / a["dir"] / f"_elide_{a['xml']}"
    txt = src.read_text()
    txt = re.sub(r"<keyframe>.*?</keyframe>", "", txt, flags=re.S)
    dst.write_text(txt)
    return dst.name


class GenericArm:
    """DLS-Jacobian IK to the configured ee body; everything else is
    read off the compiled model. Port of sim_stack.Arm with the Panda
    assumptions removed."""

    def __init__(self, m, d, name):
        cfg = ARMS[name]
        self.m, self.d, self.name = m, d, name
        self.ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg["ee"])
        assert self.ee >= 0, f"{name}: no body {cfg['ee']}"
        self.grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      cfg["grip_act"])
        assert self.grip >= 0, f"{name}: no actuator {cfg['grip_act']}"
        # arm actuators = every actuator except the gripper, via the
        # joint it drives; ordered by qpos address = kinematic order
        pairs = []
        for a in range(m.nu):
            if a == self.grip:
                continue
            j = m.actuator_trnid[a, 0]
            pairs.append((m.jnt_qposadr[j], a, j))
        pairs.sort()
        self.act = [a for _, a, _ in pairs]
        self.jnt = [j for _, _, j in pairs]
        self.qadr = [m.jnt_qposadr[j] for j in self.jnt]
        self.dof = [m.jnt_dofadr[j] for j in self.jnt]
        jn = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): i
              for i, j in enumerate(self.jnt)}
        self.home = np.zeros(len(self.jnt))
        for k, v in cfg["home"].items():
            assert k in jn, f"{name}: home names joint {k} not driven"
            self.home[jn[k]] = v
        fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
              for n in cfg["fingers"]]
        assert min(fb) >= 0, f"{name}: finger body missing"
        self.finger_geoms = [g for g in range(m.ngeom)
                             if m.geom_bodyid[g] in fb]
        self.grip_open, self.grip_close = self._measure_grip()
        self.to_home()
        self.hand = self.ee                   # sim_chains compatibility
        # how deep the tips straddle a block, in units of its half
        # height. 0.4 suits the Panda's long pads; the xArm's shorter
        # linkage fingers grip the top edge at 0.4 (measured: spread
        # closed to 0.056 on a 0.050 block ONLY at ~1.0) and foul the
        # table beyond ~1.2.
        self.straddle = ARMS[name].get("straddle", 0.4)

    def gripper(self, open_):
        self.d.ctrl[self.grip] = self.grip_open if open_ else self.grip_close

    def grip_ramp(self, frac):
        """close -> open in this arm's own ctrl units. The corpus code
        ramped a literal 0..255 (Panda units); other grippers span
        0.021..0.057 m or -1.52..0 rad, so the ramp must be relative."""
        self.d.ctrl[self.grip] = (self.grip_close
                                  + (self.grip_open - self.grip_close)
                                  * float(np.clip(frac, 0, 1)))

    # -- derived properties ------------------------------------------
    def _spread(self):
        """Distance BETWEEN the two finger bodies. Distance-from-wrist
        was wrong for the xArm: its linkage carries the fingers AWAY
        from the wrist as it closes, so that metric inverted open and
        close and the gripper opened onto every block."""
        m = self.m
        groups = {}
        for g in self.finger_geoms:
            groups.setdefault(m.geom_bodyid[g], []).append(
                self.d.geom_xpos[g])
        cs = [np.mean(v, axis=0) for v in groups.values()]
        if len(cs) < 2:
            return 0.0
        return float(np.linalg.norm(cs[0] - cs[1]))

    def _measure_grip(self):
        """Which ctrl extreme is OPEN, measured, not assumed."""
        lo, hi = self.m.actuator_ctrlrange[self.grip]
        out = {}
        for v in (lo, hi):
            mujoco.mj_resetData(self.m, self.d)
            for a, q, h in zip(self.act, self.qadr, self.home):
                self.d.qpos[q] = h
                self.d.ctrl[a] = h
            self.d.ctrl[self.grip] = v
            for _ in range(300):
                mujoco.mj_step(self.m, self.d)
            out[v] = self._spread()
        o = max(out, key=out.get)
        c = min(out, key=out.get)
        return float(o), float(c)

    def to_home(self):
        for a, q, v in zip(self.act, self.qadr, self.home):
            self.d.qpos[q] = v
            self.d.ctrl[a] = v
        self.q_cmd = self.home.copy()
        self.d.ctrl[self.grip] = self.grip_open
        mujoco.mj_forward(self.m, self.d)
        self.down_quat = self.d.xquat[self.ee].copy()
        tip_z = min(self.d.geom_xpos[g][2] for g in self.finger_geoms)
        self.tip_off = float(self.d.xpos[self.ee][2] - tip_z) + 0.010

    def open_(self):
        self.d.ctrl[self.grip] = self.grip_open

    def close_(self):
        self.d.ctrl[self.grip] = self.grip_close

    def step_ik(self, target, gain=4.0):
        m, d = self.m, self.d
        err_p = target - d.xpos[self.ee]
        rq = np.zeros(3)
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, d.xquat[self.ee])
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, self.down_quat, neg)
        mujoco.mju_quat2Vel(rq, rel, 1.0)
        err = np.concatenate([err_p, 0.15 * rq])
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, self.ee)
        J = np.vstack([jacp, jacr])[:, self.dof]
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + 1e-4 * np.eye(6), err)
        self.q_cmd = self.q_cmd + np.clip(gain * dq * m.opt.timestep,
                                          -0.05, 0.05)
        lo = m.jnt_range[self.jnt, 0]
        hi = m.jnt_range[self.jnt, 1]
        limited = m.jnt_limited[self.jnt].astype(bool)
        self.q_cmd[limited] = np.clip(self.q_cmd[limited],
                                      lo[limited], hi[limited])
        for a, v in zip(self.act, self.q_cmd):
            self.d.ctrl[a] = v
        mujoco.mj_step(m, d)
        return float(np.linalg.norm(err_p))

    def touching(self, geom_id):
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g = (c.geom1, c.geom2)
            if geom_id in g and (g[0] in self.finger_geoms
                                 or g[1] in self.finger_geoms):
                return True
        return False


# ------------------------------------------------------------------ smoke

def smoke(name, record=True):
    """One arm: reach 3 zones, pick a block, stack it, verify by the
    same settled-contact rule the corpus uses. Reports wall-clock."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import sim_stack

    rng = np.random.default_rng(0)
    inc = stripped_model(name)
    old_scene, old_dir = sim_stack.SCENE, sim_stack.PANDA_DIR
    sim_stack.SCENE = sim_stack.SCENE.replace(
        '<include file="panda.xml"/>', f'<include file="{inc}"/>')
    sim_stack.PANDA_DIR = MEN / ARMS[name]["dir"]
    try:
        path, meta, _ = sim_stack.build_scene(rng, 2)
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
    finally:
        sim_stack.SCENE, sim_stack.PANDA_DIR = old_scene, old_dir
    arm = GenericArm(m, d, name)
    print(f"[{name}] {len(arm.act)} arm actuators, grip open={arm.grip_open:g}"
          f" close={arm.grip_close:g}, tip_off {arm.tip_off:.3f}")

    t0 = time.time()
    frames = []
    ren = mujoco.Renderer(m, 240, 320) if record else None

    def sim(seconds, target=None, gain=4.0):
        n = int(seconds / m.opt.timestep)
        r = np.inf
        for i in range(n):
            if target is not None:
                r = arm.step_ik(np.asarray(target, float), gain)
            else:
                mujoco.mj_step(m, d)
            if ren and i % int(1 / (10 * m.opt.timestep)) == 0:
                ren.update_scene(d, camera="cam0")
                frames.append(ren.render().copy())
        return r

    # reach check across the zone extremes
    from sim_chains import ZONES
    ok_reach = True
    for zx, zy in (ZONES["left_near"], ZONES["right_far"], ZONES["mid_far"]):
        r = sim(2.2, (zx, zy, 0.2 + 0.12 + arm.tip_off))
        print(f"[{name}]   reach ({zx:.2f},{zy:.2f}) residual {r * 100:.1f}cm")
        ok_reach &= r < 0.03

    # pick block 0, stack on block 1 - the corpus primitives
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"blk{i}")
           for i in range(2)]
    gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"gblk{i}")
           for i in range(2)]
    bp = d.xpos[bid[0]].copy()
    top0 = d.geom_xpos[gid[0]][2] + m.geom_size[gid[0]][
        2 if m.geom_type[gid[0]] == mujoco.mjtGeom.mjGEOM_BOX else 1]
    arm.open_()
    sim(2.0, (bp[0], bp[1], top0 + arm.tip_off + 0.10))
    sim(1.6, (bp[0], bp[1], top0 + arm.tip_off - 0.012), gain=2.5)
    arm.close_()
    sim(0.8, (bp[0], bp[1], top0 + arm.tip_off - 0.012), gain=1.0)
    z_before = d.xpos[bid[0]][2]
    sim(1.6, (bp[0], bp[1], 0.55), gain=3.0)
    carried = (d.xpos[bid[0]][2] - z_before) > 0.10 and arm.touching(gid[0])
    tp = d.xpos[bid[1]].copy()
    top1 = d.geom_xpos[gid[1]][2] + m.geom_size[gid[1]][
        2 if m.geom_type[gid[1]] == mujoco.mjtGeom.mjGEOM_BOX else 1]
    half0 = m.geom_size[gid[0]][
        2 if m.geom_type[gid[0]] == mujoco.mjtGeom.mjGEOM_BOX else 1]
    sim(2.2, (tp[0], tp[1], 0.55), gain=3.0)
    sim(1.8, (tp[0], tp[1], top1 + 2 * half0 + arm.tip_off + 0.002),
        gain=2.0)
    arm.open_()
    sim(0.4, None)
    sim(1.4, (tp[0], tp[1], 0.55))
    sim(sim_stack.SETTLE_S, None)
    # settled-contact verdict, same species as the corpus check
    on = False
    for i in range(d.ncon):
        c = d.contact[i]
        if {c.geom1, c.geom2} == {gid[0], gid[1]}:
            on = True
    dz = d.xpos[bid[0]][2] - d.xpos[bid[1]][2]
    stacked = on and dz > half0 * 0.8
    dt = time.time() - t0
    simsec = d.time
    print(f"[{name}]   carried={carried}  stacked={stacked}  "
          f"({simsec:.0f}s sim in {dt:.0f}s wall = {simsec / dt:.1f}x)")
    if ren:
        out = ROOT / f"eval/smoke_{name}.mp4"
        import imageio
        imageio.mimwrite(out, frames, fps=10, codec="libx264",
                         quality=7)
        print(f"[{name}]   wrote {out.name} ({len(frames)} frames)")
        ren.close()
    return dict(name=name, reach=ok_reach, carried=bool(carried),
                stacked=bool(stacked), xreal=simsec / dt)


def main():
    args = sys.argv[1:]
    names = [a for a in args if a in ARMS] or \
        ([args[args.index("--arm") + 1]] if "--arm" in args else list(ARMS))
    res = []
    for n in names:
        try:
            res.append(smoke(n))
        except Exception as e:                            # noqa: BLE001
            import traceback
            traceback.print_exc()
            res.append(dict(name=n, error=f"{type(e).__name__}: {e}"))
    print("\nsummary")
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<8} ERROR {r['error'][:80]}")
        else:
            print(f"  {r['name']:<8} reach={r['reach']} carried={r['carried']}"
                  f" stacked={r['stacked']}  {r['xreal']:.1f}x realtime")


if __name__ == "__main__":
    main()
