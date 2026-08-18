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
        straddle=1.0, carry_gain=1.5, dir="ufactory_xarm7", xml="xarm7.xml",
        ee="link7",
        fingers=("left_finger", "right_finger"), grip_act="gripper",
        home={"joint1": 0, "joint2": -0.55, "joint3": 0, "joint4": 0.9,
              "joint5": 0, "joint6": 1.45, "joint7": 0}),
    "vx300s": dict(
        # cmd_lead 0.15 -> 0.05 (the Panda's value): letting the
        # command run 3x further ahead of the measured servo state
        # wound up into bang-bang thrash - MEASURED 2.9 reversals/s
        # and 10x the Panda's jerk, visible as constant shaking.
        # 0.05 costs travel speed, which time_scale pays back.
        # NOTE 2026-08-10: planned=True gives this arm by far the
        # smoothest motion of any (0.0 reversals/s, jerk 0.6, vs the
        # Panda's 0.72 / 9.8) and solve_ik reaches every zone to 1mm,
        # but the grasp DESCENT loop still ends 65-185mm off in xy and
        # tasks do not complete. Until that is closed the arm stays on
        # the greedy controller, which completes but visibly shakes
        # (4.75 reversals/s, jerk 43.7 - the owner called it out).
        # cmd_lead 0.15 -> 0.08 + time_scale: the wide clamp let the
        # command wind up ahead of the servo into bang-bang thrash
        # (MEASURED 4.75 reversals/s, jerk 43.7 vs the Panda's 0.72 /
        # 9.8 - the owner saw it as constant shaking). Halving it
        # halves the jerk; time_scale buys back the travel time the
        # gentler command costs, so tasks still complete.
        straddle=0.8, base_z=0.10, base_x=0.06, cmd_lead=0.08,
        carry_gain=1.5, place_tol=1.6, time_scale=1.35,
        root_body="base_link",
        base_bodies=("base_link", "shoulder_link"),
        dir="trossen_vx300s", xml="vx300s.xml", ee="gripper_link",
        fingers=("left_finger_link", "right_finger_link"),
        grip_act="gripper",
        # home sits in the FAR-REACH posture basin (shoulder forward,
        # elbow negative): kinematic IK reaches every zone exactly, but
        # greedy DLS from the old elbow-positive home ran wrist_angle
        # into its 2.23 limit and stalled 15cm short of the far zones.
        home={"waist": 0, "shoulder": 0.13, "elbow": -0.31,
              "forearm_roll": 0, "wrist_angle": 1.73, "wrist_rotate": 0}),
    "piper": dict(
        # AgiLex Piper: compact 6-DoF industrial, integrated parallel
        # jaw. Added 2026-08-10 as the third STRUCTURALLY different
        # arm (owner ruled out xarm7 as a panda look-alike, and the
        # vx300s controller oscillates 3x the Panda's reversal rate).
        # Home = elbow-down reach posture inside every joint range.
        straddle=0.5, gain_scale=8.0,
        root_body="base_link", base_bodies=("base_link",),
        dir="agilex_piper", xml="piper.xml", ee="link6",
        fingers=("link7", "link8"), grip_act="gripper",
        # home solved by multi-restart IK against all four extreme
        # zone targets at once (worst residual 2.5mm): a hand-guessed
        # home left greedy DLS 32cm short at the far zones - the same
        # local-minimum trap documented for z1 and vx300s.
        home={"joint1": 0.253, "joint2": 2.496, "joint3": -2.049,
              "joint4": -1.249, "joint5": -0.823, "joint6": -0.561}),
    "ur5e": dict(
        # UR5e + Robotiq 2F-85, composed via MJCF attach (ur5e_rq.xml
        # generated 2026-08-10): the structurally-different third arm
        # after xarm7 was ruled a Panda look-alike, piper could not
        # find collision-free grasp poses at bench height, z1's claw
        # roll is under-determined at 5-DoF, and the SO-arms cannot
        # reach the zones (0.38m vs 0.52 needed - measured).
        # mounted BACK from the workspace (base_x=-0.14): 0.34m is
        # inside a UR5e's crowded near-field - tool-down IK there
        # lands in far-away elbow branches (measured xy 0.7m misses);
        # at 0.48-0.66m every zone is in the sweet range.
        straddle=0.9, base_x=-0.10,
        dir="universal_robots_ur5e", xml="ur5e_rq.xml",
        ee="rq_base_mount", root_body="base",
        base_bodies=("base", "shoulder_link"),
        fingers=("rq_left_pad", "rq_right_pad"),
        grip_act="rq_fingers_actuator",
        # tool-down REQUIRES lift+elbow+wrist1 = -pi (the captured
        # down_quat is whatever home gives; at -pi/2 the gripper
        # descended 35 deg tilted and bulldozed blocks - seen on film)
        # home solved NUMERICALLY for tool-down at the workspace
        # centre (pos 1.6mm, axis 0.038) - two hand-guessed homes both
        # left the captured down_quat tilted (35 deg descent plows,
        # seen on film; the "fix" pointed the tool horizontally)
        home={"shoulder_pan_joint": 2.8851,
              "shoulder_lift_joint": -0.0121,
              "elbow_joint": -1.5832, "wrist_1_joint": 0.0627,
              "wrist_2_joint": -1.5669, "wrist_3_joint": -0.0347}),
    "z1": dict(
        straddle=0.8, lock_joints=("joint6",), ori_axis_only=True,
        joint_attrs={"jointGripper": 'damping="5" armature="0.05"'},
        dir="unitree_z1", xml="z1_gripper.xml", ee="link06",
        fingers=("gripperMover", "link06"), grip_act="motorGripper",
        # home = the mid-zone IK solution posture: the old shallower
        # home left greedy DLS in a local minimum 130-190mm short of
        # the far zones (no joint at a limit; the exact solution sits
        # one basin over at deeper joint2/joint3).
        home={"joint1": 0, "joint2": 1.61, "joint3": -1.39,
              "joint4": 0.35, "joint5": 0, "joint6": 0}),
}


def stripped_model(name):
    """Vendor XML minus <keyframe> blocks, written beside the original
    so relative asset paths keep resolving. Arms with base_z get their
    root body raised: the scene's table top (z=0.2) was placed for the
    Panda's tall base, and the vx300s home had its shoulder link
    EMBEDDED 12mm in the tabletop - the waist servo commanded -0.53
    moved 0.02 because the arm was physically wedged. Short arms are
    mounted on a pedestal exactly like real bench setups."""
    a = ARMS[name]
    src = MEN / a["dir"] / a["xml"]
    dst = MEN / a["dir"] / f"_elide_{a['xml']}"
    txt = src.read_text()
    txt = re.sub(r"<keyframe>.*?</keyframe>", "", txt, flags=re.S)
    # numerical-stability attributes for vendor joints: the z1 jaw is
    # a kp~1000 servo on 0.0003 kgm^2 - resonance faster than the 2ms
    # timestep, measured 22 rad/s limit cycle that shook the wrist and
    # tripped the thrash re-baseline 60% of steps. Armature is the
    # standard cure for stiff-servo instability at coarse timesteps.
    # per-arm servo gain scaling: the piper ships kp 10-80, which
    # cannot even hold the arm against gravity at bench height
    # (MEASURED: commanded poses missed by 300-535mm at settle - the
    # arm simply droops). Real deployments retune servo gains per
    # payload; same here, in the working copy only.
    gs = a.get("gain_scale")
    if gs:
        txt = re.sub(r'kp="([\d.]+)"',
                     lambda mm: f'kp="{float(mm.group(1)) * gs:g}"',
                     txt)
        txt = re.sub(r'kv="([\d.]+)"',
                     lambda mm: f'kv="{float(mm.group(1)) * (gs ** 0.5):g}"',
                     txt)
    for jname, attrs in a.get("joint_attrs", {}).items():
        pat = rf'(<joint name="{jname}" )'
        txt, n = re.subn(pat, rf'\1{attrs} ', txt, count=1)
        assert n == 1, f"{name}: joint {jname} not found for joint_attrs"
    bz = a.get("base_z", 0.0)
    bx = a.get("base_x", 0.0)
    if bz or bx:
        root = a["root_body"]
        pat = rf'(<body name="{root}")(?![^>]*\bpos=)'
        txt2, n = re.subn(pat, rf'\1 pos="{bx} 0 {bz}"', txt, count=1)
        assert n == 1, f"{name}: could not move root body {root}"
        txt = txt2
    dst.write_text(txt)
    return dst.name


def scene_patch(name, scene_txt):
    """Swap the Panda include for this arm and, when the arm is mounted
    on a pedestal (base_z), draw the pedestal so the film shows a
    mounted arm rather than one floating above the floor."""
    inc = stripped_model(name)
    out = scene_txt.replace('<include file="panda.xml"/>',
                            f'<include file="{inc}"/>')
    bz = ARMS[name].get("base_z", 0.0)
    if bz:
        bx = ARMS[name].get("base_x", 0.0)
        ped = (f'<geom name="pedestal" type="cylinder" '
               f'size="0.07 {bz / 2:.3f}" pos="{bx} 0 {bz / 2:.3f}" '
               f'rgba="0.30 0.30 0.33 1"/>\n    ')
        out = out.replace('<geom name="floor"', ped + '<geom name="floor"', 1)
    # the table's near edge (x=0.05) cuts 12mm into this arm's base
    # column - a HORIZONTAL interpenetration no pedestal height fixes.
    # qfrc_constraint exactly cancelled the waist servo and the arm
    # could not rotate at all. A real bench cuts the table around the
    # mount, so table x base-column collisions are excluded; every
    # WORKING link keeps full collision.
    bb = ARMS[name].get("base_bodies", ())
    if bb:
        exc = "".join(f'<exclude body1="table" body2="{b}"/>' for b in bb)
        out = out.replace("</mujoco>", f"<contact>{exc}</contact>\n</mujoco>")
    return out


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
        # lock_joints: held at home, excluded from IK. The z1's wrist
        # roll (joint6) is REDUNDANT with the waist for a down-pointing
        # claw; its orientation-error component flips sign near the
        # wrap and the kp=1000 servo limit-cycled at 20 rad/s - which
        # also tripped the qvel>3 thrash re-baseline EVERY step and
        # killed the integral action for the whole arm.
        locked = set(cfg.get("lock_joints", ()))
        self.locked = []
        for a in range(m.nu):
            if a == self.grip:
                continue
            j = m.actuator_trnid[a, 0]
            jn_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jn_name in locked:
                self.locked.append((a, m.jnt_qposadr[j],
                                    float(cfg["home"].get(jn_name, 0.0))))
                continue
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
            if k in locked:
                continue
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
        # loaded-transit gain. The xArm's pinch lets the block swing on
        # the 164mm hand->tip lever when the wrist tilts under fast
        # transit (measured +-50mm hand-to-block xy swing at gain 3.0;
        # the release loop then chased the swing into the table and
        # every far-zone place landed a zone short). 1.5 kills the
        # swing; smooth beats fast while holding, per embodiment.
        self.carry_gain = ARMS[name].get("carry_gain")
        # how far the integrated command may LEAD the measured joints.
        # 0.05 is the Panda's anti-windup clamp; the vx300s wrist_angle
        # servo is kp=8 and 0.05 rad of lead is only 0.4Nm against
        # ~0.7Nm of gravity on the gripper - the wrist drooped and
        # every z=0.40 target settled 60mm low. Weak-servo arms get a
        # wider lead so the integral action can actually cancel sag.
        self.cmd_lead = ARMS[name].get("cmd_lead", 0.05)
        # PLANNED mode: solve the pose offline, then walk the command
        # toward it in joint space. Arms whose geometry does not suit
        # greedy DLS (measured: vx300s) stop thrashing at the clamp.
        self.planned = ARMS[name].get("planned", False)
        self.plan_rate = ARMS[name].get("plan_rate", 0.02)
        self._plan_q = None
        self._plan_tgt = None
        # release-loop tolerance scale. The corpus gates (5/10/15mm)
        # encode the Panda's 1-2mm tracking; the vx300s hobby servos
        # track 10-20mm and the fine descent gate never opened - stacks
        # timed out and dropped on the base's edge. Tolerances scale
        # with the arm's measured tracking class, not per task.
        self.place_tol = ARMS[name].get("place_tol", 1.0)
        self.time_scale = ARMS[name].get("time_scale", 1.0)
        # with a locked roll the arm is 5-DoF; demanding full 3D
        # orientation makes DLS trade position for a roll error the
        # arm cannot produce (z1 stalled 130-190mm short, high). Axis-
        # only holds the TOOL AXIS down and leaves roll free.
        self.ori_axis_only = bool(ARMS[name].get("ori_axis_only"))

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
        for a, q, v in getattr(self, "locked", ()):
            self.d.qpos[q] = v
            self.d.ctrl[a] = v
        self.q_cmd = self.home.copy()
        self.d.ctrl[self.grip] = self.grip_open
        mujoco.mj_forward(self.m, self.d)
        self.down_quat = self.d.xquat[self.ee].copy()
        # body-frame axis that points straight DOWN at home (for the
        # axis-only orientation objective)
        R0 = self.d.xmat[self.ee].reshape(3, 3)
        self.tool_axis = R0.T @ np.array([0.0, 0.0, -1.0])
        tip_z = min(self.d.geom_xpos[g][2] for g in self.finger_geoms)
        self.tip_off = float(self.d.xpos[self.ee][2] - tip_z) + 0.010

    def open_(self):
        self.d.ctrl[self.grip] = self.grip_open

    def close_(self):
        self.d.ctrl[self.grip] = self.grip_close

    def solve_ik(self, target, restarts=6, iters=220):
        """Solve the WHOLE pose offline (scratch MjData, no physics),
        with random restarts to escape local minima. The per-step
        greedy DLS in step_ik cannot do this: it takes one gradient
        step per physics tick against a clamp, so a bad basin becomes
        a stall, and the stall becomes bang-bang thrash at the clamp -
        MEASURED as the vx300s's 4.75 reversals/s and 43.7 jerk (the
        Panda, whose geometry happens to suit the greedy path, sits at
        0.72 and 9.8). Kinematically the vx300s reaches every zone
        tool-down at 0mm; only the controller was failing."""
        import copy
        m = self.m
        d2 = mujoco.MjData(m)
        d2.qpos[:] = self.d.qpos
        lo = m.jnt_range[self.jnt, 0].copy()
        hi = m.jnt_range[self.jnt, 1].copy()
        lim = m.jnt_limited[self.jnt].astype(bool)
        lo[~lim], hi[~lim] = -np.pi, np.pi
        best_q, best_e, best_pos = None, 9e9, 9e9
        rng = np.random.default_rng(0)
        q_start = np.array([self.d.qpos[a] for a in self.qadr])
        for r in range(restarts):
            q = q_start.copy() if r == 0 else \
                lo + 0.05 + rng.random(len(self.qadr)) * (hi - lo - 0.1)
            for a, v in zip(self.qadr, q):
                d2.qpos[a] = v
            for _ in range(iters):
                mujoco.mj_kinematics(m, d2)
                mujoco.mj_comPos(m, d2)
                err_p = target - d2.xpos[self.ee]
                if self.ori_axis_only:
                    w = d2.xmat[self.ee].reshape(3, 3) @ self.tool_axis
                    rq = np.cross(w, np.array([0.0, 0.0, -1.0]))
                else:
                    rq = np.zeros(3)
                    neg = np.zeros(4)
                    mujoco.mju_negQuat(neg, d2.xquat[self.ee])
                    rel = np.zeros(4)
                    mujoco.mju_mulQuat(rel, self.down_quat, neg)
                    mujoco.mju_quat2Vel(rq, rel, 1.0)
                e = float(np.linalg.norm(err_p))
                if e < 0.002:
                    break
                err = np.concatenate([err_p, 0.30 * rq])
                jp = np.zeros((3, m.nv))
                jr = np.zeros((3, m.nv))
                mujoco.mj_jacBody(m, d2, jp, jr, self.ee)
                J = np.vstack([jp, jr])[:, self.dof]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6),
                                           err)
                qn = np.array([d2.qpos[a] for a in self.qadr])
                qn = np.clip(qn + 0.6 * dq, lo + 0.02, hi - 0.02)
                for a, v in zip(self.qadr, qn):
                    d2.qpos[a] = v
            mujoco.mj_kinematics(m, d2)
            mujoco.mj_comPos(m, d2)
            e = float(np.linalg.norm(target - d2.xpos[self.ee]))
            # prefer accurate AND near the current pose (short travel)
            qn = np.array([d2.qpos[a] for a in self.qadr])
            # rank by ACCURACY first; travel only breaks ties among
            # solutions that already reach (a composite score hid a
            # 15mm miss behind a short-travel bonus)
            score = e + (0.004 * float(np.linalg.norm(qn - q_start))
                         if e < 0.005 else 0.0)
            if score < best_e:
                best_e, best_q, best_pos = score, qn.copy(), e
            if e < 0.003:
                break
        return best_q, best_pos

    def step_ik(self, target, gain=4.0):
        if self.planned:
            return self._step_planned(target, gain)
        return self._step_greedy(target, gain)

    def _step_planned(self, target, gain=4.0):
        """Solve once per target, then approach it smoothly in joint
        space. Re-solves only when the target actually moves."""
        t = np.asarray(target, float)
        if self._plan_tgt is None \
                or float(np.linalg.norm(t - self._plan_tgt)) > 0.004:
            # CONTINUITY: once tracking, re-solve LOCALLY (warm start
            # from the current pose, no restarts). Multi-restart IK
            # mid-motion is free to return an equally-accurate but
            # far-away basin, and the arm then swings across the table
            # to reach the same point - measured as the descent ending
            # 65-185mm off in xy while the height was correct.
            first = self._plan_tgt is None
            q_goal, e = self.solve_ik(t, restarts=8 if first else 1)
            if (not first) and e > 0.02:
                # local refinement genuinely failed (target moved out
                # of this basin): fall back to a global solve
                q_goal, e = self.solve_ik(t, restarts=8)
            self._plan_q, self._plan_tgt = q_goal, t.copy()
        qm = np.array([self.d.qpos[a] for a in self.qadr])
        step = self.plan_rate * gain / 4.0
        dq = self._plan_q - self.q_cmd
        n = float(np.linalg.norm(dq))
        if n > step:
            dq = dq * (step / n)
        q = self.q_cmd + dq
        # never command further than the servo can currently follow
        q = np.clip(q, qm - self.cmd_lead, qm + self.cmd_lead)
        self.q_cmd = q
        for a, v in zip(self.act, q):
            self.d.ctrl[a] = v
        return float(np.linalg.norm(target - self.d.xpos[self.ee]))

    def _step_greedy(self, target, gain=4.0):
        """Faithful port of sim_stack.Arm.step_ik - the first port
        dropped the x20 command integration and stepped physics
        internally, which made every arm move at 1/20 the Panda's task
        speed. The corpus descent loop leads the hand by only 3.5mm per
        step, so at 1/20 speed the 2.5s descent timed out 35mm above
        the grasp depth and the close fired in FREE AIR - the measured
        xarm7 0/6, misdiagnosed twice (grip direction, straddle depth)
        before instrumentation showed no contact at all."""
        m, d = self.m, self.d
        err_p = target - d.xpos[self.ee]
        if self.ori_axis_only:
            w_cur = d.xmat[self.ee].reshape(3, 3) @ self.tool_axis
            rq = np.cross(w_cur, np.array([0.0, 0.0, -1.0]))
        else:
            rq = np.zeros(3)
            neg = np.zeros(4)
            mujoco.mju_negQuat(neg, d.xquat[self.ee])
            rel = np.zeros(4)
            mujoco.mju_mulQuat(rel, self.down_quat, neg)
            mujoco.mju_quat2Vel(rq, rel, 1.0)
        # 0.30 vs the Panda's 0.15: the xArm's block hung 60-100mm
        # behind the hand after far transits (wrist tilt x 164mm
        # tip lever + grip pivot), and far-zone places timed out on an
        # unreachable compensated target. Holding the wrist down twice
        # as hard halves the tilt; position residual stays <5mm at all
        # six zones (measured by the reach sweep).
        err = np.concatenate([err_p, 0.30 * rq])
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, self.ee)
        J = np.vstack([jacp, jacr])[:, self.dof]
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + 1e-4 * np.eye(6), err)
        qm = np.array([d.qpos[a] for a in self.qadr])
        # collision + windup clamp can lock into bang-bang thrash;
        # re-baseline on measured state to break the limit cycle
        if float(np.max(np.abs(d.qvel[self.dof]))) > 3.0:
            self.q_cmd = qm.copy()
        q = self.q_cmd + gain * dq * m.opt.timestep * 20
        # ERROR-SCHEDULED LEAD. A fixed clamp forces one compromise
        # for the whole motion: wide enough to cross the table fast
        # (vx300s 0.15) and the command winds up ahead of the servo
        # into bang-bang thrash near the target - MEASURED 2.9
        # reversals/s and 10x the Panda's jerk. Narrow enough to
        # settle (0.05) and the arm never reaches the 5mm descent
        # gate at all (grasp fails outright). Scheduling by DISTANCE
        # TO TARGET gives both: full lead while travelling, tight
        # lead in the last few centimetres where precision matters.
        q = np.clip(q, qm - self.cmd_lead, qm + self.cmd_lead)
        lo = m.jnt_range[self.jnt, 0]
        hi = m.jnt_range[self.jnt, 1]
        limited = m.jnt_limited[self.jnt].astype(bool)
        q[limited] = np.clip(q[limited], lo[limited] + 0.02,
                             hi[limited] - 0.02)
        self.q_cmd = q
        for a, v in zip(self.act, q):
            self.d.ctrl[a] = v
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
    old_scene, old_dir = sim_stack.SCENE, sim_stack.PANDA_DIR
    sim_stack.SCENE = scene_patch(name, sim_stack.SCENE)
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


# ---------------- spline motion (SDX_MOTION=spline) ----------------
# Minimum-jerk joint-space trajectories through IK waypoints - how
# industrial arms actually move. There is no feedback loop to
# oscillate and nothing to converge: the REFERENCE is smooth, the
# servo tracks it, so the motion is smooth by construction. This
# replaces only the TRANSITS; the slow contact-driven fine phases
# (final descent, release) keep their measured behaviour.

def mj_solve_ik(m, d, arm, target, restarts=6, iters=240):
    """Damped-LS IK with restarts on a scratch MjData. Position +
    full down-orientation for the Panda (parallel pads must keep
    their yaw against box faces), position + tool-axis-down for the
    generic arms (their solve already worked this way)."""
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    jnt = np.asarray(arm.jnt)
    lo = m.jnt_range[jnt, 0].copy()
    hi = m.jnt_range[jnt, 1].copy()
    lim = m.jnt_limited[jnt].astype(bool)
    lo[~lim], hi[~lim] = -np.pi, np.pi
    ee = getattr(arm, "ee", None) or arm.hand
    axis_mode = getattr(arm, "ori_axis_only", False) \
        or getattr(arm, "down_quat", None) is None
    tool_axis = getattr(arm, "tool_axis", np.array([0.0, 0.0, 1.0]))
    q_start = np.array([d.qpos[a] for a in arm.qadr])
    rng = np.random.default_rng(0)
    best_q, best_e, best_score = None, 9e9, 9e9
    for r in range(restarts):
        q = q_start.copy() if r == 0 else \
            lo + 0.05 + rng.random(len(q_start)) * (hi - lo - 0.1)
        for a, v in zip(arm.qadr, q):
            d2.qpos[a] = v
        for _ in range(iters):
            mujoco.mj_kinematics(m, d2)
            mujoco.mj_comPos(m, d2)
            err_p = target - d2.xpos[ee]
            if axis_mode:
                w = d2.xmat[ee].reshape(3, 3) @ tool_axis
                rq = np.cross(w, np.array([0.0, 0.0, -1.0]))
            else:
                rq = np.zeros(3)
                neg = np.zeros(4)
                mujoco.mju_negQuat(neg, d2.xquat[ee])
                rel = np.zeros(4)
                mujoco.mju_mulQuat(rel, arm.down_quat, neg)
                mujoco.mju_quat2Vel(rq, rel, 1.0)
            if float(np.linalg.norm(err_p)) < 0.0015 \
                    and float(np.linalg.norm(rq)) < 0.05:
                break
            err = np.concatenate([err_p, 0.5 * rq])
            jp = np.zeros((3, m.nv))
            jr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d2, jp, jr, ee)
            J = np.vstack([jp, jr])[:, arm.dof]
            dq = J.T @ np.linalg.solve(
                J @ J.T + 1e-4 * np.eye(6), err)
            qn = np.array([d2.qpos[a] for a in arm.qadr])
            qn = np.clip(qn + 0.5 * dq, lo + 0.02, hi - 0.02)
            for a, v in zip(arm.qadr, qn):
                d2.qpos[a] = v
        mujoco.mj_kinematics(m, d2)
        mujoco.mj_comPos(m, d2)
        e = float(np.linalg.norm(target - d2.xpos[ee]))
        qn = np.array([d2.qpos[a] for a in arm.qadr])
        # COLLISION-AWARE ranking: an accurate pose that drives a link
        # through the table is not a solution - the servo stalls on
        # the contact and the arm parks 100-500mm away (measured on
        # the piper: joint2 held 0.43 rad from its command by 6 active
        # contacts). Finger geoms are exempt: straddling the block at
        # grasp depth IS the goal.
        mujoco.mj_forward(m, d2)
        # exempt the whole GRIPPER subtree, not only the pads: at
        # grasp depth a linkage gripper's knuckles legitimately graze
        # the block, and counting that as collision pushed the solver
        # into far elbow branches (measured 0.7m xy misses on the
        # UR5e+2F85)
        hand_geoms = _hand_geoms(m, arm)
        armg = _arm_geoms(m, arm) - hand_geoms
        ncol = 0
        for ci in range(d2.ncon):
            g1, g2 = d2.contact[ci].geom1, d2.contact[ci].geom2
            if (g1 in armg or g2 in armg) \
                    and d2.contact[ci].dist < -1e-4:
                ncol += 1
        # branch-sticky: ALWAYS prefer solutions near the current
        # pose. A 6-DoF arm has many IK branches; hopping between
        # them makes the joint-space path sweep the gripper through
        # the table mid-transit (measured on the UR5e: endpoints
        # clean, arm stalled ~1m off on the way)
        # penalize only BRANCH-SCALE travel (>0.5 rad): a flat
        # travel term biased short solves 10-15mm off target, and a
        # biased descent clips the block and shoves it (measured:
        # spline on-target to 13mm, block scooted 60mm)
        travel = float(np.linalg.norm(qn - q_start))
        score = e + 0.30 * ncol + 0.020 * max(travel - 0.5, 0.0)
        if score < best_score:
            best_score, best_q, best_e = score, qn.copy(), e
        if e < 0.003 and ncol == 0 and r == 0:
            break
    return best_q, best_e


_AG_CACHE = {}
_HG_CACHE = {}


def _hand_geoms(m, arm):
    """Geoms at or below the end-effector body (the whole hand)."""
    key = (id(m), id(arm))
    if key in _HG_CACHE:
        return _HG_CACHE[key]
    ee = getattr(arm, "ee", None) or arm.hand
    hand_bodies = set()
    for bb in range(m.nbody):
        anc = bb
        while anc != 0:
            if anc == ee:
                hand_bodies.add(bb)
                break
            anc = m.body_parentid[anc]
    gs = {g for g in range(m.ngeom)
          if int(m.geom_bodyid[g]) in hand_bodies}
    _HG_CACHE[key] = gs
    return gs


def _arm_geoms(m, arm):
    """Geom ids belonging to the arm's kinematic chain (cached)."""
    key = (id(m), id(arm))
    if key in _AG_CACHE:
        return _AG_CACHE[key]
    ee = getattr(arm, "ee", None) or arm.hand
    # collect bodies from ee up to the world, then all bodies whose
    # ancestor set intersects that chain
    chain = set()
    b = ee
    while b != 0:
        chain.add(b)
        b = m.body_parentid[b]
    arm_bodies = set()
    for bb in range(m.nbody):
        anc = bb
        while anc != 0:
            if anc in chain:
                arm_bodies.add(bb)
                break
            anc = m.body_parentid[anc]
    gs = {g for g in range(m.ngeom)
          if int(m.geom_bodyid[g]) in arm_bodies}
    _AG_CACHE[key] = gs
    return gs


def spline_to(m, d, arm, target, frames_cb, spf,
              speed=1.4, min_s=0.35, max_s=2.4, settle=0.10,
              z_floor=None, _direct=False):
    """Execute one minimum-jerk joint move to the IK solution of
    `target`. Long HORIZONTAL transits are decomposed up-over-down
    (rise, traverse at safe height, descend): a joint-space
    interpolation between distant configurations can sweep the tool
    through the table even when both endpoints are collision-free
    (measured on the UR5e). Duration scales with the largest joint
    excursion. Returns the final EE position error."""
    target = np.asarray(target, float)
    ee = getattr(arm, "ee", None) or arm.hand
    mujoco.mj_kinematics(m, d)
    cur = d.xpos[ee].copy()
    horiz = float(np.hypot(target[0] - cur[0], target[1] - cur[1]))
    if not _direct and horiz > 0.10:
        # the traverse floor must be SCENE-DERIVED: a fixed 0.32 was
        # set under a wrong table-height assumption and dragged the
        # fingertips at block-top level across the table (measured:
        # the panda plowed blocks it had cleanly grasped before)
        z_safe = max(cur[2], target[2],
                     z_floor if z_floor is not None else 0.0)
        for wp in ([cur[0], cur[1], z_safe],
                   [target[0], target[1], z_safe]):
            spline_to(m, d, arm, wp, frames_cb, spf, speed=speed,
                      min_s=min_s, max_s=max_s, settle=0.02,
                      _direct=True)
    q_goal, e_ik = mj_solve_ik(m, d, arm, target)
    if e_ik > 0.008:
        # escalate rather than execute a known-bad waypoint: a local
        # minimum here becomes a visible re-approach in the film
        q_goal, e_ik = mj_solve_ik(m, d, arm, target,
                                   restarts=14, iters=320)
    q0 = np.array(arm.q_cmd, float)
    dq = q_goal - q0
    T = float(np.clip(np.abs(dq).max() / speed, min_s, max_s))
    n = max(int(T / m.opt.timestep), 8)
    for s in range(n):
        t = (s + 1) / n
        prof = 10 * t**3 - 15 * t**4 + 6 * t**5
        q = q0 + dq * prof
        for a, v in zip(arm.act, q):
            d.ctrl[a] = v
        mujoco.mj_step(m, d)
        if s % spf == 0:
            frames_cb()
    arm.q_cmd = q_goal.copy()
    ns = int(settle / m.opt.timestep)
    for s in range(ns):
        mujoco.mj_step(m, d)
        if s % spf == 0:
            frames_cb()
    ee = getattr(arm, "ee", None) or arm.hand
    mujoco.mj_kinematics(m, d)
    return float(np.linalg.norm(np.asarray(target) - d.xpos[ee]))
