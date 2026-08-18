"""SIMULATED CORPUS: a Panda arm stacks coloured shapes into towers.

Why this exists, beyond the demo: it is the second corpus every open
question has been waiting for -

  - cross-corpus proof that the QbE mechanism carries (it reads only
    seeds + corpus statistics; this is the first chance to measure that
    claim off the kitchen),
  - a truthset with FULL coverage: the simulator knows every verdict,
    so prec == prec_g by construction and the pool-limit caveat dies,
  - known object KINDS (shape x colour) for verifying the kind-descriptor
    program against ground truth instead of a 65-episode bank.

THE DATABASE GETS PIXELS ONLY. The sidecar truth (shapes, colours,
phase timestamps, success flags) goes to data/sim_stack/truth.parquet
and meta.json - EVAL-ONLY artifacts, never ingested; the no-metadata
rule applies to the store, and the store will meet this corpus as four
mp4 streams like any other capture. data/ is gitignored (so is
mujoco_menagerie/); only this generator is committed.

Episode: 2-4 blocks (box/cylinder, palette colour, random size/pose)
spawn on the table; the arm picks each and stacks at a random tower
site; four fixed cameras record at RFPS. Control is damped-least-
squares Jacobian IK on the 7-DoF arm with a phase machine per block
(hover, descend, grasp, lift, transfer, place, release, retreat).
Success is verified FROM SIM STATE at episode end (xy within half a
block width of the tower base, z within tolerance of its slot), and
recorded per block - a failed grasp is data too, labelled as such.

    python scripts/sim_stack.py --episodes 3 --out data/sim_stack
    python scripts/sim_stack.py --episodes 300
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_DIR = ROOT / "mujoco_menagerie/franka_emika_panda"
RFPS = 10
RES = (480, 640)                      # H, W
SETTLE_S = 0.5

PALETTE = {                            # generator-side labels; eval-only
    "red": (0.85, 0.10, 0.10), "green": (0.10, 0.65, 0.15),
    "blue": (0.15, 0.25, 0.85), "yellow": (0.90, 0.80, 0.10),
    "orange": (0.95, 0.55, 0.10), "purple": (0.55, 0.15, 0.65),
    "cyan": (0.10, 0.70, 0.75), "white": (0.92, 0.92, 0.92),
}
SHAPES = ("box", "cylinder")

SCENE = """
<mujoco model="stack_scene">
  <include file="panda.xml"/>
  <!-- elliptic cone + impratio 10: the Robotiq 2F-85 model REQUESTS
       these for grip friction and the scene was silently overriding
       them to defaults (attach warning) - measured: cylinders slipped
       out of the closed gripper 6/8 (owner caught it on film).
       These are better contact physics for every gripper, not a
       Robotiq special case. -->
  <option cone="elliptic" impratio="10"/>
  <statistic center="0.4 0 0.3" extent="1.1"/>
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35"/>
    <global azimuth="120" elevation="-20" offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0.3 0.4 1.6" dir="0 -0.2 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="0.6 -0.5 1.2" dir="-0.2 0.4 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="3 3 0.05" rgba="0.45 0.45 0.48 1"/>
    <body name="table" pos="0.5 0 0.1">
      <geom type="box" size="0.45 0.6 0.1" rgba="0.55 0.42 0.30 1"
            friction="1.2 0.01 0.001"/>
    </body>
    {CAMS}
    {BLOCKS}
  </worldbody>
</mujoco>
"""

BLOCK = """
    <body name="blk{i}" pos="{x} {y} {z}">
      <freejoint/>
      <geom name="gblk{i}" type="{shape}" size="{size}" rgba="{r} {g} {b} 1"
            friction="2.0 0.01 0.005" condim="6" mass="0.045"/>
    </body>
"""

# MuJoCo combines contact friction as the element-wise MAX of the two
# geoms, so raising BLOCK friction raises the pad-block friction even
# though panda.xml is untouched. Grippier + lighter (0.045 vs 0.06) is
# the transit-drop fix: the slip plane was pad-on-block under carry
# acceleration (14% of blocks left the gripper mid-flight in the v1
# batch). condim=6 because rolling friction is IGNORED below it - a
# knocked cylinder rolled forever, ended somewhere awkward, and the
# next pick burned 20s failing on it (measured cascade in the pilot).

CAM_BASE = (
    ("cam0", (1.45, 0.0, 0.75), "0 1 0 -0.5 0 1"),
    ("cam1", (0.5, 1.25, 0.85), "-1 0 0 0 -0.55 1"),
    ("cam2", (0.5, -1.25, 0.85), "1 0 0 0 0.55 1"),
    ("cam3", (1.15, 0.95, 1.15), "-0.68 0.73 0 -0.4 -0.38 0.9"),
)


def build_scene(rng, n_blocks, spec=None):
    """spec: optional [(shape, color), ...] so the caller can control
    episode identity (chains dedupes on it); sizes/poses stay random.
    Colors are drawn WITHOUT replacement either way - two red boxes in
    one episode are indistinguishable on film and in any query."""
    cnames = list(rng.choice(list(PALETTE), size=n_blocks, replace=False))
    blocks, meta = [], []
    spots = []
    for i in range(n_blocks):
        # spawn apart from each other and the tower site
        # spawn range pulled inside the dexterous workspace: beyond
        # x~0.56 the wrist-down pose is near reach limit and IK error
        # grows to several cm
        for _ in range(200):
            x = rng.uniform(0.30, 0.56)
            y = rng.uniform(-0.26, 0.26)
            if all(np.hypot(x - a, y - b) > 0.13 for a, b in spots):
                break
        spots.append((x, y))
        if spec is not None:
            shape, cname = spec[i]
        else:
            shape = SHAPES[rng.integers(len(SHAPES))]
            cname = cnames[i]
        r, g, b = PALETTE[cname]
        # SQUAT blocks: towers were built and then COLLAPSED (seen in
        # the frames - a 3-story tower stood at t=53 and was rubble by
        # the end). Height never exceeds width, so every story is
        # stable under the small impulses later placements add.
        w = rng.uniform(0.024, 0.032)          # half-width / radius
        h = rng.uniform(0.018, min(0.026, w))  # half-height <= w
        size = f"{w} {w} {h}" if shape == "box" else f"{w} {h}"
        blocks.append(BLOCK.format(i=i, x=f"{x:.3f}", y=f"{y:.3f}",
                                   z=f"{0.2 + h + 0.002:.3f}", shape=shape,
                                   size=size, r=r, g=g, b=b))
        meta.append({"name": f"blk{i}", "shape": shape, "color": cname,
                     "rgba": [r, g, b, 1.0], "half_w": w, "half_h": h,
                     "spawn": [x, y]})
    # tower site clear of every spawn
    for _ in range(200):
        tx = rng.uniform(0.34, 0.52)
        ty = rng.uniform(-0.20, 0.20)
        if all(np.hypot(tx - a, ty - b) > 0.15 for a, b in spots):
            break
    # per-episode camera jitter: same four viewpoints, never the same
    # framing twice - part of the "no identical-looking setup" rule
    cams = "".join(
        f'    <camera name="{n}" pos="'
        + " ".join(f"{p + rng.uniform(-0.05, 0.05):.3f}" for p in pos)
        + f'" xyaxes="{xy}"/>\n'
        for n, pos, xy in CAM_BASE)
    xml = SCENE.replace("{BLOCKS}", "".join(blocks)).replace("{CAMS}", cams)
    path = PANDA_DIR / "_stack_scene.xml"
    path.write_text(xml)
    return path, meta, (tx, ty)


class Arm:
    """DLS-Jacobian IK on the 7-DoF Panda + tendon gripper."""

    def __init__(self, m, d):
        self.m, self.d = m, d
        # menagerie's panda.xml ships no sites; the HAND BODY is the
        # IK target and grip_off below is hand-origin -> fingertips
        self.hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.jnt = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                      f"joint{i+1}") for i in range(7)]
        self.dof = [m.jnt_dofadr[j] for j in self.jnt]
        self.qadr = [m.jnt_qposadr[j] for j in self.jnt]
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      f"actuator{i+1}") for i in range(7)]
        self.grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      "actuator8")
        self.home = np.array([0, -0.4, 0, -2.2, 0, 1.9, 0.785])
        self.down_quat = None                 # captured at home
        # finger geoms, for contact-triggered grasping
        fingers = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in ("left_finger", "right_finger")]
        self.finger_geoms = [g for g in range(m.ngeom)
                             if m.geom_bodyid[g] in fingers]

    def to_home(self):
        for a, q, v in zip(self.act, self.qadr, self.home):
            self.d.qpos[q] = v
            self.d.ctrl[a] = v
        self.q_cmd = self.home.copy()         # integrator state (see step_ik)
        self.d.ctrl[self.grip] = 255          # open
        mujoco.mj_forward(self.m, self.d)
        # the DOWNWARD gripper orientation is whatever home gives us -
        # captured from the model, not hand-written
        self.down_quat = self.d.xquat[self.hand].copy()
        # hand-origin -> fingertip, MEASURED from the model at home
        # (the first pilot's hand-written 0.103 hovered above the block)
        tip_z = min(self.d.geom_xpos[g][2] for g in self.finger_geoms)
        self.tip_off = float(self.d.xpos[self.hand][2] - tip_z) + 0.010

    def step_ik(self, target, gain=4.0):
        """6-DoF DLS update: position to `target`, orientation held at
        the home DOWN pose. Position-only IK let the wrist drift and
        the first pilot's gripper arrived sideways, hovering - visible
        in the recorded frames, which is what the frames are for."""
        m, d = self.m, self.d
        err_p = target - d.xpos[self.hand]
        rq = np.zeros(3)
        cur = d.xquat[self.hand]
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, cur)
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, self.down_quat, neg)
        mujoco.mju_quat2Vel(rq, rel, 1.0)
        # orientation is a PREFERENCE, position is the job: at 0.5 the
        # DLS compromise left 1-7cm of steady-state position error
        # (measured across the table - releases were happening 5cm off
        # target), because holding the wrist perfectly down fights the
        # reach. At 0.15 the wrist tilts a few degrees at the extremes
        # and the hand actually arrives.
        err = np.concatenate([err_p, 0.15 * rq])
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, self.hand)
        J = np.vstack([jacp, jacr])[:, self.dof]
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + 1e-4 * np.eye(6), err)
        # integrate on the COMMAND, not the measured q: the position
        # servos sag ~0.005-0.01 rad under gravity torque, and stepping
        # from measured q re-baselines onto that sag every step - a
        # steady-state Cartesian error of 1-7cm across the table
        # (measured; releases landed 5cm off target). Command-side
        # integration is integral action: the command leads the sag
        # until the error is gone. Clamp keeps it from winding up when
        # the arm is blocked by contact.
        qm = np.array([d.qpos[a] for a in self.qadr])
        # the windup clamp is also the speed limiter, and a collision
        # can turn it into bang-bang drive: command rides the clamp in
        # alternating directions and the arm THRASHES (one pilot
        # episode spent 60s flailing and flung a block off the table).
        # When joints exceed real Panda velocity limits (~2.6 rad/s),
        # re-baseline the integrator on the measured state - that
        # breaks the limit cycle instead of feeding it.
        if float(np.max(np.abs(d.qvel[self.dof]))) > 3.0:
            self.q_cmd = qm.copy()
        q = self.q_cmd + gain * dq * m.opt.timestep * 20
        q = np.clip(q, qm - 0.05, qm + 0.05)
        lo = m.jnt_range[self.jnt, 0]
        hi = m.jnt_range[self.jnt, 1]
        q = np.clip(q, lo + 0.02, hi - 0.02)
        self.q_cmd = q
        for a, v in zip(self.act, q):
            d.ctrl[a] = v
        return float(np.linalg.norm(err_p))

    def touching(self, geom_id):
        """Is any finger geom in contact with `geom_id`?"""
        d = self.d
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 == geom_id and g2 in self.finger_geoms) or \
                    (g2 == geom_id and g1 in self.finger_geoms):
                return True
        return False

    def gripper(self, open_):
        self.d.ctrl[self.grip] = 255 if open_ else 0


def run_episode(ep_id, rng, out_dir, log):
    n_blocks = int(rng.integers(2, 4))    # 2-3 stories
    scene, meta, (tx, ty) = build_scene(rng, n_blocks)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    arm = Arm(m, d)
    arm.to_home()
    mujoco.mj_forward(m, d)

    r = mujoco.Renderer(m, RES[0], RES[1])
    cams = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"cam{i}")
            for i in range(4)]
    enc = []
    ep_dir = out_dir / f"ep{ep_id:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{RES[1]}x{RES[0]}",
             "-r", str(RFPS), "-i", "-", "-c:v", "libx264",
             "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
             str(ep_dir / f"cam{i}.mp4")], stdin=subprocess.PIPE)
        enc.append(p)

    steps_per_frame = max(1, int(round(1 / (RFPS * m.opt.timestep))))
    frame_n = 0
    events = []

    def sim(seconds, ctrl=None):
        nonlocal frame_n
        n = int(seconds / m.opt.timestep)
        for s in range(n):
            if ctrl is not None:
                ctrl()
            mujoco.mj_step(m, d)
            if s % steps_per_frame == 0:
                for cam, p in zip(cams, enc):
                    r.update_scene(d, camera=cam)
                    p.stdin.write(r.render().tobytes())
                frame_n += 1

    def move_to(target, tol=0.012, timeout=3.0):
        t_end = timeout
        done = {"v": False}

        def ctrl():
            if arm.step_ik(np.asarray(target)) < tol:
                done["v"] = True
        n = int(t_end / m.opt.timestep)
        for s in range(n):
            ctrl()
            mujoco.mj_step(m, d)
            if s % steps_per_frame == 0:
                nonlocal_frames()
            if done["v"]:
                break

    def nonlocal_frames():
        nonlocal frame_n
        for cam, p in zip(cams, enc):
            r.update_scene(d, camera=cam)
            p.stdin.write(r.render().tobytes())
        frame_n += 1

    def descend_to_grasp(bid, half_h, timeout=3.5):
        """Descend TRACKING the block's live centre until the pads
        straddle it (hand z = block z + tip_off), so closing traps the
        block instead of nudging it - one finger touching first was the
        cylinder-miss mode. Contact with the TABLE aborts."""
        nonlocal frame_n
        n = int(timeout / m.opt.timestep)
        for st in range(n):
            bp = d.xpos[bid]
            # grip the UPPER half: with the pads at block centre the
            # fingertips reached BELOW its bottom face, and lowering
            # onto the tower rammed the fingers into the top block
            # before the held block could land - slot 0 only worked
            # because a table cannot topple. Tips at centre + half_h/2
            # keeps the block's bottom the lowest thing descending.
            # centre grip: the upper-half variant halved ram risk but
            # tripled TRANSIT SLIPS (releases measured 56-328mm from
            # the slot - the block left the gripper mid-flight), and a
            # slipped block never stacks. Hold beats clearance.
            zt = bp[2] + arm.tip_off - half_h * 0.4
            hp = d.xpos[arm.hand]
            xy_err = float(np.hypot(hp[0] - bp[0], hp[1] - bp[1]))
            # descend only while centred; hold altitude until xy is in.
            # Arriving at depth while off-centre was the remaining miss
            # mode: the pads closed BESIDE the block.
            z = max(hp[2] - 0.0025, zt) if xy_err < 0.012 else hp[2]
            arm.step_ik(np.array([bp[0], bp[1], z]))
            mujoco.mj_step(m, d)
            if st % steps_per_frame == 0:
                nonlocal_frames()
            if hp[2] <= zt + 0.002 and xy_err < 0.008:
                return True
        return False

    sim(SETTLE_S)
    # stack order: biggest base first, boxes below cylinders - the
    # difference between a tower and a demolition, per the pilot frames
    order = sorted(range(n_blocks),
                   key=lambda i: (meta[i]["shape"] == "cylinder",
                                  -meta[i]["half_w"]))
    z_top = 0.2                                  # table surface
    placed = []
    for bi in order:
        b = meta[bi]
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b["name"])
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                f"gblk{bi}")
        t0 = frame_n / RFPS
        carried = False
        for attempt in range(3):                 # up to two retries
            bpos = d.xpos[bid].copy()
            move_to([bpos[0], bpos[1], bpos[2] + 0.22])
            arm.gripper(True)
            sim(0.2)
            descend_to_grasp(bid, b["half_h"])
            events.append({"block": b["name"], "kind": "grasp",
                           "t": frame_n / RFPS, "attempt": attempt})
            arm.gripper(False)
            sim(0.6)
            cur = d.xpos[arm.hand].copy()
            move_to([cur[0], cur[1], 0.45], timeout=2.0)
            carried = d.xpos[bid][2] > z_top + b["half_h"] + 0.05
            if carried:
                break
        # PLACE ONTO THE TOWER AS IT ACTUALLY STANDS, not the ideal
        # site: each placement had targeted (tx, ty) while the tower
        # drifted a few mm, errors accumulated, and the TOP block of
        # every failed episode ended half-on and tipped (the purple
        # cylinder lying beside the yellow box in the pilot frames).
        # the live top = highest ALREADY-PLACED block still at the
        # site. slot_z built from the PLANNED list counted toppled
        # stories as real: every release after a topple happened 2.5-5cm
        # high (measured dz +9..+26mm at release, then off the edge),
        # and correct stacks on the shorter real tower were booked low.
        top_id, top_z, top_h = None, z_top, 0.0
        for j in placed:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                    meta[j]["name"])
            jp = d.xpos[jid]
            if np.hypot(jp[0] - tx, jp[1] - ty) < 0.045 and \
                    jp[2] > top_z:
                top_id, top_z, top_h = jid, float(jp[2]), meta[j]["half_h"]
        slot_z = (top_z + top_h if top_id is not None else z_top) \
            + b["half_h"]
        if top_id is not None:
            px_, py_ = d.xpos[top_id][0], d.xpos[top_id][1]
        else:
            px_, py_ = tx, ty
        # LOW, WAYPOINTED transfer: a single long move at height was a
        # pendulum, and the pendulum shed blocks. Half-way waypoint at
        # 0.45 keeps accelerations small.
        cur = d.xpos[arm.hand].copy()
        move_to([(cur[0] + px_) / 2, (cur[1] + py_) / 2, 0.45],
                timeout=2.0)
        move_to([px_, py_, 0.45], timeout=2.5)
        move_to([px_, py_, 0.45], tol=0.006, timeout=1.5)  # settle swing
        # lower until the HELD BLOCK contacts what is below it; descend
        # only while centred over the tower - lowering mid-swing was
        # the knock-the-tower mode (carried 25/30, stacked 13/30)
        n = int(4.0 / m.opt.timestep)
        contact_at = None
        for st in range(n):
            if top_id is not None:                 # track the live top
                px_, py_ = d.xpos[top_id][0], d.xpos[top_id][1]
            hp = d.xpos[arm.hand]
            xy_err = float(np.hypot(hp[0] - px_, hp[1] - py_))
            z = hp[2] - 0.002 if xy_err < 0.008 else hp[2]
            arm.step_ik(np.array([px_, py_, z]))
            mujoco.mj_step(m, d)
            if st % steps_per_frame == 0:
                nonlocal_frames()
            hit = False
            for c in range(d.ncon):
                g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                if gid in (g1, g2) and \
                        (g1 not in arm.finger_geoms) and \
                        (g2 not in arm.finger_geoms):
                    hit = True
                    break
            if hit or d.xpos[bid][2] <= slot_z + 0.003:
                if contact_at is None:
                    contact_at = st
                # HOLD position briefly to damp before release
                if st - contact_at > int(0.25 / m.opt.timestep):
                    break
        rel = d.xpos[bid].copy()
        arm.gripper(True)
        sim(0.5)
        after = d.xpos[bid].copy()
        events.append({"block": b["name"], "kind": "place",
                       "t": frame_n / RFPS, "carried": bool(carried),
                       "at_release": [round(float(x), 4) for x in rel],
                       "after_0.5s": [round(float(x), 4) for x in after],
                       "slot": [round(float(px_), 4), round(float(py_), 4),
                                round(float(slot_z), 4)]})
        cur = d.xpos[arm.hand].copy()
        move_to([cur[0], cur[1], 0.60], timeout=1.5)
        events.append({"block": b["name"], "kind": "pick_to_place_span",
                       "t0": t0, "t": frame_n / RFPS})
        placed.append(bi)
    move_to([0.35, 0.0, 0.6], timeout=2.0)
    sim(0.8)

    # ---- verify from ACHIEVED state ------------------------------
    # a block is stacked if it rests at the tower site at a height
    # consistent with the members below it; judged from where things
    # ENDED, not from the plan - a re-ordered tower is still a tower
    finals = []
    for bi in order:
        b = meta[bi]
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b["name"])
        finals.append((bi, d.xpos[bid].copy()))
    at_tower = [(bi, p) for bi, p in finals
                if np.hypot(p[0] - tx, p[1] - ty) < 0.045]
    at_tower.sort(key=lambda t: t[1][2])
    ok_map = {bi: False for bi in order}
    exp_z = z_top
    for bi, p in at_tower:
        h = meta[bi]["half_h"]
        if abs(p[2] - (exp_z + h)) < 0.028:
            ok_map[bi] = True
            exp_z += 2 * h
        else:
            break
    ok = [ok_map[bi] for bi in order]
    for p in enc:
        p.stdin.close()
    for p in enc:
        p.wait()
    rec = {"episode": ep_id, "n_blocks": n_blocks,
           "blocks": [meta[i] for i in order], "tower_xy": [tx, ty],
           "stacked_ok": ok, "success": all(ok),
           "frames_per_cam": frame_n, "seconds": round(frame_n / RFPS, 1),
           "events": events}
    (ep_dir / "meta.json").write_text(json.dumps(rec, indent=1))
    log.append(rec)
    return rec


def main():
    argv = sys.argv
    n_ep = int(argv[argv.index("--episodes") + 1]) if "--episodes" in argv \
        else 3
    out = ROOT / (argv[argv.index("--out") + 1] if "--out" in argv
                  else "data/sim_stack")
    out.mkdir(parents=True, exist_ok=True)
    start = int(argv[argv.index("--start") + 1]) if "--start" in argv else 0
    log = []
    t0 = time.time()
    from tqdm import tqdm
    for ep in tqdm(range(start, start + n_ep), desc="episodes", unit="ep"):
        rng = np.random.default_rng(1000 + ep)   # reproducible per episode
        rec = run_episode(ep, rng, out, log)
        tqdm.write(f"  ep{ep:04d} blocks={rec['n_blocks']} "
                   f"success={rec['success']} ({rec['seconds']}s sim)")
    n_ok = sum(1 for r in log if r["success"])
    print(json.dumps({"episodes": len(log), "full_success": n_ok,
                      "rate": round(n_ok / max(len(log), 1), 2),
                      "wall_min": round((time.time() - t0) / 60, 1)}))
    # sidecar truthset (append across batches)
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    for r in log:
        for i, (b, okk) in enumerate(zip(r["blocks"], r["stacked_ok"])):
            rows.append({"episode": r["episode"], "order": i,
                         "shape": b["shape"], "color": b["color"],
                         "stacked_ok": okk, "success": r["success"]})
    t = pa.Table.from_pylist(rows)
    f = out / "truth.parquet"
    if f.exists():
        t = pa.concat_tables([pq.read_table(f), t])
    pq.write_table(t, f)
    print(f"truth sidecar: {len(t)} rows -> {f}")


if __name__ == "__main__":
    main()
