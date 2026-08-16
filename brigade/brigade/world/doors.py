"""Physical cabinet-door manipulation. The gripper moves the door, or it stays shut.

This module exists because of an owner ruling: no state writes in the robot's
action path. `fixture.open_door(env)` teleports the hinge qpos and looks exactly
like what it is. Everything here works by contact forces, and the door angle you
see is the angle the arm actually achieved.

The recipe was found empirically, eight experiments deep, on the real cabinets
(HingeCabinet, vertical bar handle 24 cm tall standing 2 cm proud of the door,
hinge at the outer edge). What survived measurement:

  CRACK   pinch the bar and pull along the hinge arc. Reliably reaches
          0.13-0.39 normalized before the fingers cam off the rotating bar.
          (Straight-line pulling is WORSE: 0.13-0.17. Hooking the closed blade
          into the 19 mm bar-door gap fails outright: the fingertip is thicker
          than the gap.)
  SWEEP   re-point the gripper along ±x and press the door's INNER face from
          the wedge between door and cabinet, leading the measured angle by
          0.3 rad. The subtlety that cost three experiments: with the gripper
          pointing +y the FOREARM crosses the door plane and pushes the door
          shut with a longer lever than the fingertip opening it — the sweep
          must enter with the forearm trailing through the open space in front
          of the NEIGHBOURING door, i.e. gripper along (s,0,0) for door side s.

Measured outcome: 0.46-0.58 normalized (42-52 degrees of actual swing) before
the contact geometry saturates. So OPEN_ENOUGH is 0.40: visibly, physically
open — not RoboCasa's 0.9 teleport threshold, and we do not pretend otherwise;
every result carries the achieved angle.
"""

from __future__ import annotations

import numpy as np
import robosuite.utils.transform_utils as T

from .sim import SkillResult

GRIP_SITE = "gripper0_right_grip_site"
FULL_RAD = 1.57          # hinge range; normalized state 1.0 == this many radians
OPEN_ENOUGH = 0.40       # normalized: the door is genuinely, visibly open
CLOSED_ENOUGH = 0.12
SEE_INSIDE = 0.35        # below this, survey must not report contents

# Handle geometry measured from the models: bar center sits 16 mm outside the
# handle site, door face 11 mm inside it.
_BAR_FROM_SITE_Y = -0.016
_FACE_FROM_SITE_Y = +0.011


def _obs(env):
    return env._get_observations()


def _eef(env):
    return np.array(_obs(env)["robot0_eef_pos"])


def _zax(env):
    return np.array(env.sim.data.get_site_xmat(GRIP_SITE)).reshape(3, 3)[:, 2]


def _rot(v, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class DoorPlan:
    """Everything geometric about one door, read from the live model."""

    def __init__(self, env, cab_name: str, side: str):
        fx = env.fixtures[cab_name]
        self.cab, self.side = cab_name, side
        self.s = +1.0 if side == "right" else -1.0
        site = f"{cab_name}_{side}_door_handle_default_site"
        hp = np.array(env.sim.data.get_site_xpos(site))
        self.bar = hp + np.array([0.0, _BAR_FROM_SITE_Y, 0.0])
        half = float(fx.size[0]) / 2.0
        self.hinge = np.array([float(fx.pos[0]) + self.s * half, hp[1] + _FACE_FROM_SITE_Y])
        self.v_edge = np.array([-self.s * half, 0.0])
        self.v_bar = self.bar[:2] - self.hinge
        self.z = float(hp[2])

    def norm(self, env) -> float:
        st = self.cab and env.fixtures[self.cab].get_door_state(env)
        return float(st[f"{self.side}_door"])

    def rad(self, env) -> float:
        return self.norm(env) * FULL_RAD

    def arc_bar(self, th):
        xy = self.hinge + _rot(self.v_bar, self.s * th)
        return np.array([xy[0], xy[1], self.z])

    def pocket(self, th, radius):
        scale = radius / abs(self.v_edge[0])
        xy = self.hinge + _rot(self.v_edge * scale, self.s * th)
        return np.array([xy[0], xy[1], self.z])

    @property
    def sweep_axis(self):
        # Forearm must trail through the open space in front of the NEIGHBOUR
        # door; pointing the gripper along +y makes the forearm close the door
        # it is trying to open (measured: 0.30 -> 0.00).
        return np.array([self.s, 0.0, 0.0])


def _hold_axis(env, axis):
    return np.clip(np.cross(_zax(env), np.asarray(axis, float)) * 1.8, -0.4, 0.4)


def _servo(env, tgt, tol, grip, axis=(0, 1, 0), max_steps=90):
    for _ in range(max_steps):
        err = np.asarray(tgt, float) - _eef(env)
        if float(np.linalg.norm(err)) < tol:
            break
        o = _obs(env)
        R = T.quat2mat(np.array(o["robot0_base_quat"]))
        a = np.zeros(env.action_dim)
        a[0:3] = np.clip(R.T @ err / 0.05, -1.0, 1.0)
        a[6] = grip
        a[3:6] = R.T @ _hold_axis(env, axis)
        yield a


def _face(env, axis, grip):
    for _ in range(80):
        if float(_zax(env) @ np.asarray(axis, float)) > 0.93:
            break
        o = _obs(env)
        R = T.quat2mat(np.array(o["robot0_base_quat"]))
        a = np.zeros(env.action_dim)
        a[6] = grip
        a[3:6] = R.T @ _hold_axis(env, axis)
        yield a


def _stand(env, x, y=-1.05, torso_steps=35):
    """Drive the base to a stance and raise the torso.

    The velocity command TAPERS with distance. A constant-magnitude command
    orbits the target without ever entering the acceptance radius — measured:
    the base hovering 7-9 cm off stance for 220 straight steps, which then
    skewed the pinch enough to drop the crack from 0.35 to 0.06.
    """
    tgt = np.array([x, y])
    settled = 0
    for _ in range(220):
        o = _obs(env)
        here = np.array(o["robot0_base_pos"])[:2]
        d = tgt - here
        dist = float(np.linalg.norm(d))
        if dist < 0.06:
            settled += 1
            if settled >= 5:
                break
            yield np.zeros(env.action_dim)
            continue
        settled = 0
        R = T.quat2mat(np.array(o["robot0_base_quat"]))[:2, :2]
        mag = 0.6 * min(1.0, dist / 0.30)  # taper inside 30 cm
        a = np.zeros(env.action_dim)
        a[7:9] = np.clip(R.T @ d / max(dist, 1e-6) * mag, -1.0, 1.0)
        a[11] = -1.0
        yield a
    for _ in range(torso_steps):
        a = np.zeros(env.action_dim)
        a[10] = 1.0
        yield a


def _crack(env, plan: DoorPlan):
    """Pinch the handle bar WHERE IT NOW IS, pull along the hinge arc.

    `plan` must be freshly built: the handle site moves with the door, so a
    retry that reuses the plan from before the first pull pinches the air where
    the bar used to be. Rebuilding per attempt is what turned the retry from a
    coin flip into a ratchet — each crack starts from the door's current angle
    and adds to it.

    Wide-spaced waypoints, deliberately: a big positional lead saturates the
    OSC and pulls HARD while the pinch lasts. A denser, gentler arc was tried
    and opened LESS (0.19 vs 0.35+) — the pinch cams off at about the same
    angle either way, so the win is force while engaged, not tracking accuracy.
    """
    th_now = plan.rad(env)
    yield from _servo(env, plan.bar + np.array([0, -0.14, 0]), 0.02, -1.0)
    yield from _servo(env, plan.bar + np.array([0, 0.010, 0]), 0.012, -1.0)
    a = np.zeros(env.action_dim)
    a[6] = 1.0
    for _ in range(24):
        yield a
    for th in np.linspace(th_now + 0.15, th_now + 1.15, 7):
        yield from _servo(env, plan.arc_bar(min(th, 1.45)), 0.035, 1.0, max_steps=45)
    a = np.zeros(env.action_dim)
    a[6] = -1.0
    for _ in range(12):
        yield a
    yield from _servo(env, _eef(env) + np.array([0, -0.10, 0]), 0.04, -1.0, max_steps=30)


def open_door(cab_name: str, side: str = "right"):
    """Physically open one door: crack, then wedge-sweep, closed-loop on angle."""

    def controller(env):
        plan = DoorPlan(env, cab_name, side)
        yield from _stand(env, plan.bar[0] - plan.s * 0.05)
        yield from _face(env, (0, 1, 0), -1.0)

        # Crack up to four times, JITTERING THE STANCE between failures. The
        # outcome is bimodal (0.6+ or ~0.1) and the failure mode is systematic
        # within a run: a few millimetres of stance error make every retry miss
        # the bar the same way. Moving the base a few centimetres between
        # attempts is what decorrelates them. Each attempt also rebuilds the
        # plan, because the handle site moves with the door.
        jitters = (0.0, +0.04, -0.04, +0.08)
        for jit in jitters:
            if plan.norm(env) >= 0.35:
                break
            if jit:
                yield from _stand(env, plan.bar[0] - plan.s * 0.05 + jit, torso_steps=0)
            plan = DoorPlan(env, cab_name, side)
            yield from _crack(env, plan)

        # The sweep only helps if the crack fell short; a good crack (>=0.55)
        # is already open enough, and the sweep is where past regressions lived.
        if 0.12 <= plan.norm(env) < 0.55:
            # Radii are FRACTIONS of the measured half-door, not absolutes. The
            # absolute radii this shipped with (0.20/0.30) were tuned when the
            # door was believed 0.25 m wide; it is 0.50 m, so those pressed the
            # thin end of the wedge near the hinge — the FRONT face — and
            # deterministically closed the door the crack had just opened
            # (0.64 after crack, 0.14 after "improvement", three runs in a row).
            half = abs(plan.v_edge[0])
            r_in, r_out = 0.80 * half, 1.10 * half
            yield from _servo(env, _eef(env) + np.array([-plan.s * 0.05, -0.15, 0]),
                              0.04, -1.0, max_steps=40)
            yield from _face(env, plan.sweep_axis, 1.0)
            th0 = plan.rad(env)
            start = plan.pocket(max(th0 - 0.10, 0.06), r_out)
            start[0] -= plan.s * 0.10
            yield from _servo(env, start, 0.03, 1.0, axis=plan.sweep_axis, max_steps=60)
            yield from _servo(env, plan.pocket(max(th0 - 0.05, 0.06), r_in),
                              0.03, 1.0, axis=plan.sweep_axis, max_steps=60)
            stall = 0
            last = plan.rad(env)
            for _ in range(28):
                cur = plan.rad(env)
                if cur > 1.30:
                    break
                stall = stall + 1 if cur - last < 0.01 else 0
                if stall >= 6:
                    break  # the contact has saturated; pushing longer changes nothing
                last = cur
                yield from _servo(env, plan.pocket(min(cur + 0.30, 1.45), r_in),
                                  0.05, 1.0, axis=plan.sweep_axis, max_steps=26)
            yield from _servo(env, _eef(env) + np.array([-plan.s * 0.10, -0.22, 0]),
                              0.05, -1.0, axis=plan.sweep_axis, max_steps=50)

        got = plan.norm(env)
        deg = got * FULL_RAD * 57.3
        return SkillResult(
            got >= OPEN_ENOUGH, "open_door", 0.0, "crack+sweep",
            f"{cab_name}/{side} at {got:.2f} ({deg:.0f}°) by contact",
            dict(cab=cab_name, side=side, angle_norm=got),
        )

    return controller


def close_door(cab_name: str, side: str = "right"):
    """Push the door shut from outside — outer-face contact along the closing arc."""

    def controller(env):
        plan = DoorPlan(env, cab_name, side)
        if plan.norm(env) <= CLOSED_ENOUGH:
            return SkillResult(True, "close_door", 0.0, "push", "already closed",
                               dict(cab=cab_name, side=side, angle_norm=plan.norm(env)))
        yield from _stand(env, plan.bar[0] - plan.s * 0.05)
        yield from _face(env, (0, 1, 0), 1.0)
        th = plan.rad(env)
        entry = plan.pocket(th + 0.10, 0.22) + np.array([0, -0.10, 0])
        yield from _servo(env, entry, 0.04, 1.0, max_steps=60)
        while th > 0.06:
            th = max(th - 0.22, 0.03)
            yield from _servo(env, plan.pocket(th, 0.22), 0.05, 1.0, max_steps=30)
            if plan.rad(env) > th + 0.45:
                break  # not tracking; stop rather than flail
        yield from _servo(env, _eef(env) + np.array([0, -0.18, 0]), 0.05, -1.0, max_steps=40)
        got = plan.norm(env)
        return SkillResult(got <= CLOSED_ENOUGH * 1.5, "close_door", 0.0, "push",
                           f"{cab_name}/{side} at {got:.2f} after push",
                           dict(cab=cab_name, side=side, angle_norm=got))

    return controller


def door_norms(env, cab_name: str) -> dict:
    try:
        return {k: float(v) for k, v in env.fixtures[cab_name].get_door_state(env).items()}
    except Exception:
        return {}


def openable(env, name: str) -> bool:
    """Only hinged cabinets have the handle geometry this module understands."""
    if not name.startswith("cab_"):
        return False
    try:
        DoorPlan(env, name, "right")
        return True
    except Exception:
        return False
