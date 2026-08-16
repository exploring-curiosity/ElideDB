"""What the robot can physically do.

Every skill is a generator of actions, run by SimRunner at 20 Hz, returning a
SkillResult judged against the WORLD — did the object end up in the cabinet —
never against whether the controller finished its loop. A skill that runs to
completion having achieved nothing is a failure, and must say so, because the
memory layer learns from these verdicts.

Control facts, all measured on this robot (see docs — probe_ctrl2):

  action[0:3]  arm EEF delta position, OSC_POSE, **in the robot BASE frame**,
               scaled so ±1 == ±0.05 m per control step.
  action[6]    gripper: -1 open, +1 close.
  action[7:9]  base translation velocity, base frame. action[9] base yaw.
  action[11]   control mode; must be -1 for base commands to take effect.

The base-frame transform is the part that is easy to get wrong and silent when
wrong: `robots[0].base_pos` is a mount placeholder of [10,10,0], NOT the robot's
position. The real pose is in the observations (`robot0_base_pos`,
`robot0_base_quat`), and base_pos + R @ base_to_eef == eef_pos was verified to
zero residual.
"""

from __future__ import annotations

import numpy as np
import robosuite.utils.transform_utils as T
from robocasa.utils import object_utils as OU

from .sim import SkillResult

# OSC output_max for position, from the PandaOmron controller config.
_EEF_STEP = 0.05
# How close counts as "reached". Tighter than this and the servo dithers.
_REACH_TOL = 0.015
_BASE_TOL = 0.45
_LIFT = 0.22


def _obs(env) -> dict:
    return env._get_observations()


def _base_frame(env):
    o = _obs(env)
    return np.array(o["robot0_base_pos"]), T.quat2mat(np.array(o["robot0_base_quat"]))


def _eef_world(env) -> np.ndarray:
    return np.array(_obs(env)["robot0_eef_pos"])


def _to_base(env, world_vec: np.ndarray) -> np.ndarray:
    _, R = _base_frame(env)
    return R.T @ world_vec


def _arm_action(env, target_world: np.ndarray, grip: float) -> tuple[np.ndarray, float]:
    """One servo step toward a world-frame target. Returns (action, error)."""
    err_world = target_world - _eef_world(env)
    a = np.zeros(env.action_dim)
    a[0:3] = np.clip(_to_base(env, err_world) / _EEF_STEP, -1.0, 1.0)
    a[6] = grip
    return a, float(np.linalg.norm(err_world))


def _fixture_xy(env, name: str) -> np.ndarray:
    return np.array(env.fixtures[name].pos, dtype=float)[:2]


# ---------------------------------------------------------------------------
# primitive motions
# ---------------------------------------------------------------------------

def reach(target_world, grip: float = -1.0, max_steps: int = 140, tol: float = _REACH_TOL):
    """Servo the gripper to a world position. Verified to converge to ~0 m."""
    target = np.asarray(target_world, dtype=float)

    def controller(env):
        err = float("inf")
        for _ in range(max_steps):
            a, err = _arm_action(env, target, grip)
            if err < tol:
                break
            yield a
        return SkillResult(err < tol * 3, "reach", 0.0, "osc",
                           f"final error {err:.3f}m", dict(error=err))

    return controller


def set_gripper(close: bool, steps: int = 18):
    def controller(env):
        a = np.zeros(env.action_dim)
        a[6] = 1.0 if close else -1.0
        for _ in range(steps):
            yield a
        return SkillResult(True, "grip", 0.0, "close" if close else "open", "")

    return controller


# Base drive, measured: an unobstructed base covers 1.035 m in 60 control steps
# at command magnitude 0.6 — about 0.34 m/s. Commands are NOT the bottleneck.
# What looks like "the base won't move at high command" is the base pressed
# against a counter, which is why stall detection matters more than gain tuning.
_BASE_CMD = 0.6
_STALL_WINDOW = 24      # steps to look back over
_STALL_DRIFT = 0.03     # metres of travel in that window below which we are stuck


# The arm's practical grasp envelope from the base. MEASURED the hard way: with
# the base parked 1.10 m from a bowl the gripper stretched to full extension and
# still stopped 0.35 m short, closed on air, and reported "not held" — which read
# as a broken grasp when it was really a navigation error.
_ARM_REACH = 0.62


def _drive(target_xy, standoff: float, max_steps: int, what: str):
    """Shared base-servo loop. Stops at `standoff`, gives up when stalled.

    Stalling matters: the kitchen is full of counters and the straight line to a
    target often runs through one. Without detection the robot spends its whole
    step budget shoving into furniture (measured 17 s per blocked approach, and
    on camera it looks broken). A stall returns a result with a distance, so the
    caller can judge whether it is close enough to work from.
    """

    def controller(env):
        target = np.asarray(target_xy(env) if callable(target_xy) else target_xy, dtype=float)[:2]
        history: list[np.ndarray] = []
        dist, stalled = float("inf"), False

        for _ in range(max_steps):
            o = _obs(env)
            here = np.array(o["robot0_base_pos"])[:2]
            to_target = target - here
            dist = float(np.linalg.norm(to_target))
            if dist <= standoff:
                break

            history.append(here)
            if len(history) > _STALL_WINDOW:
                if float(np.linalg.norm(history[-1] - history[-_STALL_WINDOW])) < _STALL_DRIFT:
                    stalled = True
                    break

            R = T.quat2mat(np.array(o["robot0_base_quat"]))[:2, :2]
            err_base = R.T @ to_target
            cmd = err_base / max(float(np.linalg.norm(err_base)), 1e-6) * _BASE_CMD
            a = np.zeros(env.action_dim)
            a[7:9] = cmd
            a[11] = -1.0  # base control mode
            yield a

        reached = dist <= standoff * 1.6
        return SkillResult(
            reached, "drive_to", 0.0, "base",
            f"{dist:.2f}m from {what}" + (" (blocked)" if stalled else ""),
            dict(distance=dist, stalled=stalled, reached=reached),
        )

    return controller


def drive_to(fixture_name: str, standoff: float = 0.85, max_steps: int = 260):
    """Park within working distance of a fixture."""
    return _drive(lambda env: _fixture_xy(env, fixture_name), standoff, max_steps,
                  _plain(fixture_name))


def _approach_point(env, instance_id: str, reach: float = _ARM_REACH) -> np.ndarray:
    """Where to stand in order to touch this object.

    NOT the object's own position. An object on a counter has the counter under
    it, so driving to its XY means driving into the furniture: the base stalls
    and reports "1.33 m away, blocked". Instead stand back along the line from
    the object toward where the robot already is — that direction is, by
    construction, floor the robot is currently standing on.
    """
    obj = np.array(env.sim.data.body_xpos[env.obj_body_id[instance_id]])[:2]
    here = np.array(_obs(env)["robot0_base_pos"])[:2]
    away = here - obj
    n = float(np.linalg.norm(away))
    if n < 1e-6:
        return obj
    return obj + away / n * reach


def drive_to_object(instance_id: str, standoff: float = 0.14, max_steps: int = 260):
    """Park within arm's reach of an object, on accessible floor.

    Distinct from drive_to(fixture), and the distinction is load-bearing: a
    counter is metres long, so being 'at the counter' says nothing about whether
    the thing on it can be touched. `standoff` here is tolerance around the
    computed approach point, not distance to the object.
    """
    return _drive(
        lambda env: _approach_point(env, instance_id), standoff, max_steps, instance_id,
    )


def _plain(name: str) -> str:
    return name.replace("_main_group", "")


# ---------------------------------------------------------------------------
# composite skills — the ones the agent actually calls
# ---------------------------------------------------------------------------

# Panda parallel gripper aperture. Objects wider than this cannot be grasped
# across their body at all — MEASURED: a RoboCasa bowl has horizontal_radius
# 0.163 m (0.33 m across) against an ~0.08 m aperture, so a centre grasp closes
# the fingers inside the empty bowl and holds nothing. That is a geometry fact,
# not a controller-tuning problem, and it is why `rim` exists.
_GRIPPER_APERTURE = 0.08


def _radius(env, instance_id: str) -> float:
    try:
        return float(getattr(env.objects[instance_id], "horizontal_radius", 0.05))
    except Exception:
        return 0.05


def _top_z(env, instance_id: str) -> float:
    obj = env.sim.data.body_xpos[env.obj_body_id[instance_id]]
    try:
        return float(obj[2] + env.objects[instance_id].top_offset[2])
    except Exception:
        return float(obj[2]) + 0.04


def strategies_for(env, instance_id: str) -> tuple[str, ...]:
    """Which grasps are worth trying on this object, widest-first by geometry.

    Not a hard-coded per-object table: it is read from the object's own
    horizontal_radius. Anything that fits the aperture can be taken centrally;
    anything wider has to be taken by an edge.
    """
    if _radius(env, instance_id) * 2 > _GRIPPER_APERTURE:
        # Too wide to take centrally. `side` is listed first because it is what
        # actually worked on the one wide object we could lift (a mug: side
        # succeeded, raised 0.196 m, where rim and top both came up empty).
        return ("side", "rim")
    return ("top", "side")


def pick(instance_id: str, strategy: str = "top"):
    """Approach, descend, close, lift. Judged by whether the object is grasped.

    `strategy` changes the approach geometry, and is the thing the agent flips
    after a failure:
      top   — descend vertically onto the object's centre (narrow objects only)
      rim   — descend onto the edge, so the fingers straddle a wall the gripper
              can actually close on (the only option for bowls and other things
              wider than the aperture)
      side  — approach horizontally at mid-height
    """

    def controller(env):
        # Close the distance first. A pick that begins out of reach fails with a
        # grasp error message and hides a navigation problem.
        here = np.array(_obs(env)["robot0_base_pos"])[:2]
        obj0 = np.array(env.sim.data.body_xpos[env.obj_body_id[instance_id]], dtype=float)
        if float(np.linalg.norm(obj0[:2] - here)) > _ARM_REACH:
            yield from drive_to_object(instance_id)(env)

        obj = np.array(env.sim.data.body_xpos[env.obj_body_id[instance_id]], dtype=float)
        gap = float(np.linalg.norm(obj[:2] - np.array(_obs(env)["robot0_base_pos"])[:2]))
        if gap > _ARM_REACH * 1.45:
            return SkillResult(
                False, "pick", 0.0, strategy,
                f"cannot get within reach — {gap:.2f}m away (arm reaches ~{_ARM_REACH}m)",
                dict(out_of_reach=True, distance=gap, instance_id=instance_id),
            )

        base_xy = np.array(_obs(env)["robot0_base_pos"])[:2]
        toward = obj[:2] - base_xy
        toward = toward / max(float(np.linalg.norm(toward)), 1e-6)

        if strategy == "side":
            approach = obj + np.array([-toward[0] * 0.16, -toward[1] * 0.16, 0.02])
            grasp_at = obj.copy()
        elif strategy == "rim":
            # Take the near edge: offset by the object's own radius toward the
            # robot, at the height of its top lip. The fingers then close across
            # a wall a few millimetres thick instead of across the whole object.
            r = _radius(env, instance_id)
            rim = np.array([obj[0] - toward[0] * r, obj[1] - toward[1] * r,
                            _top_z(env, instance_id) - 0.01])
            approach = rim + np.array([0.0, 0.0, 0.13])
            grasp_at = rim
        else:  # top
            approach = obj + np.array([0.0, 0.0, 0.14])
            grasp_at = obj + np.array([0.0, 0.0, 0.015])

        # 1. move above / beside, gripper open
        for _ in range(110):
            a, err = _arm_action(env, approach, -1.0)
            if err < 0.03:
                break
            yield a
        # 2. close in on the grasp point
        for _ in range(90):
            a, err = _arm_action(env, grasp_at, -1.0)
            if err < 0.012:
                break
            yield a
        # 3. close
        a = np.zeros(env.action_dim)
        a[6] = 1.0
        for _ in range(22):
            yield a
        # 4. lift, keeping the gripper shut
        lift_to = _eef_world(env) + np.array([0.0, 0.0, _LIFT])
        for _ in range(70):
            a, err = _arm_action(env, lift_to, 1.0)
            if err < 0.02:
                break
            yield a

        held = bool(OU.check_obj_grasped(env, instance_id))
        obj_now = np.array(env.sim.data.body_xpos[env.obj_body_id[instance_id]], dtype=float)
        raised = float(obj_now[2] - obj[2])
        # Grasped OR visibly lifted: the contact test is strict, and an object
        # carried in a scooped bowl grip can read as ungrasped while plainly up.
        ok = held or raised > 0.05
        return SkillResult(
            ok, "pick", 0.0, strategy,
            f"{'held' if held else 'not held'}, raised {raised:.3f}m",
            dict(grasped=held, raised=raised, instance_id=instance_id),
        )

    return controller


def place(instance_id: str, fixture_name: str):
    """Put whatever is held onto/into a fixture. Judged by where it ends up."""

    def controller(env):
        fxtr = env.fixtures[fixture_name]
        if hasattr(fxtr, "open_door"):
            try:
                if not fxtr.is_open(env):
                    fxtr.open_door(env)
                    env.sim.forward()
            except Exception:
                pass

        try:
            regions = fxtr.get_reset_regions(env=env)
        except TypeError:
            regions = fxtr.get_reset_regions(env)
        except Exception:
            regions = {}
        if regions:
            reg = regions[sorted(regions)[0]]
            drop = np.asarray(
                OU.get_pos_after_rel_offset(fxtr, np.array(reg["offset"])), dtype=float
            )
        else:
            drop = np.array(fxtr.pos, dtype=float)

        above = drop + np.array([0.0, 0.0, 0.20])
        for _ in range(150):
            a, err = _arm_action(env, above, 1.0)
            if err < 0.04:
                break
            yield a
        for _ in range(80):
            a, err = _arm_action(env, drop + np.array([0, 0, 0.06]), 1.0)
            if err < 0.03:
                break
            yield a
        a = np.zeros(env.action_dim)
        a[6] = -1.0
        for _ in range(20):
            yield a
        # retreat so the gripper is not resting on what it just put down
        retreat = _eef_world(env) + np.array([0.0, 0.0, 0.18])
        for _ in range(50):
            a, err = _arm_action(env, retreat, -1.0)
            if err < 0.03:
                break
            yield a
        for _ in range(60):  # let it settle before judging
            yield None

        where = env.locate_detail(instance_id)
        ok = where["location"] == fixture_name
        return SkillResult(
            ok, "place", 0.0, "overhead",
            f"{instance_id} ended in {where['location']} (by {where['method']})",
            dict(location=where["location"], instance_id=instance_id,
                 fixture=fixture_name, method=where["method"]),
        )

    return controller


def open_fixture(name: str):
    """Open a door. The robot drives to it; the latch itself is actuated.

    Being explicit: the door opening is programmatic (`fixture.open_door`), not
    a learned manipulation. The robot really does travel to the fixture first,
    and the result says `assisted` so nothing downstream can mistake this for a
    grasped door handle.
    """

    def controller(env):
        fxtr = env.fixtures.get(name)
        if fxtr is None or not hasattr(fxtr, "open_door"):
            return SkillResult(False, "open", 0.0, "assisted", f"{name} has no door")
        yield from drive_to(name)(env)
        fxtr.open_door(env)
        env.sim.forward()
        for _ in range(10):
            yield None
        return SkillResult(bool(fxtr.is_open(env)), "open", 0.0, "assisted", f"opened {name}")

    return controller


def close_fixture(name: str):
    def controller(env):
        fxtr = env.fixtures.get(name)
        if fxtr is None or not hasattr(fxtr, "close_door"):
            return SkillResult(False, "close", 0.0, "assisted", f"{name} has no door")
        fxtr.close_door(env)
        env.sim.forward()
        for _ in range(10):
            yield None
        return SkillResult(not fxtr.is_open(env), "close", 0.0, "assisted", f"closed {name}")

    return controller


def survey(fixture_name: str):
    """Go and look inside something. This is how exploration happens.

    Returns what was found in `payload["found"]` — the list of object instances
    now locatable in that fixture. An empty list is a real answer: the robot
    looked and there was nothing there.
    """

    def controller(env):
        yield from drive_to(fixture_name)(env)
        fxtr = env.fixtures.get(fixture_name)
        opened = False
        if fxtr is not None and hasattr(fxtr, "open_door"):
            try:
                if not fxtr.is_open(env):
                    fxtr.open_door(env)
                    env.sim.forward()
                    opened = True
            except Exception:
                pass
        for _ in range(12):  # dwell so the cameras actually see the inside
            yield None
        snap = env.world_snapshot()
        found = [i for i, o in snap["objects"].items() if o.get("location") == fixture_name]
        return SkillResult(
            True, "survey", 0.0, "look",
            f"{fixture_name}: {', '.join(found) if found else 'empty'}",
            dict(fixture=fixture_name, found=found, opened=opened,
                 labels={i: snap["objects"][i]["label"] for i in found}),
        )

    return controller


SKILLS = {
    "pick": pick,
    "place": place,
    "drive_to": drive_to,
    "open": open_fixture,
    "close": close_fixture,
    "survey": survey,
}
