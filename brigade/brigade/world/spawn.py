"""Teleporting objects around the kitchen.

This is how the demo pokes the world: "a bowl appears on the counter", "someone
moved the bowl while the robot wasn't looking". Both are the same mechanic —
write an object's free joint qpos and let physics settle.

Why teleport rather than spawn: MuJoCo cannot add a body to a compiled model at
runtime. Recompiling mid-run would drop the render context and every joint state
with it. So the whole cast exists from boot (see kitchen.POOL) and "appearing" is
a move from somewhere out of frame. This is the standard trick, and it is honest:
the object is really there, really has mass, and really has to be picked up.

Placement targets come from RoboCasa's own reset regions, so an object lands
where the scene generator would have put it — on a counter's top surface, on a
cabinet's shelf — rather than at a coordinate we invented.
"""

from __future__ import annotations

import numpy as np
from robocasa.utils import object_utils as OU

from .sim import SimRunner

# Drop height above the target surface. Enough that settling is visible and the
# object cannot spawn interpenetrating a shelf; small enough that it does not
# bounce off the counter.
_DROP_CLEARANCE = 0.06
# Settle until the object is actually at rest, not for a fixed count. A fixed 60
# steps (~0.12 s of sim time) is almost exactly the freefall time for a 6 cm drop,
# so it returned "still moving" about half the time and locate() reported
# 'unknown' for an object that was a moment away from resting on the counter.
# Spatial memory is written from that read, so it has to be a settled one.
#
# Rest is measured by POSITION STABILITY, not velocity. An object resting on a
# stack of other objects keeps a persistent contact jitter — a bowl parked on two
# other bowls never gets below 1e-3 m/s and looked "never settled" for 600 steps
# while sitting perfectly still. What we actually care about is whether it has
# stopped going anywhere.
_SETTLE_MAX_STEPS = 800
_SETTLE_MIN_STEPS = 40
_REST_WINDOW = 25       # steps to look back over
_REST_DRIFT = 5e-4      # metres of travel across that window that still counts as still


def _region_world_pos(env, fixture, rng: np.random.Generator) -> np.ndarray | None:
    """A world-frame point on one of the fixture's placement regions."""
    try:
        try:
            regions = fixture.get_reset_regions(env=env)
        except TypeError:
            # Counter's signature takes env positionally and wants a reference.
            regions = fixture.get_reset_regions(env)
    except Exception:
        return None
    if not regions:
        return None

    name = sorted(regions)[int(rng.integers(len(regions)))]
    reg = regions[name]
    ox, oy, oz = reg["offset"]
    w, d = reg["size"]
    # Jitter within the region so repeated drops are not stacked in one spot —
    # but biased to the FRONT half. This is where a person actually leaves a
    # mug, and it matters mechanically: an object against the backsplash is
    # ~0.75 m from any legal stance, beyond the arm's 0.62 m envelope, so a
    # back-of-counter drop creates a task no stance can complete.
    ox += float(rng.uniform(-0.30, 0.30)) * w
    oy += float(rng.uniform(-0.42, -0.10)) * d
    try:
        # Respects the fixture's rotation; the naive pos+offset does not.
        return np.asarray(OU.get_pos_after_rel_offset(fixture, np.array([ox, oy, oz])), dtype=float)
    except Exception:
        return np.asarray(fixture.pos, dtype=float) + np.array([ox, oy, oz])


def teleport(runner: SimRunner, instance_id: str, fixture_name: str, seed: int | None = None) -> dict:
    """Move an object onto a fixture. Returns where it actually ended up.

    The return value is deliberately the *settled* pose read back from physics,
    not the pose we asked for — if the object rolled off, the caller (and the
    event log) should say so.
    """

    def _do(env):
        rng = np.random.default_rng(seed)
        fixture = env.fixtures.get(fixture_name)
        if fixture is None:
            raise KeyError(f"no fixture named {fixture_name!r}")
        if instance_id not in env.objects:
            raise KeyError(f"no object named {instance_id!r} in the pool")

        target = _region_world_pos(env, fixture, rng)
        if target is None:
            target = np.asarray(fixture.pos, dtype=float)

        # Doors are NEVER touched here. This is the demo god-hand: an object may
        # materialise inside a closed cabinet (that is where "someone put the
        # mugs away last week" comes from), but a door swinging with nobody near
        # it is exactly the fakery this project was told to remove. Only the
        # robot's gripper moves doors (world/doors.py).
        opened = False

        joint = env.objects[instance_id].joints[0]
        qpos = np.array(env.sim.data.get_joint_qpos(joint), dtype=float)
        qpos[:3] = target + np.array([0.0, 0.0, _DROP_CLEARANCE])
        # Upright, so a bowl lands as a bowl rather than on its rim.
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.sim.data.set_joint_qpos(joint, qpos)
        # Zero the velocity too: a free joint keeps whatever it had, so an object
        # nudged a moment ago would launch itself off the new surface.
        env.sim.data.set_joint_qvel(joint, np.zeros(6))
        env.sim.forward()

        body = env.obj_body_id[instance_id]
        history: list[np.ndarray] = []
        steps, at_rest = 0, False
        for i in range(_SETTLE_MAX_STEPS):
            env.sim.step()
            steps = i + 1
            history.append(np.array(env.sim.data.body_xpos[body], dtype=float))
            if steps < max(_SETTLE_MIN_STEPS, _REST_WINDOW):
                continue
            drift = float(np.linalg.norm(history[-1] - history[-_REST_WINDOW]))
            if drift < _REST_DRIFT:
                at_rest = True
                break
        env.sim.forward()

        settled = np.array(env.sim.data.body_xpos[body], dtype=float)
        where = env.locate_detail(instance_id)
        return dict(
            instance_id=instance_id,
            label=env.object_label(instance_id),
            requested_fixture=fixture_name,
            settled_location=where["location"],
            located_by=where["method"],
            location_confidence=where["confidence"],
            settled_steps=steps,
            at_rest=at_rest,
            pos=settled.tolist(),
            opened_door=opened,
        )

    return runner.call(_do, timeout_s=60.0)


def drop_into_play(runner: SimRunner, instance_id: str, fixture_name: str | None = None, seed: int | None = None) -> dict:
    """'A bowl appears on the counter.' The user-facing poke."""
    if fixture_name is None:
        fixture_name = runner.call(lambda env: env.counter.name)
    return teleport(runner, instance_id, fixture_name, seed=seed)


def move_secretly(runner: SimRunner, instance_id: str, fixture_name: str, seed: int | None = None) -> dict:
    """'Someone moved it while the robot wasn't looking.'

    Mechanically identical to a drop. It is a separate function because the
    caller must NOT write a perception event for it — the whole point is that
    the robot's memory is now wrong and has to discover that for itself.
    """
    return teleport(runner, instance_id, fixture_name, seed=seed)
