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
# Steps of physics to let it come to rest before anyone looks.
_SETTLE_STEPS = 60


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
    # Jitter within the region so repeated drops are not stacked in one spot,
    # but stay clear of the edges.
    ox += float(rng.uniform(-0.30, 0.30)) * w
    oy += float(rng.uniform(-0.30, 0.30)) * d
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

        # An openable fixture must be open or the object lands inside a closed
        # door and the scene is nonsense. Opening it is a world change, so it is
        # reported back and the caller writes it to memory.
        opened = False
        if hasattr(fixture, "open_door"):
            try:
                if not fixture.is_open(env):
                    fixture.open_door(env)
                    opened = True
            except Exception:
                pass

        joint = env.objects[instance_id].joints[0]
        qpos = np.array(env.sim.data.get_joint_qpos(joint), dtype=float)
        qpos[:3] = target + np.array([0.0, 0.0, _DROP_CLEARANCE])
        # Upright, so a bowl lands as a bowl rather than on its rim.
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        for _ in range(_SETTLE_STEPS):
            env.sim.step()
        env.sim.forward()

        settled = np.array(env.sim.data.body_xpos[env.obj_body_id[instance_id]], dtype=float)
        return dict(
            instance_id=instance_id,
            label=env.object_label(instance_id),
            requested_fixture=fixture_name,
            settled_location=env.locate(instance_id),
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
