"""World-layer acceptance test.

This is the gate: if these fail, nothing above them is worth building.

Structural note — the tests are functions called by a *driver*, not pytest test
functions, and there is exactly one pytest test that runs them all. That is
forced by MuJoCo: on macOS the render context is only valid on the process's
main thread, so the simulation loop must own it. `run_with_driver` boots on the
main thread, runs our checks on a worker, and re-raises any failure here. Booting
once also matters practically — boot is tens of seconds.

Run:  python -m pytest brigade/tests/test_world.py -v -s
"""

from __future__ import annotations

import time

import numpy as np

from brigade.config import CFG
from brigade.world.kitchen import POOL
from brigade.world.sim import SimRunner
from brigade.world.spawn import teleport


def check_kitchen_boots_with_pool(r: SimRunner):
    names = r.call(lambda env: sorted(env.objects.keys()))
    assert names == sorted(i.instance_id for i in POOL), names

    n_fixtures = r.call(lambda env: len(env.fixtures))
    assert n_fixtures > 20, f"only {n_fixtures} fixtures — did the layout load?"

    fx = r.call(lambda env: env.get_ep_meta()["brigade_fixtures"])
    assert fx["home_cabinet"] != fx["stage_cabinet"], (
        "staging shares the home cabinet; dropped bowls would start beside the stack"
    )
    print(f"  [pool] {len(names)} objects, {n_fixtures} fixtures")
    print(f"  [pool] home={fx['home_cabinet']} stage={fx['stage_cabinet']} counter={fx['counter']}")


def check_world_is_readable(r: SimRunner):
    snap = r.call(lambda env: env.world_snapshot())
    assert set(snap) == {"objects", "doors"}

    for iid, o in snap["objects"].items():
        assert "error" not in o, f"{iid}: {o.get('error')}"
        assert len(o["pos"]) == 3
        assert o["label"], f"{iid} has no semantic label"
    labels = sorted({o["label"] for o in snap["objects"].values()})
    assert "bowl" in labels, labels
    assert snap["doors"], "no openable fixtures found"

    located = [o["location"] for o in snap["objects"].values()]
    assert any(loc != "unknown" for loc in located), "nothing could be located"
    print(f"  [read] labels={labels}")
    print(f"  [read] locations={sorted(set(located))}")


def check_doors_actuate_and_report(r: SimRunner):
    cab = r.call(lambda env: env.cab.name)

    def _open(env):
        env.cab.open_door(env)
        env.sim.forward()
        return env.cab.is_open(env)

    def _close(env):
        env.cab.close_door(env)
        env.sim.forward()
        return env.cab.is_open(env)

    assert r.call(_open) is True, f"{cab} did not report open after open_door"
    assert r.call(_close) is False, f"{cab} did not report closed after close_door"
    print(f"  [doors] {cab} opens and closes, state readable")


def check_teleport_onto_counter(r: SimRunner):
    """The 'drop a bowl on the counter' mechanic."""
    counter = r.call(lambda env: env.counter.name)
    before = np.array(r.call(lambda env: env.object_pose("bowl_c")))

    result = teleport(r, "bowl_c", counter, seed=7)

    moved = float(np.linalg.norm(np.array(result["pos"]) - before))
    assert moved > 0.01, f"bowl_c did not move (delta {moved:.4f} m)"
    assert result["settled_location"] != "unknown", result
    assert result["label"] == "bowl"
    print(f"  [teleport] bowl_c moved {moved:.2f}m -> {result['settled_location']}")


def check_teleport_into_cabinet_is_containment(r: SimRunner):
    """Spatial memory needs locate() to name a container, not a surface."""
    cab = r.call(lambda env: env.cab.name)
    result = teleport(r, "bowl_d", cab, seed=3)
    assert result["settled_location"] == cab, (
        f"expected bowl_d inside {cab}, got {result['settled_location']!r}"
    )
    print(f"  [teleport] bowl_d -> {result['settled_location']} (containment detected)")


def check_control_rate_is_realtime(r: SimRunner):
    hold = r.call(lambda env: np.zeros(env.action_dim))

    def controller(env):
        for _ in range(60):
            yield hold

    t0 = time.time()
    res = r.run_skill("bench_hold", controller, timeout_s=30.0)
    achieved = 60 / (time.time() - t0)

    assert res.ok, res.detail
    assert achieved >= CFG.world.control_freq * 0.8, f"only {achieved:.1f} Hz"
    print(f"  [rate] {achieved:.1f} Hz sustained (target {CFG.world.control_freq} Hz)")
    print(f"  [rate] status: {r.status()}")


def check_frames_are_published(r: SimRunner):
    for cam in CFG.world.cameras:
        f = r.frame(cam)
        assert f is not None, f"no frame for {cam}"
        assert f.ndim == 3 and f.shape[2] == 3, f.shape
        assert f.dtype == np.uint8
        assert int(f.max()) > 0, f"{cam} frame is entirely black"
    print(f"  [frames] {len(CFG.world.cameras)} cameras publishing {r.frame().shape}")


def check_skill_failure_is_reported_not_raised(r: SimRunner):
    """A controller that explodes must produce a failure result, not kill the sim."""

    def bad(env):
        yield None
        raise RuntimeError("deliberate")

    res = r.run_skill("bad_skill", bad, timeout_s=10.0)
    assert res.ok is False
    assert "deliberate" in res.detail
    assert r.running, "sim died on a controller error"
    assert r.call(lambda env: env.action_dim) == 12
    print(f"  [resilience] {res.as_text('bowl_c')!r}; sim still up")


CHECKS = (
    check_kitchen_boots_with_pool,
    check_world_is_readable,
    check_doors_actuate_and_report,
    check_teleport_onto_counter,
    check_teleport_into_cabinet_is_containment,
    check_control_rate_is_realtime,
    check_frames_are_published,
    check_skill_failure_is_reported_not_raised,
)


def test_world_layer():
    runner = SimRunner()
    t0 = time.time()

    def driver(r: SimRunner):
        print(f"\n[boot] kitchen up in {time.time() - t0:.1f}s")
        passed = []
        for check in CHECKS:
            print(f"\n--- {check.__name__}")
            check(r)
            passed.append(check.__name__)
        return passed

    passed = runner.run_with_driver(driver, timeout_s=900.0)
    assert len(passed) == len(CHECKS)
    print(f"\n[world] {len(passed)}/{len(CHECKS)} checks passed")
