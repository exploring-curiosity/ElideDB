"""SimRunner — owns the MuJoCo kitchen on the process's MAIN thread.

MEASURED CONSTRAINT, not a style preference: on macOS MuJoCo renders through a
CGL context that is only valid on the main thread. Building the env on a worker
thread does not raise — the process dies silently, no traceback, mid-render. We
reproduced that, and confirmed the same build succeeds on the main thread.
(`MUJOCO_GL=osmesa`, which would have allowed off-thread rendering, is not
available in this MuJoCo build.)

So the shape is inverted from the usual web app: **the simulation runs the main
thread, and everything else — the HTTP server, the agent loop, the perception
pass — runs in background threads** and talks to the world through two
thread-safe primitives:

    call(fn)      -- run fn(env) on the sim thread, get the return value back
    submit(job)   -- run a multi-step controller to completion, get a result back

Never touch `runner.env` from another thread. That is the whole rule, and the
failure mode for breaking it is a silent process death rather than an exception.

The loop paces itself to wall-clock at control_freq. We measured 60-71 Hz of
capability against 20 Hz of need, so pacing is a sleep, not a struggle; the
headroom is what lets perception share the machine without the kitchen going
slow-motion on camera.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

from ..config import CFG, WorldConfig


@dataclass
class SkillResult:
    """What a controller did. This is the unit that becomes a memory."""

    ok: bool
    skill: str
    seconds: float
    strategy: str = "default"
    detail: str = ""
    payload: dict = field(default_factory=dict)

    def as_text(self, subject: str = "") -> str:
        """The self-description that gets embedded and stored.

        Deliberately terse and factual — this string is the thing the robot will
        later search against, so it should read like something you would type
        into a search box.
        """
        head = f"{self.skill} {subject}".strip()
        verdict = "success" if self.ok else "failure"
        bits = [head, verdict, f"{self.seconds:.1f}s", f"strategy={self.strategy}"]
        if self.detail:
            bits.append(self.detail)
        return " — ".join(bits)


# A controller is a generator: given the env, yield an action per control step,
# and return (by raising StopIteration via `return`) when finished. Yielding None
# means "hold still this step".
Controller = Callable[[Any], Iterator[np.ndarray | None]]


@dataclass
class _Job:
    name: str
    controller: Controller
    timeout_s: float
    subject: str = ""
    strategy: str = "default"
    done: threading.Event = field(default_factory=threading.Event)
    result: SkillResult | None = None


class _Call:
    __slots__ = ("fn", "done", "value", "error")

    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.value = None
        self.error: BaseException | None = None


class SimRunner:
    """The world, running."""

    def __init__(self, cfg: WorldConfig | None = None):
        self.cfg = cfg or CFG.world
        self.env = None
        self._stop = threading.Event()
        self._booted = threading.Event()
        self._loop_thread_id: int | None = None

        self._jobs: "queue.Queue[_Job]" = queue.Queue()
        self._calls: "queue.Queue[_Call]" = queue.Queue()

        self._frames: dict[str, np.ndarray] = {}
        self._frames_lock = threading.Lock()
        self._tick = 0
        self._hz = 0.0
        self._current_job: str | None = None
        self._looping = False

    # ---- lifecycle ----------------------------------------------------------

    def boot(self) -> None:
        """Build the kitchen. MUST be called on the thread that will run the loop.

        Slow — tens of seconds, dominated by loading object meshes; placement
        itself is fast now that the sampling windows fit the shelves. Brigade
        resets exactly once, so this is a startup cost, not a per-task one. It is
        worth being loud about: a caller who thinks this is instant sees a hang.
        """
        if self.env is not None:
            return
        self._loop_thread_id = threading.get_ident()
        self.env = self._build()
        obs = self.env.reset()
        self._store_frames(obs)
        self._booted.set()

    def run_forever(self) -> None:
        """Run the control loop until stop() is called. Blocks the main thread."""
        if self.env is None:
            self.boot()
        self._loop_thread_id = threading.get_ident()
        self._looping = True
        try:
            self._loop()
        finally:
            self._looping = False

    def run_with_driver(self, driver: Callable[["SimRunner"], Any], timeout_s: float = 900.0) -> Any:
        """Boot on this (main) thread, run `driver` on a worker, loop until it finishes.

        This is how both the tests and the app get to "sim owns the main thread"
        without every caller having to invert its own control flow. The driver's
        return value comes back here, and its exception is re-raised here, so a
        failure inside the worker still fails the caller.
        """
        self.boot()
        box: dict[str, Any] = {}

        def _wrapped():
            try:
                box["value"] = driver(self)
            except BaseException as exc:  # surfaced on the caller's thread below
                box["error"] = exc
            finally:
                self._stop.set()

        # Mark the loop live BEFORE the driver starts. Otherwise the driver's
        # first call() races us and sees "sim loop is not running"; queued calls
        # made in that window are simply serviced once the loop turns.
        self._loop_thread_id = threading.get_ident()
        self._looping = True
        deadline = time.time() + timeout_s
        worker = threading.Thread(target=_wrapped, name="brigade-driver", daemon=True)
        worker.start()
        try:
            self._loop(deadline=deadline)
        finally:
            self._looping = False
        worker.join(timeout=10.0)
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        """True when the world exists and the loop is turning."""
        return self.env is not None and self._looping and not self._stop.is_set()

    # ---- talking to the world ----------------------------------------------

    def call(self, fn: Callable[[Any], Any], timeout_s: float = 60.0) -> Any:
        """Run fn(env) on the sim thread and return its value.

        Every read of the world goes through here. It is deliberately the only
        door: if you find yourself holding `runner.env` on another thread, that
        is the bug, and on macOS it is a silent process death rather than a
        stack trace.
        """
        if self.env is None:
            raise RuntimeError("kitchen has not booted")
        # Called from the loop thread itself (or before the loop starts): run it
        # inline, because queueing here would wait for a loop that is us.
        if threading.get_ident() == self._loop_thread_id and not self._looping:
            return fn(self.env)
        if not self.running:
            raise RuntimeError("sim loop is not running")
        c = _Call(fn)
        self._calls.put(c)
        if not c.done.wait(timeout_s):
            raise TimeoutError(f"sim call timed out after {timeout_s}s")
        if c.error is not None:
            raise c.error
        return c.value

    def submit(
        self,
        name: str,
        controller: Controller,
        timeout_s: float | None = None,
        subject: str = "",
        strategy: str = "default",
    ) -> _Job:
        job = _Job(
            name=name,
            controller=controller,
            timeout_s=timeout_s if timeout_s is not None else CFG.agent.skill_timeout_s,
            subject=subject,
            strategy=strategy,
        )
        self._jobs.put(job)
        return job

    def run_skill(self, name: str, controller: Controller, **kw) -> SkillResult:
        """submit + wait. The normal way to do something."""
        job = self.submit(name, controller, **kw)
        # +5s so we observe the runner's own timeout rather than racing it.
        if not job.done.wait(job.timeout_s + 5.0):
            return SkillResult(False, name, job.timeout_s, kw.get("strategy", "default"),
                               "controller did not return")
        assert job.result is not None
        return job.result

    # ---- observation --------------------------------------------------------

    def frame(self, camera: str | None = None) -> np.ndarray | None:
        """Latest RGB frame for a camera (HWC uint8), or None before first step."""
        cam = camera or self.cfg.cameras[0]
        with self._frames_lock:
            f = self._frames.get(cam)
            return None if f is None else f.copy()

    def status(self) -> dict:
        return dict(
            running=self.running,
            tick=self._tick,
            hz=round(self._hz, 1),
            current_job=self._current_job,
            queued_jobs=self._jobs.qsize(),
        )

    # ---- the loop -----------------------------------------------------------

    def _build(self):
        import robosuite
        import robocasa  # noqa: F401  (registers kitchen assets)

        # Importing any kitchen module registers all 396 envs; without this the
        # registry only has base robosuite tasks.
        import robocasa.environments.kitchen.atomic.kitchen_pick_place  # noqa: F401
        from robosuite.controllers import load_composite_controller_config

        from .kitchen import BrigadeKitchen  # noqa: F401  (registers BrigadeKitchen)

        cfg = self.cfg
        ctrl = load_composite_controller_config(robot=cfg.robot)
        return robosuite.make(
            env_name="BrigadeKitchen",
            robots=cfg.robot,
            controller_configs=ctrl,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=list(cfg.cameras),
            camera_heights=cfg.camera_h,
            camera_widths=cfg.camera_w,
            control_freq=cfg.control_freq,
            ignore_done=True,
            layout_ids=cfg.layout_id,
            style_ids=cfg.style_id,
            seed=cfg.seed,
        )

    def _loop(self, deadline: float | None = None) -> None:
        dt = 1.0 / self.cfg.control_freq
        hold = np.zeros(self.env.action_dim)
        active: Iterator[np.ndarray | None] | None = None
        job: _Job | None = None
        job_started = 0.0
        last = time.time()

        while not self._stop.is_set():
            if deadline is not None and time.time() > deadline:
                break
            t0 = time.time()

            # Service world reads first: they are cheap and callers are blocked.
            while True:
                try:
                    c = self._calls.get_nowait()
                except queue.Empty:
                    break
                try:
                    c.value = c.fn(self.env)
                except BaseException as exc:
                    c.error = exc
                finally:
                    c.done.set()

            if active is None and job is None:
                try:
                    job = self._jobs.get_nowait()
                    active = job.controller(self.env)
                    job_started = time.time()
                    self._current_job = job.name
                except queue.Empty:
                    pass
                except BaseException as exc:
                    self._finish(job, False, 0.0, f"controller failed to start: {exc}")
                    job, active = None, None

            action = hold
            if active is not None and job is not None:
                elapsed = time.time() - job_started
                if elapsed > job.timeout_s:
                    self._finish(job, False, elapsed, "timeout")
                    job, active = None, None
                else:
                    try:
                        nxt = next(active)
                        action = hold if nxt is None else np.asarray(nxt, dtype=float)
                    except StopIteration as stop:
                        # A controller may `return SkillResult(...)` to report
                        # its own verdict; otherwise finishing means success.
                        val = stop.value
                        if isinstance(val, SkillResult):
                            val.seconds = elapsed
                            self._finish_with(job, val)
                        else:
                            self._finish(job, True, elapsed, "")
                        job, active = None, None
                    except BaseException as exc:
                        self._finish(job, False, elapsed, f"error: {exc}")
                        job, active = None, None

            try:
                obs, _, _, _ = self.env.step(action)
            except BaseException as exc:
                # A physics blow-up must not take the process with it. Report it,
                # drop the job, keep the kitchen alive.
                if job is not None:
                    self._finish(job, False, time.time() - job_started, f"sim error: {exc}")
                    job, active = None, None
                continue

            self._store_frames(obs)
            self._tick += 1
            now = time.time()
            inst = 1.0 / max(now - last, 1e-6)
            self._hz = inst if self._hz == 0 else 0.9 * self._hz + 0.1 * inst
            last = now

            slack = dt - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)

        # Shutting down: fail anything still waiting rather than leaving callers
        # blocked on a loop that has stopped turning.
        self._current_job = None
        if job is not None:
            self._finish(job, False, 0.0, "sim stopped")
        while True:
            try:
                self._finish(self._jobs.get_nowait(), False, 0.0, "sim stopped")
            except queue.Empty:
                break
        while True:
            try:
                c = self._calls.get_nowait()
            except queue.Empty:
                break
            c.error = RuntimeError("sim stopped")
            c.done.set()

    # ---- internals ----------------------------------------------------------

    def _store_frames(self, obs: dict) -> None:
        frames = {}
        for cam in self.cfg.cameras:
            img = obs.get(f"{cam}_image")
            if img is not None:
                # robosuite renders bottom-up; flip so humans and vision models
                # both see the kitchen the right way up.
                frames[cam] = np.ascontiguousarray(img[::-1])
        if frames:
            with self._frames_lock:
                self._frames.update(frames)

    def _finish(self, job: _Job | None, ok: bool, seconds: float, detail: str) -> None:
        if job is None:
            return
        self._finish_with(job, SkillResult(ok, job.name, seconds, job.strategy, detail))

    def _finish_with(self, job: _Job, result: SkillResult) -> None:
        result.payload.setdefault("subject", job.subject)
        job.result = result
        self._current_job = None
        job.done.set()
