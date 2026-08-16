"""The Pass — Brigade's operator console.

An HTTP surface over the running kitchen so a human can watch it and poke it.
Runs in a background thread; the simulation owns the main thread (see
world/sim.py for why that is not negotiable on macOS).

Every endpoint reaches the world through `runner.call(...)`. None of them touch
`runner.env`. Frames are the one exception and they are safe: `runner.frame()`
returns a copy taken under a lock by the sim thread.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..config import CFG
from ..world.sim import SimRunner
from ..world.spawn import move_secretly, teleport

log = logging.getLogger("brigade.api")

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class DropRequest(BaseModel):
    instance_id: str
    fixture: str | None = None
    secret: bool = False


class DoorRequest(BaseModel):
    fixture: str
    action: str  # open | close


class JogRequest(BaseModel):
    axis: str  # base_x | base_y | base_yaw | arm_x | arm_y | arm_z | gripper
    amount: float = 1.0
    steps: int = 12


# Which slice of the 12-d action vector each control drives.
# Verified layout: arm 0:6, gripper 6:7, base 7:10, torso 10, mode 11.
_JOG_AXES = {
    "base_x": (7, 1.0),
    "base_y": (8, 1.0),
    "base_yaw": (9, 1.0),
    "arm_x": (0, 1.0),
    "arm_y": (1, 1.0),
    "arm_z": (2, 1.0),
    "gripper": (6, 1.0),
}


class InstructRequest(BaseModel):
    """A human telling the robot something. Two shapes only.

    A rule ("bowls go in the cabinet") writes a NORM and changes future
    unprompted behaviour. A job ("put the bowl away") writes a high-priority
    TASK. Both enter the same queue the robot writes to itself.
    """

    text: str
    label: str | None = None
    location: str | None = None
    instance_id: str | None = None


def create_app(runner: SimRunner, memory=None, agent=None) -> FastAPI:
    app = FastAPI(title="Brigade — The Pass", docs_url="/api/docs")
    started = time.time()

    # ---- page ------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(_STATIC, "index.html")) as fh:
            return fh.read()

    # ---- world -----------------------------------------------------------

    def _require_running():
        if not runner.running:
            raise HTTPException(503, "kitchen is not running")

    @app.get("/api/state")
    def state():
        if not runner.running:
            return dict(ready=False, status=runner.status(), uptime_s=time.time() - started)
        snap = runner.call(lambda env: env.world_snapshot())
        meta = runner.call(lambda env: env.get_ep_meta()["brigade_fixtures"])
        return dict(
            ready=True,
            uptime_s=round(time.time() - started, 1),
            status=runner.status(),
            fixtures=meta,
            cameras=list(CFG.world.cameras),
            objects=snap["objects"],
            doors=snap["doors"],
            kitchen_id=CFG.world.kitchen_id,
            robot_id=CFG.world.robot_id,
        )

    @app.get("/api/targets")
    def targets():
        _require_running()
        return runner.call(lambda env: env.placement_targets())

    # ---- poke the world --------------------------------------------------

    @app.post("/api/drop")
    def drop(req: DropRequest):
        _require_running()
        fixture = req.fixture or runner.call(lambda env: env.counter.name)
        fn = move_secretly if req.secret else teleport
        try:
            return fn(runner, req.instance_id, fixture)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/door")
    def door(req: DoorRequest):
        _require_running()
        if req.action not in ("open", "close"):
            raise HTTPException(400, "action must be open or close")

        def _do(env):
            fxtr = env.fixtures.get(req.fixture)
            if fxtr is None:
                raise KeyError(req.fixture)
            if not hasattr(fxtr, "open_door"):
                raise ValueError(f"{req.fixture} has no door")
            (fxtr.open_door if req.action == "open" else fxtr.close_door)(env)
            env.sim.forward()
            return dict(fixture=req.fixture, is_open=bool(fxtr.is_open(env)))

        try:
            return runner.call(_do)
        except KeyError as exc:
            raise HTTPException(404, f"no fixture {exc}") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/jog")
    def jog(req: JogRequest):
        """Manual teleop, so a human can confirm the robot really is controllable.

        Not a skill — skills are built on top of this same submit() path, but
        this one exists purely so the console can prove the loop is live.
        """
        _require_running()
        if req.axis not in _JOG_AXES:
            raise HTTPException(400, f"axis must be one of {sorted(_JOG_AXES)}")
        idx, scale = _JOG_AXES[req.axis]
        amount = float(np.clip(req.amount, -1.0, 1.0)) * scale
        steps = int(np.clip(req.steps, 1, 60))

        def controller(env):
            a = np.zeros(env.action_dim)
            a[idx] = amount
            # The base needs its control mode selected; mode is the last slot.
            if req.axis.startswith("base"):
                a[11] = -1.0
            for _ in range(steps):
                yield a

        res = runner.run_skill("jog", controller, timeout_s=15.0, subject=req.axis)
        return dict(ok=res.ok, seconds=round(res.seconds, 2), detail=res.detail, axis=req.axis)

    # ---- vision ----------------------------------------------------------

    def _jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    @app.get("/snapshot/{camera}")
    def snapshot(camera: str):
        frame = runner.frame(camera)
        if frame is None:
            raise HTTPException(404, f"no frame for {camera}")
        return Response(_jpeg(frame), media_type="image/jpeg")

    # ---- the 3D view -----------------------------------------------------

    class ViewRequest(BaseModel):
        azimuth: float | None = None
        elevation: float | None = None
        distance: float | None = None
        lookat_x: float | None = None
        lookat_y: float | None = None
        lookat_z: float | None = None

    @app.post("/api/view")
    def set_view(req: ViewRequest):
        _require_running()
        return runner.set_view(**req.model_dump())

    @app.get("/api/view")
    def get_view():
        return runner.view()

    @app.get("/stream3d")
    def stream3d(width: int = 0, height: int = 0):
        """MJPEG from the free orbit camera — the whole kitchen in 3D.

        Rendered on demand rather than every control step: this is a second
        render pass and there is no reason to pay for it when nobody is looking.
        Capped at 12 fps so dragging the view can never starve the 20 Hz control
        loop it shares a thread with.

        `width`/`height` resize the RESULT. They do not resize the renderer —
        rebuilding it would destroy the GL context and permanently break every
        later render.
        """
        def gen():
            period = 1.0 / 12.0
            while runner.running:
                t0 = time.time()
                try:
                    frame = runner.render_free()
                except Exception:
                    break
                if frame is not None:
                    if width and height and (frame.shape[1], frame.shape[0]) != (width, height):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    payload = _jpeg(frame, quality=72)
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
                        + payload + b"\r\n"
                    )
                slack = period - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stream/{camera}")
    def stream(camera: str):
        """MJPEG. Deliberately capped below the control rate — the console must
        never be the reason the kitchen misses a control step."""
        if camera not in CFG.world.cameras:
            raise HTTPException(404, f"unknown camera {camera}")

        def gen():
            period = 1.0 / 15.0
            while runner.running:
                t0 = time.time()
                frame = runner.frame(camera)
                if frame is not None:
                    payload = _jpeg(frame)
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
                        + payload + b"\r\n"
                    )
                slack = period - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    # ---- events ----------------------------------------------------------

    # ---- the agent -------------------------------------------------------

    @app.get("/api/agent")
    def agent_state():
        """What the robot is doing and why — the console's whole point."""
        if agent is None:
            return dict(present=False, feed=[], status=dict(running=False))
        return dict(present=True, status=agent.status(), feed=agent.feed(50))

    @app.post("/api/agent/pause")
    def agent_pause(on: bool = True):
        if agent is None:
            raise HTTPException(503, "no agent")
        agent.paused = bool(on)
        return dict(paused=agent.paused)

    @app.get("/api/memory")
    def memory_state():
        if memory is None:
            raise HTTPException(503, "memory offline")
        return dict(
            stats=memory.stats(),
            beliefs=memory.beliefs(),
            norms=memory.norms(),
            tasks=memory.task_board(12),
            decisions=memory.decision_log(8),
            events=memory.recent(14),
            skills=memory.skill_table(),
        )

    @app.get("/api/memory/recall")
    def memory_recall(q: str, k: int = 5):
        """Search the robot's memory the way the robot does."""
        if memory is None:
            raise HTTPException(503, "memory offline")
        t0 = time.time()
        hits = memory.recall(q, k=k, floor=0.0)
        return dict(
            query=q,
            latency_ms=round((time.time() - t0) * 1000, 1),
            hits=[dict(score=round(h.score, 3), text=h.text, kind=h.kind,
                       outcome=h.outcome, subject=h.subject) for h in hits],
        )

    @app.post("/api/instruct")
    def instruct(req: InstructRequest):
        """Give the robot a rule or a job. Both are memory writes."""
        if memory is None:
            raise HTTPException(503, "memory offline")
        if req.label and req.location:
            norm = memory.instruct_norm(req.label, req.location)
            return dict(kind="norm", label=req.label, location=req.location,
                        confidence=norm["confidence"], source=norm["source"])
        if req.instance_id:
            label = runner.call(lambda env: env.object_label(req.instance_id))
            tid = memory.enqueue(req.text or f"put the {label} away", origin="user",
                                 priority=9, subject=req.instance_id,
                                 payload=dict(instance_id=req.instance_id, label=label))
            memory.remember(f"asked to: {req.text}", kind="instruction",
                            subject=req.instance_id)
            return dict(kind="task", task_id=tid, priority=9)
        raise HTTPException(400, "give either (label, location) for a rule, or instance_id")

    @app.post("/api/forget")
    def forget():
        """Wipe memory so the cold-start behaviour can be shown again."""
        if memory is None:
            raise HTTPException(503, "memory offline")
        memory.wipe()
        if agent is not None:
            agent._known_locations.clear()
        return dict(ok=True, stats=memory.stats())

    @app.get("/api/health")
    def health():
        return dict(
            sim=runner.status(),
            memory=_memory_health(memory),
            agent=(agent.status() if agent is not None else dict(running=False)),
            uptime_s=round(time.time() - started, 1),
        )

    return app


def _memory_health(memory=None) -> dict:
    """Report the memory layer honestly, including when it is absent.

    The console shows this in red when it is down, because "the agent stops when
    its memory stops" is the claim being made and it should be visible, not
    hidden behind a fallback.
    """
    try:
        from ..memory.db import DB

        ok = DB.healthy()
        out = dict(connected=ok, dsn=DB._safe_dsn())
        if ok:
            out["flavor"] = DB.flavor
            out["native_vectors"] = DB.supports_vector()
        return out
    except Exception as exc:
        return dict(connected=False, error=str(exc)[:200])


def serve_in_background(runner: SimRunner, host: str = "127.0.0.1", port: int = 8080,
                        memory=None, agent=None) -> threading.Thread:
    """Start uvicorn on a worker thread and return it.

    The main thread must go on to run the simulation loop.
    """
    import uvicorn

    app = create_app(runner, memory=memory, agent=agent)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="brigade-http", daemon=True)
    thread.start()
    return thread
