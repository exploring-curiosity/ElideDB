"""The dashboard: a web thread reading a sim that owns the main thread.

The thread split is not a style choice. MuJoCo on macOS renders through CGL and
a GL context belongs to the thread that created it, so the policy's camera
observations must be produced on the process's main thread. Uvicorn therefore
runs in a daemon thread and the simulation keeps the main one — the inverse of
the usual arrangement, and the reason `run.py` is the entry point rather than
`uvicorn brigade.api.server:app`.

Between the two sits `Live`, one lock and a few slots. The sim never writes to a
socket and never awaits anything: it drops the latest transform buffer into a
slot and moves on. A broadcast task in the event loop samples that slot at its
own rate. Consequences worth stating, because they are the design:

  * a slow or absent browser cannot slow the robot down — frames are dropped,
    never queued, so the viewer shows the present rather than a backlog;
  * the sim's clock is the robot's control loop (57 ms/step measured), and the
    viewer's is 30 Hz, and neither has to know about the other.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("brigade.api")
STATIC = Path(__file__).parent / "static"


def _dumps(obj) -> str:
    """JSON for the wire, with `default=str`.

    Rows come back from psycopg2 with real `datetime` and `Decimal` values in
    them — a norm carries `updated_at` — and one of those anywhere inside a
    status frame raises mid-send, killing the websocket and freezing the whole
    dashboard for a field nothing was going to display anyway.
    """
    return json.dumps(obj, default=str)


@dataclass
class Live:
    """Everything the web thread may look at. Guarded by one lock."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    frame: bytes | None = None
    seq: int = 0
    scene_header: dict | None = None
    scene_blob: bytes = b""
    scene_epoch: int = 0            # bumped when the geometry itself changes
    status: dict = field(default_factory=lambda: dict(state="booting", detail=""))
    feed: deque = field(default_factory=lambda: deque(maxlen=200))
    ledger: list = field(default_factory=list)
    commands: "queue.Queue[dict]" = field(default_factory=queue.Queue)

    # ---- written by the sim thread -----------------------------------------

    def publish_frame(self, buf: bytes) -> None:
        with self.lock:
            self.frame = buf
            self.seq += 1

    def publish_scene(self, header: dict, blob: bytes) -> None:
        with self.lock:
            self.scene_header, self.scene_blob = header, blob
            self.scene_epoch += 1
            self.frame = None

    def set_status(self, state: str, detail: str = "", **extra) -> None:
        with self.lock:
            self.status = dict(state=state, detail=detail, ts=time.time(), **extra)

    def say(self, kind: str, text: str, **extra) -> None:
        """One line for the on-screen feed. This is the narration a viewer reads."""
        with self.lock:
            self.feed.append(dict(ts=time.time(), kind=kind, text=text, **extra))

    def record(self, row: dict) -> None:
        with self.lock:
            self.ledger.append(row)

    # ---- read by the web thread --------------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            return dict(status=self.status, seq=self.seq,
                        scene_epoch=self.scene_epoch,
                        feed=list(self.feed)[-60:], ledger=list(self.ledger))


LIVE = Live()


def create_app(live: Live, memory_getter) -> FastAPI:
    app = FastAPI(title="Brigade")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/vendor", StaticFiles(directory=STATIC / "vendor"), name="vendor")

    # ---- the scene ---------------------------------------------------------

    @app.get("/api/scene")
    def scene():
        with live.lock:
            if live.scene_header is None:
                return JSONResponse(dict(ready=False), status_code=503)
            return JSONResponse(dict(ready=True, epoch=live.scene_epoch, **live.scene_header))

    @app.get("/api/scene.bin")
    def scene_bin():
        with live.lock:
            blob = live.scene_blob
        return Response(blob, media_type="application/octet-stream")

    # ---- live feed ---------------------------------------------------------

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        last_seq, last_epoch, last_status = -1, -1, 0.0
        try:
            while True:
                with live.lock:
                    seq, frame = live.seq, live.frame
                    epoch = live.scene_epoch
                if epoch != last_epoch:
                    # Geometry changed under the client; tell it to refetch
                    # rather than drawing new transforms onto the old scene.
                    await sock.send_text(_dumps(dict(t="scene", epoch=epoch)))
                    last_epoch, last_seq = epoch, -1
                elif frame is not None and seq != last_seq:
                    await sock.send_bytes(frame)
                    last_seq = seq
                now = time.time()
                if now - last_status > 0.25:
                    await sock.send_text(_dumps(dict(t="status", **live.snapshot())))
                    last_status = now
                await asyncio.sleep(1 / 30)
        except (WebSocketDisconnect, RuntimeError):
            return

    # ---- commands ----------------------------------------------------------

    @app.post("/api/command")
    async def command(body: dict):
        req = (body.get("request") or "").strip()
        if not req:
            return JSONResponse(dict(error="empty request"), status_code=400)
        live.commands.put(dict(request=req, use_memory=bool(body.get("use_memory", True))))
        return dict(queued=True, request=req)

    @app.post("/api/teach")
    async def teach(body: dict):
        """A human states where something belongs. Outranks anything learned."""
        mem = memory_getter()
        label, place = (body.get("label") or "").strip(), (body.get("location") or "").strip()
        if not (label and place):
            return JSONResponse(dict(error="label and location required"), status_code=400)
        norm = mem.instruct_norm(label, place)
        live.say("memory", f"told: {label} belongs in {place}")
        return dict(norm=dict(norm) if norm else None)

    # ---- what is in the memory --------------------------------------------

    @app.get("/api/memory")
    def memory():
        mem = memory_getter()
        return dict(
            stats=mem.stats(),
            norms=[dict(r) for r in mem.norms()],
            beliefs=[_jsonable(r) for r in mem.beliefs()],
            events=[_jsonable(r) for r in mem.recent(24)],
            decisions=[_jsonable(r) for r in mem.decision_log(12)],
        )

    @app.post("/api/recall")
    def recall(body: dict):
        """Run a recall by hand — the search box over the robot's own past."""
        mem = memory_getter()
        q = (body.get("query") or "").strip()
        t0 = time.perf_counter()
        hits = mem.recall(q, k=int(body.get("k", 8)), floor=0.0)
        ms = (time.perf_counter() - t0) * 1e3
        return dict(query=q, latency_ms=round(ms, 2), hits=[
            dict(id=h.id, text=h.text, score=round(h.score, 4),
                 outcome=h.outcome, subject=h.subject, kind=h.kind) for h in hits])

    @app.get("/api/similar")
    def similar(recording_id: str = "", k: int = 5):
        """RelMo: the same question asked of video instead of text."""
        from ..memory.relmo import similar_recordings

        return dict(hits=similar_recordings(memory_getter().db, recording_id, k))

    @app.post("/api/forget")
    def forget():
        memory_getter().wipe()
        live.say("memory", "memory wiped — the robot has no history")
        return dict(ok=True)

    return app


def _jsonable(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        out[k] = v.isoformat() if hasattr(v, "isoformat") else (
            str(v) if not isinstance(v, (str, int, float, bool, type(None), list, dict)) else v)
    return out


def serve_in_thread(app: FastAPI, host: str = "0.0.0.0", port: int = 8099) -> threading.Thread:
    """Run uvicorn off the main thread, which the simulator needs for itself.

    Bound to 0.0.0.0 rather than 127.0.0.1 because `localhost` resolves to ::1
    first on macOS: an IPv4-only bind then refuses the browser's connection
    while `lsof` cheerfully shows the port listening, which reads as "the server
    crashed" and is not that at all.
    """
    import uvicorn

    cfg = uvicorn.Config(app, host=host, port=port, log_level="warning", ws_ping_interval=None)
    server = uvicorn.Server(cfg)
    th = threading.Thread(target=server.run, daemon=True, name="brigade-web")
    th.start()
    return th
