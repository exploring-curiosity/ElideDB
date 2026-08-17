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

    @app.get("/api/scene.pack")
    def scene_pack():
        """Header and geometry in ONE response, under ONE lock.

        Fetching /api/scene then /api/scene.bin is two requests with a gap in
        the middle, and the scene can be republished in that gap — the client
        then builds meshes using one epoch's byte offsets against another
        epoch's bytes. Serving both together makes that impossible rather than
        unlikely.

            uint32 little-endian   length of the JSON header, INCLUDING padding
            bytes                  the header, space-padded to a 4-byte boundary
            bytes                  the geometry blob

        The padding is not tidiness. The blob is read on the client as
        Float32Array views at `4 + header_length + offset`, and a typed-array
        view whose byte offset is not a multiple of its element size throws
        `RangeError: start offset of Float32Array should be a multiple of 4`.
        A JSON header is whatever length it happens to be, so three times in
        four the whole scene fails to build. Trailing spaces are legal JSON
        whitespace, so the header still parses as-is.
        """
        with live.lock:
            if live.scene_header is None:
                return Response(status_code=503)
            head = _dumps(dict(ready=True, epoch=live.scene_epoch, **live.scene_header))
            blob = live.scene_blob
        hb = head.encode()
        hb += b" " * (-len(hb) % 4)
        return Response(len(hb).to_bytes(4, "little") + hb + blob,
                        media_type="application/octet-stream")

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

    # ---- what is in the memory ---------------------------------------------
    #
    # There is no /api/teach and no /api/forget any more, and their absence is
    # the design. Teaching a norm wrote a fact ("the bowl belongs in the
    # cabinet") into a table, which the owner's ruling removes: the memory is
    # video, and where the bowl belongs is derived at read time or not at all.
    # Nothing here can be told anything.

    @app.get("/api/memory")
    def memory():
        """What the store holds. Note what a viewer CANNOT be shown: there is
        no belief, no norm and no label, because no such column exists."""
        store = memory_getter()
        return dict(stats=store.stats(),
                    clips=[_clip(c) for c in store.recent(24)])

    @app.get("/api/clip")
    def clip(clip_id: str = ""):
        """The video itself. A memory you can watch is the point of a video
        memory — a similarity score is a claim, the clip is the evidence."""
        store = memory_getter()
        row = store.db.query("SELECT path FROM clips WHERE clip_id = %s", (clip_id,))
        if not row or not Path(row[0]["path"]).exists():
            return JSONResponse(dict(error="no such clip"), status_code=404)
        return FileResponse(row[0]["path"], media_type="video/mp4")

    @app.post("/api/recall")
    def recall(body: dict):
        """Retrieval by hand, with the two stages timed SEPARATELY.

        The split is the architecture, so the UI reports it rather than a single
        latency: stage 1 is the database's vector index computing RelMo's
        prefilter over the whole store; stage 2 is RelMo's DTW over the traces
        of the shortlist. One is a SQL query, the other is the part a pooled
        vector cannot do.
        """
        store = memory_getter()
        qid = (body.get("clip_id") or "").strip()
        if not qid:
            recent = store.recent(1)
            if not recent:
                return JSONResponse(dict(error="nothing in the store yet"),
                                    status_code=404)
            qid = recent[0].clip_id
        vec = store.await_vector(qid, timeout=5.0)
        if vec is None:
            return JSONResponse(dict(error="that clip is not indexed yet"),
                                status_code=409)
        hits, timing = store.similar(vec, k=int(body.get("k", 8)), query_id=qid)
        return dict(query=qid, **timing,
                    hits=[_clip(c) for c in hits if c.clip_id != qid])

    @app.get("/api/turns")
    def turns(k: int = 20):
        """The audit trail: what was heard, which clips were read, what was
        commanded. Written after the fact; the recall path never reads it."""
        store = memory_getter()
        return dict(turns=[_jsonable(r) for r in store.db.query(
            "SELECT * FROM turns WHERE kitchen_id = %s ORDER BY ts DESC LIMIT %s",
            (store.kitchen, k))])

    return app


def _clip(c) -> dict:
    return dict(clip_id=c.clip_id, t0=c.t0, t1=c.t1,
                score=round(float(c.score), 4), stage=getattr(c, "stage", ""),
                seconds=round(c.t1 - c.t0, 1))


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
