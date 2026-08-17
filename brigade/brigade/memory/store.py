"""The video store. Cameras in, segments out, nothing else.

This replaces the previous `store.py`, which held beliefs, norms, skill stats
and text-embedded events. Every one of those was a derived fact saved as state,
which the owner's ruling forbids. What is left is deliberately thin: write
video, index it by RelMo vector and time, retrieve by similarity.

**Segments, not episodes.** The cameras run on a clock and are cut every
`SEGMENT_S` seconds whether or not the robot is doing anything. A memory that
only records during tasks cannot answer a question about the time between them,
and it also makes the memory depend on the agent correctly deciding when
something interesting is happening — which is the very judgement the memory
exists to support.

**The encode is not on the write path.** Cutting a segment writes an mp4 and
enqueues it; a background worker asks the RelMo sidecar for the vector and
fills it in. Encoding takes ~3 s and the simulator's clock does not stop for
it. Rows therefore exist briefly with a NULL embedding, which retrieval skips —
a clip is in the store the moment it is on disk and becomes *findable* a few
seconds later.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

from ..config import CFG
from .db import DB, Database

log = logging.getLogger("brigade.store")

# RelMo's encoder window is 4.0 s and it REFUSES a shorter clip outright
# ("clip is 3.0s; the encoder window is 4.0s"). A 3 s segment therefore wrote
# 75 rows and indexed none of them — present in the store, invisible to
# retrieval. 5 s leaves margin for the last, short segment of an episode.
SEGMENT_S = 5.0          # seconds of video per stored clip
FPS = 10.0               # memory-camera rate (every other 20 Hz control step)


@dataclass
class Clip:
    clip_id: str
    path: str
    t0: float
    t1: float
    score: float = 0.0


class VideoStore:
    """Continuous camera -> segments -> RelMo vectors -> the database."""

    def __init__(self, db: Database | None = None, relmo=None,
                 root: str | None = None, camera: str = "agentview"):
        self.db = db or DB
        self.relmo = relmo
        self.camera = camera
        self.root = root or os.path.join(CFG.artifacts, "clips")
        self.kitchen = CFG.world.kitchen_id
        self.robot = CFG.world.robot_id

        self._buf: list[np.ndarray] = []
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._encode_q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self.n_written = 0
        self.n_encoded = 0

    # ---- schema -------------------------------------------------------------

    def setup(self, fresh: bool = False) -> dict:
        """Create the v2 schema. `fresh=True` drops the forbidden v1 tables."""
        import re

        from .db import to_postgres

        if fresh:
            for t in ("object_beliefs", "norms", "norm_evidence", "events",
                      "skill_stats", "decisions", "relmo_recordings"):
                self.db.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
            log.info("dropped v1 fact tables — memory is video only now")

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema_v2.sql")
        body = open(path).read()
        native = self.db.supports_vector()
        if self.db.flavor != "cockroach":
            body = to_postgres(body, native_vectors=native)
        for stmt in [s.strip() for s in body.split(";") if s.strip()]:
            if not re.match(r"^\s*--", stmt):
                self.db.execute(stmt)
        if self.db.flavor != "cockroach":
            self.db.execute("CREATE INDEX IF NOT EXISTS clips_by_time "
                            "ON clips (kitchen_id, t0 DESC)")
            self.db.execute("CREATE INDEX IF NOT EXISTS turns_by_time "
                            "ON turns (kitchen_id, ts DESC)")
        idx = []
        if native:
            try:
                self.db.execute(
                    "CREATE VECTOR INDEX IF NOT EXISTS ON clips (embedding)"
                    if self.db.flavor == "cockroach" else
                    "CREATE INDEX IF NOT EXISTS clips_embedding_hnsw ON clips "
                    "USING hnsw (embedding vector_cosine_ops)")
                idx.append("clips.embedding")
            except Exception as exc:
                log.warning("no vector index on clips: %s", exc)
        return dict(native_vectors=native, flavor=self.db.flavor, vector_indexes=idx)

    # ---- the write path -----------------------------------------------------

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._encode_loop, daemon=True,
                                            name="brigade-encode")
            self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    def write(self, frame: np.ndarray) -> str | None:
        """Feed one memory-camera frame. Returns a clip_id when a segment cuts.

        Called from the simulator's own loop, so it does nothing expensive: it
        appends to a list and, every SEGMENT_S, hands a finished buffer to the
        writer thread.
        """
        with self._lock:
            self._buf.append(frame)
            if len(self._buf) < int(SEGMENT_S * FPS):
                return None
            buf, t0 = self._buf, self._t0
            self._buf, self._t0 = [], time.time()
        return self._cut(buf, t0, time.time())

    def _cut(self, buf: list[np.ndarray], t0: float, t1: float) -> str:
        from .relmo import write_clip

        clip_id = f"c-{uuid.uuid4().hex[:12]}"
        path = os.path.join(self.root, f"{clip_id}.mp4")
        write_clip(buf, path, fps=int(FPS))
        self.db.execute(
            """INSERT INTO clips (clip_id, kitchen_id, robot_id, camera,
                                  t0, t1, n_frames, fps, path)
               VALUES (%s,%s,%s,%s, to_timestamp(%s), to_timestamp(%s), %s,%s,%s)""",
            (clip_id, self.kitchen, self.robot, self.camera,
             t0, t1, len(buf), FPS, path),
        )
        self.n_written += 1
        self._encode_q.put((clip_id, path))
        return clip_id

    def _encode_loop(self) -> None:
        """Fill in vectors behind the simulator. See the module docstring."""
        while not self._stop.is_set():
            try:
                clip_id, path = self._encode_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.relmo is None or not self.relmo.ready:
                continue
            try:
                vec, meta = self.relmo.encode(path, clip_id)
                if vec is None:
                    log.warning("encode failed for %s: %s", clip_id, meta.get("error"))
                    continue
                self.db.execute(
                    "UPDATE clips SET embedding = %s, basis_id = %s WHERE clip_id = %s",
                    (self.db.vector_param(vec), self.relmo.basis, clip_id))
                self.n_encoded += 1
            except Exception as exc:  # noqa: BLE001 — a bad clip must not end the shift
                log.warning("encode error on %s: %s", clip_id, exc)

    def pending(self) -> int:
        return self._encode_q.qsize()

    # ---- the read path ------------------------------------------------------

    def similar(self, vector, k: int = 8) -> tuple[list[Clip], float]:
        """Nearest segments by what they LOOKED like. -> (clips, latency_ms)."""
        sim = self.db.similarity_expr("embedding")
        t = time.perf_counter()
        rows = self.db.query(
            f"""SELECT clip_id, path, extract(epoch from t0) AS a,
                       extract(epoch from t1) AS b, {sim} AS score
                FROM clips
                WHERE kitchen_id = %s AND embedding IS NOT NULL
                ORDER BY score DESC LIMIT %s""",
            (self.db.vector_param(vector), self.kitchen, k),
        )
        ms = (time.perf_counter() - t) * 1e3
        return [Clip(r["clip_id"], r["path"], float(r["a"]), float(r["b"]),
                     float(r["score"])) for r in rows], ms

    def recent(self, k: int = 8) -> list[Clip]:
        rows = self.db.query(
            """SELECT clip_id, path, extract(epoch from t0) AS a,
                      extract(epoch from t1) AS b
               FROM clips WHERE kitchen_id = %s AND embedding IS NOT NULL
               ORDER BY t0 DESC LIMIT %s""", (self.kitchen, k))
        return [Clip(r["clip_id"], r["path"], float(r["a"]), float(r["b"]))
                for r in rows]

    def vectors_for(self, clip_ids: list[str]) -> np.ndarray:
        rows = self.db.query(
            "SELECT clip_id, embedding FROM clips WHERE clip_id = ANY(%s)",
            (list(clip_ids),))
        by = {r["clip_id"]: _parse_vec(r["embedding"]) for r in rows}
        return np.stack([by[c] for c in clip_ids if c in by]) if by else np.zeros((0, 512))

    def stats(self) -> dict:
        row = self.db.query(
            """SELECT count(*) AS n,
                      count(embedding) AS n_vec,
                      coalesce(sum(n_frames)/nullif(max(fps),0), 0) AS seconds
               FROM clips WHERE kitchen_id = %s""", (self.kitchen,))[0]
        return dict(clips=int(row["n"]), indexed=int(row["n_vec"]),
                    seconds=float(row["seconds"] or 0), pending=self.pending(),
                    written=self.n_written, encoded=self.n_encoded)

    # ---- audit --------------------------------------------------------------

    def log_turn(self, heard: str, read_clips: list[str], weights, margin: float,
                 retrieval_ms: float, reason_ms: float) -> str:
        rows = self.db.execute(
            """INSERT INTO turns (kitchen_id, heard, read_clips, weights, margin,
                                  retrieval_ms, reason_ms)
               VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (self.kitchen, heard, list(read_clips),
             [float(x) for x in (weights if weights is not None else [])],
             float(margin), float(retrieval_ms), float(reason_ms)))
        return str(rows[0]["id"])

    def finish_turn(self, turn_id: str, acted: bool, ok: bool | None,
                    seconds: float) -> None:
        self.db.execute(
            "UPDATE turns SET acted=%s, succeeded=%s, seconds=%s WHERE id=%s",
            (acted, ok, float(seconds), turn_id))


def _parse_vec(v) -> np.ndarray:
    if isinstance(v, str):
        return np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(v, dtype=np.float32)
