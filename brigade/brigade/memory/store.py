"""The video store. Cameras in, spans out, nothing else.

This replaces the belief/norm/skill-stat store, every column of which was a
derived fact saved as state and forbidden by the owner's ruling. What is left is
deliberately thin: write video, index it by RelMo, retrieve by similarity.

**Spans, not episodes.** The cameras run on a clock and the store cuts on that
clock, whether or not the robot is doing anything. A memory that only records
during tasks cannot answer a question about the time between them, and it makes
the memory depend on the agent correctly deciding when something interesting is
beginning — the very judgement the memory exists to support.

**A sliding span, not a tile.** Every HOP_S seconds the store indexes the last
SPAN_S seconds. Consecutive rows therefore overlap, which is not redundancy: it
is what stops an event from being cut in half by an arbitrary boundary, and it
is why every indexed row has real temporal extent. The first version cut
disjoint 5 s tiles, and 5 s is below RelMo's resolution — the encoder tiles
4.0 s windows on a 2.0 s hop, so a 5 s clip yielded ONE window, eight 0.25 s
descriptor steps, and the DTW stage had nothing to align. Measured consequence:
1-NN behaviour match 0.300 against a 0.100 chance. SPAN_S = 15 s gives six
windows and 48 steps.

**Retrieval is two-stage, and the first stage is the database's job.**

    stage 1   SELECT ... ORDER BY embedding <=> q LIMIT M      (pgvector HNSW)
    stage 2   DTW over the traces of those M                   (RelMo, sidecar)

That split is RelMo's own (`vjstore.Store.query`), and moving stage 1 into SQL
is the whole architectural claim: the coarse stage belongs in the database once
a corpus outgrows one process's memory, which is the same reason CockroachDB
ships a vector index at all. Stage 2 stays in the sidecar because it needs the
full trace, and a pooled vector cannot tell apart two behaviours that visit the
same pixels in a different order.

**The encode is not on the write path.** Cutting a span writes an mp4 and
enqueues it; a background worker asks RelMo for the vector and the trace and
fills them in. Encoding takes ~1 s and the simulator's clock does not stop for
it. Rows exist briefly with a NULL embedding, which retrieval skips — a clip is
in the store the moment it is on disk and becomes *findable* a second later.
"""

from __future__ import annotations

import collections
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

# RelMo tiles 4.0 s encoder windows on a 2.0 s hop, so descriptor steps arrive
# in blocks of eight per window and a span shorter than ~8 s has no temporal
# extent worth aligning. 15 s -> 6 windows -> 48 steps.
SPAN_S = 15.0            # seconds of video each indexed row covers
HOP_S = 5.0              # seconds between rows; rows overlap by SPAN_S - HOP_S
FPS = 10.0               # memory-camera rate (every other 20 Hz control step)
PREFILTER_M = 32         # candidates the database hands stage 2


@dataclass
class Clip:
    clip_id: str
    path: str
    t0: float
    t1: float
    score: float = 0.0
    stage: str = "prefilter"     # which stage produced this score


class VideoStore:
    """Continuous camera -> spans -> RelMo -> the database."""

    def __init__(self, db: Database | None = None, relmo=None,
                 root: str | None = None, camera: str = "agentview"):
        self.db = db or DB
        self.relmo = relmo
        self.camera = camera
        self.root = root or os.path.join(CFG.artifacts, "clips")
        self.kitchen = CFG.world.kitchen_id
        self.robot = CFG.world.robot_id

        self._buf: collections.deque = collections.deque(
            maxlen=int(SPAN_S * FPS))
        self._since = 0                      # frames since the last cut
        # Timestamps come from the VIDEO clock — frames written / FPS — not
        # from wall time. The two disagree by however long encoding, rendering
        # and policy inference took, and it is video time a human means by
        # "when did that happen", since it is the only clock the stored frames
        # are actually on.
        self._epoch = time.time()
        self._n = 0
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
        # CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so the
        # stage-2 columns have to be added explicitly for a store that predates
        # them. Rows written before the change keep a NULL trace and are simply
        # never re-projected, so they retire into the old basis namespace and
        # drop out of retrieval on their own.
        text = "STRING" if self.db.flavor == "cockroach" else "TEXT"
        vec768 = "VECTOR(768)" if native else "FLOAT8[]"
        for col, typ in (("trace_path", text), ("steps", "INT"),
                         ("motion", vec768)):
            self.db.execute(f"ALTER TABLE clips ADD COLUMN IF NOT EXISTS {col} {typ}")
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
            try:
                self.db.execute(
                    "CREATE VECTOR INDEX IF NOT EXISTS ON clips (motion)"
                    if self.db.flavor == "cockroach" else
                    "CREATE INDEX IF NOT EXISTS clips_motion_hnsw ON clips "
                    "USING hnsw (motion vector_cosine_ops)")
                idx.append("clips.motion")
            except Exception as exc:
                log.warning("no vector index on motion: %s", exc)
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
        """Feed one memory-camera frame. Returns a clip_id when a span cuts.

        Called from the simulator's own loop, so it does nothing expensive: it
        appends to a ring and, every HOP_S, hands a copy of the ring to the
        writer. The ring is why spans overlap — the buffer is never emptied,
        only advanced.
        """
        with self._lock:
            self._buf.append(frame)
            self._since += 1
            self._n += 1
            if self._since < int(HOP_S * FPS) or len(self._buf) < int(SPAN_S * FPS):
                return None
            buf, t0, t1 = self._span()
        return self._cut(buf, t0, t1)

    def _span(self) -> tuple[list, float, float]:
        buf = list(self._buf)
        self._since = 0
        t1 = self._epoch + self._n / FPS
        return buf, t1 - len(buf) / FPS, t1

    def flush(self) -> str | None:
        """Cut whatever is buffered. For the end of a session, not the loop."""
        with self._lock:
            if len(self._buf) < int(8.0 * FPS):      # below one useful trace
                return None
            buf, t0, t1 = self._span()
        return self._cut(buf, t0, t1)

    def new_session(self) -> str | None:
        """Flush and clear the ring: the next span starts fresh.

        Used when the video stream is genuinely discontinuous — a different
        scene, a restarted simulator — because a sliding span across a cut
        would index fifteen seconds that never happened consecutively. It is
        NOT for episode boundaries in a continuous kitchen, where the whole
        point is that the camera does not care where one task ends.
        """
        cid = self.flush()
        with self._lock:
            self._buf.clear()
            self._since = 0
        return cid

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
                # The motion view is derived from the trace the sidecar just
                # wrote, in this process, with numpy — no second encode and no
                # round trip. Two reductions of one pass over the video.
                from . import traces as TR

                mot = TR.feature(clip_id, blocks=("std_sig",))
                self.db.execute(
                    """UPDATE clips SET embedding = %s, basis_id = %s,
                                        trace_path = %s, steps = %s, motion = %s
                       WHERE clip_id = %s""",
                    (self.db.vector_param(vec), self.relmo.basis,
                     meta.get("trace"), int(meta.get("steps") or 0),
                     self.db.vector_param(mot) if mot is not None else None,
                     clip_id))
                self.n_encoded += 1
            except Exception as exc:  # noqa: BLE001 — a bad clip must not end the shift
                log.warning("encode error on %s: %s", clip_id, exc)

    def pending(self) -> int:
        return self._encode_q.qsize()

    def drain(self, timeout: float = 300.0) -> int:
        """Block until every written span is INDEXED. -> rows still missing.

        Waits on the artefact, not on the queue. An empty queue means the last
        clip has been dequeued, not that its UPDATE has landed — and a fixed
        sleep after it is a guess. Measured cost of guessing: one span of a
        59-span collection missed the basis refit by milliseconds and stayed in
        the old namespace, which `basis_id` correctly hid from retrieval and
        which nothing else would have noticed.
        """
        t = time.time()
        while time.time() - t < timeout:
            if self.relmo is None or not self.relmo.ready:
                break
            missing = int(self.db.query(
                "SELECT count(*) AS n FROM clips WHERE kitchen_id = %s "
                "AND embedding IS NULL", (self.kitchen,))[0]["n"])
            if not missing and not self.pending():
                return 0
            time.sleep(0.5)
        return self.pending()

    # ---- the basis ----------------------------------------------------------

    def backfill_motion(self) -> int:
        """Fill the motion view for rows written before the column existed.

        Reads the traces already on disk — no video is decoded twice, and no
        model runs. -> number of rows filled.
        """
        from . import traces as TR

        rows = self.db.query(
            "SELECT clip_id FROM clips WHERE kitchen_id = %s "
            "AND trace_path IS NOT NULL AND motion IS NULL", (self.kitchen,))
        n = 0
        for r in rows:
            v = TR.feature(r["clip_id"], blocks=("std_sig",))
            if v is None:
                continue
            self.db.execute("UPDATE clips SET motion = %s WHERE clip_id = %s",
                            (self.db.vector_param(v), r["clip_id"]))
            n += 1
        if n:
            log.info("backfilled the motion view for %d spans", n)
        return n

    def refit_basis(self, reset: bool = False) -> dict:
        """Change the whitening basis, then re-project every row.

        Both halves matter. Changing the basis alone would leave the table
        holding vectors in the old namespace while `basis_id` claimed the new
        one, which is exactly the silent corruption the id exists to prevent.

        `reset=True` returns to the bootstrap (RoboCasa) basis, which is the
        DEFAULT for this deployment and not an escape hatch: measured over 111
        spans with overlapping spans barred, RoboCasa scores 0.685 on
        behaviour matching and a basis fitted on this kitchen scores 0.324. A
        whitening basis has to be fitted on a corpus wider than the question
        asked of it, and 111 overlapping spans of one room are not.
        """
        if self.relmo is None or not self.relmo.ready:
            return dict(error="RelMo is not running")
        ids = [r["clip_id"] for r in self.db.query(
            "SELECT clip_id FROM clips WHERE kitchen_id = %s AND trace_path IS NOT NULL",
            (self.kitchen,))]
        r = self.relmo.reset_basis() if reset else self.relmo.fit(ids)
        if not r.get("ok"):
            return r
        self.relmo.basis = r["basis"]
        vecs = self.relmo.project(ids)
        for cid, v in vecs.items():
            self.db.execute(
                "UPDATE clips SET embedding = %s, basis_id = %s WHERE clip_id = %s",
                (self.db.vector_param(v), r["basis"], cid))
        log.info("refit basis %s over %d clips; re-projected %d",
                 r["basis"], r["n"], len(vecs))
        return dict(**r, reprojected=len(vecs))

    # ---- the read path ------------------------------------------------------

    def similar(self, vector, k: int = 8, query_id: str | None = None,
                prefilter_m: int = PREFILTER_M, rerank: bool = True,
                since: float | None = None, until: float | None = None,
                view: str = "motion") -> tuple[list[Clip], dict]:
        """Nearest spans by what they LOOKED like. -> (clips, timing).

        Stage 1 is the SQL below — RelMo's prefilter, computed by the vector
        index rather than by numpy. Stage 2 re-ranks that shortlist by DTW,
        which needs the query's own trace and is therefore skipped when the
        caller has only a vector to search with.

        `since`/`until` bound the search in time. This is the other half of what
        the owner's ruling allows a video memory to return — "similar clips or
        its timestamps" — and it is a pushdown, not a post-filter: the predicate
        runs beside the vector scan, so a question about last Tuesday never
        touches the rest of the corpus.
        """
        # WHICH VIEW STAGE 1 RANKS BY. `embedding` is RelMo's canonical
        # prefilter — a mean over the trace, and the thing `selfcheck` proves
        # the database computes exactly. `motion` is the per-channel spread over
        # the same trace. Measured on this kitchen, nearest-neighbour behaviour
        # match with overlapping spans barred: mean 0.475, motion 0.712. The
        # default is the one that measures better; the other stays reachable so
        # the comparison can be re-run rather than remembered.
        col = "motion" if view == "motion" else "embedding"
        sim = self.db.similarity_expr(col)
        t = time.perf_counter()
        m = max(k, prefilter_m) if rerank and query_id else k
        where, args = f" AND {col} IS NOT NULL", []
        if since is not None:
            where += " AND t0 >= to_timestamp(%s)"
            args.append(float(since))
        if until is not None:
            where += " AND t1 <= to_timestamp(%s)"
            args.append(float(until))
        # The basis filter guards the WHITENED view only. A whitening basis is a
        # namespace and comparing across two of them returns confident nonsense;
        # the motion view is a raw per-channel spread with no basis in it, so
        # applying the filter there would exclude rows for a reason that does
        # not apply to them.
        if col == "embedding":
            where += " AND basis_id = %s"
            args.append(getattr(self.relmo, "basis", None) or self._basis())
        rows = self.db.query(
            f"""SELECT clip_id, path, extract(epoch from t0) AS a,
                       extract(epoch from t1) AS b, {sim} AS score
                FROM clips
                WHERE kitchen_id = %s{where}
                ORDER BY score DESC LIMIT %s""",
            (self.db.vector_param(vector), self.kitchen, *args, m),
        )
        sql_ms = (time.perf_counter() - t) * 1e3
        hits = [Clip(r["clip_id"], r["path"], float(r["a"]), float(r["b"]),
                     float(r["score"]), stage=f"stage1:{col}") for r in rows]
        timing = dict(stage1_ms=round(sql_ms, 2), candidates=len(hits),
                      stage2_ms=0.0, reranked=False, view=col)

        if rerank and query_id and self.relmo is not None and self.relmo.ready and hits:
            scores, ms = self.relmo.rank(query_id, [h.clip_id for h in hits])
            if scores:
                for h in hits:
                    if h.clip_id in scores:
                        h.score, h.stage = scores[h.clip_id], "dtw"
                hits.sort(key=lambda h: (h.stage == "dtw", h.score), reverse=True)
                timing.update(stage2_ms=round(ms, 2), reranked=True)
        return hits[:k], timing

    def _basis(self) -> str:
        r = self.db.query(
            "SELECT basis_id FROM clips WHERE kitchen_id = %s AND basis_id IS NOT NULL "
            "ORDER BY t0 DESC LIMIT 1", (self.kitchen,))
        return r[0]["basis_id"] if r else ""

    def recent(self, k: int = 8) -> list[Clip]:
        rows = self.db.query(
            """SELECT clip_id, path, extract(epoch from t0) AS a,
                      extract(epoch from t1) AS b
               FROM clips WHERE kitchen_id = %s AND embedding IS NOT NULL
               ORDER BY t0 DESC LIMIT %s""", (self.kitchen, k))
        return [Clip(r["clip_id"], r["path"], float(r["a"]), float(r["b"]))
                for r in rows]

    def await_vector(self, clip_id: str, timeout: float = 60.0,
                     view: str = "motion") -> np.ndarray | None:
        """Block until this clip is findable. -> its vector, or None.

        The write path is deliberately asynchronous, so a span exists on disk
        seconds before it exists in the index. A query that uses the robot's
        LIVE view has to wait for exactly one encode, and only that one.
        """
        col = "motion" if view == "motion" else "embedding"
        t = time.time()
        while time.time() - t < timeout:
            r = self.db.query(
                f"SELECT {col} AS v FROM clips WHERE clip_id = %s AND {col} IS NOT NULL",
                (clip_id,))
            if r:
                return _parse_vec(r[0]["v"])
            if self.relmo is None or not self.relmo.ready:
                return None
            time.sleep(0.25)
        return None

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
                      coalesce(sum(n_frames)/nullif(max(fps),0), 0) AS seconds,
                      coalesce(avg(steps), 0) AS steps
               FROM clips WHERE kitchen_id = %s""", (self.kitchen,))[0]
        return dict(clips=int(row["n"]), indexed=int(row["n_vec"]),
                    seconds=float(row["seconds"] or 0),
                    mean_steps=round(float(row["steps"] or 0), 1),
                    basis=self._basis(), pending=self.pending(),
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
