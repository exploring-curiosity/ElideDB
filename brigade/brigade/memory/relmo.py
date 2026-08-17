"""Video memory: RelMo's retrieval, with its first stage inside the database.

Text memory answers "have I been *told* anything like this?". This answers a
question text cannot: **"have I ever seen anything that looked like this?"** —
a query by example over what the robot's own cameras recorded, with no caption,
no label and no name for what happened.

The split of labour is the interesting part:

    RelMo (sidecar)   video -> 512-d whitened, pooled V-JEPA 2 + SigLIP 2
    Postgres/CRDB     that vector, HNSW-indexed, ranked by `<=>`
    RelMo (sidecar)   [optional] exact DTW re-rank over the survivors

RelMo's own prefilter scores candidates with `0.5*(pf@qf + ps@qs)`. Storing
`concat(qf, qs)/sqrt(2)` makes that quantity exactly the inner product of two
unit vectors, so the database's cosine operator computes RelMo's prefilter
rather than approximating it. Measured, not assumed: `selfcheck` in the sidecar
reports max abs error **2.3e-08** against RelMo's internal path.

That is what makes this more than "we also put some embeddings in a table". The
coarse stage of a real retrieval system was moved into the database, which is
where a coarse stage belongs when the corpus outgrows one process's memory —
the same reason CockroachDB ships C-SPANN at all.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("brigade.relmo")

_ROOT = Path(__file__).resolve().parents[3]
_SIDECAR = _ROOT / "brigade" / "sidecar" / "relmo_encode.py"
# RelMo's interpreter, not Brigade's: it needs transformers 4.57 for VJEPA2Model
# while the control loop is pinned to 4.53.2 for pi0.5. See the sidecar's docstring.
_PYTHON = os.environ.get("BRIGADE_RELMO_PYTHON", str(_ROOT / "myenv" / "bin" / "python"))


class RelMoSidecar:
    """A persistent RelMo process. Start once; encode many.

    Loading the encoders and the whitening basis costs ~85 s. Per clip is ~3 s.
    A subprocess per episode would therefore spend 96% of its life loading, and
    video memory would stop being an online stage — which is the one thing it is
    not allowed to stop being.
    """

    def __init__(self, python: str = _PYTHON, basis: str | None = None):
        self.python = python
        self.basis = basis
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.ready = False
        self.n_basis = 0
        self.error: str | None = None

    def available(self) -> bool:
        return Path(self.python).exists() and _SIDECAR.exists()

    def start(self, wait: bool = True, timeout: float = 240.0) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return self.ready
        if not self.available():
            self.error = f"no RelMo interpreter at {self.python}"
            log.warning("%s — video memory disabled", self.error)
            return False
        log.info("starting RelMo sidecar (loads V-JEPA 2 + basis, ~85s)")
        self.proc = subprocess.Popen(
            [self.python, "-u", str(_SIDECAR)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, cwd=str(_ROOT),
        )
        self.proc.stdout.readline()          # the booting line
        if not wait:
            return False
        r = self._rpc(dict(cmd="ping", basis=self.basis), timeout=timeout)
        if r.get("ok"):
            self.ready = True
            self.basis = r.get("basis")
            self.n_basis = int(r.get("n_basis", 0))
            log.info("RelMo ready: basis %r over %d recordings", self.basis, self.n_basis)
        else:
            self.error = str(r.get("error", "sidecar did not answer"))
            log.warning("RelMo sidecar failed: %s", self.error)
        return self.ready

    def _rpc(self, req: dict, timeout: float = 120.0) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            return dict(error="sidecar not running")
        with self.lock:
            try:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
            except (BrokenPipeError, ValueError) as exc:
                return dict(error=f"sidecar pipe closed: {exc}")
        if not line:
            return dict(error="sidecar closed its output")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return dict(error=f"unparseable reply: {line[:120]}")

    def encode(self, clip_path: str, rec_id: str) -> tuple[np.ndarray | None, dict]:
        r = self._rpc(dict(clip=str(clip_path), id=rec_id))
        if "vec" not in r:
            return None, r
        return np.asarray(r["vec"], dtype=np.float32), r

    # ---- the store's own basis ---------------------------------------------

    def fit(self, ids: list[str] | None = None) -> dict:
        """Refit the whitening on this kitchen's own video. NOT the default.

        RelMo fits per store because whitening removes the variance a corpus
        SHARES, which is a property of the deployment. It still loses here, and
        the reason is instructive: measured over 111 spans with overlapping
        spans barred, RoboCasa's basis scores 0.685 on behaviour matching and a
        basis fitted on this kitchen scores 0.324. 111 heavily overlapping spans
        of one room share the very variance that separates the behaviours, so
        fitting on them whitens the signal away. A basis needs a corpus wider
        than the question asked of it.

        An earlier measurement on 5 s clips said the opposite (0.300 -> 0.400).
        It was scoring near-duplicate matching, and the sliding-span overlap was
        not yet barred. Kept here so the reversal is on the record.
        """
        return self._rpc(dict(cmd="fit", ids=ids), timeout=300.0)

    def reset_basis(self) -> dict:
        """Return to the bootstrap (RoboCasa) basis — the measured default."""
        return self._rpc(dict(cmd="fit", reset=True), timeout=300.0)

    def project(self, ids: list[str]) -> dict[str, np.ndarray]:
        """Re-project stored traces under the current basis.

        A new basis is a new namespace, so every vector written under the old
        one is stale the moment `fit` returns. Re-projecting reads the traces
        already on disk — no video is decoded twice.
        """
        r = self._rpc(dict(cmd="project", ids=list(ids)), timeout=300.0)
        return {k: np.asarray(v, dtype=np.float32)
                for k, v in (r.get("vecs") or {}).items()}

    def text(self, texts: list[str]) -> np.ndarray:
        """Embed what the human said. Never stored, never indexed."""
        r = self._rpc(dict(cmd="text", texts=list(texts)), timeout=180.0)
        v = r.get("vecs")
        return np.asarray(v, dtype=np.float32) if v else np.zeros((0, 768), np.float32)

    # ---- stage 2 ------------------------------------------------------------

    def rank(self, query_id: str, candidates: list[str], band: float = 0.25
             ) -> tuple[dict[str, float], float]:
        """DTW over the traces of the database's shortlist. -> (scores, ms)."""
        r = self._rpc(dict(cmd="rank", query=query_id,
                           candidates=list(candidates), band=band))
        return {k: float(v) for k, v in (r.get("scores") or {}).items()}, \
            float(r.get("ms", 0.0))

    def selfcheck(self) -> dict:
        return self._rpc(dict(cmd="selfcheck"))

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"cmd":"quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc, self.ready = None, False


# ---------------------------------------------------------------- persistence

def write_clip(frames, path: str, fps: int = 20) -> str:
    """Save an episode's camera frames as the artefact the encoder reads."""
    import imageio.v3 as iio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    # Already upright: Pilot.memory_frame() un-flips MuJoCo's bottom-up render.
    iio.imwrite(path, arr, fps=fps, codec="libx264",
                out_pixel_format="yuv420p", plugin="pyav")
    return path


def index_recording(db, recording_id: str, vector: np.ndarray, *, store: str,
                    basis_id: str, kitchen_id: str, robot_id: str,
                    duration_s: float, n_steps: int, lang: str, outcome: str,
                    event_id: str | None, clip_path: str) -> None:
    """One row: the vector plus everything needed to trust it later."""
    db.execute(
        """INSERT INTO relmo_recordings
             (recording_id, store, basis_id, robot_id, kitchen_id, duration_s,
              n_steps, embedding, bank_key, lang, outcome, event_id)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (recording_id) DO UPDATE SET
             embedding = EXCLUDED.embedding, basis_id = EXCLUDED.basis_id,
             outcome = EXCLUDED.outcome, lang = EXCLUDED.lang""",
        (recording_id, store, basis_id, robot_id, kitchen_id, float(duration_s),
         int(n_steps), db.vector_param(vector), clip_path, lang, outcome, event_id),
    )


def similar_recordings(db, recording_id: str, k: int = 5) -> list[dict]:
    """Nearest past episodes by what they LOOKED like.

    The basis filter is not defensive tidiness. A whitening basis is fitted over
    a store's contents, so two vectors written under different bases are
    incomparable — ranking across them returns confident nonsense rather than an
    error. Comparing only within a basis is what the `basis_id` column is for.
    """
    rows = db.query(
        "SELECT embedding, basis_id, store FROM relmo_recordings WHERE recording_id = %s",
        (recording_id,),
    )
    if not rows:
        return []
    sim = db.similarity_expr("embedding")
    return [
        dict(recording_id=r["recording_id"], score=round(float(r["score"]), 4),
             lang=r["lang"], outcome=r["outcome"], duration_s=r["duration_s"],
             clip=r["bank_key"])
        for r in db.query(
            f"""SELECT recording_id, lang, outcome, duration_s, bank_key,
                       {sim} AS score
                FROM relmo_recordings
                WHERE basis_id = %s AND recording_id <> %s AND embedding IS NOT NULL
                ORDER BY score DESC LIMIT %s""",
            (rows[0]["embedding"], rows[0]["basis_id"], recording_id, k),
        )
    ]


def search_by_vector(db, vector, basis_id: str, k: int = 5) -> tuple[list[dict], float]:
    """Rank the whole video memory against a vector. Returns (hits, latency_ms)."""
    sim = db.similarity_expr("embedding")
    t0 = time.perf_counter()
    rows = db.query(
        f"""SELECT recording_id, lang, outcome, duration_s, bank_key, {sim} AS score
            FROM relmo_recordings
            WHERE basis_id = %s AND embedding IS NOT NULL
            ORDER BY score DESC LIMIT %s""",
        (db.vector_param(vector), basis_id, k),
    )
    ms = (time.perf_counter() - t0) * 1e3
    return [dict(recording_id=r["recording_id"], score=round(float(r["score"]), 4),
                 lang=r["lang"], outcome=r["outcome"],
                 duration_s=r["duration_s"], clip=r["bank_key"]) for r in rows], ms
