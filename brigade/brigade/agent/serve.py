"""The serve loop: one continuous kitchen, cameras always on, memory in charge.

    human says one thing
        -> the store cuts the robot's LIVE view and indexes it
        -> stage 1: pgvector ranks the whole memory by RelMo's prefilter
        -> stage 2: RelMo re-ranks that shortlist by DTW over the traces
        -> the reasoning head turns the retrieved clips into a COMMAND
        -> pi0.5 executes the command as a latent prefix, no words involved
        -> the kitchen keeps whatever changed, and the cameras never stopped

Every constraint the owner set is structural here rather than promised:

  NOTHING TEXTUAL IS STORED. The store holds video, vectors and timestamps.
  What the human said is embedded, used, and dropped; it is written to `turns`
  afterwards for the audit trail and never read back by the recall path.

  THE MEMORY IS RELMO. Retrieval is RelMo's own two-stage read path, with
  stage 1 moved into the database. There is no second index, no fact table, no
  cache of what the robot concluded last time.

  THE REASONING LAYER IS NOT GENERATIVE. It emits K weights over frozen
  prototypes. Its output space is exactly the span of behaviours the robot can
  perform, so it cannot invent an instruction, and it cannot drift.

  THE WORLD DOES NOT RESET. `carry`/`restore` splice the persistent kitchen
  onto a canonical robot pose between commands, so what the last command
  achieved is what the next one starts from — which is the only thing that
  makes a spatial memory load-bearing rather than decorative.

  MEMORY IS LOAD-BEARING BY CONSTRUCTION. The head's only view of the kitchen
  comes from the retrieved clips. With the memory switched off there are no
  clips, so there is no command and the robot cannot act at all — which is a
  much stronger claim than acting worse.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("brigade.serve")


@dataclass
class Turn:
    """One human utterance and everything that followed it."""
    heard: str
    clips: list = field(default_factory=list)
    weights: np.ndarray | None = None
    margin: float = 0.0
    chose: str = ""
    acted: bool = False
    succeeded: bool | None = None
    retrieval_ms: float = 0.0
    reason_ms: float = 0.0
    act_s: float = 0.0
    note: str = ""

    def to_json(self) -> dict:
        return dict(heard=self.heard, clips=[c.clip_id for c in self.clips],
                    scores=[round(c.score, 4) for c in self.clips],
                    stage=[c.stage for c in self.clips],
                    weights=None if self.weights is None
                    else [round(float(w), 4) for w in self.weights],
                    margin=round(self.margin, 4), chose=self.chose,
                    acted=self.acted, succeeded=self.succeeded,
                    retrieval_ms=round(self.retrieval_ms, 2),
                    reason_ms=round(self.reason_ms, 2),
                    act_s=round(self.act_s, 1), note=self.note)


class Kitchen:
    """The continuous world, its cameras, its memory, and the agent on top."""

    def __init__(self, reasoner_path: str, scene: int = 0, device: str = "mps"):
        self.reasoner_path = reasoner_path
        self.scene = scene
        self.device = device
        self.pilot = self.store = self.relmo = self.reasoner = None
        self.state = None                 # the persistent kitchen
        self.turns: list[Turn] = []

    # ---- setup --------------------------------------------------------------

    def open(self) -> dict:
        from ..memory.relmo import RelMoSidecar
        from ..memory.store import VideoStore
        from .latent import Reasoner
        from .pilot import Pilot

        self.relmo = RelMoSidecar()
        if not self.relmo.start():
            raise RuntimeError(f"RelMo unavailable: {self.relmo.error}")
        self.store = VideoStore(relmo=self.relmo)
        self.store.setup()
        self.store.start()

        self.pilot = Pilot(device=self.device)
        self.pilot.load()
        self.pilot.open_scene("libero_goal", self.scene)
        # A fresh simulator is a genuine discontinuity in the video: the ring
        # must not bridge a scene that no longer exists.
        self.store.new_session()
        self.reasoner = Reasoner.load(self.reasoner_path)
        return dict(basis=self.relmo.basis, store=self.store.stats(),
                    behaviours=self.reasoner.bank.k)

    def close(self) -> None:
        if self.store is not None:
            self.store.stop()
        if self.relmo is not None:
            self.relmo.stop()
        if self.pilot is not None:
            self.pilot.close()

    # ---- the camera ---------------------------------------------------------

    def _stream(self, on_frame=None):
        """The write path, as a callback. Returns (fn, list-of-cut-ids)."""
        cut: list[str] = []

        def fn(_payload, step):
            # Every other control step at 20 Hz is the store's 10 fps.
            if step % 2 == 0:
                cid = self.store.write(self.pilot.memory_frame())
                if cid:
                    cut.append(cid)
            if on_frame is not None:
                on_frame(_payload, step)

        return fn, cut

    def watch(self, seconds: float) -> list[str]:
        """Record the kitchen doing nothing. Spans cut here are real memories."""
        cut: list[str] = []

        def fn(frame, _i):
            cid = self.store.write(frame)
            if cid:
                cut.append(cid)

        self.pilot.idle(seconds, on_frame=fn)
        return cut

    # ---- the turn -----------------------------------------------------------

    def hear(self, heard: str, on_frame=None, k: int = 6,
             since: float | None = None, until: float | None = None,
             memory: bool = True, act: bool = True) -> Turn:
        """One human utterance, start to finish.

        `memory=False` is the control arm: the recall path is skipped entirely,
        which is what an empty database looks like from here. It does not
        degrade the command — there is no command.
        """
        t = Turn(heard=heard)
        if not memory:
            t.note = "memory OFF — no clips, so no command and nothing to execute"
            self.turns.append(t)
            return t

        # 1. THE QUERY IS THE ROBOT'S OWN VIEW. Cut the live buffer and index
        #    it. Text never touches the index: RelMo's text channel is measured
        #    weak (open/close sit at cosine 0.957, so the words erase the very
        #    distinction the query needs), and the robot's eyes are the one
        #    query that is always available and always current.
        # WHICH VIEW THE QUERY USES, and it is not the one that scores best on
        # the behaviour-identification benchmark. The question here is "when did
        # the kitchen last look like it looks NOW", and the appearance vector is
        # what encodes where things are sitting. The motion vector encodes how
        # much moved, and at the moment a human speaks the robot is usually
        # idle — a motion query from a still kitchen retrieves other still
        # moments, which carry no evidence about anything. Retrieval matches on
        # STATE; the head then reads the motion of what came back.
        t0 = time.perf_counter()
        qid = self.store.flush()
        qvec = self.store.await_vector(qid, view="embedding") if qid else None
        if qvec is None:
            t.note = ("no live view to search with — the camera has not filled "
                      "a span yet")
            self.turns.append(t)
            return t

        # 2. TWO-STAGE RETRIEVAL. Stage 1 is the SQL vector index; stage 2 is
        #    RelMo's DTW over the traces of the shortlist.
        clips, timing = self.store.similar(qvec, k=k, query_id=qid,
                                           since=since, until=until,
                                           view="embedding")
        clips = [c for c in clips if c.clip_id != qid][:k]
        t.retrieval_ms = timing["stage1_ms"] + timing["stage2_ms"]
        t.clips = clips
        if not clips:
            t.note = "memory returned nothing — with no clips there is no command"
            self.turns.append(t)
            return t

        # 3. REASON. The head reads the retrieved clips and the request. It
        #    emits weights over frozen prototypes, never a sentence.
        t1 = time.perf_counter()
        from ..memory import traces as TR

        req = self.relmo.text([heard])[0] if self.reasoner.head.use_request else None
        # The head reads the TRACE, not the indexed column — see traces.py. The
        # column is a mean, which is what a prefilter must be and is measured at
        # 0.305 on this question; the per-channel spread over the same trace is
        # 0.712. Retrieval and reasoning want different reductions of one clip.
        vecs, kept = TR.features([c.clip_id for c in clips])
        keep = {c: i for i, c in enumerate(kept)}
        clips = [c for c in clips if c.clip_id in keep]
        t.clips = clips
        if not len(vecs):
            t.note = "retrieved spans have no trace on disk"
            self.turns.append(t)
            return t
        cmd = self.reasoner.command(vecs, req, evidence=[c.clip_id for c in clips],
                                    match=np.array([c.score for c in clips], np.float32))
        t.reason_ms = (time.perf_counter() - t1) * 1e3
        if cmd is None:
            t.note = "no command"
            self.turns.append(t)
            return t
        t.weights, t.margin = cmd.weights, cmd.margin
        top = int(np.argmax(cmd.weights))
        t.chose = (self.reasoner.instructions[top]
                   if top < len(self.reasoner.instructions) else str(top))

        # 4. ABSTAIN RATHER THAN GUESS. Blends succeed at 0.5 and fail at 0.3
        #    (bench/a_interp.py), so a top weight under half is not a command,
        #    it is a question.
        if not cmd.confident:
            t.note = (f"not confident (top {cmd.weights.max():.2f}) — asking "
                      f"rather than guessing")
            self._log_turn(t, qid)
            self.turns.append(t)
            return t

        if not act:
            t.note = "decided but not executed"
            self._log_turn(t, qid)
            self.turns.append(t)
            return t

        # 5. ACT. The command is installed as pi0.5's language prefix. No string
        #    is tokenized; nothing in this path has a vocabulary.
        from .latent import LatentPrefix

        fn, _ = self._stream(on_frame)
        a0 = time.perf_counter()
        with LatentPrefix(self.pilot.policy, cmd):
            ep = self.pilot.run(t.chose, keep_frames=False, on_frame=fn,
                                from_state=self.state)
        t.act_s = time.perf_counter() - a0
        t.acted, t.succeeded = True, bool(ep.success)
        # THE KITCHEN KEEPS WHAT CHANGED. Captured mid-rollout because the
        # vector env autoresets on the step that reports termination.
        self.state = ep.end_state
        self.pilot.restore(self.state)
        self._log_turn(t, qid)
        self.turns.append(t)
        return t

    def _log_turn(self, t: Turn, qid: str) -> None:
        """The audit trail. Written after the fact, never read by recall."""
        try:
            tid = self.store.log_turn(t.heard, [c.clip_id for c in t.clips],
                                      t.weights, t.margin, t.retrieval_ms,
                                      t.reason_ms)
            self.store.finish_turn(tid, t.acted, t.succeeded, t.act_s)
        except Exception as exc:                                  # noqa: BLE001
            log.warning("turn not logged: %s", exc)
