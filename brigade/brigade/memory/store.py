"""The robot's memory, as four kinds of remembering.

  episodic   events        — what happened, as text the robot wrote about itself
  spatial    beliefs       — where things are, with confidence and staleness
  procedural norms         — where things BELONG, learned or instructed
  procedural skill_stats   — which way of doing a thing actually works
  working    tasks         — what to do next, claimed with SKIP LOCKED
  audit      decisions     — what was chosen, and which memories were read

The distinction that carries the whole project is between a *belief* ("the bowl
is on the counter") and a *norm* ("bowls go in the cabinet"). Beliefs are about
the present and go stale. Norms outlive the objects they describe and are what
makes the robot behave differently tomorrow.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import CFG
from . import embed
from .db import DB, Database, retry_serializable

log = logging.getLogger("brigade.memory")


@dataclass
class Recalled:
    id: str
    text: str
    kind: str
    score: float
    outcome: str | None
    subject: str | None
    payload: dict = field(default_factory=dict)
    ts: Any = None


class Memory:
    """Everything the robot knows, over one database."""

    def __init__(self, db: Database | None = None, kitchen_id: str | None = None,
                 robot_id: str | None = None):
        self.db = db or DB
        self.kitchen = kitchen_id or CFG.world.kitchen_id
        self.robot = robot_id or CFG.world.robot_id

    def setup(self) -> dict:
        info = self.db.apply_schema()
        log.info("memory ready: %s (native vectors: %s)", info["flavor"], info["native_vectors"])
        return info

    # ---- episodic -----------------------------------------------------------

    @retry_serializable
    def remember(self, text: str, kind: str = "observation", *, subject: str | None = None,
                 outcome: str | None = None, task_id: str | None = None,
                 payload: dict | None = None, embed_text: bool = True) -> str:
        """Write one event. Returns its id.

        The embedding is of `text` — the robot's own description — which is why
        that string is written to read like a search query rather than a log line.
        """
        vec = self.db.vector_param(embed.encode(text)[0]) if embed_text else None
        rows = self.db.execute(
            """INSERT INTO events (robot_id, kitchen_id, kind, text, embedding,
                                   payload, task_id, subject, outcome)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (self.robot, self.kitchen, kind, text, vec,
             json.dumps(payload or {}), task_id, subject, outcome),
        )
        return str(rows[0]["id"])

    def recall(self, query: str, k: int = 5, *, kind: str | None = None,
               floor: float | None = None) -> list[Recalled]:
        """Semantic recall over the robot's own history.

        `floor` matters more than it looks: a recall that returns nothing above
        threshold is the signal that the robot has NOT been here before, and that
        is what makes it explore instead of going straight somewhere. An empty
        recall is a result, not a failure.
        """
        floor = CFG.memory.recall_floor if floor is None else floor
        qv = self.db.vector_param(embed.encode(query)[0])
        sim = self.db.similarity_expr("embedding")
        where = "kitchen_id = %s AND embedding IS NOT NULL"
        params: list[Any] = [qv, self.kitchen]
        if kind:
            where += " AND kind = %s"
            params.append(kind)
        params.append(k)
        rows = self.db.query(
            f"""SELECT id, text, kind, outcome, subject, payload, ts, {sim} AS score
                FROM events WHERE {where}
                ORDER BY score DESC NULLS LAST LIMIT %s""",
            params,
        )
        return [
            Recalled(str(r["id"]), r["text"], r["kind"], float(r["score"] or 0.0),
                     r["outcome"], r["subject"], r["payload"] or {}, r["ts"])
            for r in rows if (r["score"] or 0.0) >= floor
        ]

    def recent(self, limit: int = 20) -> list[dict]:
        return self.db.query(
            """SELECT id, ts, kind, text, outcome, subject FROM events
               WHERE kitchen_id = %s ORDER BY ts DESC LIMIT %s""",
            (self.kitchen, limit),
        )

    # ---- spatial ------------------------------------------------------------

    @retry_serializable
    def see(self, instance_id: str, label: str, location: str, pos: Sequence[float],
            confidence: float = 1.0, event_id: str | None = None) -> None:
        """Record that an object was observed somewhere. Clears staleness."""
        self.db.execute(
            """INSERT INTO object_beliefs
                 (kitchen_id, instance_id, label, location, pos, confidence,
                  last_seen, last_verified, stale, evidence_event_id)
               VALUES (%s,%s,%s,%s,%s,%s,now(),now(),false,%s)
               ON CONFLICT (kitchen_id, instance_id) DO UPDATE SET
                 label=EXCLUDED.label, location=EXCLUDED.location, pos=EXCLUDED.pos,
                 confidence=EXCLUDED.confidence, last_seen=now(), last_verified=now(),
                 stale=false, evidence_event_id=EXCLUDED.evidence_event_id""",
            (self.kitchen, instance_id, label, location, list(map(float, pos)),
             float(confidence), event_id),
        )

    def where_is(self, label_or_instance: str) -> list[dict]:
        """Where does the robot believe things of this kind are?

        Matches an instance id first, then a label, so "bowl" and "bowl_c" both
        work. Ordered by confidence then recency: the best guess first.
        """
        return self.db.query(
            """SELECT instance_id, label, location, pos, confidence, stale,
                      last_seen, last_verified
               FROM object_beliefs
               WHERE kitchen_id = %s AND (instance_id = %s OR label = %s)
               ORDER BY stale ASC, confidence DESC, last_seen DESC""",
            (self.kitchen, label_or_instance, label_or_instance),
        )

    @retry_serializable
    def mark_stale(self, instance_id: str, why: str = "") -> None:
        """The robot looked where it believed, and the thing was not there."""
        self.db.execute(
            """UPDATE object_beliefs SET stale = true, confidence = confidence * 0.4
               WHERE kitchen_id = %s AND instance_id = %s""",
            (self.kitchen, instance_id),
        )
        self.remember(
            f"expected {instance_id} but it was not there{(' — ' + why) if why else ''}",
            kind="observation", subject=instance_id, outcome="failure",
        )

    def beliefs(self) -> list[dict]:
        return self.db.query(
            """SELECT instance_id, label, location, confidence, stale, last_seen
               FROM object_beliefs WHERE kitchen_id = %s ORDER BY label, instance_id""",
            (self.kitchen,),
        )

    # ---- procedural: norms --------------------------------------------------

    def home_for(self, label: str) -> dict | None:
        rows = self.db.query(
            "SELECT * FROM norms WHERE kitchen_id = %s AND label = %s", (self.kitchen, label)
        )
        return rows[0] if rows else None

    @retry_serializable
    def learn_norm(self, label: str, location: str) -> dict:
        """Reinforce 'things of this kind live here' from one more episode.

        An instructed norm is never overwritten by observation — being told
        outranks having noticed, and it must keep outranking it or act 3 would
        silently decay back to what the robot used to do.
        """
        existing = self.home_for(label)
        if existing and existing["source"] == "instructed" and existing["home_location"] != location:
            return existing
        if existing and existing["home_location"] == location:
            self.db.execute(
                """UPDATE norms SET n_episodes = n_episodes + 1,
                     confidence = LEAST(0.99, confidence + (1 - confidence) * 0.5),
                     updated_at = now()
                   WHERE kitchen_id = %s AND label = %s""",
                (self.kitchen, label),
            )
        else:
            self.db.execute(
                """INSERT INTO norms (kitchen_id, label, home_location, confidence,
                                      n_episodes, source)
                   VALUES (%s,%s,%s,%s,1,'learned')
                   ON CONFLICT (kitchen_id, label) DO UPDATE SET
                     home_location = EXCLUDED.home_location, confidence = 0.5,
                     n_episodes = 1, source = 'learned', updated_at = now()""",
                (self.kitchen, label, location, 0.5),
            )
        return self.home_for(label)

    @retry_serializable
    def instruct_norm(self, label: str, location: str) -> dict:
        """A human said where these go. Outranks anything learned."""
        self.db.execute(
            """INSERT INTO norms (kitchen_id, label, home_location, confidence,
                                  n_episodes, source)
               VALUES (%s,%s,%s,0.95,1,'instructed')
               ON CONFLICT (kitchen_id, label) DO UPDATE SET
                 home_location = EXCLUDED.home_location, confidence = 0.95,
                 source = 'instructed', updated_at = now()""",
            (self.kitchen, label, location),
        )
        self.remember(
            f"instructed: {label} belongs in {location}", kind="instruction", subject=label,
        )
        return self.home_for(label)

    def norms(self) -> list[dict]:
        return self.db.query(
            "SELECT * FROM norms WHERE kitchen_id = %s ORDER BY label", (self.kitchen,)
        )

    # ---- procedural: skill stats -------------------------------------------

    @retry_serializable
    def record_attempt(self, skill: str, label: str, strategy: str, ok: bool) -> None:
        self.db.execute(
            """INSERT INTO skill_stats (robot_id, skill, object_label, strategy, n_try, n_ok)
               VALUES (%s,%s,%s,%s,1,%s)
               ON CONFLICT (robot_id, skill, object_label, strategy) DO UPDATE SET
                 n_try = skill_stats.n_try + 1,
                 n_ok = skill_stats.n_ok + EXCLUDED.n_ok, updated_at = now()""",
            (self.robot, skill, label, strategy, 1 if ok else 0),
        )

    def best_strategy(self, skill: str, label: str, options: Sequence[str]) -> tuple[str, str]:
        """Pick how to attempt something, from what has worked before.

        Returns (strategy, why). Untried strategies are preferred over ones known
        to fail — that is what turns a failure into a different attempt next time
        rather than the same attempt forever.
        """
        rows = self.db.query(
            """SELECT strategy, n_try, n_ok FROM skill_stats
               WHERE robot_id = %s AND skill = %s AND object_label = %s""",
            (self.robot, skill, label),
        )
        stats = {r["strategy"]: (r["n_try"], r["n_ok"]) for r in rows}
        if not stats:
            return options[0], "no experience; using default"

        untried = [o for o in options if o not in stats]
        failing = {s for s, (t, ok) in stats.items() if t >= 1 and ok == 0}
        if untried and stats and all(s in failing for s in stats):
            return untried[0], f"{sorted(failing)} failed before; trying {untried[0]}"

        def rate(o: str) -> float:
            t, ok = stats.get(o, (0, 0))
            return (ok + 1) / (t + 2)  # Laplace: an untried option is not 0

        best = max(options, key=rate)
        t, ok = stats.get(best, (0, 0))
        return best, f"{best} succeeded {ok}/{t} before"

    def skill_table(self) -> list[dict]:
        return self.db.query(
            """SELECT skill, object_label, strategy, n_try, n_ok FROM skill_stats
               WHERE robot_id = %s ORDER BY skill, object_label, strategy""",
            (self.robot,),
        )

    # ---- working: tasks -----------------------------------------------------

    @retry_serializable
    def enqueue(self, goal: str, *, origin: str = "self", priority: int = 5,
                subject: str | None = None, payload: dict | None = None) -> str:
        rows = self.db.execute(
            """INSERT INTO tasks (kitchen_id, goal, priority, origin, subject, payload)
               VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
            (self.kitchen, goal, priority, origin, subject, json.dumps(payload or {})),
        )
        return str(rows[0]["id"])

    def has_open_task(self, subject: str) -> bool:
        rows = self.db.query(
            """SELECT 1 FROM tasks WHERE kitchen_id = %s AND subject = %s
               AND state IN ('pending','claimed','running') LIMIT 1""",
            (self.kitchen, subject),
        )
        return bool(rows)

    @retry_serializable
    def claim(self) -> dict | None:
        """Take the next task. SKIP LOCKED so N workers never collide.

        This is one transaction, not a select-then-update: two robots polling the
        same queue must not both get the same job, and the database is the only
        thing that can promise that.
        """
        with self.db.cursor(commit=True) as cur:
            cur.execute(
                """SELECT id, goal, priority, origin, subject, payload FROM tasks
                   WHERE kitchen_id = %s AND state = 'pending'
                   ORDER BY priority DESC, created_at ASC
                   LIMIT 1 FOR UPDATE SKIP LOCKED""",
                (self.kitchen,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                """UPDATE tasks SET state='running', claimed_by=%s, claimed_at=now()
                   WHERE id = %s""",
                (self.robot, row["id"]),
            )
            return dict(row)

    @retry_serializable
    def finish(self, task_id: str, ok: bool, result: str = "") -> None:
        self.db.execute(
            """UPDATE tasks SET state=%s, result=%s, finished_at=now() WHERE id=%s""",
            ("done" if ok else "failed", result[:400], task_id),
        )

    @retry_serializable
    def preempt(self, task_id: str) -> None:
        """Put a running task back so something more urgent can go first."""
        self.db.execute(
            "UPDATE tasks SET state='pending', claimed_by=NULL, claimed_at=NULL WHERE id=%s",
            (task_id,),
        )

    def task_board(self, limit: int = 20) -> list[dict]:
        return self.db.query(
            """SELECT id, goal, state, origin, priority, subject, result, created_at
               FROM tasks WHERE kitchen_id = %s
               ORDER BY (state='running') DESC, priority DESC, created_at DESC LIMIT %s""",
            (self.kitchen, limit),
        )

    # ---- audit --------------------------------------------------------------

    @retry_serializable
    def decide(self, chose: str, rationale: str, recalled: Sequence[Recalled] = (),
               latency_ms: float = 0.0, task_id: str | None = None,
               source: str = "planner") -> str:
        """Log a decision together with the memories that produced it.

        If `recalled` is empty, the robot was not using its memory for this
        decision — which is exactly the thing this table exists to make visible.
        """
        ids = [r.id for r in recalled] or None
        rows = self.db.execute(
            # ::uuid[] is required: the ids arrive as Python strings, which
            # psycopg2 adapts to text[], and Postgres will not implicitly cast
            # text[] to uuid[].
            """INSERT INTO decisions (robot_id, kitchen_id, task_id, chose, rationale,
                                      recalled_event_ids, recall_latency_ms, source)
               VALUES (%s,%s,%s,%s,%s,%s::uuid[],%s,%s) RETURNING id""",
            (self.robot, self.kitchen, task_id, chose, rationale[:800], ids,
             float(latency_ms), source),
        )
        return str(rows[0]["id"])

    def decision_log(self, limit: int = 20) -> list[dict]:
        return self.db.query(
            """SELECT id, ts, chose, rationale, recall_latency_ms,
                      COALESCE(array_length(recalled_event_ids,1),0) AS n_recalled
               FROM decisions WHERE kitchen_id = %s ORDER BY ts DESC LIMIT %s""",
            (self.kitchen, limit),
        )

    # ---- housekeeping -------------------------------------------------------

    def stats(self) -> dict:
        def n(t: str) -> int:
            try:
                return int(self.db.query(f"SELECT count(*) AS c FROM {t}")[0]["c"])
            except Exception:
                return -1

        return dict(
            events=n("events"), beliefs=n("object_beliefs"), norms=n("norms"),
            tasks=n("tasks"), decisions=n("decisions"), skills=n("skill_stats"),
            flavor=self.db.flavor, native_vectors=self.db.supports_vector(),
        )

    def wipe(self) -> None:
        """Forget everything. Demo reset; never called by the agent."""
        for t in ("decisions", "tasks", "skill_stats", "norms", "object_beliefs", "events"):
            self.db.execute(f"DELETE FROM {t}")
