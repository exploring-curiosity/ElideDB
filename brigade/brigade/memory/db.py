"""CockroachDB access: pool, retries, schema.

Two things here are not boilerplate and are worth reading.

**Retries.** CockroachDB runs SERIALIZABLE by default, so a transaction can be
aborted with SQLSTATE 40001 purely because it raced another one. That is not an
error, it is the contract: the client is required to retry. Brigade has several
writers (the agent loop, the perception pass, the dashboard, the consolidator)
touching the same rows, so this is load-bearing, not defensive decoration.

**No fallback.** If the database is unreachable, or the server has no VECTOR
support, this module raises. It does not quietly fall back to an in-process dict.
An agent whose memory is offline should stop, visibly — that is the premise of
the whole project, and a fallback cache would make the demo a lie.
"""

from __future__ import annotations

import functools
import logging
import os
import random
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable

import psycopg2
import psycopg2.extras
import psycopg2.pool

from ..config import CFG, MemoryConfig

log = logging.getLogger("brigade.memory")

SERIALIZATION_FAILURE = "40001"
_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

# One schema, several backends. Two INDEPENDENT axes, which is why this is not
# a single if/else:
#
#   dialect   crdb      — STRING, inline INDEX / INVERTED INDEX, CREATE VECTOR INDEX
#             postgres  — TEXT, separate CREATE INDEX, GIN for JSONB
#   vectors   native    — a real VECTOR(n) column and the `<=>` cosine operator.
#                         True for CockroachDB (C-SPANN) and for Postgres with
#                         pgvector (HNSW). Same operator, same query shape.
#             arrays    — FLOAT8[] plus a cosine function we install. The
#                         fallback when neither is available.
#
# The whole point is that NOTHING above this module changes when the DSN moves
# to CockroachDB: same tables, same queries, same call sites. The differences
# live in three small functions here.
CRDB = "cockroach"
POSTGRES = "postgres"

_COSINE_FN = """
CREATE OR REPLACE FUNCTION brigade_cosine(a FLOAT8[], b FLOAT8[])
RETURNS FLOAT8 AS $$
  SELECT CASE
    WHEN a IS NULL OR b IS NULL OR array_length(a,1) IS DISTINCT FROM array_length(b,1)
      THEN NULL
    ELSE (
      SELECT SUM(x*y) / NULLIF(SQRT(SUM(x*x)) * SQRT(SUM(y*y)), 0)
      FROM unnest(a, b) AS t(x, y)
    )
  END
$$ LANGUAGE SQL IMMUTABLE;
"""


class MemoryUnavailable(RuntimeError):
    """The memory layer is not usable. The robot must stop, not improvise."""


class Database:
    """A connection pool with the retry semantics CockroachDB requires."""

    def __init__(self, cfg: MemoryConfig | None = None):
        self.cfg = cfg or CFG.memory
        self._pool: psycopg2.pool.ThreadedConnectionPool | None = None
        self._lock = threading.Lock()
        self._flavor: str | None = None
        self._native_vectors: bool | None = None

    # ---- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        if self._pool is not None:
            return
        with self._lock:
            if self._pool is not None:
                return
            try:
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    self.cfg.pool_min, self.cfg.pool_max, self.cfg.dsn,
                    application_name="brigade",
                )
            except psycopg2.Error as exc:
                raise MemoryUnavailable(
                    f"cannot reach CockroachDB at {self._safe_dsn()}: {exc}"
                ) from exc

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.closeall()
                self._pool = None

    def _safe_dsn(self) -> str:
        """DSN with any password removed — this ends up in logs and on screen."""
        return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", self.cfg.dsn)

    # ---- primitives ---------------------------------------------------------

    @contextmanager
    def connection(self):
        self.connect()
        assert self._pool is not None
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def cursor(self, commit: bool = True):
        """A cursor in its own transaction. Rolls back on any exception."""
        with self.connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            try:
                yield cur
                if commit:
                    conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                cur.close()

    def query(self, sql: str, params: Iterable | None = None) -> list[dict]:
        with self.cursor(commit=False) as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return []
            return [dict(r) for r in cur.fetchall()]

    def execute(self, sql: str, params: Iterable | None = None) -> list[dict]:
        with self.cursor(commit=True) as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return []
            return [dict(r) for r in cur.fetchall()]

    # ---- health -------------------------------------------------------------

    def healthy(self) -> bool:
        try:
            self.query("SELECT 1")
            return True
        except Exception:
            return False

    def server_version(self) -> str:
        rows = self.query("SELECT version() AS v")
        return rows[0]["v"] if rows else "unknown"

    @property
    def flavor(self) -> str:
        """Which dialect are we actually talking to? Asked, never assumed."""
        if self._flavor is None:
            v = self.server_version().lower()
            self._flavor = CRDB if "cockroach" in v else POSTGRES
        return self._flavor

    def supports_vector(self) -> bool:
        """Does this server have a native VECTOR type?

        Asked of the server rather than inferred from a version string, because
        the string differs across self-hosted, Cloud Basic and Cloud Standard.
        Cached — this sits on the recall path and a round trip per query to
        re-answer a fact that cannot change mid-connection is pure waste.
        """
        if self._native_vectors is None:
            try:
                self.query("SELECT '[1,2,3]'::VECTOR(3)")
                self._native_vectors = True
            except Exception:
                self._native_vectors = False
        return self._native_vectors

    def supports_vector_index(self) -> bool:
        """Can this server actually build a C-SPANN vector index?

        Separate from supports_vector: some deployments expose the type but not
        the index. We find out by building one on a scratch table and dropping
        it — a definite answer beats a documented one.
        """
        probe = "brigade_vecprobe"
        try:
            self.execute(f"DROP TABLE IF EXISTS {probe}")
            self.execute(f"CREATE TABLE {probe} (id INT PRIMARY KEY, e VECTOR(3))")
            self.execute(f"CREATE VECTOR INDEX ON {probe} (e)")
            return True
        except Exception as exc:
            log.warning("vector index unsupported: %s", exc)
            return False
        finally:
            try:
                self.execute(f"DROP TABLE IF EXISTS {probe}")
            except Exception:
                pass

    # ---- schema -------------------------------------------------------------

    def apply_schema(self, with_vector_indexes: bool = True) -> dict:
        """Create tables, then vector indexes on the still-empty tables.

        Splitting the two is deliberate: CockroachDB's guidance is that building
        a vector index over an already-populated table is expensive, so we build
        while there is nothing to backfill.
        """
        native = self.supports_vector()
        crdb = self.flavor == CRDB
        with open(_SCHEMA_PATH) as fh:
            body = fh.read()
        if not crdb:
            body = to_postgres(body, native_vectors=native)

        statements = [s.strip() for s in body.split(";") if s.strip() and not _only_comments(s)]
        applied = []
        for stmt in statements:
            self.execute(stmt)
            applied.append(_stmt_label(stmt))

        if not crdb:
            if not native:
                self.execute(_COSINE_FN)
            # Secondary indexes, created separately because CockroachDB's inline
            # INDEX syntax is stripped for Postgres.
            for ddl in _PG_INDEXES:
                self.execute(ddl)

        indexes: list[str] = []
        if native and with_vector_indexes:
            for table, column in (("events", "embedding"), ("relmo_recordings", "embedding")):
                # CockroachDB: distributed C-SPANN. pgvector: HNSW. Different
                # DDL, same operator (`<=>`) and therefore the same query above.
                ddl = (
                    f"CREATE VECTOR INDEX IF NOT EXISTS ON {table} ({column})"
                    if crdb else
                    f"CREATE INDEX IF NOT EXISTS {table}_{column}_hnsw ON {table} "
                    f"USING hnsw ({column} vector_cosine_ops)"
                )
                try:
                    self.execute(ddl)
                    indexes.append(f"{table}.{column}")
                except Exception as exc:
                    # Loud, not silent: without the index, recall still works by
                    # exact scan but the "vector index" claim does not hold, and
                    # the caller must know which world it is in.
                    log.warning("no vector index on %s.%s: %s", table, column, exc)

        return dict(
            tables=applied,
            vector_indexes=indexes,
            native_vectors=native,
            flavor=self.flavor,
            version=self.server_version().split(" on ")[0],
        )

    # ---- dialect ------------------------------------------------------------

    def vector_param(self, values):
        """Bind a vector for this backend: VECTOR literal, or a float list."""
        if self.supports_vector():
            return vector_literal(values)
        return [float(v) for v in values]

    def similarity_expr(self, column: str, param: str = "%s") -> str:
        """SQL that yields cosine similarity in [-1,1], higher = closer.

        CockroachDB's `<=>` is cosine DISTANCE, so it is subtracted from 1 to
        keep one orientation everywhere: callers always ORDER BY ... DESC.
        """
        if self.supports_vector():
            return f"1 - ({column} <=> {param})"
        return f"brigade_cosine({column}, {param})"

    def drop_all(self) -> None:
        """Tear the memory down. Used by tests; never by the agent."""
        for t in ("decisions", "tasks", "skill_stats", "norms",
                  "object_beliefs", "relmo_recordings", "events"):
            self.execute(f"DROP TABLE IF EXISTS {t} CASCADE")


_PG_INDEXES = (
    "CREATE INDEX IF NOT EXISTS events_by_time ON events (kitchen_id, ts DESC)",
    "CREATE INDEX IF NOT EXISTS events_by_subject ON events (kitchen_id, subject, ts DESC)",
    "CREATE INDEX IF NOT EXISTS events_payload ON events USING GIN (payload)",
    "CREATE INDEX IF NOT EXISTS beliefs_by_label ON object_beliefs (kitchen_id, label, stale)",
    "CREATE INDEX IF NOT EXISTS tasks_queue ON tasks (kitchen_id, state, priority DESC, created_at ASC)",
    "CREATE INDEX IF NOT EXISTS decisions_by_time ON decisions (kitchen_id, ts DESC)",
    "CREATE INDEX IF NOT EXISTS relmo_by_store ON relmo_recordings (store, basis_id)",
)


def to_postgres(sql: str, native_vectors: bool = False) -> str:
    """Translate the CockroachDB schema into Postgres dialect.

    Three differences, and they are the only three:
      * STRING is CockroachDB's spelling of TEXT.
      * VECTOR(n) survives when pgvector is installed; otherwise it becomes
        FLOAT8[] and cosine becomes a function we install.
      * CockroachDB allows INDEX / INVERTED INDEX inside CREATE TABLE; Postgres
        requires separate CREATE INDEX statements, which apply_schema issues.

    Removing those inline index lines is what makes the trailing comma on the
    preceding column illegal, so the comma is cleaned up here rather than left
    for the server to complain about at line 35 of a generated statement.
    """
    out = re.sub(r"\bSTRING\b", "TEXT", sql)
    if not native_vectors:
        out = re.sub(r"VECTOR\(\d+\)", "FLOAT8[]", out)

    # Work line-wise. A regex for the trailing comma is not enough: the comma is
    # usually followed by an end-of-line comment ("recording_id TEXT, -- joins…"),
    # so a ",\s*\)" pattern never matches and Postgres reports a syntax error at
    # the closing paren instead.
    lines: list[str] = []
    for raw in out.splitlines():
        code = raw.split("--", 1)[0].rstrip()  # no '--' appears inside a literal here
        if re.match(r"^\s*(?:INVERTED\s+)?INDEX\s+\w+\s*\(", code):
            continue
        if not code.strip():
            continue
        lines.append(code)

    # Drop a comma left dangling on the last member of a CREATE TABLE.
    for i, line in enumerate(lines):
        if line.strip().startswith(")") and i > 0:
            prev = lines[i - 1].rstrip()
            if prev.endswith(","):
                lines[i - 1] = prev[:-1]
    return "\n".join(lines)


def _only_comments(stmt: str) -> bool:
    return all(not ln.strip() or ln.strip().startswith("--") for ln in stmt.splitlines())


def _stmt_label(stmt: str) -> str:
    m = re.search(r"(CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?)\s+(\w+)", stmt, re.I)
    return m.group(2) if m else stmt.split("\n", 1)[0][:48]


def retry_serializable(fn=None, *, attempts: int | None = None):
    """Retry a function whose transaction was aborted by contention (40001).

    CockroachDB is serializable; a 40001 means "you raced someone, run it again",
    and the client is *required* to handle it. Backoff is exponential with
    jitter so two contending writers do not resynchronise on the retry.
    """

    def decorate(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            n = attempts if attempts is not None else CFG.memory.retry_max
            last: BaseException | None = None
            for i in range(n):
                try:
                    return f(*a, **kw)
                except psycopg2.errors.SerializationFailure as exc:
                    last = exc
                    sleep = (0.05 * (2 ** i)) * (0.5 + random.random())
                    log.debug("40001 on %s, retry %d/%d in %.3fs", f.__name__, i + 1, n, sleep)
                    time.sleep(sleep)
                except psycopg2.Error as exc:
                    if getattr(exc, "pgcode", None) == SERIALIZATION_FAILURE:
                        last = exc
                        time.sleep(0.05 * (2 ** i) * (0.5 + random.random()))
                        continue
                    raise
            raise MemoryUnavailable(
                f"{f.__name__} still contended after {n} attempts"
            ) from last

        return wrapper

    return decorate(fn) if fn is not None else decorate


# One process-wide handle. Brigade is a single-node agent; a module-level pool is
# the right amount of machinery.
DB = Database()


def vector_literal(values) -> str:
    """Format a float sequence as a CockroachDB VECTOR literal.

    psycopg2 has no VECTOR adapter, so vectors travel as a string and are cast
    server-side. Kept in one place so the cast is impossible to forget.
    """
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"
