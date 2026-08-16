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


class MemoryUnavailable(RuntimeError):
    """The memory layer is not usable. The robot must stop, not improvise."""


class Database:
    """A connection pool with the retry semantics CockroachDB requires."""

    def __init__(self, cfg: MemoryConfig | None = None):
        self.cfg = cfg or CFG.memory
        self._pool: psycopg2.pool.ThreadedConnectionPool | None = None
        self._lock = threading.Lock()

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

    def supports_vector(self) -> bool:
        """Does this server have the VECTOR type at all?

        Checked by asking the server rather than by parsing a version string,
        because the version string differs between self-hosted, Cloud Basic and
        Cloud Standard.
        """
        try:
            self.query("SELECT '[1,2,3]'::VECTOR(3)")
            return True
        except Exception:
            return False

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
        if not self.supports_vector():
            raise MemoryUnavailable(
                f"server at {self._safe_dsn()} has no VECTOR type "
                f"(version: {self.server_version()}). CockroachDB v25.2+ is required."
            )

        with open(_SCHEMA_PATH) as fh:
            body = fh.read()
        statements = [s.strip() for s in body.split(";") if s.strip() and not _only_comments(s)]

        applied = []
        for stmt in statements:
            self.execute(stmt)
            applied.append(_stmt_label(stmt))

        indexes = []
        if with_vector_indexes:
            for table, column in (("events", "embedding"), ("relmo_recordings", "embedding")):
                try:
                    self.execute(f"CREATE VECTOR INDEX IF NOT EXISTS ON {table} ({column})")
                    indexes.append(f"{table}.{column}")
                except Exception as exc:
                    # Loud, not silent: without the index, recall still works
                    # (exact scan) but the "distributed vector index" claim does
                    # not, and the caller must know which world it is in.
                    log.warning("could not create vector index on %s.%s: %s", table, column, exc)

        return dict(tables=applied, vector_indexes=indexes, version=self.server_version())

    def drop_all(self) -> None:
        """Tear the memory down. Used by tests; never by the agent."""
        for t in ("decisions", "tasks", "skill_stats", "norms",
                  "object_beliefs", "relmo_recordings", "events"):
            self.execute(f"DROP TABLE IF EXISTS {t} CASCADE")


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
