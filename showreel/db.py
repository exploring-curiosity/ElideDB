#!/usr/bin/env python3
"""One connection layer, two engines. The port is a URL, not a rewrite.

PostgreSQL + pgvector locally; CockroachDB in production. Both speak the same
wire protocol and almost the same SQL, so the differences are isolated here
rather than sprinkled through the agent:

    vector index     pgvector  CREATE INDEX ... USING hnsw (col vector_cosine_ops)
                     cockroach CREATE VECTOR INDEX ... ON t (col)
    retries          cockroach can return 40001 (serialisation) on a contended
                     transaction and EXPECTS the client to retry. Postgres under
                     READ COMMITTED rarely does. The retry lives here so every
                     caller gets it for free: omitting it is the single most
                     common way a CockroachDB app is wrong in production.

Everything else in this codebase writes plain SQL.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

DSN = os.environ.get("PRECEDENT_DSN", "postgresql://localhost:5433/brigade")
# Set PRECEDENT_ENGINE=cockroach, or let it be sniffed from the server version.
_ENGINE = os.environ.get("PRECEDENT_ENGINE", "")

# 40001 serialization_failure, 40P01 deadlock_detected. Both mean "retry me".
RETRYABLE = ("40001", "40P01")
MAX_TRIES = 5


def engine() -> str:
    global _ENGINE
    if not _ENGINE:
        with connect() as (con, cur):
            cur.execute("SELECT version()")
            v = (cur.fetchone() or {}).get("version", "")
        _ENGINE = "cockroach" if "CockroachDB" in str(v) else "postgres"
    return _ENGINE


@contextmanager
def connect(autocommit: bool = True):
    con = psycopg2.connect(DSN)
    con.autocommit = autocommit
    try:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield con, cur
    finally:
        con.close()


def q(sql: str, args=()) -> list[dict]:
    """One statement, autocommitted. For reads and single writes.

    Passing NO second argument when there are no parameters is not tidiness. An
    empty tuple still puts psycopg2 into interpolation mode, where a literal %
    in the SQL is read as a placeholder, so

        SELECT count(*) FROM moments WHERE video LIKE 's3://%'

    raises IndexError rather than running. It is a good failure in a script and
    a bad one at 3 a.m., and doubling the % to hide it moves the trap instead of
    removing it. Omitting args means psycopg2 sends the statement untouched.
    """
    with connect() as (_c, cur):
        cur.execute(sql, args) if args else cur.execute(sql)
        return [dict(r) for r in cur.fetchall()] if cur.description else []


def tx(fn, tries: int = MAX_TRIES):
    """Run fn(cur) inside ONE transaction, retrying on serialisation failure.

    CockroachDB uses SERIALIZABLE isolation and will abort a transaction that
    would otherwise interleave badly, expecting the client to run it again. An
    application without this loop looks fine on a laptop and drops writes the
    first time two workers touch the same rows. The cascade below is exactly
    such a transaction, so it is the reason this exists.
    """
    last = None
    for n in range(tries):
        try:
            with connect(autocommit=False) as (con, cur):
                out = fn(cur)
                con.commit()
                return out
        except psycopg2.Error as exc:                              # noqa: PERF203
            if getattr(exc, "pgcode", None) not in RETRYABLE:
                raise
            last = exc
            time.sleep(0.05 * (2 ** n))       # exponential backoff, jitter-free
    raise last


def apply_schema() -> dict:
    """Create the memory. Idempotent, and safe to run on every boot."""
    here = os.path.dirname(os.path.abspath(__file__))
    body = open(os.path.join(here, "schema.sql")).read()
    # Strip line comments BEFORE splitting on ';'. A prose comment containing a
    # semicolon ("a sequence is one hot range; every insert queues") otherwise
    # splits mid-sentence and the server reports a syntax error on English.
    code = "\n".join(l.split("--", 1)[0] for l in body.splitlines())
    for stmt in [s.strip() for s in code.split(";") if s.strip()]:
        q(stmt)
    made = []
    # Secondary indexes: the queue is read by state and the graph is walked
    # backwards, so both need covering the other way round.
    for name, sql in (
        ("inbox_pending", "CREATE INDEX IF NOT EXISTS inbox_pending "
                          "ON inbox (fleet, state, arrived)"),
        ("filings_by_rec", "CREATE INDEX IF NOT EXISTS filings_by_rec "
                           "ON filings (fleet, rec_id)"),
        ("filings_live", "CREATE INDEX IF NOT EXISTS filings_live "
                         "ON filings (fleet, superseded, disposition)"),
        # The cascade walks precedent_id -> filing_id, i.e. the reverse of the
        # primary key. Without this the recursive step is a full scan per level.
        ("prec_reverse", "CREATE INDEX IF NOT EXISTS prec_reverse "
                         "ON filing_precedents (precedent_id)"),
        ("verdicts_by_filing", "CREATE INDEX IF NOT EXISTS verdicts_by_filing "
                               "ON verdicts (filing_id)"),
    ):
        try:
            q(sql)
            made.append(name)
        except psycopg2.Error as exc:
            print(f"  index {name}: {exc}")
    return dict(engine=engine(), indexes=made)


def vector_index(table: str, column: str) -> str:
    """The one statement that genuinely differs between the two engines.

    CockroachDB does NOT accept IF NOT EXISTS on CREATE VECTOR INDEX: it is a
    syntax error at the ON, not a no-op, so the name is explicit and callers
    swallow the already-exists error. pgvector wants an access method and an
    operator class; CockroachDB infers both. Found by running it against a real
    node rather than by reading about it.
    """
    if engine() == "cockroach":
        return f"CREATE VECTOR INDEX {table}_{column}_vec ON {table} ({column})"
    return (f"CREATE INDEX IF NOT EXISTS {table}_{column}_hnsw ON {table} "
            f"USING hnsw ({column} vector_cosine_ops)")


def ensure_vector_index(table: str, column: str) -> bool:
    """Idempotent across both engines. -> True if it exists now."""
    try:
        q(vector_index(table, column))
        return True
    except psycopg2.Error as exc:
        # 42P07 duplicate_object / duplicate relation: already there, fine.
        if getattr(exc, "pgcode", "") in ("42P07", "42710"):
            return True
        print(f"  vector index {table}.{column}: {exc}")
        return False


def vec(a) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in a) + "]"


if __name__ == "__main__":
    print(apply_schema())
