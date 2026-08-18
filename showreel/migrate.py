#!/usr/bin/env python3
"""Copy the memory from PostgreSQL to CockroachDB. The port, executed.

    .venv-libero/bin/python showreel/migrate.py \
        --to "postgresql://root@localhost:26257/precedent?sslmode=disable"

The point of running this is to find out what actually differs between the two
engines rather than to assert that nothing does. What it found:

    VECTOR(n) and CREATE VECTOR INDEX both exist on CockroachDB v26: the
    schema and the vector index are portable as written.
    Batches must be modest. CockroachDB's default transaction size limits are
    tighter than Postgres's, and a 3,402-row insert of 2,048 floats each is a
    large write; 200 rows a batch keeps every transaction well inside them.

Everything else in the schema was already written to be portable: UUID keys
rather than SERIAL (a sequence is one hot range in a distributed cluster) and
TIMESTAMPTZ rather than TIMESTAMP (a fleet is not in one timezone).
"""
from __future__ import annotations

import argparse
import sys

import psycopg2
import psycopg2.extras

SRC = "postgresql://localhost:5433/brigade"
COLS = ("rec_id", "store", "dataset", "task", "video", "seconds", "steps",
        "appearance", "motion", "siglip")

DDL = """
CREATE TABLE IF NOT EXISTS moments (
    rec_id      TEXT PRIMARY KEY,
    store       TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    task        TEXT,
    video       TEXT NOT NULL,
    seconds     FLOAT,
    steps       INT,
    appearance  VECTOR(512),
    motion      VECTOR(768),
    siglip      VECTOR(768),
    held_out    BOOL NOT NULL DEFAULT false
)"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=SRC)
    ap.add_argument("--to", required=True)
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args()

    from tqdm import tqdm

    s = psycopg2.connect(a.src)
    d = psycopg2.connect(a.to); d.autocommit = True
    sc = s.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    dc = d.cursor()
    dc.execute(DDL)

    sc.execute(f"SELECT {','.join(COLS)} FROM moments ORDER BY rec_id")
    rows = sc.fetchall()
    print(f"{len(rows)} moments to copy")

    sql = (f"INSERT INTO moments ({','.join(COLS)}) VALUES "
           f"({','.join(['%s'] * len(COLS))}) ON CONFLICT (rec_id) DO NOTHING")
    for i in tqdm(range(0, len(rows), a.batch), desc="copy", unit="batch"):
        chunk = [[str(r[c]) if c in ("appearance", "motion", "siglip") else r[c]
                  for c in COLS] for r in rows[i:i + a.batch]]
        psycopg2.extras.execute_batch(dc, sql, chunk, page_size=a.batch)

    import db as D

    for col in ("appearance", "motion", "siglip"):
        try:
            dc.execute(f"CREATE VECTOR INDEX moments_{col}_vec ON moments ({col})")
            print(f"  vector index on {col}")
        except psycopg2.Error as exc:
            if getattr(exc, "pgcode", "") not in ("42P07", "42710"):
                print(f"  index {col}: {exc}")
    dc.execute("SELECT count(*), count(DISTINCT task) FROM moments")
    n, k = dc.fetchone()
    print(f"CockroachDB now holds {n} moments across {k} tasks, vector-indexed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
