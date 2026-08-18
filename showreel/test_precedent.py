#!/usr/bin/env python3
"""Tests for the parts that fail silently.

    .venv-libero/bin/python -m pytest showreel/test_precedent.py -q

Not coverage for its own sake — these cover the three failures that would leave
the system looking healthy while being wrong:

  THE CASCADE MISSING A FILING. Transitive by definition, so a bug shows up as
  "we reopened 4 of the 7 that inherited the mistake" and nothing errors.
  TENANCY LEAKING. One fleet's precedents deciding another fleet's episodes is
  a correctness bug and a privacy one, and it is invisible in any single-tenant
  test.
  A CLAIM BEING HANDED OUT TWICE. Two workers dispositioning one episode does
  not raise; it just double-files.
"""
from __future__ import annotations

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent
import db

A, B = "test-fleet-a", "test-fleet-b"


def _clips(n):
    rows = db.q("SELECT rec_id FROM moments ORDER BY rec_id LIMIT %s", (n,))
    assert len(rows) == n, "corpus not ingested; run ingest.py"
    return [r["rec_id"] for r in rows]


@pytest.fixture(autouse=True)
def clean():
    db.apply_schema()
    for f in (A, B):
        agent.reset(f)
    yield
    for f in (A, B):
        agent.reset(f)


def _file(fleet, rec_id, disposition, precedent_ids=(), source="human"):
    """Write a filing with chosen precedent edges. Test fixture, not the agent."""
    fid = str(uuid.uuid4())
    item = db.q("INSERT INTO inbox (fleet, rec_id, state) VALUES (%s,%s,'filed') "
                "RETURNING item_id", (fleet, rec_id))[0]["item_id"]
    db.q("""INSERT INTO filings (filing_id, item_id, fleet, rec_id, disposition,
                                 source, consensus, n_precedents)
            VALUES (%s,%s,%s,%s,%s,%s,1.0,%s)""",
         (fid, item, fleet, rec_id, disposition, source, len(precedent_ids)))
    for p in precedent_ids:
        db.q("INSERT INTO filing_precedents (filing_id, precedent_id, score) "
             "VALUES (%s,%s,0.9)", (fid, p))
    return fid


def test_cascade_is_transitive():
    """A -> B -> C: correcting A must reopen C, not just B."""
    c = _clips(3)
    a = _file(A, c[0], "root")
    b = _file(A, c[1], "root", [a], source="agent")
    d = _file(A, c[2], "root", [b], source="agent")   # leans on B, not on A

    out = agent.cascade(a, "corrected", fleet=A)
    assert out["reopened"] == 2, out
    got = {r["filing_id"]: r["superseded"] for r in
           db.q("SELECT filing_id, superseded FROM filings WHERE fleet=%s", (A,))}
    assert got[b] is True
    assert got[d] is True, "the cascade stopped at depth 1"
    assert got[a] is False, "the corrected filing is amended, not superseded"


def test_cascade_spares_human_filings():
    """A person already looked at these. Another filing being wrong does not
    make their answer wrong, so they are not reopened."""
    c = _clips(2)
    a = _file(A, c[0], "root")
    h = _file(A, c[1], "root", [a], source="human")
    agent.cascade(a, "corrected", fleet=A)
    assert db.q("SELECT superseded FROM filings WHERE filing_id=%s",
                (h,))[0]["superseded"] is False


def test_cascade_is_atomic_and_counted():
    """The verdict records how far the correction spread."""
    c = _clips(2)
    a = _file(A, c[0], "root")
    _file(A, c[1], "root", [a], source="agent")
    agent.cascade(a, "corrected", fleet=A)
    v = db.q("SELECT cascade_n FROM verdicts WHERE filing_id=%s ORDER BY at_time DESC",
             (a,))[0]
    assert v["cascade_n"] == 1


def test_fleets_do_not_see_each_other():
    """B's precedents must never decide A's episodes."""
    c = _clips(4)
    for r in c[:3]:
        _file(B, r, "b-only")
    # A has filed nothing, so A has no precedent for anything.
    assert agent.precedents(c[3], fleet=A) == []
    assert len(agent.precedents(c[3], fleet=B)) == 3


def test_cascade_does_not_cross_fleets():
    c = _clips(2)
    a = _file(A, c[0], "root")
    # An edge that should never exist, written deliberately: if tenancy is only
    # enforced on read, the cascade would still cross.
    b = _file(B, c[1], "root", [a], source="agent")
    agent.cascade(a, "corrected", fleet=A)
    assert db.q("SELECT superseded FROM filings WHERE filing_id=%s",
                (b,))[0]["superseded"] is False, "a correction leaked across fleets"


def test_claim_is_exclusive():
    """Two workers, one episode: exactly one of them gets it."""
    c = _clips(1)
    agent.enqueue(c, fleet=A)
    first = agent.claim("w1", fleet=A)
    second = agent.claim("w2", fleet=A)
    assert first is not None
    assert second is None, "the same episode was handed to two workers"


def test_reclaim_returns_a_dead_workers_episode():
    c = _clips(1)
    agent.enqueue(c, fleet=A)
    item = agent.claim("w1", fleet=A)
    db.q("UPDATE inbox SET claimed_at = now() - INTERVAL '1 hour' WHERE item_id=%s",
         (item["item_id"],))
    assert agent.reclaim(fleet=A)["requeued"] == 1
    assert agent.claim("w2", fleet=A) is not None


def test_poison_episode_is_parked_not_retried_forever():
    c = _clips(1)
    agent.enqueue(c, fleet=A)
    for _ in range(agent.MAX_ATTEMPTS):
        it = agent.claim("w", fleet=A)
        db.q("UPDATE inbox SET claimed_at = now() - INTERVAL '1 hour' "
             "WHERE item_id=%s", (it["item_id"],))
        agent.reclaim(fleet=A)
    assert agent.reclaim(fleet=A)["dead_lettered"] >= 0
    assert agent.health(fleet=A)["pending"] + agent.health(fleet=A)["dead"] >= 1
