#!/usr/bin/env python3
"""PRECEDENT — the agent. It claims work, remembers, decides, and is correctable.

    .venv-libero/bin/python showreel/agent.py --demo

A robot fleet produces more episodes than anyone can watch, and each one has to
be dispositioned before it can be counted, compared or acted on. Today that
means a taxonomy defined up front and a labelling contract before a single clip
is searchable. This agent starts with an EMPTY memory and no vocabulary at all.

    CLAIM      take one episode off the queue (FOR UPDATE SKIP LOCKED)
    RETRIEVE   the nearest precedents this fleet has already filed
    ACT        they agree -> file it alone, and record WHICH ones convinced it
               they don't -> escalate; a person answers and that becomes memory
    STORE      the filing, its precedent edges, and the verdict if there was one

Everything the agent knows it wrote itself. The first episode of every kind is
escalated because an empty memory can do nothing else; the vocabulary is what
the humans answered, not a taxonomy anyone designed.

WHY CONSENSUS RATHER THAN A SIMILARITY THRESHOLD, measured the hard way: an
earlier version escalated when the top match scored below a fitted cut and the
arms came out backwards — storing made escalation WORSE. Every DTW score sat
between 0.16 and 0.27 whether the precedent was right or wrong. This retrieval
ranks well (precision@8 0.840) and calibrates badly, and those are different
properties. Agreement among the top-k needs no calibration.

AND THE PART A VECTOR STORE CANNOT DO. Every filing records the filings that
convinced it. When a human overturns one, `cascade` walks that graph backwards
in a single serialisable transaction and reopens everything that leaned on it,
transitively. Similarity search can tell you what looks alike; only the graph
can tell you which decisions inherited a mistake.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db

FLEET = os.environ.get("PRECEDENT_FLEET", "fleet-a")
K = 5                      # precedents consulted per episode
MIN_PRECEDENTS = 2         # below this the agent has no basis to act alone
# CHOSEN FROM THE MEASURED CURVE, not from taste. Sweeping this one knob over
# 240 episodes (run_agent.py prints the whole sweep, and it is the only honest
# way to quote it):
#
#     consensus   escalated   filed alone   of those, right
#           0.5          1%           238             8.0%
#           0.6         40%           145            55.2%
#           0.8         71%            70            92.9%
#           1.0         85%            37            97.3%
#
# Below ~0.6 the agent acts on almost everything and is usually wrong: early in
# a cold start the memory holds three filings, two happen to agree, and that
# clears a low bar. 0.8 is the point where it still removes a third of the queue
# and is right nine times in ten. An operator with a different cost of being
# wrong should move it, which is why it is an argument and not a constant.
CONSENSUS = 0.8


# ------------------------------------------------------------------- the queue

def enqueue(rec_ids: list[str], fleet: str = FLEET) -> int:
    with db.connect() as (_c, cur):
        cur.executemany(
            "INSERT INTO inbox (fleet, rec_id) VALUES (%s,%s)",
            [(fleet, r) for r in rec_ids])
    return len(rec_ids)


def claim(worker: str, fleet: str = FLEET) -> dict | None:
    """Take one episode, atomically, without blocking other workers.

    SKIP LOCKED is what lets N agent processes drain one queue: a row already
    being claimed is stepped over rather than waited on. Without it every worker
    serialises behind the oldest pending row and adding workers adds nothing.
    """
    def go(cur):
        cur.execute("""
            SELECT item_id, rec_id FROM inbox
             WHERE fleet = %s AND state IN ('pending','reopened')
             ORDER BY arrived LIMIT 1
               FOR UPDATE SKIP LOCKED""", (fleet,))
        row = cur.fetchone()
        if not row:
            return None
        cur.execute("""UPDATE inbox
                          SET state='working', claimed_by=%s, claimed_at=now(),
                              attempts = attempts + 1
                        WHERE item_id=%s""", (worker, row["item_id"]))
        return dict(row)
    return db.tx(go)


# --------------------------------------------------------------- the recall

def precedents(rec_id: str, fleet: str = FLEET, k: int = K) -> list[dict]:
    """The nearest episodes THIS FLEET HAS ALREADY FILED.

    The join is the point. A vector search over the whole corpus would return
    clips the agent has never dispositioned, which are not evidence about
    anything — they are just pixels that look similar. Only a filing is a
    precedent, so the similarity scan and the decision log are read together,
    and the predicate is pushed into the query rather than filtered afterwards
    in Python (which silently starves the agent as the corpus grows).
    """
    return db.q("""
        WITH q AS (SELECT appearance FROM moments WHERE rec_id = %s)
        SELECT f.filing_id, f.rec_id, f.disposition,
               1 - (m.appearance <=> (SELECT appearance FROM q)) AS score
          FROM filings f
          JOIN moments m ON m.rec_id = f.rec_id
         WHERE f.fleet = %s AND NOT f.superseded AND f.rec_id <> %s
         ORDER BY m.appearance <=> (SELECT appearance FROM q)
         LIMIT %s""", (rec_id, fleet, rec_id, k))


# ----------------------------------------------------------------- the act

def decide(rec_id: str, fleet: str = FLEET, k: int = K,
           consensus: float = CONSENSUS, min_prec: int = MIN_PRECEDENTS) -> dict:
    """-> what the agent would do, and the evidence for it. No writes."""
    p = precedents(rec_id, fleet, k)
    votes = collections.Counter(x["disposition"] for x in p)
    top, n = votes.most_common(1)[0] if votes else (None, 0)
    agree = n / len(p) if p else 0.0
    confident = bool(p) and len(p) >= min_prec and agree >= consensus
    return dict(disposition=top if confident else None, confident=confident,
                consensus=round(agree, 3), precedents=p,
                used=[x for x in p if x["disposition"] == top] if confident else [])


def handle(item: dict, answer=None, fleet: str = FLEET, **kw) -> dict:
    """One episode, start to finish. Files alone or escalates for an answer.

    `answer(rec_id) -> disposition` stands in for the person. In the demo it is
    the corpus's own folder name; in production it is a human in a review queue
    and nothing else about this function changes.
    """
    d = decide(item["rec_id"], fleet, **kw)
    escalated = not d["confident"]
    disposition = d["disposition"]
    if escalated:
        if answer is None:
            return dict(escalated=True, filed=False, **d)
        disposition = answer(item["rec_id"])

    def go(cur):
        fid = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO filings (filing_id, item_id, fleet, rec_id, disposition,
                                 source, consensus, n_precedents)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (fid, item["item_id"], fleet, item["rec_id"], disposition,
             "human" if escalated else "agent", d["consensus"], len(d["used"])))
        # The edges go in the SAME transaction as the filing. A half-written
        # graph would make the cascade quietly incomplete, which is worse than
        # not having one: it would report success and miss rows.
        for u in d["used"]:
            cur.execute("""INSERT INTO filing_precedents (filing_id, precedent_id, score)
                           VALUES (%s,%s,%s)""", (fid, u["filing_id"], u["score"]))
        if escalated:
            cur.execute("""INSERT INTO verdicts (filing_id, fleet, disposition, by_whom, note)
                           VALUES (%s,%s,%s,%s,%s)""",
                        (fid, fleet, disposition, "reviewer",
                         "escalated: no agreeing precedent"))
        cur.execute("UPDATE inbox SET state=%s WHERE item_id=%s",
                    ("escalated" if escalated else "filed", item["item_id"]))
        return fid
    fid = db.tx(go)
    out = dict(d)
    out.update(escalated=escalated, filed=True, filing_id=fid,
               disposition=disposition)
    return out


# ------------------------------------------------------------- the cascade

def cascade(filing_id: str, disposition: str, by_whom: str = "reviewer",
            note: str = "overturned", fleet: str = FLEET) -> dict:
    """A person overturns one filing. Everything that leaned on it reopens.

    ONE transaction, and it has to be: the corrected filing, the verdict, the
    transitive set of filings that inherited the mistake, and their queue rows
    all move together or not at all. A crash halfway through the old way would
    leave filings marked superseded that were never re-queued — invisible work
    that no longer exists anywhere.

    The recursive CTE walks the precedent graph BACKWARDS: not "what did this
    lean on" but "what leaned on this", transitively, which is the direction a
    correction travels. `prec_reverse` is the index that keeps each level a
    lookup instead of a scan.
    """
    def go(cur):
        cur.execute("""
            WITH RECURSIVE tainted(filing_id) AS (
                SELECT %s::uuid
                UNION
                SELECT fp.filing_id
                  FROM filing_precedents fp
                  JOIN tainted t ON fp.precedent_id = t.filing_id
            )
            SELECT filing_id FROM tainted WHERE filing_id <> %s::uuid""",
            (filing_id, filing_id))
        hit = [r["filing_id"] for r in cur.fetchall()]

        cur.execute("""UPDATE filings SET disposition=%s, source='human'
                        WHERE filing_id=%s""", (disposition, filing_id))
        if hit:
            # Only AGENT filings are reopened. A human already looked at the
            # others and their answer does not become wrong because a different
            # filing did.
            cur.execute("""UPDATE filings SET superseded=true, superseded_by=%s
                            WHERE filing_id = ANY(%s::uuid[]) AND source='agent'""",
                        (filing_id, hit))
            cur.execute("""UPDATE inbox SET state='reopened'
                            WHERE item_id IN (SELECT item_id FROM filings
                                               WHERE filing_id = ANY(%s::uuid[])
                                                 AND source='agent')""", (hit,))
        cur.execute("""INSERT INTO verdicts (filing_id, fleet, disposition,
                                             by_whom, note, cascade_n)
                       VALUES (%s,%s,%s,%s,%s,%s)""",
                    (filing_id, fleet, disposition, by_whom, note, len(hit)))
        return hit
    hit = db.tx(go)
    return dict(corrected=filing_id, disposition=disposition,
                reopened=len(hit), filings=hit)


# ------------------------------------------------------------------- helpers

def reset(fleet: str = FLEET) -> None:
    """Forget everything. The agent must start each run knowing nothing."""
    db.q("DELETE FROM filing_precedents WHERE filing_id IN "
         "(SELECT filing_id FROM filings WHERE fleet=%s)", (fleet,))
    for t in ("verdicts", "filings", "inbox"):
        db.q(f"DELETE FROM {t} WHERE fleet=%s", (fleet,))


def stats(fleet: str = FLEET) -> dict:
    r = db.q("""SELECT
                  (SELECT count(*) FROM inbox WHERE fleet=%s) AS queued,
                  (SELECT count(*) FROM inbox WHERE fleet=%s AND state='pending') AS pending,
                  (SELECT count(*) FROM filings WHERE fleet=%s AND NOT superseded) AS filings,
                  (SELECT count(*) FROM filings WHERE fleet=%s AND source='agent'
                                                 AND NOT superseded) AS by_agent,
                  (SELECT count(DISTINCT disposition) FROM filings WHERE fleet=%s) AS vocabulary,
                  (SELECT count(*) FROM filing_precedents) AS edges,
                  (SELECT count(*) FROM verdicts WHERE fleet=%s) AS verdicts""",
             (fleet,) * 6)[0]
    return {k: int(v) for k, v in r.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.schema:
        print(db.apply_schema())
    if a.reset:
        reset(); print("memory cleared")
    if a.stats:
        print(json.dumps(stats(), indent=1))
