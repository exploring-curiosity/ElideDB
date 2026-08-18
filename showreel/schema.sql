-- PRECEDENT: the agent's memory. Four tables, four different jobs.
--
-- Written to be valid on BOTH PostgreSQL+pgvector and CockroachDB, because the
-- port is meant to be a connection string and not a rewrite. That constrains a
-- few choices and each one is deliberate:
--
--   UUID keys, never SERIAL. A sequence is a single hot range in a distributed
--   cluster; every insert in the fleet would queue behind one leaseholder.
--   TIMESTAMPTZ everywhere, never TIMESTAMP. A fleet is not in one timezone.
--   No foreign keys across the hot path. They are correct and they serialise
--   writes; the cascade below does the integrity work explicitly and in one
--   transaction, which is the thing worth demonstrating anyway.
--
-- The vector index is the one statement that genuinely differs between the two
-- engines, so it lives in db.py rather than here.

-- ---------------------------------------------------------------------------
-- 1. THE INBOX: task state. Episodes arrive here and wait to be dispositioned.
--
-- Claimed with FOR UPDATE SKIP LOCKED so any number of agent workers can drain
-- one queue without two of them taking the same episode. This is the table that
-- makes the system a production queue rather than a for-loop over a list.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inbox (
    item_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fleet       TEXT NOT NULL,            -- tenant. Every read is scoped by it.
    rec_id      TEXT NOT NULL,            -- the clip, in `moments`
    arrived     TIMESTAMPTZ NOT NULL DEFAULT now(),
    state       TEXT NOT NULL DEFAULT 'pending',
                -- pending -> working -> filed | escalated ; reopened by a cascade
    claimed_by  TEXT,
    claimed_at  TIMESTAMPTZ,
    attempts    INT NOT NULL DEFAULT 0    -- a worker that dies mid-episode
);

-- ---------------------------------------------------------------------------
-- 2. FILINGS: what the agent decided, and how sure it was.
--
-- `disposition` is a string the AGENT coined or a human supplied. Nothing seeds
-- it: on an empty memory the first episode of every kind is escalated and the
-- human's answer becomes the vocabulary. There is no taxonomy in this schema.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS filings (
    filing_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id       UUID NOT NULL,
    fleet         TEXT NOT NULL,
    rec_id        TEXT NOT NULL,
    disposition   TEXT NOT NULL,
    source        TEXT NOT NULL,          -- 'agent' | 'human'
    consensus     FLOAT,                  -- agreement among the precedents used
    n_precedents  INT NOT NULL DEFAULT 0,
    decided_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- set by the cascade when the ground under this filing moves
    superseded    BOOL NOT NULL DEFAULT false,
    superseded_by UUID
);

-- ---------------------------------------------------------------------------
-- 3. THE PRECEDENT GRAPH, which filings supported which other filing.
--
-- THIS IS THE TABLE THAT MAKES THE MEMORY CORRECTABLE. A vector store can tell
-- you what is similar; it cannot tell you which decisions LEANED ON a decision
-- that later turned out to be wrong. Every edge is written in the same
-- transaction as the filing it belongs to, so the graph is never half-built.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS filing_precedents (
    filing_id     UUID NOT NULL,
    precedent_id  UUID NOT NULL,          -- another filing
    score         FLOAT,                  -- how close it was
    PRIMARY KEY (filing_id, precedent_id)
);

-- ---------------------------------------------------------------------------
-- 4. VERDICTS: a person's answer. The only place a human writes.
--
-- An escalation produces one of these. So does an overturn, and an overturn is
-- what triggers the cascade: every filing that leaned on the corrected one is
-- reopened in the same transaction, transitively.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS verdicts (
    verdict_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id   UUID NOT NULL,
    fleet       TEXT NOT NULL,
    disposition TEXT NOT NULL,
    by_whom     TEXT NOT NULL,
    at_time     TIMESTAMPTZ NOT NULL DEFAULT now(),
    note        TEXT,
    -- how many filings this one answer invalidated. Observability, not audit
    -- theatre: it is the number that tells an operator a bad call spread.
    cascade_n   INT NOT NULL DEFAULT 0
);
