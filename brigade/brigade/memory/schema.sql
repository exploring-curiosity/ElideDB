-- Brigade's memory. Four kinds, one database.
--
-- Ordering matters: CockroachDB's docs warn that building a vector index on a
-- populated table is expensive and that bulk paths (IMPORT INTO) are not
-- supported on tables carrying one. So every table is created first and every
-- vector index is created immediately after, while the table is still empty.
--
-- Nothing here is Brigade-specific SQL trickery. It is a plain relational schema
-- because the interesting claim is not "we used a database", it is "the robot
-- cannot act without reading this".

-- ---------------------------------------------------------------------------
-- EPISODIC: everything that happened, as text the robot wrote about itself.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    robot_id      STRING NOT NULL,
    kitchen_id    STRING NOT NULL,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind          STRING NOT NULL,      -- observation|action|outcome|instruction|reflection
    -- The self-description. This is what gets embedded, and what a human sees
    -- in the dashboard. Written to read like something you'd type in a search
    -- box: "place bowl_c in cab_2_main_group — success — 12.4s — strategy=top".
    text          STRING NOT NULL,
    embedding     VECTOR(384),
    payload       JSONB NOT NULL DEFAULT '{}'::JSONB,
    task_id       UUID,
    subject       STRING,               -- the object or fixture this concerns
    outcome       STRING,               -- success|failure|NULL for non-actions
    clip_key      STRING,               -- S3 key of the video clip, if recorded
    recording_id  STRING,               -- joins to relmo_recordings
    INDEX events_by_time (kitchen_id, ts DESC),
    INDEX events_by_subject (kitchen_id, subject, ts DESC),
    INVERTED INDEX events_payload (payload)
);

-- ---------------------------------------------------------------------------
-- SPATIAL: where the robot believes each object is. Beliefs, not facts —
-- `stale` and `confidence` exist because the world changes when nobody is
-- looking, and a memory layer that cannot represent "I might be wrong" will
-- send a robot confidently to an empty shelf.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS object_beliefs (
    kitchen_id    STRING NOT NULL,
    instance_id   STRING NOT NULL,
    label         STRING NOT NULL,
    location      STRING NOT NULL,      -- fixture name
    pos           FLOAT8[] NOT NULL,
    confidence    FLOAT NOT NULL DEFAULT 1.0,
    last_seen     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_verified TIMESTAMPTZ,
    stale         BOOL NOT NULL DEFAULT false,
    evidence_event_id UUID,
    PRIMARY KEY (kitchen_id, instance_id),
    INDEX beliefs_by_label (kitchen_id, label, stale)
);

-- ---------------------------------------------------------------------------
-- PROCEDURAL (1): where things BELONG. The difference between this and
-- object_beliefs is the whole point of act 3 — a belief is "the bowl is on the
-- counter", a norm is "bowls go in the cabinet". Norms outlive the objects.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS norms (
    kitchen_id    STRING NOT NULL,
    label         STRING NOT NULL,
    home_location STRING NOT NULL,
    confidence    FLOAT NOT NULL DEFAULT 0.5,
    n_episodes    INT NOT NULL DEFAULT 1,
    source        STRING NOT NULL DEFAULT 'learned',   -- learned|instructed
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kitchen_id, label)
);

-- ---------------------------------------------------------------------------
-- PROCEDURAL (2): which way of doing a thing actually works. Act 4 reads this
-- to flip strategy after a failure.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skill_stats (
    robot_id      STRING NOT NULL,
    skill         STRING NOT NULL,
    object_label  STRING NOT NULL,
    strategy      STRING NOT NULL,
    n_try         INT NOT NULL DEFAULT 0,
    n_ok          INT NOT NULL DEFAULT 0,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (robot_id, skill, object_label, strategy)
);

-- ---------------------------------------------------------------------------
-- WORKING: the task queue. `origin='self'` is what makes Brigade an agent
-- rather than a remote control — the robot writes most of these rows itself.
-- Claimed with SELECT ... FOR UPDATE SKIP LOCKED so N workers never collide.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tasks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kitchen_id    STRING NOT NULL,
    goal          STRING NOT NULL,
    priority      INT NOT NULL DEFAULT 5,
    state         STRING NOT NULL DEFAULT 'pending',
    origin        STRING NOT NULL DEFAULT 'self',      -- self|user
    subject       STRING,
    claimed_by    STRING,
    claimed_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    result        STRING,
    payload       JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    INDEX tasks_queue (kitchen_id, state, priority DESC, created_at ASC)
);

-- ---------------------------------------------------------------------------
-- AUDIT: what was decided, and which memories were read to decide it. This is
-- the table that proves memory is load-bearing rather than decorative — if
-- recalled_event_ids is empty, the robot was not using its memory.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decisions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    robot_id      STRING NOT NULL,
    kitchen_id    STRING NOT NULL,
    task_id       UUID,
    chose         STRING NOT NULL,
    rationale     STRING,
    recalled_event_ids UUID[],
    recall_latency_ms  FLOAT,
    source        STRING NOT NULL DEFAULT 'planner',   -- planner|llm
    INDEX decisions_by_time (kitchen_id, ts DESC)
);

-- ---------------------------------------------------------------------------
-- ELIDEDB / RELMO stage 1. One row per encoded recording.
--
-- `embedding` is concat(pf, ps)/sqrt(2): the store's PCA-whitened, L2-normalised
-- pooled V-JEPA and SigLIP2 channels. Inner product on this vector reproduces
-- RelMo's own prefilter score exactly (verified to 5.55e-17), so this index is
-- not an approximation of RelMo's retrieval — it IS stage 1 of it.
--
-- `basis_id` is load-bearing: the PCA basis is fitted over store contents, so a
-- refit silently invalidates every vector already stored. Carrying the basis
-- hash as a column turns that from corruption into a visible namespace change.
--
-- The per-step DTW bank (~1.8 MB/recording) exceeds CockroachDB's guidance for
-- a single value, so it lives in S3 and the row carries only the key.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS relmo_recordings (
    recording_id  STRING PRIMARY KEY,
    store         STRING NOT NULL,
    basis_id      STRING NOT NULL,
    robot_id      STRING,
    kitchen_id    STRING,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_s    FLOAT,
    n_steps       INT,
    embedding     VECTOR(512),
    bank_key      STRING,
    lang          STRING,
    outcome       STRING,
    event_id      UUID,
    INDEX relmo_by_store (store, basis_id)
);
