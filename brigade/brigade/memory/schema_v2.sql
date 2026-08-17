-- Brigade's memory, second design: VIDEO AND NOTHING ELSE.
--
-- The first design stored beliefs (bowl -> cabinet, xyz, confidence), norms
-- (label -> home_location with episode counts) and text embeddings of events.
-- All three are derived facts saved as state, and the owner's ruling removes
-- them outright: "the memory is reIMO... no other info goes in", "no saving
-- hard numbers/ state mem", and finally, closing the last loophole, the human's
-- own words are not stored either.
--
-- So what remains is a video database. Segments of camera, their RelMo vectors,
-- and time. Where the bowl is kept is not a column here and never will be — it
-- is derived at read time by the reasoning layer from the clips this table
-- returns.
--
-- WHAT IS DELIBERATELY ABSENT, so that a later hand does not helpfully add it
-- back: no label, no caption, no instruction, no object name, no location, no
-- outcome, no success flag. A column here that a human could read as a fact
-- about the kitchen is a bug.

-- ---------------------------------------------------------------------------
-- THE STORE. One row per segment of video written by the cameras.
--
-- Segments, not episodes: the cameras stream continuously and are cut on a
-- fixed clock, so the memory does not depend on the robot telling it when
-- something interesting began. A memory that only records during tasks cannot
-- answer questions about the time between them.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clips (
    clip_id       STRING PRIMARY KEY,
    kitchen_id    STRING NOT NULL,
    robot_id      STRING NOT NULL,
    camera        STRING NOT NULL,

    -- when this segment covers. The ONLY index into content a human gets
    -- besides similarity, and the thing "or its timestamps" refers to.
    t0            TIMESTAMPTZ NOT NULL,
    t1            TIMESTAMPTZ NOT NULL,
    n_frames      INT NOT NULL,
    fps           FLOAT NOT NULL,

    -- the video itself. Local path now; an S3 key when this moves to AWS,
    -- which is why it is a string and not a blob.
    path          STRING NOT NULL,

    -- RelMo STAGE 1. concat(whitened pooled V-JEPA, whitened pooled SigLIP2)
    -- /sqrt(2), so the database's cosine IS RelMo's prefilter rather than an
    -- approximation of it — verified to 2.3e-08. This column exists to be
    -- INDEXED; it is a coarse quantiser, not the memory.
    embedding     VECTOR(512),
    -- The whitening basis is fitted over a store's contents, so vectors under
    -- different bases are incomparable. Carrying the id turns a silent
    -- corruption into a visible namespace. It changes when the kitchen refits
    -- the basis on its own video, which is how RelMo is deployed on a new site.
    basis_id      STRING,

    -- THE MOTION VIEW. The same trace reduced a different way: the per-channel
    -- standard deviation over time, i.e. how much each feature MOVED while the
    -- span ran, rather than where it sat. The mean is the same for every
    -- behaviour here because it is the same kitchen; the spread is not.
    -- Measured on 59 spans, nearest-neighbour behaviour match with overlapping
    -- spans barred, chance 0.100: mean 0.475, DTW 0.576, this 0.712.
    -- Indexed like the other, so a query can be asked of either view.
    motion        VECTOR(768),

    -- RelMo STAGE 2. The descriptor trace: (steps, 1024) V-JEPA + (steps, 768)
    -- SigLIP2, on disk beside the video. Stage 1 narrows to a shortlist and DTW
    -- over these traces ranks it, because neither reduction above sees ORDER —
    -- a mean and a spread are both invariant to shuffling the frames. Still not
    -- a fact about the kitchen: two float arrays with no name for anything.
    trace_path    STRING,
    steps         INT,

    INDEX clips_by_time (kitchen_id, t0 DESC)
);

-- ---------------------------------------------------------------------------
-- OPERATIONAL AUDIT. Not memory — the recall path never reads these. They
-- exist so a human can see what the agent did and which clips it read to
-- decide, which is the only way "memory was load-bearing" is checkable rather
-- than assertable.
--
-- `heard` is what the human said. It is written here AFTER the fact, for the
-- transcript, and is never retrieved, never embedded, and never an input to
-- any decision. If that ever changes, this table has become memory.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS turns (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kitchen_id    STRING NOT NULL,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    heard         STRING NOT NULL,
    -- which clips the reasoning layer read
    read_clips    STRING[],
    -- the command it emitted, as weights over prototypes. NOT a sentence.
    weights       FLOAT8[],
    margin        FLOAT,
    acted         BOOL NOT NULL DEFAULT false,
    succeeded     BOOL,
    seconds       FLOAT,
    retrieval_ms  FLOAT,
    reason_ms     FLOAT,
    INDEX turns_by_time (kitchen_id, ts DESC)
);
