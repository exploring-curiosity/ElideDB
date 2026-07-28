# Deploying the ElideDB demo

Everything here is prepared and locally verified. What remains is the
one thing only an account owner can do: create the hosting target and
push. No code changes are needed at deploy time.

## What the demo is

The real product surface, read only: the Desk console serving the
standalone demo store (1,122 robot manipulation episodes, 1.0 GB,
media materialized inside the store). Search runs the full fitted
channel stack on CPU.

Measured in the container end to end: first boot warms every text
tower before serving (about 15 minutes of downloads and loads, once
per container); the first query on a store pays a few minutes of
one-time index touches; warm queries then run in about 5 seconds
with all channels live. The demo-environment benchmark measured
0.36 precision / 0.33 yield against the 0.38 / 0.36 reference row
(two borderline clips flip under bfloat16 arithmetic; hosts with
memory to spare can set ELIDEDB_DTYPE=float32). Maintenance and
index mutations return 403.

## Option A: Hugging Face Spaces (free)

Free CPU basic hardware (2 vCPU, 16 GB RAM) is enough, and every
model the demo loads is ungated, so no tokens or secrets are needed.

    python scripts/build_demo_store.py      # once, if not built
    python deploy/stage_space.py            # assembles deploy/space/

The stager prints the exact login, create, and push commands. First
boot downloads about 5 GB of weights and warms the towers (about 15
minutes); restarts refetch them (free Spaces have no persistent
disk). That costs minutes at boot, nothing at query time.

## Option B: any Docker host (a few dollars a month)

Hetzner CX32 or an equivalent 4 GB box is too small; use 16 GB
(about 12 EUR). Then:

    docker build -f deploy/Dockerfile -t elidedb-demo .
    docker run -d -p 80:7860 elidedb-demo

## Local verification (already run)

    python scripts/build_demo_store.py
    docker build -f deploy/Dockerfile -t elidedb-demo .
    docker run -p 7860:7860 elidedb-demo
    curl localhost:7860/api/stores

The CPU query path was verified natively as well:
`ELIDEDB_DEVICE=cpu ELIDEDB_TEXT_BACKEND=torch` runs the same
`search_set` with cosine parity above 0.9999 against the Apple
Silicon path.

## What is intentionally not in the demo

The geometry audit tier needs the SAM 3 tracker stack, which is not
in this image; audited queries fall back to the fused set and say so
in the response. Everything else is the full system.
