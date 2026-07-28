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

## Hosting options, priced honestly

Hugging Face now requires a PRO subscription (9 USD a month) to host
Docker Spaces, even on free CPU hardware (verified 2026-07: repo
create returns 402 without PRO). So the real choices are:

### Option A: a small VPS (recommended, ~7 to 13 EUR a month)

Better than a Space for a demo anyway: weights persist across
restarts, so the 15 minute warm happens once, and you can point a
custom domain at it. Hetzner CAX31 (ARM, 8 vCPU, 16 GB, ~13 EUR) or
CAX21 (4 vCPU, 8 GB, ~7 EUR; workable since the query path resides
in about 5.4 GB, but 16 GB is the comfortable choice). On the box:

    apt install -y docker.io git git-lfs
    git clone <your repo> && cd <repo>
    python3 scripts/build_demo_store.py    # or scp deploy/demo/ over
    docker build -f deploy/Dockerfile -t elidedb-demo .
    docker run -d --restart unless-stopped -p 80:7860 elidedb-demo

The store is not in git; either rebuild it on the box from the lake
or copy `deploy/demo/` over with scp/rsync.

### Option B: Hugging Face Spaces (9 USD a month for PRO)

Zero ops once subscribed. The stager does everything:

    python scripts/build_demo_store.py      # once, if not built
    python deploy/stage_space.py            # assembles deploy/space/

The stager prints the exact login, create, and push commands. Every
model is ungated, so no tokens or secrets. Note: free Spaces have no
persistent disk, so every restart repeats the 15 minute warm.

### Option C: Oracle Cloud Always Free (0 USD, more setup)

Oracle's Always Free tier includes an ARM VM with 4 OCPUs and 24 GB
RAM, which fits this workload with room to spare. The demo image
already builds and runs on ARM (the locally verified container is
linux/arm64). Follow Option A's commands on that VM. The trade is
Oracle's signup friction and capacity availability by region.

Whichever host: the landing page itself is static and free anywhere
(Netlify Drop, Cloudflare Pages, GitHub Pages).

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
