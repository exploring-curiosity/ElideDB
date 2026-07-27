# Deploying the ElideDB demo

Everything here is prepared and locally verified. What remains is the
one thing only an account owner can do: create the hosting target and
push. No code changes are needed at deploy time.

## What the demo is

The real product surface, read only: the Desk console serving the
standalone demo store (1,122 robot manipulation episodes, 1.0 GB,
media materialized inside the store). Search runs the full fitted
channel stack on CPU. Measured on 2 CPU cores: about 27 s cold start
for model loads, then under a second per warm query. Quality is
identical to the benchmarked configuration; the fitted weights ship
with the store. Maintenance and index mutations return 403.

## Option A: Hugging Face Spaces (free)

Free CPU basic hardware (2 vCPU, 16 GB RAM) is enough. Steps:

1. Create a Space, SDK type "Docker", visibility public.
2. In the Space repo, place this repository's `python/`,
   `scripts/get_iv2.py`, `deploy/` (including `deploy/demo/lake`,
   built by `python scripts/build_demo_store.py`), and `.dockerignore`.
   Copy `deploy/Dockerfile` to the repo root as `Dockerfile` and fix
   its COPY paths accordingly, or keep the layout and set
   `dockerfile_path: deploy/Dockerfile` in the Space README metadata.
3. The Space README needs this front matter:

       ---
       title: ElideDB
       emoji: "0"
       sdk: docker
       app_port: 7860
       ---

4. Push with git lfs for the store parquet and media files
   (`git lfs track "*.parquet" "*.h264"`).
5. First boot downloads about 5 GB of model weights from public
   Hugging Face repos. No tokens or secrets are required; every model
   the demo loads is ungated.

Restarts refetch weights (free Spaces have no persistent disk). That
costs minutes at boot, nothing at query time.

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
