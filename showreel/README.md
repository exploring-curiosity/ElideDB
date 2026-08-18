# Showreel

**You cannot describe the moment you are looking for. You can point at one.**

Type *"taking something out of the drawer"* into 3,402 clips of robot video and
you get drawers **closing**, fridges **closing**, and a fridge **opening**.
Precision: **0%**. Then hand the system one clip of it happening instead, and it
returns five more of exactly that, at **63%** against a 2.6% chance.

That is the whole demo, and it takes about eight seconds to watch. Nobody needs
the paragraph.

---

## Why typing words at video fails

```
cos( "open the drawer" , "close the drawer" )  =  0.9766
```

To a text encoder those are the same sentence. Every text-to-video product on
the market inherits that, and no amount of prompt engineering fixes it: the
distinction was destroyed at the embedding, before search began.

Across all 57 kinds of moment in the corpus:

| arm | precision@8 | chance | |
|---|---|---|---|
| **TYPE IT**: text query | 0.285 | 0.018 | 16.2× |
| **SHOW IT**: query by clip | **0.840** | 0.021 | **40.7×** |

**24 of 57 typed queries land at or below chance.** Six return nothing correct
at all: *arrange tea*, *garnish pancake*, *gather tableware*, *load dishwasher*,
*make ice lemonade*, *navigate kitchen*.

The text arm is not a straw man. It is SigLIP 2's text tower against raw
mean-pooled SigLIP 2 image embeddings: the standard zero-shot method, run
against the raw column rather than a whitened one the text tower knows nothing
about, because handicapping it would make the comparison worthless.

```bash
.venv-libero/bin/python showreel/bench.py
```

---

## How it works

```
   3,402 clips, encoded once by RelMo (V-JEPA 2 + SigLIP 2)
                    │
      ┌─────────────┴──────────────┐
      ▼                            ▼
  appearance (512)            motion (768)          siglip (768)
  RelMo's pooled prefilter    what MOVED             the text arm's target
      │                            │
      └──────────┬─────────────────┘
                 ▼
        Postgres + pgvector, HNSW on all three
                 │  STAGE 1: one SQL query, 98.6% of the corpus never scored
                 ▼
          48 candidates
                 │  STAGE 2. RelMo's DTW over the full descriptor traces
                 ▼
             top 8, ranked
```

**Stage 1 is the database doing real work**, not storing embeddings. The
indexed vector is `concat(qf, qs)/√2` of RelMo's whitened pooled channels, which
makes the database's cosine *exactly* RelMo's own prefilter: verified against
RelMo's internal score to **2.3e-08**. Stage 2 is the part a single vector
cannot do: DTW over the trace sees the order things happened in.

Median latency: **34 ms** stage 1, **350 ms** stage 2, 98.6% elided.

### Which view to prefilter by is a measured decision, and it flips

| corpus | appearance | motion |
|---|---|---|
| 57 tasks across different rooms and appliances (here) | **0.786** | 0.391 |
| 10 behaviours inside one kitchen (LIBERO) | 0.649 | **0.748** |

Match on **appearance** when the corpus varies in scene; on **motion** when it
varies only in action. In one kitchen every task looks alike and only the
movement differs, so the mean is nearly useless and the per-channel temporal
spread carries it. Both columns are indexed; `view` picks one per query.

---

## Grading

Every result carries a ✓ or ✗ and the precision on screen is computed from the
query you just ran: not quoted from this file.

The ground truth is the folder RoboCasa filed the episode under. **It is never
matched against and never read by retrieval**; deleting the `task` column would
not change a single ranking. It exists so a viewer sees a verdict instead of
taking the ordering on faith.

---

## Run it

```bash
brew services start postgresql@18
```

```bash
.venv-libero/bin/python showreel/ingest.py
```

```bash
.venv-libero/bin/python showreel/server.py
```

Then open <http://localhost:8100>.

## Or run it against the cloud

The same code, with the corpus in S3 and the memory in CockroachDB. `blob.py`
decides which, from one environment variable, and nothing above it changes.

```bash
BUCKET=precedent-corpus-yourname ./showreel/deploy/setup_bucket.sh
```

```bash
BUCKET=precedent-corpus-yourname .venv-libero/bin/python showreel/deploy/upload_assets.py
```

```bash
./showreel/space/build_space.sh ~/precedent-space
```

The container runs on ONE interpreter, because the version conflict that forced
two is a LIBERO problem and there is no LIBERO in it: the ranker imports ten
RelMo modules and all of them are numpy. Measured on the assembled tree, one
interpreter, empty trace cache, against CockroachDB Cloud and S3:

| | |
|---|---|
| resident | 1.37 GB (0.10 server + 1.27 sidecar) |
| clip query | P@8 1.000, reranked, 98.6% elided |
| stage 2 | 2,123 ms cold, 143 ms warm |
| S3 pulled | 17.2 MB for 49 traces, of 1.1 GB on offer |
| video | 307 to a presigned URL, never through the app |

Cold stage 2 is the price of a first-touch S3 fetch over home broadband. In
region it is a fraction of that, and every later query that shares a candidate
pays nothing.

Two interpreters, as everywhere in this repo: SigLIP 2's text tower needs
`transformers` 4.57, which the control venv is pinned away from. `sidecar.py`
is that boundary. `ingest.py` does **not** need it: loading a RelMo store reads
npz off disk, and only the encoder needs the newer library.

`ingest.py` encodes nothing. RelMo has already seen this video; the script reads
its traces and writes two reductions per recording.

---

## What this is honest about

**The corpus is one domain.** 3,402 RoboCasa kitchen episodes. The claim is
about text-vs-example retrieval, not about generalising to your CCTV.

**The text arm gets one phrasing per task**, derived mechanically from the task
name so it cannot be tuned. A human trying five rewordings would do better on
some queries, and would still be unable to separate *open* from *close*,
because that failure is in the encoder, not the phrasing.

**Precision@8 is not recall.** Classes hold ~90 examples, so k=8 cannot exceed
1.0 and a low score may mean the right clips ranked 9–20. The chance line is
printed beside every number for exactly this reason.

**Stage 2 is 350 ms** and scales with trace length squared. Composite episodes
(208 steps) dominate it. The Sakoe-Chiba band is on at 0.25; it has not been
swept.
