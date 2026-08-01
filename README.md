# ElideDB

**A Parquet-native, timestamp-first database for camera and sensor
recordings.** Put any timestamped data in (sensor rows, video, GPS,
audio, logs) and get back time-window reads, SQL, and event search,
while the engine reads as few bytes as physically possible.
*The best read is the read elided.*

- **Everything is Parquet.** Every table is plain Parquet files under a
  Delta-Lake-style transaction log. No custom formats; DuckDB, Spark,
  pandas, anything that reads Parquet reads your database.
- **Timestamps are the law.** Every table has a `ts` column (int64
  nanoseconds). That one rule is what makes cross-modal queries,
  alignment, and pruning work.
- **The store is standalone.** Ingest transcodes media once into the
  store directory and indexes it to the byte; the database keeps
  serving after the original files are archived or deleted. Raw
  sources are never modified.
- **Search by describing the event.** "The arm closes the drawer"
  returns playable clips where it actually happens. No labels, no
  language model in the query path, and a question the footage cannot
  answer returns nothing, with the reason.

## Install

```bash
pip install -e ".[ml]"     # from the repo root; [ml] adds search
brew install ffmpeg        # clip playback + video indexing
```

## 60 seconds to your first database

```bash
elidedb create lake/mydb --name "my project"

# any timestamped rows: CSV/Parquet, ISO dates or epoch s/ms/us/ns
elidedb add lake/mydb readings sensor_log.csv --ts-col time

# any video
elidedb video lake/mydb dashcam.mp4 --stream front

elidedb ls lake/mydb                                   # what's inside
elidedb sql lake/mydb "SELECT count(*) FROM readings"  # SQL via DuckDB
elidedb embed lake/mydb                                # local ML, once
elidedb desk                                           # browse it
```

## Event search

The query model is a set, not a top hit: a robotics team wants every
clip where the thing happened, clean enough to review or retrain on.

```python
from elidedb import Store
from elidedb.scenario import search_set

db = Store.open("lake/mydb")
r = search_set(db, "the robot arm closes the drawer", k_max=10)
for c in r["clips"]:
    print(c["stream"], c["t0"], c["t1"], c["score"])
```

Under the hood, per-store fitted retrieval over open world models and
video-native encoders (V-JEPA 2, InternVideo2, SigLIP 2, Perception
Encoder, X-CLIP, FastSAM regions), fused by learned weights and passed
through a fitted selection chain: no-match gate, direction filter,
event dedup, and a confidence cut, so the returned set ends where the
evidence does. An opt-in geometry tier verifies spatial relations with
the SAM 3 video tracker. Nothing in the engine is tuned to a dataset;
whatever matters in your corpus is learned from your corpus, and every
change to retrieval lands with a benchmark row against a frozen,
hand-graded truth set ([BENCHMARKS.md](BENCHMARKS.md)).

## ElideDB Desk

`elidedb desk` opens the console: store overview with honest size and
coverage tiles, an analytics view (storage, row density over time,
vector inventory, the fitted retrieval profile, write history), the
search console where every result plays, schema and Parquet-layout
browsers, a live architecture schematic drawn from what is actually
on disk, and index/maintenance operations. Set `DESK_READONLY=1` to
serve it publicly with mutations disabled.

## Deploy the demo

`deploy/` holds a verified Docker package that runs the full search
stack read-only on two CPU cores (about 5 s per warm query), plus a
one-command stager for a free Hugging Face Space. `site/` is the
landing page. See [deploy/README.md](deploy/README.md).

## The numbers that matter (measured, [BENCHMARKS.md](BENCHMARKS.md))

- 2 s window over a **14.16 M-row** audio table: touches **4 MB of
  286 MB (98.6 % elided)**, 165 ms.
- Multimodal 2 s window across 15 tables: 10/56 files touched,
  **98.9 % elided, 13.8 ms** (sensor-only).
- Event search over 1,122 episodes: **seconds warm on two CPU
  cores**, eight model channels fused, no GPU in the query path.
- Retrieval precision and yield are tracked per commit on a frozen
  truth set; the ledger in BENCHMARKS.md is appended by the benchmark
  script, never by hand.

## Documentation

| doc | what it covers |
|---|---|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | step-by-step: install, create, add data, query, browse |
| [docs/API.md](docs/API.md) | every class, method, and CLI verb |
| [DESIGN.md](DESIGN.md) | architecture + which idea came from which system (Delta, Spark, C-Store, warehouses) |
| [deploy/README.md](deploy/README.md) | the cloud demo: container, costs, one-command staging |
| [notebooks/elidedb_demo.ipynb](notebooks/elidedb_demo.ipynb) | query styles, executed on real data |
| [BENCHMARKS.md](BENCHMARKS.md) | measured numbers and the per-commit retrieval ledger |

## Repository layout

```
python/elidedb/     THE ENGINE. Everything that runs: store, log, video,
                    planner, retrieval, cli, desk. Python on Parquet.
scripts/            entry points only — ingest, benchmark, fit. A file with
                    a main() is a program; anything imported lives in the
                    package above. Never both.
tests/              the live test suite (pytest)
artifacts/          fitted state: teacher scores, thresholds, verb partitions
                    — see artifacts/README.md
eval/               truth set. EVAL ONLY: nothing is ever fitted on it.
deploy/             the Hugging Face Space: Dockerfile, store builder, stager
site/               landing page (static, single file)
desk/               macOS app bundle (thin launcher for elidedb.desk)
notebooks/          executed demo notebook
FDNN_BrainModel/    reference MLP carrying the FDNN architecture, kept as the
                    canonical statement of the three rules the students obey
rust/               a vertical slice against the same on-disk format — store
                    open, tx log, parquet scan, zone-map prune, counted
                    reads. NOT WIRED: no pyo3, no ctypes, nothing in Python
                    imports it. It reads what Python writes; it does not
                    serve. See deprecated/README.md for the language history.
deprecated/         the C++20 era and its Python sidecar. Off every code
                    path, kept as provenance — see deprecated/README.md
```

Raw data (`data/`), generated databases (`lake/`), model weights
(`models/`), and build output are git-ignored: the repo carries code,
docs, and fitted state only.
