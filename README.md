# ElideDB

**A Parquet-native, timestamp-first database for multimodal data.** Put any
timestamped data in — sensor rows, video, GPS, audio, logs — and get back
time-window reads, SQL, and semantic search, while the engine reads as few
bytes as physically possible. *The best read is the read elided.*

- **Everything is Parquet.** Every table is plain Parquet files under a
  Delta-Lake-style transaction log. No custom formats; DuckDB, Spark, pandas —
  anything that reads Parquet reads your database.
- **Timestamps are the law.** Every table has a `ts` column (int64
  nanoseconds). That one rule is what makes cross-modal queries, alignment,
  and pruning work.
- **Media is indexed, never copied.** Video files stay exactly where they
  are; ElideDB stores byte ranges into them and decodes only the frames a
  query touches.
- **Search by meaning.** Local SigLIP embeddings (no cloud) turn video into
  searchable windows: `"a person crossing the street"` → playable clips.

## Install

```bash
pip install -e ".[ml]"     # from the repo root; [ml] adds semantic search
brew install ffmpeg        # clip playback + video indexing
```

## 60 seconds to your first database

```bash
elidedb create lake/mydb --name "my project"

# any timestamped rows: CSV/Parquet, ISO dates or epoch s/ms/us/ns (auto-detected)
elidedb add lake/mydb readings sensor_log.csv --ts-col time

# any video (the file is indexed in place, not copied)
elidedb video lake/mydb dashcam.mp4 --stream front

elidedb ls lake/mydb                                   # what's inside
elidedb sql lake/mydb "SELECT count(*) FROM readings"  # SQL via DuckDB
elidedb embed lake/mydb                                # local ML, one command
elidedb search lake/mydb "a cyclist passing a bus"     # → time windows
elidedb desk                                           # browse it
```

The same five verbs in Python:

```python
from elidedb import Store
db = Store.create("lake/mydb", "my project")
db.ingest_rows("readings", df, ts_column="time")     # DataFrame/CSV/Parquet
db.ingest_video("frames", "dashcam.mp4", stream="front")
db.embed_windows(); import elidedb; elidedb.cluster(db)
hits, _ = db.search_text("a cyclist passing a bus")  # → (stream, t0, t1)
window, stats = db.window(hits[0]["t0"], hits[0]["t1"])
print(stats)                                          # bytes touched vs corpus
```

## Contextual search

Semantic search finds *what is in frame*. Contextual search finds *what is
happening* — and it does so without a model in the query path, because the
expensive part runs once at ingest:

```python
db.index_context()                       # VLM captions -> caption index -> tower
db.search_context("a robot putting a pot in the sink")
db.search_context("crossing red car", weights={"lexical": 2.0})
db.search_context("...", rerank=True, explain_top=5)   # why each hit is here
```

Three rankers vote and are fused by **reciprocal rank** — appearance (SigLIP),
context (the clip's caption), and lexical (exact terms). Fusing by rank rather
than by score means a hit has to convince more than one ranker, which is why
`crossing red car` no longer returns everything with a person crossing.
Full design, ablations, and the parts that do not work yet:
[docs/CONTEXT.md](docs/CONTEXT.md).

## Documentation

| doc | what it covers |
|---|---|
| [docs/CONTEXT.md](docs/CONTEXT.md) | contextual retrieval: the caption index, RRF, the temporal tower, and its limits |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | step-by-step: install → create → add your data → query → browse |
| [docs/API.md](docs/API.md) | every class, method, and CLI verb |
| [DESIGN.md](DESIGN.md) | architecture + which idea came from which system (Delta, Spark, C-Store, warehouses) |
| [notebooks/elidedb_demo.ipynb](notebooks/elidedb_demo.ipynb) | every query style, executed on real data with outputs baked in |
| [BENCHMARKS.md](BENCHMARKS.md) | measured numbers on 28.5 GB of real captures |

## ElideDB Desk

`elidedb desk` (or double-click `desk/ElideDB Desk.app` on macOS) opens the
database browser: every store's tables and timelines, a semantic map where
hovering any point decodes its frame live, click-to-play clips (with the
sensor's audio track when the store has one), and a query console for
SQL / semantic / window queries.

## The numbers that matter (measured, [BENCHMARKS.md](BENCHMARKS.md))

- 2 s window over a **14.16 M-row** audio table: touches **4 MB of 286 MB
  (98.6 % elided)**, 165 ms.
- Multimodal 2 s window across 15 tables: 10/56 files touched, **98.9 %
  elided, 13.8 ms** (sensor-only).
- Semantic search: ranking is **microseconds** at thousands of windows;
  end-to-end text query ≈ 2.8 s (model load dominates, then it stays warm).
- Store is *smaller* than the raw input in both real corpora, while adding
  random access, SQL, and search.

## Repository layout

```
python/elidedb/     the database (store, log, video, embeddings, cli, desk)
notebooks/          executed demo notebook
desk/               macOS app bundle (thin launcher for elidedb.desk)
src/, tests/        v1: the original C++20 engine with hand-built formats
                    (SDX/SFI) — the mechanisms ElideDB now hosts on Parquet
scripts/            dataset ETL adapters (REIP, Oxford RobotCar) + tooling
```

Raw data (`data/`), generated databases (`lake/`, `store*/`), and build
output are git-ignored — the repo carries code and docs only.
