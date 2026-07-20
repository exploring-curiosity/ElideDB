# ElideDB API reference

Python package `elidedb` and the `elidedb` CLI. Everything below operates on
a **store** — a directory of Parquet tables under transaction logs.
Timestamps are always int64 nanoseconds since the Unix epoch (UTC) once
inside the database; ingest converts for you.

---

## Store

```python
from elidedb import Store
```

| method | description |
|---|---|
| `Store.create(path, name)` | make a new empty database directory |
| `Store.open(path)` | open an existing one |
| `db.name` / `db.dir` / `db.meta` | identity + `_store.json` contents |
| `db.tables()` | table names |
| `db.table(name)` | a `Table` handle (see below) |
| `db.describe()` | one dict per table: kind, rows, bytes, min/max ts, files, version, meta |
| `db.snapshot()` | pin every table's version in one call; pass as `version=` to any query for a consistent multi-table read |
| `db.vacuum(retain_versions=3, dry_run=False)` | garbage-collect parquet parts and managed media unreachable from the last N versions |

### Ingest

| method | description |
|---|---|
| `db.ingest_rows(table, data, ts_column="ts", ts_unit="auto", meta=None)` | append rows from a DataFrame, dict of arrays, pyarrow Table, or a CSV/Parquet **path**. `ts_column` may hold datetimes, ISO strings, or epoch numbers; `"auto"` infers s/ms/us/ns by magnitude. Returns the new version. |
| `db.ingest_video(table, video_path, timestamps_ns=None, stream=None, meta=None, copy=True)` | packet-scan a video into a `frame_index` table: one row per frame with its byte range. `copy=True` (default) copies the media into the store's `media/` dir — the store directory IS the complete database. `copy=False` references the file in place. |
| `db.adopt_media(table="frames")` | copy every externally-referenced media file into `media/` and rewrite the index (one replace-commit) — makes an existing store standalone |

`ingest_video(..., transcode="hevc"|"h264", gop_s=1.0, crf=26)` re-encodes the
managed copy ~10-25x smaller with a keyframe every `gop_s` seconds; decode
becomes GOP-granular (reads one GOP span per window).

### Secondary & vector indexes

| method | description |
|---|---|
| `t.create_index(column, order=256)` | immutable bulk-loaded **B+ tree** (BPT1) over any numeric column; rebuild after appends |
| `t.where(column, op, value, value2=None, columns=None)` | predicate pushdown on a non-time column (`==`,`>=`,`<=`,`between`) — with a B+ index reads only the row groups with hits; without one, honest full-scan fallback |
| `elidedb.ann.build_hnsw(db)` / `build_ivfpq(db)` | **HNSW** graph / **IVF-PQ** (product-quantized shortlist + exact rerank) over the embeddings table |
| `search_text/clip(..., method="auto"\|"exact"\|"hnsw"\|"ivfpq", t0=, t1=, streams=)` | tier auto-selected; **hybrid**: time-range & stream predicates pushed into candidate selection |

### Maintenance

| method | description |
|---|---|
| `t.compact(target_rows_per_file=8_000_000)` | OPTIMIZE: rewrite the active file set into few large ts-sorted files with current encodings — one atomic replace-commit |
| `t.delete_range(t0, t1)` | delete rows in a time range (scrub a run / PII / retention); untouched files stay, covered files drop, overlapping files rewrite — one atomic commit; earlier versions still see the data |
| `t.files(version=None)` | absolute paths of the snapshot's Parquet files — hand them to any engine (DuckDB, Spark, Polars, distributed dataframes) with zero export |
| `t.append_batches(iterable)` | many files, ONE atomic commit — bounded-memory bulk loads |

### Queries

| method | returns |
|---|---|
| `db.window(t0, t1, tables=None, columns=None, version=None)` | `(dict, QueryStats)` — every requested table filtered to the window; `frame_index` tables come back as `FrameSet` |
| `db.aligned(t0, t1, rate_hz, tables=None, interp="nearest"\|"linear", version=None)` | `(dict, QueryStats)` — `{"timeline_ns": array, table: {column: array}}`, all numeric columns resampled onto one timeline |
| `db.sql(query, version=None)` | pandas DataFrame — DuckDB over the store's own Parquet files; SQL table names = store table names |
| `db.search(text, k=10, method="auto", min_score=None, percentile=None, neg_weight=0.5, t0=, t1=, streams=)` | **compositional** text search. `text` supports `AND` (every term must match — min-pooled, so a clip with people but no laptop is rejected), `NOT`/`-term` (vector-subtracted exclusion). `min_score`/`percentile` = precision floor. Returns dynamic segments. |
| `db.search_text(text, k=10, nprobe=3, merge=True)` | `(hits, stats)` — hits are **dynamic segments** `{"stream", "t0", "t1", "score", "windows"}`: consecutive matching windows merged into one hit of its true duration (an event never comes back as duplicated sub-clips; a narrow match comes back tight). `merge=False` returns raw fixed windows. `stats` includes the per-query score threshold and segment counts. |
| `db.search_clip(stream, t0, t1, k=10, nprobe=3, merge=True)` | same, ranked by similarity to the probe range (probe excluded) |
| `db.embed_windows(frame_table="frames", window_s=2.0, frames_per_window=2, model=None, batch=16, stride_s=None, incremental=True)` | embed video streams in windows (local SigLIP via MLX). **Incremental by default**: only windows past what the embeddings table already covers get embedded — new footage costs new embedding, never a re-run of history |

Module-level helpers:

```python
import elidedb
elidedb.cluster(db, pca_dims=50, min_cluster_size=8)  # HDBSCAN cells + centroids
elidedb.embed_text("query text")                      # raw 1152-d unit vector
```

## Table

```python
t = db.table("readings")
```

| method | description |
|---|---|
| `t.scan(t0=None, t1=None, columns=None, version=None, stats=None)` | pyarrow Table, pruned two ways: whole files via the log's min/max stats, then row groups via Parquet footer stats; `columns` projects (`ts` always included) |
| `t.append(arrow_table, kind="timeseries", meta=None)` | low-level append (must contain `ts` int64; sorted for you) |
| `t.state(version=None)` | `TableState`: kind, schema, files, rows, bytes, min/max ts at that version |
| `t.history()` | the commit log: version, time, op, rows added, meta |

**Time travel:** pass `version=` to `scan` / `window` / `sql` — a version is
the fold of the log up to that commit and is immutable forever.

## FrameSet  (what `window()` returns for video tables)

| member | description |
|---|---|
| `len(fs)` / `fs.streams()` | frames in window / camera names present |
| `fs.decode(stream=None, stride=1, width=None, limit=None)` | `[(ts_ns, HxWx3 uint8 ndarray), ...]` — preads exactly the selected frames' byte ranges; `width` downscales, `stride` skips |
| `fs.last_bytes_read` | bytes actually read by the last decode |
| `fs.rows` | the underlying pyarrow rows (byte offsets, sizes, sources) |

## QueryStats

Every read reports itself: `files_touched/files_total`, `bytes_touched` vs
`corpus_bytes`, `elided_pct`, `rows_returned`, `wall_ms`. Bytes are counted
at row-group granularity from Parquet footer statistics — the same units the
reader actually materializes.

---

## CLI

Every verb maps 1:1 onto the API above. Times in CLI output and arguments
use `+SECONDS` relative to the store's earliest timestamp (raw ns also
accepted).

```
elidedb create <store> [--name NAME]
elidedb ls     <store>
elidedb add    <store> <table> <file.csv|.parquet> [--ts-col COL] [--ts-unit auto|s|ms|us|ns]
elidedb video  <store> <video> [--table frames] [--stream NAME] [--timestamps FILE]
elidedb embed  <store> [--window-s 2.0] [--frames-per-window 2] [--no-cluster]
elidedb search <store> "text" [-k 8]
elidedb adopt  <store>                 (pull referenced media into the store)
elidedb optimize <store> [--table T]   (compact into fewer, delta-encoded files)
elidedb vacuum   <store> [--retain N] [--dry-run]
elidedb index    <store> [--table T --column C | --ann hnsw|ivfpq]
elidedb sql    <store> "SELECT ..."
elidedb window <store> <t0> <t1> [--dump DIR] [--width 640]
elidedb desk   [--root lake] [--port 8787] [--no-open]
```

## Desk HTTP API (what the UI uses; yours to script against)

| endpoint | returns |
|---|---|
| `GET /api/stores` | every store under the root, with table summaries |
| `GET /api/map?store=` | 2-D layout + cluster per embedded window (UMAP, cached) |
| `GET /api/geo?store=` | lat/lon trace if any table has latitude+longitude |
| `GET /api/thumb?store=&stream=&t=&w=` | JPEG — one frame near `t`, decoded on demand |
| `GET /api/storage?store=&table=` | the table's commit log + every Parquet file's row-group layout (rows/bytes/ts range per group) |
| `GET /api/clip?store=&stream=&t0=&t1=&w=` | MP4 — the window as H.264 (+AAC mic track when available), Range-request capable |
| `POST /api/query` | `{store, type: "text"\|"clip"\|"sql"\|"window", ...}` → results |

## On-disk contract (for other tools)

A table directory is self-describing: fold `_log/*.json` in filename order —
each commit lists `add`/`remove` files with `rows/bytes/min_ts/max_ts` — and
read the surviving Parquet files with any engine. Nothing else is required
to consume an ElideDB store. Full rationale: [DESIGN.md](../DESIGN.md).
