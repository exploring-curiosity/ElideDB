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

### Ingest

| method | description |
|---|---|
| `db.ingest_rows(table, data, ts_column="ts", ts_unit="auto", meta=None)` | append rows from a DataFrame, dict of arrays, pyarrow Table, or a CSV/Parquet **path**. `ts_column` may hold datetimes, ISO strings, or epoch numbers; `"auto"` infers s/ms/us/ns by magnitude. Returns the new version. |
| `db.ingest_video(table, video_path, timestamps_ns=None, stream=None, meta=None)` | packet-scan a video (ffprobe) into a `frame_index` table: one row per frame with its byte range. `timestamps_ns` (list, one per frame) overrides container time. The media file is referenced, never copied. |

### Queries

| method | returns |
|---|---|
| `db.window(t0, t1, tables=None, columns=None, version=None)` | `(dict, QueryStats)` — every requested table filtered to the window; `frame_index` tables come back as `FrameSet` |
| `db.aligned(t0, t1, rate_hz, tables=None, interp="nearest"\|"linear", version=None)` | `(dict, QueryStats)` — `{"timeline_ns": array, table: {column: array}}`, all numeric columns resampled onto one timeline |
| `db.sql(query, version=None)` | pandas DataFrame — DuckDB over the store's own Parquet files; SQL table names = store table names |
| `db.search_text(text, k=10, nprobe=3)` | `(hits, stats)` — hits are `{"stream", "t0", "t1", "score"}` |
| `db.search_clip(stream, t0, t1, k=10, nprobe=3)` | same, ranked by similarity to the probe window (probe excluded) |
| `db.embed_windows(frame_table="frames", window_s=2.0, frames_per_window=2, model=None, batch=16)` | embed every video stream in tumbling windows (local SigLIP via MLX); commits an `embeddings` table version |

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
| `GET /api/clip?store=&stream=&t0=&t1=&w=` | MP4 — the window as H.264 (+AAC mic track when available), Range-request capable |
| `POST /api/query` | `{store, type: "text"\|"clip"\|"sql"\|"window", ...}` → results |

## On-disk contract (for other tools)

A table directory is self-describing: fold `_log/*.json` in filename order —
each commit lists `add`/`remove` files with `rows/bytes/min_ts/max_ts` — and
read the surviving Parquet files with any engine. Nothing else is required
to consume an ElideDB store. Full rationale: [DESIGN.md](../DESIGN.md).
