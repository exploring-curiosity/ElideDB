# Getting started with ElideDB

This guide takes you from nothing to a searchable multimodal database with
**your own data**. Every step works with the CLI or Python — use whichever
you prefer; they call the same engine.

## 1 · Install

```bash
git clone <this repo> && cd <repo>
pip install -e ".[ml]"      # [ml] = semantic search (Apple Silicon, local models)
brew install ffmpeg          # video indexing + clip playback
```

Check it worked:

```bash
elidedb --help
```

## 2 · Create a database

A database is just a directory. Put them wherever you like; `lake/` is the
convention this repo uses (and where `elidedb desk` looks by default).

```bash
elidedb create lake/mydb --name "field study 2026"
```

## 3 · Add timestamped rows (any CSV, Parquet, or DataFrame)

The **one rule** in ElideDB: every table has a timestamp column. Everything
else about your schema is yours. Timestamps can be:

- ISO-8601 strings (`2026-07-18 10:00:00.125`)
- epoch numbers in seconds, milliseconds, microseconds, or nanoseconds —
  the unit is **auto-detected** by magnitude
- pandas datetime columns (Python path)

```bash
# --ts-col names YOUR timestamp column; everything else is stored as-is
elidedb add lake/mydb gps      track.csv     --ts-col utc_time
elidedb add lake/mydb readings sensors.parquet --ts-col t
```

```python
from elidedb import Store
db = Store.open("lake/mydb")
db.ingest_rows("heartrate", df, ts_column="measured_at")
```

Appending more rows later is the same command — each append is a new
**version** of the table (see time travel below). If the unit auto-detect
ever guesses wrong for your data, pass `--ts-unit s|ms|us|ns` explicitly.

## 4 · Add video

By default the video file is **copied into the store** (`media/`) and
indexed there, so the store directory is the complete, portable database.
Pass `copy=False` / index in place if you'd rather reference the original
file (then don't move or delete it). An existing store becomes standalone
with `elidedb adopt <store>`.

```bash
# use the container's own timestamps:
elidedb video lake/mydb walk.mp4 --stream phone

# or supply exact per-frame times (text file, one ns value per line):
elidedb video lake/mydb cam0.avi --stream cam0 --timestamps cam0_times.txt
```

Add as many videos as you like; each `--stream` name identifies a camera in
query results. Best supported today: MJPEG-family codecs (every frame
independent). H.264/H.265 index fine but Python-side decode of arbitrary
windows is on the roadmap.

## 5 · Look at what you have

```bash
elidedb ls lake/mydb
```

```
my project  (lake/mydb)
  frames    frame_index      12,004 rows   0.3 MB  v2  [+0.0 .. +680.2]s
  gps       timeseries        8,113 rows   0.7 MB  v1  [+0.0 .. +681.0]s
  readings  timeseries      512,220 rows  11.2 MB  v1  [+2.1 .. +679.8]s
```

Times display as `+seconds` from the database's earliest timestamp, and every
command accepts the same `+N` form.

## 6 · Query

**Time windows** — the core read. Everything overlapping [t0, t1], plus the
honest byte accounting:

```bash
elidedb window lake/mydb +120 +125 --dump /tmp/frames   # frames as JPEGs too
```

```python
w, stats = db.window(t0, t1)          # ns timestamps
print(stats)                          # files touched, bytes, % elided
frames = w["frames"].decode(stream="phone", width=640)   # [(ts, ndarray), ...]
gps    = w["gps"].to_pandas()
```

**SQL** — DuckDB runs directly on the database's Parquet files. Table names
in SQL are your table names:

```bash
elidedb sql lake/mydb "SELECT date_trunc('minute', to_timestamp(ts/1e9)) m,
                              avg(speed) FROM gps GROUP BY 1 ORDER BY 1"
```

**Aligned reads** — resample every numeric stream onto one timeline
(alignment is a query option, never baked into storage):

```python
al, _ = db.aligned(t0, t1, rate_hz=10, interp="linear")
al["timeline_ns"]; al["gps"]["speed"]; al["readings"]["temp"]
```

**Time travel** — every append is a version; old versions stay readable:

```python
db.table("readings").history()        # what changed, when
db.table("readings").state(3)         # the table as of version 3
```

## 7 · Semantic search

One command embeds every video stream (2-second windows by default) with a
local SigLIP model — nothing leaves your machine — and clusters the results:

```bash
elidedb embed lake/mydb                # first run downloads the model (~2 GB)
elidedb search lake/mydb "a dog running on grass"
```

Results are **dynamic segments**, not fixed chunks: consecutive matching
windows merge into one hit of the event's true duration (a 40 s event is one
40 s hit; a 2 s match stays 2 s), and how many hits you get depends on how
many distinct moments actually match — up to `-k`. Ranking is by visual
appearance; attributes single frames can't show (speed, motion, sound) are
weakly captured.

```python
hits, stats = db.search_text("a dog running on grass", k=8)
# each hit: {"stream", "t0", "t1", "score"} → feed t0/t1 into db.window()
sim, _ = db.search_clip("phone", t0, t1)   # "find moments like this one"
```

Re-run `elidedb embed` whenever you've added more video; it writes a new
version of the embeddings table.

## 8 · Browse it — ElideDB Desk

```bash
elidedb desk           # serves http://localhost:8787 and opens your browser
```

or double-click **`desk/ElideDB Desk.app`** on macOS. You get: every store
under `lake/`, table inventories and timelines, a semantic map (hover a
point → its frame decodes live; click → the clip **plays**, with the
sensor's audio track when the store has one), and a query console for
semantic / SQL / window queries.

## Troubleshooting

- **"a `ts` int64-ns column is required"** — name your timestamp column with
  `--ts-col` / `ts_column=`.
- **"timestamps must be non-decreasing"** — your file isn't time-sorted;
  `ingest_rows` sorts each batch automatically, but rows must not go
  backwards *across* appends to the same table. Append in chronological
  order, or load out-of-order history into a fresh table.
- **Search returns nothing** — run `elidedb embed` first; search only sees
  embedded windows.
- **Clip won't play** — check `ffmpeg` is installed and the source video
  file still exists at its original path.
- **Desk shows no stores** — it lists databases under `--root` (default
  `lake/`); pass `elidedb desk --root /path/to/your/dbs`.
