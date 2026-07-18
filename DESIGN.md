# DESIGN.md — ElideDB architecture

One sentence: ElideDB is **a timestamp-first multimodal lakehouse where every table is
plain Parquet under a transaction log, and heavy media is indexed by byte
range instead of copied.**

v1 built custom formats (SDX/SFI) to *prove the mechanisms*; v2 keeps every
mechanism and re-hosts it on open formats. Nothing here is a new datatype —
it is Parquet, JSON commit files, and your original media, arranged.

## Layout

```
lake/<store>/
  _store.json                      # name, format, display hints
  tables/<table>/
    _log/00000000000000000001.json # append-only commits (the table IS this log)
    part-<uuid>.parquet            # ts-sorted data files, zstd, 64k-row groups
```

Table kinds (all just Parquet schemas; `ts` int64-ns is the one law):

| kind | schema | holds |
|---|---|---|
| `timeseries` | ts + any columns | sensors, GPS/INS, audio, anything |
| `frame_index` | ts, source, byte_offset, packet_size, keyframe, w, h, codec, stream | video: byte ranges into UNTOUCHED media files |
| `embeddings` | ts, t1, stream, vector fixed_size_list<f32>[d], cluster | semantic windows |
| `centroids` | cluster, vector | the coarse stage of retrieval |

## Which idea came from which system

| mechanism here | lineage | why it matters |
|---|---|---|
| columnar files + min/max statistics | warehouses (Vertica, BigQuery), Parquet | prune before reading; projection reads only asked-for columns |
| commit log → snapshot isolation, time travel, atomic multi-file appends | **Delta Lake / Iceberg** | `O_EXCL` on the next log entry is the whole transaction protocol; version N is immutable forever |
| file-level zone maps in the log + row-group zone maps in footers | Delta file stats + Parquet row groups (ClickHouse skip indexes) | two pruning layers with one idea; a 2 s window over 14.16 M rows touches 4 MB of 286 MB (98.6 % elided, measured) |
| ts-sorted files, time-partitioned commits | Kafka log segments, TSDBs | sorted time is what makes both zone-map layers *tight* |
| predicate + projection pushdown | Spark / DataFusion | `Table.scan(t0, t1, columns=…)` pushes both into pyarrow |
| late materialization of media | C-Store | pixels decode LAST, from `pread(byte_offset, packet_size)`; the media file is never copied, re-encoded, or even opened until a query needs it |
| open format ⇒ multi-engine | the lakehouse thesis | `Store.sql()` is DuckDB pointed at the same files — SQL came for free, zero export |
| learned-cell IVF (centroid prune → exact rank) | Faiss IVF, with HDBSCAN cells | noise rows always scanned: pruning can cost recall nothing |
| ETL adapters at the edge, generic core | every warehouse's loader ecosystem | `ingest_rows` / `ingest_video` are the core; REIP and Oxford are ~100-line adapters |

## Deliberate simplifications (know these cold)

- **Single writer** per table (`O_EXCL` conflict = clean failure, not a merge).
  Multi-writer needs Delta-style optimistic retry — out of scope on purpose.
- **No compaction/OPTIMIZE yet**: many small commits ⇒ many small files; the
  fix (rewrite N files into one, one commit that adds+removes) is exactly the
  log's `remove` field — `embeddings.cluster()` already uses it.
- **No checkpoints**: log folds are O(commits); Delta checkpoints every 10th
  commit into Parquet. Trivial to add when logs grow long.
- **Byte accounting is row-group-granular** (footer stats), matching what the
  reader materializes — honest, but page-level I/O may differ under mmap.
- **MJPEG-family decode** in Python (packets are standalone JPEGs). H.264+
  needs a GOP-aware decode step; the index schema already stores
  keyframe/codec for exactly that upgrade.

## The lifecycle, end to end

```python
from elidedb import Store
db = Store.create("lake/mydata", "my dataset")
db.ingest_rows("weather", df, ts_column="time", ts_unit="s")   # any rows
db.ingest_video("frames", "cam.avi", timestamps_ns=ts)         # any video
db.embed_windows(window_s=2.0)                                 # SigLIP, local
from elidedb import cluster; cluster(db)                     # HDBSCAN cells
db.search_text("a person crossing the street")                 # → (stream, t0, t1)
db.window(t0, t1)                                              # → aligned everything
db.sql("SELECT …")                                             # DuckDB, same files
```

Browse it: `elidedb desk` or double-click `desk/ElideDB Desk.app`. Demo of every query style: `notebooks/elidedb_demo.ipynb`.
