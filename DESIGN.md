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

## Engineering state (each of these is implemented and measured)

- **Optimistic multi-writer** (Delta-style): `O_EXCL` on the next log entry
  is the lock; losing the race retries at the next version — three racing
  writers × six appends land as exactly 18 clean commits. Commits that
  remove files revalidate against fresh state and fail with `CommitConflict`
  rather than double-applying.
- **Checkpoints** every 10th commit (Delta's `_last_checkpoint` move): state
  reads fold from the latest checkpoint — 0.23 ms regardless of log length.
- **Compaction** (`compact()` / `elidedb optimize`): active file set →
  few large ts-sorted files, one atomic replace-commit.
- **Delta-encoded timestamps**: every file writes `ts` as
  `DELTA_BINARY_PACKED` (the Gorilla/TSDB observation) — ~3× on that column;
  strings keep dictionaries; everything rides zstd.
- **Compressed video tier**: `ingest_video(transcode="hevc", gop_s=1.0)`
  re-encodes managed media **24.9× smaller** (measured: 500 MB MJPEG →
  20.1 MB HEVC) with a keyframe every `gop_s` seconds; window decode preads
  exactly one GOP span (mid-GOP 2 s window: 57 frames from 1.66 MB).
  `gop_s` is the seekability-vs-compression dial, chosen per ingest.
- **Deletes**: `delete_range(t0, t1)` — scrub a bad run / PII / retention in
  one atomic commit (drop covered files, rewrite overlapping ones); earlier
  versions remain readable as the audit trail.
- **Incremental embedding**: `embed_windows()` only embeds windows past what
  the embeddings table already covers — a new day of footage costs a new day
  of embedding, never a re-run of history.
- **Parallel decode**: packet reads are sequential, JPEG decode fans out on
  threads (cv2 releases the GIL) — 49×4K frames 618 ms → 94 ms (6.6×).
- **Daft-native**: `table.to_daft()` hands any snapshot to Daft as a
  DataFrame over the store's own Parquet — distributed scans and multimodal
  UDFs with zero export.

## Positioning: what the target user gives up (and doesn't)

For robotics teams and research groups working with large multimodal
captures, the classic "vs a traditional database" objections resolve like
this:

| objection | resolution |
|---|---|
| transactions / concurrent writers | per-table optimistic commits + snapshot isolation; capture streams are naturally per-table appends, so append-vs-append never conflicts |
| updates & deletes | append-only by design (immutable logs are the *point* for training provenance) **plus** `delete_range` for scrubs and retention |
| point-lookup latency | checkpointed state (sub-ms) + footer caching; the workload is windows and scans, not OLTP point reads — OLTP stays out of lane deliberately |
| secondary predicates | DuckDB over the same files brings a full optimizer; Parquet column stats prune non-time predicates when data is locally sorted |
| security / backup / replication | delegated to the object store (IAM, bucket versioning, cross-region replication) — the enterprise norm for every lakehouse, not a gap: the store is plain files, so S3/ADLS/GCS machinery applies unmodified |
| ecosystem lock-in | none: any Parquet reader consumes the store; Daft/DuckDB demonstrated in-repo |

Remaining honest roadmap (not yet built): fsspec/object-store URIs with
range-request coalescing, ANN indexing past ~10⁶ windows (IVF-PQ/HNSW), and
motion-aware video encoders for attributes single frames can't rank.

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
