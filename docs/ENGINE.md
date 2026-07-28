# ElideDB Engine v3 — Rust core, store independence, compression first

Date: 2026-07-28. Status: design approved by the founding directive of this date:
raw data fully separate; the store functions standalone; store size must not
exceed raw size and should be far below it, without loss of quality; the core
engine moves to Rust (Python remains the ML sidecar only); OLAP orientation
with Parquet as the baseline to beat.

Every claim in this document that has a number was measured on this machine,
on this repo's real stores, on 2026-07-28. Nothing here is quoted from a
vendor benchmark.

---

## 1. The measured baseline (what the current store actually is)

Byte audit, current stores vs their raw sources:

| store | raw source | media in store | tables | dominant cost |
|---|---|---|---|---|
| bridge-full | 84 G | 18 G (h264 renditions) | 9.3 G | frame_vectors 7.2 G fp32 |
| lab | 26 G | 24 G (**raw 4K MJPEG copied verbatim**) | 1.6 G | media, then audio tables |
| bridge4h | (subset) | 524 M | 2.0 G | **vectors are 4x the media** |
| bench | (subset) | **0 — symlinks into bridge4h** | 787 M | vectors are ~99% of the store |

Compression probes on the real data:

- zstd over fp32 vectors (current writer): **1.08x**. Float mantissas are
  entropy; a general-purpose codec cannot help. `ts` compresses 17.9x in the
  same file — the format is fine, the encoding of the vector column is wrong.
- fp16: 1.97x, cosine top-10 overlap vs fp32 = **0.9990** (200 queries,
  39,026 vectors, 1152-d, leave-one-out).
- fp16 + Parquet BYTE_STREAM_SPLIT + zstd: **2.39x**, same quality.
- 1-bit sign codes (crude binary quantization, no rotation) + Hamming
  shortlist(100) + fp32 rerank: recall@10 = **0.9805** at **32x** smaller
  scan footprint. Proper RaBitQ (random rotation + per-vector correction,
  SIGMOD'24; extended multi-bit form SIGMOD'25) does strictly better; our
  unrotated probe is the floor.
- lab media, one 500 MB 4K MJPEG segment: x264 veryfast CRF18 g30 → 1.75x
  (sensor noise dominates an intra-only source); **x265 CRF21 g60 → 5.4x**
  (500 MB → 93 MB). Projected lab media: 24 G → ~4.5 G.

Conclusion of the audit: the store today frequently *adds* bytes on top of
raw instead of subtracting them, and the two offenders are fp32 vectors and
verbatim media copies. Both have measured fixes.

## 2. Loopholes in the current engine (pure database systems view)

Storage:

1. **The flagship store is not standalone.** lake/bench's media are symlinks
   into bridge4h. Deleting bridge4h breaks bench. Violates the founding
   independence requirement.
2. **No media rendition policy.** lab copied raw MJPEG verbatim (24 G);
   bridge got h264. Whatever ingest happened to do is what the store is.
3. **Vector encoding is wrong for the data.** fp32 + zstd = 1.08x. No fp16
   tier, no BYTE_STREAM_SPLIT, no quantized scan tier.
4. **No compaction or vacuum discipline in practice**; small files accumulate
   per commit. (Compaction code exists; nothing schedules or enforces it.)
5. Sidecar caches (`_cache` npy files, UMAP artifacts) sit inside table dirs
   uninventoried — invisible to the log, unaccounted in any byte ledger.

Query:

6. **The founding metric is not enforced end to end.** `Table.scan/where`
   count bytes into QueryStats, but the semantic path builds whole-table
   `_cache` matrices on first touch (the exact "cache whole files in memory"
   the v1 rules forbade — and the direct cause of the 10-minute cloud cold
   start), and `Store.sql()` hands the files to DuckDB, bypassing counting
   entirely. No per-query bytes-read number exists for the queries the
   product actually serves.
7. **No planner.** Each query style is a hand-rolled path; predicates,
   time ranges, and vector stages don't compose. The v1 C++ B+ tree exists
   but the v2 lake's `where()` is the only pushdown, and nothing routes
   between index kinds by cost.
8. **Vector search is exact scan over in-RAM fp32.** Fine at 39k vectors;
   it is the design that made 2 vCPUs unusable, and it has no tiering
   (codes → refine) to trade bytes for recall explicitly.
9. **Python per-query overhead and the GIL.** The fusion math is numpy over
   cached matrices; every query pays Python dispatch, and nothing scans in
   parallel. CPU serving measured "pathetically slow" (user, 2026-07-28).

Robustness:

10. No checksums beyond Parquet's own page CRCs; log entries carry no
    integrity hash; no fsck-style verifier for a store.
11. No schema evolution tests; `schema` travels in the log but nothing
    exercises reading old files under a new schema.
12. Media files are outside the transaction log entirely (only the frames
    table references them); a store cannot prove its media inventory.

## 3. What we keep (deliberately)

- **Parquet as the table substrate, readable by anything.** DuckDB/pandas/
  Spark can open an ElideDB store with zero ElideDB code. For a product that
  sells against lock-in, ecosystem readability is worth more than the last
  20% of scan speed. The bytes win comes from *encodings inside Parquet*
  (fp16, BYTE_STREAM_SPLIT, delta, dictionary) plus *sidecar artifacts for
  what Parquet cannot express* — not from a bespoke container.
- **The JSON transaction log exactly as specified** (O_EXCL commits,
  checkpoints every 10, file-level zone maps, schema-on-log). It is
  language-neutral and the Rust engine reads it unchanged.
- **Raw data immutability.** The store never mutates a source file; it holds
  renditions and derived artifacts, and must function with the source gone.

## 4. Store format v3 (language-neutral; additive to v2)

1. **Vector tables become two-tier.**
   - *Refine tier:* the Parquet vector column moves to fp16 with
     BYTE_STREAM_SPLIT + zstd (2.39x, overlap 0.9990). Schema records
     `encoding: fp16-bss` in table meta.
   - *Scan tier:* a sidecar `<table>/_codes.v<N>.bin` — RaBitQ-style binary
     codes (rotation matrix seed + 1 bit/dim + per-vector correction terms),
     mmap-able, byte layout documented in docs/FORMAT.md. Versioned to the
     data version exactly like the v1 B+ tree artifacts (derived; never
     bumps the log).
   - Query contract: Hamming scan over codes → shortlist → fp16 rerank →
     (optional) exact fp32 only if a table keeps an fp32 archive tier.
     Recall gate ≥ 0.98@10 measured per store before a tier is trusted.
2. **Media renditions with a policy.** `media/` holds store-owned renditions;
   default policy `hevc-crf21-g60` (5.4x on lab; g60 keeps ≤2 s of decode per
   random seek at 30 fps). The `frames` table stays the byte-range index over
   renditions. A `_media.json` inventory (path, bytes, codec, source hash)
   makes media auditable; symlinks are forbidden in a standalone store —
   `elide fsck` fails them.
3. **Sensor/timeseries tables:** already fine (zstd, ts delta ~18x). Audio
   moves to per-chunk FLAC-in-Parquet-binary *only if* measured better than
   I16+zstd; lossless is non-negotiable for sensor data.
4. **Byte ledger per store.** `_ledger.json`: raw_source_bytes,
   store_bytes (tables/media/artifacts split), ratio. The headline number
   next to elision %: **store:raw**. Target < 0.5 for video-dominant stores.

## 5. The Rust engine

Workspace `rust/` (edition 2021), crates:

- `elide-store` — store open, log fold (checkpoint + tail), snapshot,
  describe, fsck, the byte ledger, and `CountingReader`: **every** read path
  in the engine goes through it; a byte that is read is a byte that is
  counted. No exceptions — this restores the founding metric as a type-level
  guarantee (the scan APIs only accept counted readers).
- `elide-scan` — Parquet scan: log-level file pruning (zone maps) →
  row-group pruning (footer stats) → page-index pruning where present →
  projected, predicate-pushed reads via the `parquet` crate. Parallel over
  files/row groups with rayon.
- `elide-vec` — the two-tier vector path: code building (rotation, sign,
  correction), mmap'd Hamming scan (NEON `vcnt`/popcount via portable SIMD),
  fp16 rerank, recall self-test against exact scan.
- `elide-media` — rendition transcode driver + frame-index maintenance
  (invokes ffmpeg for encode at ingest/migrate time only; *decode* on the
  query path stays byte-range disciplined and lands in a later milestone).
- `elide-query` — DataFusion integration: a `TableProvider` per store table
  that applies our pruning and counted reads, giving full SQL (joins,
  aggregates, window functions — "any query, not just contextual text")
  over every store, with per-query bytes accounting in EXPLAIN output.
- `elide-cli` — `elide stats|scan|vsearch|compress|fsck|sql|serve`.
- `elide-py` (pyo3, later) — so Desk and the ML sidecar call the Rust engine
  in-process; the sidecar keeps writing embeddings via files as today.

Boundary: Python **writes** model outputs (embeddings, probes, fitted
weights) as files; Rust **owns** every byte on the query path. The fusion
math (weights, filter, NMS, confidence cut from `_set_weights.json`) is
pure arithmetic over per-episode scores and moves to Rust in R5; model
inference never does.

## 6. Milestones (each ends runnable, with numbers)

- **R0 — parity open.** `elide stats <store>` folds the log (checkpoints
  included) and matches Python `describe()` on every lake store, to the row
  and byte. Acceptance: byte-identical inventory on 7 stores.
- **R1 — counted pruned scan.** `elide scan` with t0/t1/columns; prints
  rows, bytes_read, bytes_total, elided %. Property test: pruned scan ≡ full
  scan on randomized windows. Benchmark vs Python `Table.scan` on
  lab audio (14M rows) and bench frames: p50/p99, cold and warm.
- **R2 — vector tiers.** `elide vsearch` (query vector from .npy): exact
  fp32 baseline, then codes+rerank. Acceptance: recall@10 ≥ 0.98 vs exact on
  bench frame_vectors + pe_vectors; report latency and bytes touched vs
  Python numpy scan.
- **R3 — `elide compress` store migration.** Rewrites vector tables to
  fp16-BSS (log-committed rewrite, old files removed via the log, vacuum
  reclaims), builds code sidecars, transcodes media per policy, materializes
  symlinks; writes `_ledger.json`. Acceptance: lab store:raw < 0.35 with
  frozen-benchmark parity on bench (ret/true/sup unchanged within the ±1
  noise band already tolerated by the ledger) and recall gates passing.
- **R4 — SQL surface.** DataFusion provider; `elide sql` on any store;
  EXPLAIN carries bytes-read. DuckDB remains a supported *external* reader.
- **R5 — serve.** `elide serve` replaces the Python query path under Desk:
  loads fitted weights, computes fusion in Rust, calls the sidecar only for
  text embedding. Target: warm query well under the current 4.8 s CPU
  number; cold start seconds, not minutes (no whole-table cache builds —
  codes are mmap'd, first query pays page faults only for what it touches).

## 7. Explicitly rejected (for now, with reasons)

- **A bespoke container format** (our own SDX v3): Lance 2.1 and Vortex
  prove the ceiling is higher than Parquet, but both also prove how much
  engineering a credible format costs. We take their *ideas* where Parquet
  can express them and put the rest in documented sidecars. If measurement
  later shows Parquet's structural floor is the bottleneck (random access
  syscall amplification, string columns), the leaf format sits behind one
  trait in `elide-scan` and Vortex (a Rust crate) is the candidate — by
  benchmark, not by fashion.
- **Lossy media "enhancement"** (denoising before encode): changes pixels;
  quality gates can't prove it harmless. CRF renditions only.
- **Distributed anything.** Single node until a paying reason exists. The
  cloud translation stays: bytes elided = range GETs never issued.
