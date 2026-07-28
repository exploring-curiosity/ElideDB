# What the other systems do, and what ElideDB took from them

Date: 2026-07-28. Every ElideDB number here was measured on `lake/bench`
(1,122 robot episodes, 0.819 GB of source video) on this machine. Numbers
attributed to other systems are their own published claims and are
labelled as such — they were not reproduced here, and are not compared
head to head on the same hardware.

---

## 1. The five systems, and what each is actually for

| system | what it really is | the mechanism worth stealing |
|---|---|---|
| **Lance / LanceDB** | a columnar *file format* (Lance v2.x) plus an embedded multimodal database over it | structural encodings tuned so random access costs ~2 IOPS regardless of nesting; the format treats "fetch row N" as a first-class operation instead of an accident of scanning |
| **Daft** | a Rust dataframe/query engine (Swordfish local, Flotilla distributed) for multimodal data | a real optimizer: filter/projection/limit **pushdown into the scan**, plus the inverse rule — expensive multimodal work (decode, model inference) is deliberately kept OUT of the scan and run as late as correctness allows |
| **Apache Iceberg** | a table format: snapshots, manifest tree, hidden partitioning, schema evolution | metadata that answers "which files can possibly matter" before opening any data file |
| **Twelve Labs** | a hosted video-understanding API (Marengo embeddings, Pegasus VLM) | one **unified embedding space** across text/image/audio/video, chunked into ~6 s segments with `startSec`/`endSec`, so any modality can query any other and results are *moments*, not files |
| **Encord Index** | a data-curation platform | 40+ automatic **quality metrics** (duplicates, outliers, blur, brightness) as first-class filterable columns — curation as a query, not a manual pass |

## 2. Honest scorecard: ElideDB vs each mechanism

**Where we already matched or led (before this sprint)**

- *Iceberg's metadata tree* — we have it in simplified form: a JSON
  transaction log with per-file zone maps, snapshot isolation, time
  travel, checkpoints. A time-window query prunes whole files with
  **zero I/O**. This was never the gap.
- *LanceDB's vector index* — we have a two-tier path (1-bit EVC1 sign
  codes, mmap'd, then exact rerank): **recall@10 0.995 at shortlist 200,
  0.30 ms vs 1.75 ms exact, 27.5x fewer bytes touched**. LanceDB's
  RaBitQ is the better-engineered version of the same idea (rotation +
  correction terms); our codes are the unrotated floor and the upgrade
  path is recorded.
- *Lance's storage reduction* — we get ours from encodings inside
  Parquet: fp16 + BYTE_STREAM_SPLIT gave **2.19x on vector tables with
  cosine top-10 overlap 0.9990**, and dropping unaddressable frames plus
  episode-aligned GOPs gave **2.61x on media**. Store is now **52.4% of
  its raw source**.
- *Video, where we are genuinely ahead of all of them.* Lance stores
  video as blobs; Twelve Labs is an API you send files to; Daft decodes
  whole files in UDFs. We keep the compressed stream and a per-frame
  byte-range index, and **place the GOP boundary on the retrieval unit**
  — one IDR per episode — so reading an episode reads that episode:
  **196 KB per episode, down from 323 KB**. That is a storage-engine
  idea none of the four apply to video.
- *Counted bytes.* Every read in the Rust engine goes through a counting
  reader, so "bytes elided" is measured inside the read calls rather than
  estimated. No system above reports this to the user at all.

**Where we were genuinely behind (the real gaps)**

1. **Predicate pushdown existed only for `ts`.** Daft's headline
   capability — filter on any column, pushed into the scan — was simply
   missing. Any other filter meant reading the column and discarding it.
2. **No page-level pruning.** Our Parquet files were written *without a
   page index*, so even a perfect predicate could only prune to whole
   row groups. This is the layer that makes a columnar file behave like
   an index, and Lance's whole random-access argument lives here.
3. **No late materialization.** A window query on a 1152-d vector table
   read the entire row group — every vector — to return 48 rows.

**Where we chose not to follow**

- *Encord's quality metrics as a product surface.* Blur/brightness/
  duplicate scores are ordinary derived columns; now that arbitrary
  predicates are pushed down, any of them can be added as a column and
  filtered efficiently. Baking a fixed metric list into the engine would
  violate the no-hardwire rule, so the engine gains the *capability* and
  the metrics stay data.
- *Twelve Labs' single unified embedding space.* Their model puts every
  modality in one 1024-d space. We deliberately keep **nine separate
  channels with fitted per-query-type weights** because it was measured
  better here: a single space erases verbs (`cos("close the drawer",
  "open the drawer") = 0.951`), while our motion channel separates the
  same pair at AUC 0.98. Their *chunking* idea we already share —
  episodes are our segments, and results are moments with timestamps.
- *A distributed engine.* Daft's Flotilla matters at petabyte scale.
  Single node until there is a paying reason.

## 3. What this sprint built (bench-measured)

All three gaps closed in the Rust engine, in `elide-store`:

**Four pruning layers, cheapest first**

1. log zone maps (no I/O) → 2. row-group statistics → 3. **Parquet page
index** → 4. exact Arrow filtering. Layers 2 and 3 are new for arbitrary
columns; layer 3 is new entirely.

**Late materialization** (C-Store's idea; the reason Lance's random
access matters): predicates are evaluated against *their own columns
first* via an Arrow `RowFilter`, and the wide columns are read only for
surviving rows.

**A cost model that can decline it.** Late materialization re-reads the
predicate column, so it is not free. Measured on the narrow `frames`
table it made things **worse — elision went to −20.8%**, i.e. the engine
read 1.2x the table. The planner now requires that the protected columns
have more than one page per row group (otherwise nothing can be skipped)
and that the deciding columns are under a quarter of the projected bytes.
With that check the same queries sit at 0.00% instead of −20.8%.

### Measured result, one episode window

| table | before (row-group only) | after (full stack) | gain | elided |
|---|---|---|---|---|
| frame_vectors | 7,448,313 B | 911,535 B | **8.2x** | 98.69% |
| object_vectors | 8,131,168 B | 956,299 B | **8.5x** | 99.06% |
| pe_vectors | 7,006,688 B | 884,616 B | **7.9x** | 94.22% |
| sig2_vectors | 7,805,004 B | 879,492 B | **8.9x** | 94.86% |

### Measured result, arbitrary predicates (new capability)

| predicate | rows | elided |
|---|---|---|
| `stream = '…/file-132'` | 5,725 | 82.20% |
| `codec != 'h264'` | 0 | 97.98% |
| `packet_size > 10000` | 3,015 | 0.00% (planner declined the filter) |
| `keyframe = true` | 1,122 | 0.00% (same) |

Every row count was verified equal to the Python engine's answer,
including a two-predicate AND.

**The page index costs nothing.** Rewriting all bench tables to carry it:
218,676,442 → 218,526,050 bytes, **−0.07% overall** (vector tables
+0.00%; `frames` shrank 27% because the rewrite also restored dictionary
encoding). Retrieval quality after the rewrite is unchanged at
**0.38 / 0.36, 35/91**.

## 4. What is still missing, ranked

1. **`take(row_ids)`** — Lance's actual headline. We can now *prune* to
   the rows we want, but there is no O(1) "fetch these 500 row ids"
   path for shuffled training reads. This is the next real gap.
2. **Zone maps for non-`ts` columns in the transaction log** — Iceberg
   keeps per-column bounds in the manifest, so files are eliminated
   without opening footers. We open every surviving file's footer.
3. **A query planner worth the name** — today the CLI takes predicates;
   there is no plan, no join, no aggregate. DataFusion is the intended
   host (R4 in docs/ENGINE.md).
4. **RaBitQ proper** (rotation + correction) if the recall dial needs
   headroom at scale.
5. **Deletion vectors / merge-on-read** for row-level updates without
   rewriting files.
