# ElideDB — architecture

A content-aware, timecode-native store for video + sensor data. You give it
raw video; you ask in English; it returns the exact clips. The design goal is
one line: **every query should read the minimum possible bytes — the best read
is the read elided.**

This document covers the storage engine first (that is where the specificity
is asked for), then the two models, the five elements they produce, and how a
query walks the whole stack.

---

# PART I — THE DATABASE

## 1. Physical layout

A store is a directory. There is no server, no daemon, no custom byte format
outside of the media itself.

```
lake/elide_v1/
  _store.json                     format id, name, created
  _cache/                         disposable, rebuildable, never authoritative
  media/                          store-managed renditions (see §6)
  tables/
    frames/                       one directory per table
      _log/
        00000000000000000001.json    commit 1
        00000000000000000002.json    commit 2
        ...                          version N = fold of commits 1..N
      _meta.json
      part-67283a02cdd8.parquet     immutable data file
      part-cfca668ca561.parquet
    episodes/  events/  events_s/  answers2/
    pe_vectors/  sig2_vectors/  iv2_vectors/  motion_vectors/
    vjepa_vectors/  xclip_vectors/  object_vectors/  frame_vectors/
    action_probs/  embeddings/
```

`FORMAT = "elidedb/2"`. Everything is Parquet — sensor rows, the video frame
index, embeddings, centroids. Open format is a deliberate position: another
engine can read the store without going through us, and `Store.sql()` is
DuckDB pointed at these very files.

### The one schema law

**Every table must carry `ts` (int64 nanoseconds), sorted within a file.**
That is the only rule the store enforces on your schema. Time is the primary
axis; it is what makes cross-modal alignment a query parameter instead of a
storage commitment. `scan()` silently adds `ts` to any projection that omits
it, because it is both the sort key and the alignment axis.

### Write encoding

`write_parquet()` is the single writer for every file in the store.

| knob | value | why |
|---|---|---|
| `ROW_GROUP_ROWS` | 65,536 | the amortize-vs-overfetch dial |
| `ROW_GROUP_TARGET_BYTES` | 8 MB | row groups are sized by *bytes*, not rows, so a 1152-d vector table doesn't emit 64k-row groups nobody can skip inside |
| `ts` encoding | `DELTA_BINARY_PACKED` | timestamps are near-arithmetic; delta-packing them is close to free |
| vectors | fp16 | halves the largest tables at no measured retrieval cost |

Row-group size is the central tension in this design, and it is the same one
the Lance paper is about: large groups amortize metadata but overfetch on
random access; small groups make random access cheap but bloat the footer.
8 MB is the compromise, chosen per-table by row width.

## 2. The transaction log — table state is a fold, not a directory listing

A table's state is **not** "the files in the directory". It is the fold of an
append-only log of JSON commits under `_log/`. Each commit records:

```json
{ "op": "append" | "replace" | "compact" | "delete" | "adopt-media",
  "kind": "...", "schema": "...",
  "add":    [ {"path","rows","bytes","min_ts","max_ts"}, ... ],
  "remove": [ "part-....parquet", ... ],
  "meta":   { ... } }
```

That single move buys four properties with nothing but plain files:

- **Snapshot isolation / time travel.** Version N is immutable forever. A
  reader at version N never sees version N+1's files. `scan(version=N)`,
  `window(version=N)` — reproducibility is structural, not a convention.
- **Atomic multi-file commits.** A multi-gigabyte load lands as *one* log
  entry. The lock is `O_EXCL` on the next log filename: single writer, many
  readers, no coordination service.
- **File-level zone maps.** Every `FileEntry` carries `min_ts`/`max_ts`, so a
  time-window query prunes whole files **from the log alone**, before opening
  a single Parquet footer.
- **Schema-on-log.** The schema travels with the commit, so evolution is an
  append, never a rewrite.

### Write operations

| op | meaning |
|---|---|
| `append` | add rows; the common path, one file per batch, one commit |
| `replace` | **rebuild in place** — new rows replace the current file set in one commit |
| `compact` | rewrite the active set into few large sorted files (the many-small-files tax) |
| `delete_range` | drop a time range; rewrites only the straddling files |
| `adopt-media` | take ownership of external media, rewriting `source` pointers |

`replace` exists because of a real bug: derived tables (`events`, `answers2`,
`events_s`) are *recomputed* from the frames, and recomputing one with
`append` silently doubled it — the events table went 8,263 → 16,526 rows with
`agent` at exactly 2× the demo count, leaving every demo holding both the old
and the new typing of the same transition. Nothing errored. A derived table
needs an operation that means "this is now the content", and the old files
must leave in the same commit that the new ones arrive.

## 3. Predicate pushdown — two layers, same idea

`Table.scan(t0, t1, columns, version, stats)` prunes twice.

**Layer 1 — log-level file pruning.** Zone maps live in the commit entries,
so this costs zero I/O against the data files:

```python
files = [f for f in st.files
         if not (t0 is not None and f.max_ts < t0)
         and not (t1 is not None and f.min_ts > t1)]
```

**Layer 2 — Parquet row-group statistics + projection pushdown.** Surviving
files are read with the predicate handed to the Parquet reader, so row groups
whose `ts` statistics fall outside the window are never decompressed, and
column chunks outside the projection are never touched:

```python
filt = (pc.field("ts") >= t0) & (pc.field("ts") <= t1)
pq.read_table(path, columns=columns, filters=filt)
```

This is warehouse skip-indexing (Vertica/BigQuery) plus Spark-style predicate
pushdown, with Delta/Iceberg's log as the outer layer. Two layers of the same
idea at two granularities: files, then row groups.

**Non-time predicates.** `scan_where()` handles pushdown on a non-`ts` column
by descending an optional B+ index to row ids, mapping rows → row groups via
the Parquet footer's group sizes, and reading only those groups
(`pf.read_row_groups(groups, columns=want_cols)`). Time is privileged; other
columns are supported.

## 4. Byte accounting — the elision metric, honestly

Pruning claims are worthless unless the bytes are counted. `QueryStats`
travels through the scan and records:

```
files_total, files_touched        how much of the table survived layer 1
corpus_bytes, bytes_touched       the elision numerator and denominator
rows_returned
```

`bytes_touched` is charged as **footer + only the row groups that overlap the
window, and within them only the projected column chunks** — which is exactly
what the reader materializes. Not an estimate; the same arithmetic the reader
performs. Elision % = `(corpus_bytes − bytes_touched) / corpus_bytes`.

## 5. Late materialization — pixels are produced last

This is C-Store's idea applied to video, and it is the reason the store is
small.

The `frames` table is a **frame index**, not frames:

```
ts | byte_offset | packet_size | keyframe | width | height | codec | source | stream
```

One row per frame, pointing *into* a media file. A window query walks:

1. log zone maps → candidate files
2. Parquet row-group stats → candidate rows
3. `FrameSet` — a **lazy handle** over those rows; still no pixels
4. `.decode()` — only now, and only the byte ranges for the GOPs that overlap
   the window, are read and decoded

Every stage before step 4 is metadata arithmetic. The planner prunes on time,
on structure, and on semantics, and the expensive modality is produced **only
for the survivors**. A semantic query over 1,122 demos decodes pixels for the
handful it returns, not for the corpus.

The `source` column decides where bytes come from:
- `"@media/..."` — store-managed rendition (standalone store)
- anything else — **external reference-in-place**: the raw file is never
  copied into the store

Raw data is immutable and is not ingested-by-copy. The store is an index over
your media, not a second copy of it.

## 6. Metadata and meta files — what is authoritative

| file | authority | contents |
|---|---|---|
| `_store.json` | **authoritative** | format id, store name, creation |
| `tables/*/_log/*.json` | **authoritative** | the table. Version, op, schema, file list with zone maps |
| `tables/*/_meta.json` | descriptive | table kind, hints |
| `_set_weights.json` | fitted artifact | selection weights (with `.prev.json` rotation and `.loqo.json` held-out record) |
| `_channel_weights.json` | fitted artifact | per-channel fusion weights |
| `_vocab.json` | **cache** | corpus-attested vocabulary, grown lazily per new word |
| `_split_map.json` | provenance | demo → source segment mapping |
| `_cache/` | **disposable** | ITM vision tokens (3.0 GB); deleting costs recompute time, never correctness |

Two rules that are load-bearing here:

1. **No dataset metadata is ever ingested.** Task strings, instruction fields,
   class names and per-dataset vocabularies are forbidden at index and train
   time. The truthset lives *outside* the store (`eval/truthsets/*.parquet`)
   and is evaluation-only. Everything the system knows, it derived from
   pixels.
2. **A snapshot must pin its environment.** `models/teacher_v*.json` records
   table versions, artifact digests, the stage list, **and the env flags that
   decide which stages run**, plus a `reproduce` command. This was learned the
   hard way: `teacher_v1` listed its ITM cascade but not `ELIDEDB_ITM=1`, and
   re-running "the versioned teacher" scored 0.32 against a recorded 0.42 with
   every table digest still matching. Pinning artifacts is half of
   reproducibility; the switches are the other half.

---

# PART II — THE CURRENT PLAN (fixed with the user, 2026-08-01)

This part supersedes Part III's model roster and Part IV's numbers describe the previous generation. Part I (the database) is
unchanged and current. [ELEMENTS.md](ELEMENTS.md) holds the measurements
that forced each decision here.

## The five elements, as now specified

| element | is | linked to |
|---|---|---|
| **scene** | the setting, as a scene→scene TIME SERIES (a driving corpus changes scenes; one static vector cannot) | frame embeddings over time |
| **agent** | the self-moving thing | `object_id`, its trajectory |
| **participants** | ALL objects in the scene, not just what the agent touched | `object_id` each, a trajectory each |
| **events** | typed, timestamped transitions | agent + participants, joined with trajectory PER PARTICIPANT |
| **answer** | the JOIN of the four — a collective story, not a table | everything above |

Identity is the keystone: every join runs through `object_id`, so
unreliable ids poison every element downstream. Built first, measured
hardest.

## The model roster — five backbones, one student

**The teacher/student law:** the teacher must be proven BEFORE anything is
distilled from it. FDNN (our compression concept) appears nowhere in the
teacher — the previous FDNN-V was never properly distilled, and using it
as the teacher's substrate poisons the thing the teacher exists to prove.
FDNN returns only at the end, distilled fresh from a teacher that already
works.

| task | model | why |
|---|---|---|
| frame embedding, every frame | **DINOv3 ViT-S/16** (21.6M, 384-d) | SSL, zero text tower → no category collapse by construction; +10.9 GAP instance retrieval over DINOv2 |
| scene series | DINOv3 frame vectors over time | no extra model |
| identity descriptor | **DINOv3-S** on track exemplar crops | same backbone, crop granularity; replaces yolo26n-reid (nano ceiling: AUC 0.90, 5% of trivially-same pairs below 0.465) |
| region proposal (write) | **YOLO11n-seg** `single_cls` | class-agnostic, real-time, measured |
| association → presence | **BoT-SORT**, geometry only | tracks ARE identity while continuity holds; the store is asked once per track, not per frame |
| trajectory (replaces `mot`) | tracker boxes → contact point (centroid fallback) + **2.5D box-scale z** | geometry, not an encoder. Depth Pro was measured first (1.61 s/frame MPS = 1 h compute per hour of video at even 4 frames/episode) and the user chose 2.5D outright |
| physics per participant | **V-JEPA2 ViT-L** | tubelets seeded from the ELEMENTS, never whole-frame; only significantly-moving participants, plus the agent |
| text bridge, read-time only | **SigLIP2 so400m** | the ONE text tower; maps a query's nouns into vision space at query time |
| shipped write path, later | **FDNN student** | distilled fresh against the DINOv3 teacher + the corpus's free geometric pairs |

### Dropped, and the number that killed each

- **FDNN-V as teacher substrate** — never properly distilled (user call)
- **act** (SSv2 probe) — no earned purpose; riding V-JEPA2's forward pass
  for free is not a purpose (user call)
- **PE-Core** — 0.89 rank-dup of sig2, at 36 min/build vs sig2's 23
- **X-CLIP** — 0.70 dup; its one distinct ability (cross-frame attention)
  was discarded at write anyway
- **delta-appearance `mot`** — a delta of appearance is not a trajectory
- **yolo26n-reid** — AUC 0.90 ceiling, 5% failures on same-frame pairs
- **IV2** — *named* redundancy candidate, not yet deleted: with verbs
  carried by trajectory+events and temporal attention carried by the
  elements, it does sig2's noun job at 1B params. One A/B on the new
  stack decides — it was the old fusion's strongest independent channel,
  and deletions get measurements here.
- **FastSAM** — teacher-only forever (7.5 min/h)
- MobileCLIP2 / Unicom — the ungated fallback bracket for identity; moot
  once DINOv3 access was granted

### CLIP-family policy (user, 2026-08-01)

Text-aligned encoders are allowed for SIMILARITY; **naming at write is
banned forever** — an embedding never becomes a word on the write path.
DINOv3 takes all appearance duty; exactly one text tower (sig2) survives,
for read-time query nouns only.

## Identity, settled 2026-08-01

- Free supervision is two-sided: disjoint co-existing boxes = proven
  DIFFERENT; same-region co-existing boxes (double detections) = proven
  SAME. Geometry only (`free_negatives`, `free_positives`,
  `interval_pairs`).
- The cut is fitted by sweeping **cross-episode recurrence** (`fit_cut`) —
  the quantity a persistent id exists for; it has an interior maximum.
  The singleton rate is the WRONG target: 80%→17% singletons costs 2/3 of
  the recurrence and 8x proven-wrong merges.
- Assignment order matters more than the cut (greedy gallery): the
  writer's track-close order, never global-ts order (interleaves cameras;
  worst measured).
- Store today: cut 0.692, 25,629 objects, 71.8% singletons, 6,901
  recurring. The ceiling is the nano descriptor → DINOv3 replacement.

## The joins to build (the point of all of it)

```
agent        -> object_id            events currently name no object
events       -> object_id            per participant
events       -> scene position       box computed but not stored
participants -> scene relation       the "in/on what" of the answer
trajectory   -> per participant      one per participant AND agent, 3D as
                                     affordable (contact point + depth z)
vjepa        -> per participant      tubelets seeded from elements
```

`answer` = the join of all of it. A relational query ("object like THIS
undergoing motion like THAT near THIS scene region") executes as joins
over these tables — not as one fused cosine.

## Budgets and time discipline

- Shipped write path: **1 min per hour of video**, all stages included.
  Teacher runs may exceed it (they are what gets distilled away), but
  every teacher run gets a measured cost projection BEFORE it starts, a
  per-item tqdm bar, and a subset gate: nothing runs full-corpus until a
  subset proves the win.
- Corpus reference (fresh_bench): 2,097 episodes / 3.91 h / 70,436 frames
  / 73,521 tracks. Decode+track ≈ 35 min full-corpus.

---

# PART III — THE PREVIOUS MODELS (superseded 2026-08-01, kept as record)

Both models produce the **same five elements** and answer with the same
four-stage pipeline. They differ in what they spend to do it.

## 7. The five elements

Set as the schema for what a clip *is*:

| element | definition | how it is derived |
|---|---|---|
| **scene** | the setting, from any one frame | pooled PE vector per demo (1024-d) |
| **agent** | the self-moving thing | optical-flow track with the largest coherent persistent motion; recorded with its extent |
| **participants** | what the agent contacts, in order | non-agent tracks whose motion *onset* is adjacent to the agent — causality, not pixel change. Their bbox at onset is the clean, unoccluded rest footprint |
| **events** | typed, **timestamped** state transitions | `contact`, `release`, `open`, `close`, `put_into`, `put_on`, `take_out`, `adjust`, `agent` |
| **answer** | initial→final diff, attributed | ordered verb sequence + the state change |

Timestamps are not decoration. A single verb per demo cannot express a
compositional query: "pick up a green object and put it into the drawer" is
four events, so flattening scored exactly zero on it no matter how good the
one verb was.

**Articulation** (`open`/`close`) comes from a cavity signal — the dark-
fraction inside the articulated box, per frame — and the event is the
*transition* in that series, which is what carries the timestamp. Optical-flow
sign was tried and had no signal (corr +0.04).

**Relocation** (`put_into`/`put_on`/`take_out`/`adjust`) is typed by geometry:
displacement relative to **the object's own diagonal** (not the frame's), then
a containment test. The frame-relative threshold it replaced was measuring
tracker dropout rather than motion — median displacement 0.025 of the diagonal
against a 0.05 cut, median track lifespan 3.5 of 12 frames, and
corr(lifespan, displacement) = **+0.487**.

**Known limit, stated plainly:** the containment test is degenerate. The
articulated box spans 96% of the frame at the median, so `inside()` is 97%
trivially true and `put_on` fires on ~1% of transitions. Four routes to fixing
it are measured dead (connected-component box −0.10; state-diff persistence,
starved; open→close temporal bracket, 22.7% vs 25.4% — no discrimination;
threshold-only, a wash). The honest fix is container-as-entity and it is
unbuilt.

## 8. The teacher — expensive, 2–75 s/query

| stage | what it does |
|---|---|
| 1. channels | cosine over `pe`, `sig2`, `iv2`, `obj`, `act`, `vid`, `conj` → **RRF** fusion |
| 2. PRF | Rocchio round: the head of the first pass re-queries the corpus |
| 3. ITM cascade | InternVideo2-Stage2 1B cross-encoder over the top-N. Cost-gated OFF by default (`ELIDEDB_ITM=1`), 0.4 s/episode, 3.24 GB of tokens corpus-wide |
| 4. structure | transition **anchor** + routed event gate + motion-space density |
| 5. selection | fitted filter, temporal NMS, confidence cut |

Two hard-won constraints are encoded here. ITM is a **rerank stage, not a
channel** — as a weighted RRF voter it cost 0.38 → 0.27, because RRF discards
the margin's scale. And its rerank is **distribution-preserving**: it permutes
the candidates and hands back the same sorted score values in the new order,
because writing z-scores into the fused score broke the downstream confidence
cut, which is fitted against RRF's own scale (q03 returned 83 of a 371
ceiling, yield 0.87 → 0.28).

### The transition anchor — the corpus supplies what the query cannot

The text tower cannot ask for a *direction*.
`cos(pe_text("opens the drawer"), pe_text("closes the drawer")) = 0.957`, and
0.905 even after the student's trained query tower — so both queries returned
the same list (221 of 248 shared) while their truth sets are **disjoint**.

The direction was never missing from the store, only from the question.
Held-out open-vs-close separability by space:

| space | accuracy |
|---|---|
| **motion vectors** | **0.983** |
| V-JEPA | 0.733 |
| PE | 0.647 |
| IV2 | 0.569 |
| SigLIP2 | 0.556 |

So the query *names* a transition (closed-class English), and the episodes the
store itself flagged with that transition **define its direction** in motion
space, Rocchio against the complement.

Each kind must **earn its weight with no labels**: split the class, build the
prototype on one half, check it ranks the held-out half above the complement,
bootstrap, take a 2σ lower bound. Small classes then collapse on their own
variance rather than a hand-set minimum count — `close` 0.52, `open` 0.35,
`put_into` 0.02, `put_on`/`take_out` exactly **0.00**. This matters: a flat
weight gives q04 +0.54 and q08 −0.33 simultaneously, because `close` is
attested by 654 episodes and `put_on` by 5.

## 9. The student — 3.28M params, 27 ms/query

Same five elements, same four stages, produced without a 1B model.

**Write path** (`student_elements.py`, 0.85 s/demo vs the teacher's 3.40):
the geometry is kept verbatim — it is flow and boxes and costs ~0.5 s with no
model at all. Only the expensive step is replaced: a distilled `Namer` head
maps a crop's SigLIP embedding **directly to the name vector** the teacher's
7B generator + Grounding-DINO verifier would have produced. Names are never
compared as strings anywhere in this system — matching is cosine in name space
— so predicting the vector *is* the whole job, and it turns 3.4 s of
autoregressive generation into one matmul.

**Read path** — two small networks:

```python
Tower:  Linear(d_in, 512) → GELU → Linear(512, 256) → L2 norm
Rerank: Linear(4·256, 256) → GELU → Linear(256, 64) → GELU → Linear(64, 1)
        over [q, e, q·e, |q−e|]
```

Episode side: **1.15 MB for 1,122 demos**. Retrieval is one text forward plus
one matmul over the corpus, because every video-side computation was moved to
write time.

**Stage order was measured, not inherited.** Copying the teacher's sequence
(RRF → PRF → cascade → gate) made the student *worse* — 0.30 → 0.26, q01 to
zero. The student's cascade is a small listwise head, not a cross-encoder, and
it does better *consuming* the gate's evidence than overriding it. Faithfulness
to a pipeline is not faithfulness to its behaviour. Final order:
**PRF → gate → cascade.**

Stage 4 needed no learning at all: the teacher's gate is a deterministic
corroboration test, so the student runs the same test over `events_s` — the
table it produced itself.

## 10. How a query executes, end to end

```
text
 ├─ closed-class English → required transition(s)          (no dataset priors)
 ├─ STAGE 1  text tower → 256-d → matmul over episode embeddings
 ├─ STAGE 2  PRF: head of pass 1 re-queries the corpus
 ├─ STAGE 3  structural gate, all precomputed columns:
 │             · transition anchor   (direction, reliability-weighted)
 │             · event membership    (does this demo have the transition)
 │             · motion density      (15-NN agreement in motion space)
 ├─ STAGE 4  cascade rerank over the top-100
 ├─ selection: confidence cut — return fewer than k rather than pad with junk
 └─ MATERIALIZE: FrameSet.decode() — byte ranges, survivors only
```

Note where the pixels are: the last line. Everything above is arithmetic over
columns that were computed once at write time and pruned by zone maps.

---

# PART IV — MEASURED STATE (previous generation)

Metric: `k = ceil(1.5 × support)` per query; `yield = true/support`;
`prec = true/returned`; strict grading (ungraded counts as false).

| q | support | teacher | student |
|---|---|---|---|
| q03 vessel → stove | 247 | **0.62 / 0.62** | 0.79 / 0.52 |
| q05 opens drawer | 196 | 0.76 / 0.51 | 0.81 / 0.54 |
| q04 closes drawer | 165 | **0.92 / 0.61** | **0.90 / 0.60** |
| q08 spoon | 18 | 0.39 / 0.26 | 0.11 / 0.07 |
| q01 yellow | 17 | 0.41 / 0.27 | 0.24 / 0.15 |
| q02 red | 14 | 0.50 / 0.33 | 0.07 / 0.05 |
| q07 lid | 12 | 0.58 / 0.39 | 0.00 |
| q00 green | 8 | 0.38 / 0.25 | 0.12 / 0.08 |
| q09 / q10 | 2 | 0.00 | 0.00 |
| **mean** | | **0.45 / 0.32** | **0.30 / 0.20** |
| **read** | | 2,000–75,000 ms | **27 ms** |
| **write** | | 3.40 s/demo | **0.85 s/demo** |

q06 ("fold a piece of towel") is a **no-match gate** test — the corpus contains
no folding, and the correct answer is to return nothing. It passes.

## Where the remaining loss is

Not recall, and not the cut. Measured:

- **Abstention is exhausted.** Over the ranking already produced, only q03 and
  q04 can clear 0.60/0.60 and both already do; seven of ten queries never
  reach yield 0.60 at *any* truncation.
- **The truth is already close.** For every failing query, most or all of it
  sits inside the **top 18% of the corpus** — q07 all 12, q09/q10 both, q08 13
  of 18 — and ITM already reranks the top 150.

So the failure is **ordering inside the candidate set**, and what those
queries need is object *identity*, not motion. Identity alone scores AUC 0.862
on banana and 0.790 on eggplant — precisely the failing queries — but at or
below chance on the generic colours (green 0.531, red 0.466), because the
namer writes "black object" and the colour never matches. Fusing it naively
lost at every weight (0.45 → 0.37 → 0.28 → 0.22): the combiner takes a min
over nouns, so the weakest term governs, and "the drawer" appears in nearly
every query while matching nothing in particular. It is off by default, kept
as a reproduction. The fix is a per-noun earn-it-or-score-zero gate of the
kind the transition anchor uses — not another weight sweep.
