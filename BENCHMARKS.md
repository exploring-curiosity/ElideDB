# BENCHMARKS.md

All numbers measured on Apple Silicon (macOS 25.5, Apple clang 17,
RelWithDebInfo) against the real lab capture: **28.49 GB corpus** = 8× 4K
MJPEG camera streams (26.2 GB, ~45.4k indexed frames), 4× 16-channel 48 kHz
audio SDX streams (14.16 M rows each), 8 telemetry streams, +1 synthetic 10 M-row
stream. Raw per-query rows in `bench_results.jsonl` (written by `sdx bench`).
OS page cache warm unless noted (`purge` needs privileges this run didn't have).

## Headline: bytes elided per query class

| query class | bytes read | of corpus | elided |
|---|---|---|---|
| 2 s, one 4K camera (49 frames decoded) | 25.9 MB | 28.49 GB | **99.9096 %** |
| 2 s, one 16-ch audio stream (192k rows) | 9.8 MB | 28.49 GB | **99.9657 %** |
| 2 s, ALL 20 streams (104 frames + 4×16ch audio + telemetry) | 98.9 MB | 28.49 GB | **99.653 %** |
| 2 s, 10 M-row synthetic stream | 372 KB | — | 6 of 2,442 chunks (0.25 %) |
| `sdx bench` mean over 20 random 2 s windows, all streams | 217 MB avg | 28.49 GB | **99.563 %** |

Video reads are frame-payload-exact: a 2 s window costs its 49 frames'
packet bytes (~25.9 MB) + the mmapped SFI (84 KB) and nothing else — no
demux, no container probe, no over-read. For all-intra MJPEG the theoretical
floor IS the window's byte share; measured overhead above that floor is the
SFI index, 0.015 % of source bytes (3.9 MB indexes 26.2 GB).

## The worked example made real (FORMAT.md §1.5)

Spec prediction for 10 M rows @ chunk 4096: 2,442 chunks, ~176 KB directory,
2 s query ≈ 1–2 chunks/column. Measured: **2,442 chunks, 175,879 B
directory, 0.58 ms** end-to-end for a 2 s window (6 column-chunks incl. the
±1 s edge guard), 372,487 B read. The format guarantees the number by
construction — and it came out.

## Window-retrieval latency (2 s windows)

| shape | p50 | p99 / spread | notes |
|---|---|---|---|
| audio-only (16 ch, 192k rows → 100 Hz timeline) | **8.5 ms** | 9.0 ms | 3 runs, warm |
| synthetic 10 M-row stream | **0.6 ms** | — | zone-map floor |
| one 4K camera @ 640 px (49 frames) | **458 ms** | — | JPEG decode-bound (~9 ms/frame) |
| all 20 streams @ 640 px (~189 frames/window) | **3,074 ms** | p99 3,255 ms | `sdx bench`, 20 windows |

Latency on video windows is decoder-bound, not I/O-bound: bytes arrive in
~10 ms; 4K JPEG software decode costs ~9 ms/frame. `--stride`/`--width`
trade frames for speed; sensor-only queries never touch a decoder.

## vs. the naive baseline (`sdx bench --naive`)

Naive = read every byte of every file overlapping the window (stand-in for
"decode the segment, post-filter"): window 0 read **6.27 GB where StreetDex
read 217 MB (29× fewer bytes)**; a sensor-region window read 0.16 GB vs
0.20 MB (**815× fewer**). Note the naive *wall time* (935 ms) beats ours on
video windows only because it never decodes; decoding 6.27 GB of 4K MJPEG
(~9,600 frames) would take ~90 s vs our 3.2 s — the elision number is the
honest comparison, and it is why the naive path cannot scale.

## Ingest throughput

**28.49 GB in 29.7 s = 57.5 GB/min**, covering: full libav packet scan of 54
video segments (SFI build), clock-model fit over 45k (python, global-tick)
pairs, 56.6 M audio rows written to SDX, 8 telemetry SDX streams, manifest.

## Clock / sync accuracy

- Global-clock fit: **1200.0040 Hz** tick rate (four independent per-sensor
  estimates agree < 0.1 %); sensors' local clocks were off by ~3 years
  (108/184/227) and ~93 days (122) — reconciled to one canonical timeline.
- TV-clock ground truth: Sensor 227 frame at canonical +60.004 s shows the
  on-screen stopwatch at 00:01:17.3; Sensor 184 at +60.100 s shows
  00:01:17.4. Δcanonical = 95.8 ms ↔ Δdisplayed = 100 ms. Cross-sensor sync
  error is below the stopwatch's 0.1 s display quantum: **bound ≤ ~50 ms**,
  consistent at every sampled instant. (Tightening below the display quantum
  would need the 16-ch audio cross-correlation — future work.)

## Semantic retrieval (snapshot 3, SigLIP so400m via MLX, 1,194 windows)

| query | clusters probed | vectors scanned | recall@k vs exact |
|---|---|---|---|
| text "a person sitting at a table working on a laptop" (k=8) | 3 / 12 | 321 / 1,194 = **26.9 %** | **1.000** |
| query-by-clip (k=5) | 3 / 12 | 204 / 1,194 = **17.1 %** | **1.000** |

HDBSCAN found 12 clusters + 3.0 % noise (noise is always scanned — recall is
never sacrificed to the prune). Top text hit verified visually: the returned
clip shows two people at a table over a laptop. At 1,194 windows the prune is
about the *mechanism* (learned-cell IVF), not necessity — flat exact scan of
1,194×1,152 floats is ~1 ms; the crossover argument is the interview point.

## Second dataset: Oxford RobotCar sample (`store_oxford`)

StreetDex is not tied to the lab rig: any timestamped data enters through the
generic commands (`sdx init`, `sdx index-video` + ns sidecar, `sdx
ingest-csv`); `scripts/oxford_prepare.py` is a ~120-line ETL adapter
(demosaic Bayer PNGs → packed MJPEG/AVI per camera, µs CSVs → ns CSVs, lidar
scans → per-scan range summaries). 13 snapshots document the store's growth
stream by stream.

### Size: input vs stored

| | size |
|---|---|
| raw input (3,591 files: Bayer PNGs, f64 lidar, CSVs) | **991.2 MB** |
| StreetDex store total | **373.2 MB (0.38×)** |
| — packed camera streams (6× demosaiced MJPEG-in-AVI, q92) | 357.9 MB |
| — SDX sensor streams (gps, ins 126k rows, 3 lidar summaries) | 14.4 MB |
| — SFI frame indexes (1,443 frames) | 0.13 MB |
| — embeddings + clusters + UMAP (120 windows × 1152d) | 1.1 MB |

The store is *smaller* than the input while adding random access: raw Bayer
PNG compresses worse than demosaiced JPEG, and a pile of per-frame files has
no byte-range story at all — packing + SFI is what turns it into a database.

### Time per operation

| operation | time | notes |
|---|---|---|
| prepare/pack (demosaic 1,443 imgs + remux + CSV/lidar ETL) | 12.1 s | one-time, adapter |
| `index-video` ×6 (SFI build, 371 MB scanned) | 0.11 s total | one-time |
| `ingest-csv` ×5 (140k rows → SDX) | 0.17 s total | one-time |
| embed 120 windows (240 frames, SigLIP, MLX) | 13.3 s | per embedding run |
| PCA + HDBSCAN + UMAP (120×1152) | 6.4 s | per clustering run |
| **read**: 1 s video window, 1 camera @640px (~14 frames) | **43.9 ms** warm / 31 ms cold-CLI | decode-bound |
| **read**: 1 s window, ALL streams (6 cams + gps/ins/lidar) | **209.9 ms** warm | 84 frames + aligned sensors |
| **read**: 60 s INS window (3k rows → 100 Hz timeline) | **0.11 ms** warm | 99.885 % elided |
| **semantic text search** end-to-end | 2.8 s | sidecar model load dominates |
| — of which ranking (prune + cosine + top-k) | **0.005 ms** (nprobe 1) / 0.009 ms exact | 120 windows |
| **similarity (clip) search**, pool + rank, no sidecar | **0.005 ms** | |
| same at REIP scale (1,194 windows) | 0.016 ms pruned / 0.072 ms exact | 4.5× from prune |

### Semantic quality (verified visually)

- text "pedestrians walking on the sidewalk past buildings" → top hits all
  `mono/left` (the sidewalk-facing fisheye); decoded top window shows three
  pedestrians on the pavement. recall@5 vs exact = **1.000**.
- query-by-clip on `stereo/centre` → returns the *same street moment seen by
  the other stereo cameras* at 0.994 cosine, probing 1/2 clusters (50 %
  scanned).
- HDBSCAN on this 19 s homogeneous drive finds 2 clusters (60/60 — camera
  viewpoint split); prune-then-rank matters at corpus scale, not at n=120,
  and the numbers above show exactly where the crossover lives.

## Context retrieval

Oxford RobotCar sample, 66 **held-out** windows (last 30% of the timeline, never
trained on), six relational queries, judged by an independent VLM yes/no
logprob margin. Full method and caveats in [docs/CONTEXT.md](docs/CONTEXT.md).

| method | ms/query | judge margin | judge-top5 overlap |
|---|---|---|---|
| appearance (plain semantic search) | 12.9 | +0.276 | 1.2/5 |
| **context, captions materialised at ingest** | **0.6** | **+0.347** | **2.0/5** |
| context, tower-estimated | 0.4 | +0.259 | 1.5/5 |
| fused (0.6 context + 0.4 appearance), materialised | 12.5 | +0.323 | 1.8/5 |
| fused, tower-estimated | 12.0 | +0.283 | 1.5/5 |
| VLM rerank at query time | 2508.5 | +0.306 | 1.5/5 |

Materialised context is **4,000x faster than query-time VLM reranking and
scores higher**. It also beats plain semantic search on latency, because the
lexical path runs no neural text encoder at query time — almost all of
appearance's 12.9 ms is SigLIP's text tower.

Ingest cost: VLM captioning 0.75 s/window; per-frame SigLIP 46 ms/frame;
tower inference 84 us/window.

### Choosing the context space (ablation, same judge)

| context space | judge |
|---|---|
| appearance only | +0.276 |
| caption **embedding**, SigLIP text tower — *oracle* | +0.218 |
| caption **text**, raw TF-IDF | +0.331 |
| caption **text**, LSA-48 | +0.339 |

The embedding route loses even with perfect captions: SigLIP's text tower is
trained to sit near images, not near other text.

### Cellular turnover on the context tower (PPO + reverse attention)

| | channels | params | us/window | val loss |
|---|---|---|---|---|
| before | 48 | 82,770 | 66.5 | 0.3242 |
| after | **17** | **30,504** | **42.4** | **0.2939** |

63% fewer parameters, 36% faster, validation loss 9% **better** — pruning an
over-provisioned tower on a small corpus removes memorisation, not signal.

### Known limitation

Tower-estimated context (+0.259) is below plain appearance (+0.276) and only
earns its place when fused. The corpus is 19 seconds of one drive: 153 training
windows whose captions are near-duplicates (mean pairwise cosine 0.68). The
distillation path is built and measured, not yet demonstrated.

## Ingest throughput — where the time actually goes

Measured on the Bridge store (M-series, `scripts/bench_context.py` numbers
alongside):

| stage | rate | note |
|---|---|---|
| byte-range decode, contiguous | **2.2 ms/frame** | 0.23% of a 133 MB file for one 23-frame episode |
| byte-range decode, scattered | **2.6 ms/frame** | was 188 ms/frame — see below |
| SigLIP `quality` (so400m-384) | 90.3 ms/frame | 1152-d |
| SigLIP `fast` (so400m-224) | **27.7 ms/frame** | same model + space, 256 patches vs 729 — 3.3x |
| window embeddings from frame vectors | **6.2 s for 4,574 windows** | pooled, no GPU (was 1,525 s re-embedding) |
| VLM caption | ~650 ms/window | one per captioned window |
| context tower inference | 22.8 us/window | after pruning |

### Scattered reads were 74x slower than they should have been

`stride=N` sampling produces a scattered selection, and `_decode_gop` read one
span from the first selected frame's keyframe to the LAST selected packet — so
96 frames spread over a 4103 s stream decoded all 20,515 frames of the file to
return 96. Two fixes: partition the selection into contiguous byte runs, then
concatenate those runs into a single decoder invocation (every run starts at a
keyframe, so they form one valid elementary stream).

| | ms/frame | bytes read |
|---|---|---|
| before | 188.0 | 133 MB of 133 MB |
| runs partitioned | 27.2 | 2.9 MB |
| runs partitioned + batched into one decode | **2.6** | 2.9 MB |

Verified pixel-identical against per-run decoding. This is the difference
between sampling being free and sampling costing more than reading everything.

**97% of ingest time is the image encoder, not the storage engine.** The knobs
that matter are therefore `model="fast"`, `frame_stride=N`, and
`label_fraction<1`; all three are parameters of `Store.index_context()`.

### What the full corpus actually costs

90.3 GB of BridgeData2 is **477 hours of video** (AV1 is dense: the same 28.5 GB
of the lab capture is 0.64 h, because it is raw MJPEG plus 16-channel audio).
Measured decode + encode, extrapolated:

| sampling | `quality` (so400m-384) | `fast` (so400m-224) |
|---|---|---|
| every frame (5 fps) | 112 h | 38.5 h |
| 1 frame/s | 22.5 h | 7.7 h |
| 1 per 2 s | 11.2 h | 3.9 h |
| 1 per 4 s | 5.6 h | **1.9 h** |
| 1 per 10 s | 2.2 h | **46 min** |

Sampling is the dominant term, and it is now honest — before the decode fix,
sampling sparsely made things *slower* per frame, so the knob did not work.
`frame_vectors` is the single source both the semantic and the context index
derive from, so this is paid once, not per index.

## FDNN-V: the write-path encoder (embed every frame)

Distilled from SigLIP into a 1.95M-parameter recurrent video encoder
(design + full iteration log: [docs/FDNNV.md](docs/FDNNV.md)).

| | SigLIP-384 | SigLIP-224 | **FDNN-V** |
|---|---|---|---|
| params | 428M | 428M | **1.95M** |
| ms/frame (batched) | 90.3 | 27.7 | **0.270** |
| Bridge store, every frame (58,011) | ~87 min | ~27 min | **135.7 s** |
| 477 h corpus, every frame | 112 h | 38.5 h | **5.6 h** (decode-bound) |
| held-out fidelity vs teacher | 1.0 | 0.921 (self-ceiling) | 0.902 |
| text top-10 agreement | 10/10 | 4.3/10 (self-ceiling) | 0.5/10 — not delivered |

The write path is decode-bound now (2.06 of 2.34 ms/frame); embedding keeps
up with ~200 live 5 fps streams. Student vectors serve image-image and clip
similarity; text search stays on teacher windows until the text gap closes
(levers listed in the doc).

## bridge4h: 4 hours loaded and fully embedded in 79 seconds

Hard budget: 5 minutes for ~4 h of video, every frame embedded. Result:
**79.1 s — 178x real time** (`scripts/bridge4h.py`, PASS on iteration 1).

| stage | wall | note |
|---|---|---|
| embed EVERY frame (70,436, FDNN-V) | 47.1 s | source piped straight through the encoder at 2,185 fps decode — sequential embedding needs no random access, so it never waits for transcode |
| transcode (read path) | +18.8 s | 4 files in parallel, overlapped with embedding |
| index + episodes + robot + windows | +13.2 s | one atomic append per table |
| **total** | **79.1 s** | 890 frames/s end to end |

### Sharp text search: student shortlist, teacher verdict, cracked cache

Warm query **26 ms** (after the zero-copy vector-table fix, which sped every
search path ~30x). Teacher vectors computed for a query are cached into the
store — 15% of the corpus was teacher-embedded after 100 queries, so quality
accumulates exactly where users look (database cracking).

Quality, graded on human task labels held outside the store:

| query type | student | teacher (ceiling) | verdict |
|---|---|---|---|
| category ("opening a drawer", 748 rel) | **0.80 @10** | 0.90 | usable; sharp converges to ceiling |
| category ("pot on stove", 426 rel) | **0.60 @10** | 0.50 | student ≥ teacher |
| rare category (towel, spoon) | 0.0–0.2 | 0.1–0.2 | both weak |
| singleton instruction (100 queries) | 0.000 R@10 | **0.030 R@10** | fails at EVERY tier — even teacher-everywhere shortlist recall@48 is 0.17 |

The last row is the honest boundary: instruction-level retrieval among 2,097
near-identical clips defeats SigLIP-class embedding search entirely; that is
the caption/context path's job (background-priced), not the embedding index's.
Sharp search cannot beat its shortlist — fix directions are a wider adaptive
shortlist and shortlisting from the RRF union once captions exist.

## Reproduce

```bash
./build/sdx bench --store store --windows 20 --dur 2 --naive 2
./build/sdx query --store store --text "..." --eval-recall
```

## Truthset ledger (lake/bench, k=10, strict: ungraded=false)

| when | commit | true/returned | mean prec | mean yield | per-query |
|---|---|---|---|---|---|
| 2026-07-25 23:29 | a989306 | 18/50 returned true | mean prec 0.32 | mean yield 0.18 | q00:1/3 q01:1/4 q02:4/4 q03:1/4 q04:9/10 q05:1/8 q07:1/3 q08:0/6 q09:0/4 q10:0/4 |
| 2026-07-25 23:30 | a989306 | 10/46 returned true | mean prec 0.25 | mean yield 0.10 | q00:1/3 q01:1/4 q02:4/4 q03:1/4 q04:1/10 q05:1/4 q07:1/3 q08:0/6 q09:0/4 q10:0/4 |
| 2026-07-25 23:31 | a989306 | 18/52 returned true | mean prec 0.32 | mean yield 0.18 | q00:1/3 q01:1/4 q02:4/4 q03:1/4 q04:9/10 q05:1/10 q07:1/3 q08:0/6 q09:0/4 q10:0/4 |
| 2026-07-25 23:33 | a989306 | 18/52 returned true | mean prec 0.32 | mean yield 0.18 | q00:1/3 q01:1/4 q02:4/4 q03:1/4 q04:9/10 q05:1/10 q07:1/3 q08:0/6 q09:0/4 q10:0/4 |
| 2026-07-25 23:35 | a989306 | 18/48 returned true | mean prec 0.32 | mean yield 0.18 | q00:1/3 q01:1/4 q02:4/4 q03:1/4 q04:10/10 q05:0/6 q07:1/3 q08:0/6 q09:0/4 q10:0/4 |
| 2026-07-25 23:36 | a989306 | 20/47 returned true | mean prec 0.37 | mean yield 0.20 | q00:1/3 q01:1/4 q02:4/4 q03:1/4 q04:8/9 q05:4/6 q07:1/3 q08:0/6 q09:0/4 q10:0/4 |
| 2026-07-25 23:37 | e5104f7 | 21/48 returned true | mean prec 0.39 | mean yield 0.21 | q00:1/3 q01:1/4 q02:4/4 q03:2/5 q04:8/9 q05:4/6 q07:1/3 q08:0/6 q09:0/4 q10:0/4 |
| 2026-07-26 00:10 | 64d38bd | 25/56 returned true | mean prec 0.41 | mean yield 0.25 | q00:1/3 q01:1/4 q02:4/4 q03:3/10 q04:8/9 q05:4/6 q07:1/3 q08:3/10 q09:0/3 q10:0/4 |
| 2026-07-26 00:35 | cc794dd | 16/51 returned true | mean prec 0.27 | mean yield 0.17 | q00:2/10 q01:0/3 q02:4/5 q03:3/10 q04:5/10 q05:1/3 q07:1/3 q08:0/0 q09:0/4 q10:0/3 |
| 2026-07-26 00:37 | cc794dd | 15/63 returned true | mean prec 0.25 | mean yield 0.16 | q00:2/7 q01:1/10 q02:4/5 q03:3/10 q04:1/3 q05:3/9 q07:1/3 q08:0/3 q09:0/10 q10:0/3 |
| 2026-07-26 00:40 | cc794dd | 8/40 returned true | mean prec 0.18 | mean yield 0.08 | q00:0/3 q01:0/4 q02:0/0 q03:3/10 q04:4/4 q05:0/4 q07:1/3 q08:0/3 q09:0/6 q10:0/3 |
| 2026-07-26 00:42 | cc794dd | 8/45 returned true | mean prec 0.16 | mean yield 0.08 | q00:1/3 q01:0/3 q02:0/0 q03:3/10 q04:0/3 q05:3/7 q07:1/3 q08:0/3 q09:0/10 q10:0/3 |
| 2026-07-26 00:44 | cc794dd | 9/46 returned true | mean prec 0.16 | mean yield 0.10 | q00:2/10 q01:0/3 q02:0/0 q03:3/7 q04:3/6 q05:1/3 q07:0/3 q08:0/4 q09:0/4 q10:0/6 |
| 2026-07-26 00:45 | cc794dd | 19/90 returned true | mean prec 0.21 | mean yield 0.20 | q00:2/10 q01:1/10 q02:0/0 q03:5/10 q04:6/10 q05:3/10 q07:2/10 q08:0/10 q09:0/10 q10:0/10 |
| 2026-07-26 01:10 | d88ca01 | 24/100 returned true | mean prec 0.24 | mean yield 0.25 | q00:2/10 q01:1/10 q02:5/10 q03:5/10 q04:6/10 q05:3/10 q07:2/10 q08:0/10 q09:0/10 q10:0/10 |
| 2026-07-26 05:05 | d88ca01 | 25/100 returned true | mean prec 0.25 | mean yield 0.26 | q00:2/10 q01:2/10 q02:5/10 q03:5/10 q04:5/10 q05:3/10 q07:3/10 q08:0/10 q09:0/10 q10:0/10 |
| 2026-07-27 05:21 | 42f7625 | 25/100 returned true | mean prec 0.25 | mean yield 0.26 | q00:2/10 q01:2/10 q02:5/10 q03:5/10 q04:5/10 q05:3/10 q07:3/10 q08:0/10 q09:0/10 q10:0/10 |
| 2026-07-27 05:30 | 01a4720 | 25/100 returned true | mean prec 0.25 | mean yield 0.26 | q00:2/10 q01:2/10 q02:5/10 q03:5/10 q04:5/10 q05:3/10 q07:3/10 q08:0/10 q09:0/10 q10:0/10 |
| 2026-07-27 05:43 | 0868832 | 25/100 returned true | mean prec 0.25 | mean yield 0.26 | q00:2/10 q01:2/10 q02:5/10 q03:5/10 q04:5/10 q05:3/10 q07:3/10 q08:0/10 q09:0/10 q10:0/10 |
| 2026-07-27 05:56 | 286b2e5 | 25/100 returned true | mean prec 0.25 | mean yield 0.26 | q00:2/10 q01:2/10 q02:5/10 q03:5/10 q04:5/10 q05:3/10 q07:3/10 q08:0/10 q09:0/10 q10:0/10 |

### Binding diagnosis (2026-07-27)

`scripts/diag_binding.py` instruments q08/q09/q10 per-channel: the fitter (previous
commit) rejected joining `conj` to the filter set, and this run shows why the
channels are not the whole story. q08 "place the spoon on top of the cloth"
(support 18): conj AUC +0.829, top10-by-conj-alone 3/10; obj AUC +0.837, top10
3/10 — both individually well above chance, yet the live fused+filtered pipeline
returns 0/10 for the same query across every ledger row since d88ca01. That is
the fusion/filter pattern: usable per-channel signal, discarded downstream. q10
"put the banana on top of the drawer" (support 2): conj AUC +0.933, sig2 +0.921,
vid +0.900, but top10 0/10 on every channel; with only 2 positives among ~1100
episodes, a top-10 hit by any ranker is a low-probability event regardless of
ranking quality, so this query's numbers do not distinguish fusion failure from
encoder failure. q09 "put the eggplant into the drawer" (support 2) has the same
n=2 noise problem, but also a distinct, non-noisy defect: `atoms_of` returns a
single atom, `['the eggplant into the']` — the regex swallowed the preposition
"into" as a filler word and grabbed the next determiner as the head, so the
phrase never reaches "drawer" and `conj_lookup` abstains (len(atoms) < 2) before
scoring anything. `atoms_of` on q08 and q10 shows a milder form of the same
defect: `['the spoon on top', 'the cloth']` and `['the banana on top', 'the
drawer']` — the object atom absorbs the relational "on top" phrase instead of
stopping at the noun, corrupting the text embedding fed to SigLIP2 even where
conj does run. Next lever: fix the atom boundary (stop at prepositions) before
re-attempting filter-membership search, since q08's fusion/filter finding is
only actionable once the atoms feeding conj are the intended noun phrases.
| 2026-07-27 06:07 | 04548bc | 23/100 returned true | mean prec 0.23 | mean yield 0.23 | q00:2/10 q01:2/10 q02:4/10 q03:5/10 q04:4/10 q05:3/10 q07:3/10 q08:0/10 q09:0/10 q10:0/10 |
| 2026-07-27 06:24 | 0833cdf | 23/100 returned true | mean prec 0.23 | mean yield 0.23 | q00:2/10 q01:2/10 q02:4/10 q03:5/10 q04:4/10 q05:3/10 q07:3/10 q08:0/10 q09:0/10 q10:0/10 |
| 2026-07-27 06:35 | 2cd631c | 24/100 returned true | mean prec 0.24 | mean yield 0.24 | q00:2/10 q01:2/10 q02:4/10 q03:5/10 q04:4/10 q05:3/10 q07:1/10 q08:3/10 q09:0/10 q10:0/10 |
| 2026-07-27 06:38 | 2cd631c | 25/100 returned true | mean prec 0.25 | mean yield 0.26 | q00:2/10 q01:2/10 q02:5/10 q03:5/10 q04:6/10 q05:3/10 q07:2/10 q08:0/10 q09:0/10 q10:0/10 |

### InternVideo2 negative result (2026-07-27)

Task 7 (video-native text channel via InternVideo2-Stage2 1B, arXiv 2403.15377)
bailed at Step 1 (the load probe), before any flash_attn/MPS/CPU fallback logic
was reached. `scripts/iv2_probe.py` calling
`AutoTokenizer.from_pretrained("OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
trust_remote_code=True)` fails immediately with:

```
OSError: You are trying to access a gated repo.
Make sure to have access to it at
https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4.
403 Client Error. ... Access to model
OpenGVLab/InternVideo2-Stage2_1B-224p-f4 is restricted and you are not in the
authorized list.
```

The machine's HF token (`whoami` succeeds, user SudharshanR) is valid and
authenticated, but this specific repo is gate-restricted and the account has
not been granted access — that requires a manual "request access" click on
the model page and (for OpenGVLab gates) an indeterminate wait for approval,
which is a human/account action outside this task's scope, not a code or
environment problem. No shim exists for an HTTP 403 the way one exists for a
missing `flash_attn` import, so the flash_attn-shim fallback in the bailout
protocol was never reached — this is a harder stop than the anticipated
failure mode. Total wall-clock on Task 7: under 5 minutes (probe fails on the
first `from_pretrained` call). No `iv2_vectors` table, no channel code, no
weight-fit or bench changes — `python/elidedb/scenario.py` and
`scripts/fit_set_weights.py` are unchanged by this task. `scripts/iv2_probe.py`
is committed as the scaffold; a future attempt only needs HF access granted
(https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4 → Request
access) before rerunning it.

## Binding-filter sprint record (2026-07-27)

Nine tasks, each committed with its ledger row; plan:
docs/superpowers/plans/2026-07-27-conj-binding-filter-and-video-native-iteration.md.
Thesis: rank-fusion consensus (RRF, Cormack et al. 2009) structurally drowns a
decisive minority channel, and bag-of-concepts encoders cannot rank binding
(Winoground CVPR 2022, ARO ICLR 2023) — so the conjunctive atom channel must
be eligible to FILTER, with veto authority fitted per store, never coded.

Ledger trajectory (mean prec/yield): 0.25/0.26 baseline (aa55b46) →
0.25/0.26 through the behavior-preserving unification tasks (T1 variant_max,
T2 setpath one-code-path, T3 fitted filter membership — the fitter REJECTED
conj from the filter set, an honest negative) → 0.23/0.23 after the
atoms_of preposition-boundary fix forced a refit into a worse local optimum
(T4b; q02/q04 −1 each, LOQO 0.212→0.182) → 0.25/0.26 recovered by
corpus-attested vocabulary (T6; q04 4→6, q02 4→5, q07 3→2; _HYPONYMS
deleted). Net: flat on the headline number, structurally ahead: zero hand
dictionaries, one fit-live selection code path (knee, filters, gate all
modeled in fit), the Desk context tab now serves the measured search_set
(fast + geometry-audit tiers, streams/t0/t1 honored), and a binding
diagnostics instrument (scripts/diag_binding.py).

Binding queries remain 0/10 on the ledger, and the diagnosis is now precise:
q08 conj ALONE ranks 3/10 true in its top-10 (AUC +0.829 dirty, +0.800
clean) while the fused pipeline returns 0 — the residual gap is FUSION
AUTHORITY, not decomposition (atoms are clean post-T4b: q09 conj went from
abstaining on one corrupt atom to AUC +0.981). q09/q10 have support 2; their
positives sit near rank ~20 of 1122 (AUC 0.93–0.98), so the needed lift is
rank-20 → rank-10, which reweighting alone did not find (greedy toggle,
fixed order, 4-round budget — a local search, not a categorical no).

Next levers, in order: (1) InternVideo2-Stage2 1B — BLOCKED on HF gated-repo
access (user action: Request access at
https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4), probe
scaffold committed; (2) per-query channel authority for binding queries
(the q08 conj-alone 3/10 vs fused 0/10 measurement is the case for it);
(3) _STOP compound prepositions ("out of", "on top") before q11–q13 enter
the graded set; (4) atomic _vocab.json writes (tmp+rename) if attestation
ever runs concurrently.
| 2026-07-27 14:26 | 9a8e8b4 | 19/72 returned true | mean prec 0.28 | mean yield 0.20 | q00:2/10 q01:2/10 q02:2/4 q03:3/4 q04:6/10 q05:3/10 q07:1/4 q08:0/2 q09:0/10 q10:0/8 |
| 2026-07-27 14:39 | 5e912c6 | 22/58 returned true | mean prec 0.39 | mean yield 0.23 | q00:2/4 q01:1/4 q02:3/6 q03:9/10 q04:4/4 q05:2/4 q07:1/5 q08:0/8 q09:0/3 q10:0/10 |
| 2026-07-27 17:25 | 1645da8 | 21/57 returned true | mean prec 0.38 | mean yield 0.21 | q00:2/3 q01:1/4 q02:3/6 q03:9/10 q04:2/4 q05:3/4 q07:1/5 q08:0/8 q09:0/3 q10:0/10 |
| 2026-07-27 17:27 | 6b5499b | 33/89 returned true | mean prec 0.36 | mean yield 0.34 | q00:2/10 q01:2/10 q02:3/6 q03:9/10 q04:8/10 q05:8/10 q07:1/5 q08:0/8 q09:0/10 q10:0/10 |
| 2026-07-27 17:30 | 6b5499b | 33/89 returned true | mean prec 0.36 | mean yield 0.34 | q00:2/10 q01:2/10 q02:3/6 q03:9/10 q04:8/10 q05:8/10 q07:1/5 q08:0/8 q09:0/10 q10:0/10 |
