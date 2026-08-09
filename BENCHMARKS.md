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
| 2026-07-27 17:55 | 9c0faa0 | 34/89 returned true | mean prec 0.37 | mean yield 0.35 | q00:2/10 q01:2/10 q02:3/6 q03:9/10 q04:8/10 q05:9/10 q07:1/5 q08:0/8 q09:0/10 q10:0/10 |
| 2026-07-27 18:55 | bddf2b1 | 25/47 returned true | mean prec 0.40 | mean yield 0.26 | q00:2/5 q01:2/7 q02:3/6 q03:0/0 q04:8/10 q05:9/10 q07:1/3 q08:0/1 q09:0/0 q10:0/5 |
| 2026-07-27 20:19 | bddf2b1 | 27/65 returned true | mean prec 0.38 | mean yield 0.28 | q00:2/5 q01:2/7 q02:3/6 q03:2/3 q04:8/10 q05:9/10 q07:1/4 q08:0/5 q09:0/9 q10:0/6 |
| 2026-07-27 20:20 | bddf2b1 | 35/91 returned true | mean prec 0.38 | mean yield 0.36 | q00:2/10 q01:2/10 q02:3/6 q03:9/10 q04:8/10 q05:9/10 q07:1/5 q08:1/10 q09:0/10 q10:0/10 |
| 2026-07-27 22:56 | baf2b78 | 33/91 returned true | mean prec 0.36 | mean yield 0.33 | q00:1/10 q01:2/10 q02:3/6 q03:9/10 q04:7/10 q05:9/10 q07:1/5 q08:1/10 q09:0/10 q10:0/10 |
| 2026-07-28 18:10 | 2f0684a | 10/92 returned true | mean prec 0.13 | mean yield 0.10 | q00:0/10 q01:0/10 q02:4/7 q03:2/7 q04:0/10 q05:1/10 q07:1/10 q08:2/8 q09:0/10 q10:0/10 |

### 2026-07-28 - the 10/92 ledger row is an ENVIRONMENT regression, not a model change

The `10/92, prec 0.13` row above ran after the local environment had
drifted to transformers 5.14 (pulled in by an mlx-embeddings upgrade),
whose new internals refuse the InternVideo2 port - the iv2 channel
failed silently and search ran without its strongest lever. Proof it
was the environment: the identical measurement on an fp16-compressed
COPY of the store produced the same numbers query for query, and the
fp32 original scored 0.13 in the same shell. transformers is now
pinned at 4.57.6 (requirements-local.txt), matching the demo
container. The row stays because the ledger is append-only; read it as
"what happens when a fitted channel dies silently" - and that silence
is now a recorded loophole to fix (channel-death must surface in
result meta and block ledger appends).
| 2026-07-28 18:19 | 2f0684a | 35/91 returned true | mean prec 0.38 | mean yield 0.36 | q00:2/10 q01:2/10 q02:3/6 q03:9/10 q04:8/10 q05:9/10 q07:1/5 q08:1/10 q09:0/10 q10:0/10 |
| 2026-07-28 18:22 | 299692a | 35/91 returned true | mean prec 0.38 | mean yield 0.36 | q00:2/10 q01:2/10 q02:3/6 q03:9/10 q04:8/10 q05:9/10 q07:1/5 q08:1/10 q09:0/10 q10:0/10 |
| 2026-07-28 20:58 | 45656e1 | 35/91 returned true | mean prec 0.38 | mean yield 0.36 | q00:2/10 q01:2/10 q02:3/6 q03:9/10 q04:8/10 q05:9/10 q07:1/5 q08:1/10 q09:0/10 q10:0/10 |

### 2026-07-28 - bench-truthset compression, and a wrong-frames bug it uncovered

All numbers are lake/bench (1,122 episodes), measured, source =
the four BridgeData mp4 files the store was built from.

| | bytes | GB | vs source |
|---|---|---|---|
| raw source (4 mp4) | 819,494,109 | 0.819 | 100% |
| store tables (fp16+BSS, 10 tables) | 218,676,442 | 0.219 | 26.7% |
| store media (h264 crf26, IDR per episode) | 210,798,343 | 0.211 | 25.7% |
| **the database** | **429,474,785** | **0.429** | **52.4%** |
| python whole-table caches (derived, rebuildable) | 354,552,656 | 0.355 | 43.3% |

Retrieval after both migrations: 0.38 / 0.36, 35/91 - identical to the
pre-migration reference row, query for query.

The media rebuild uncovered a defect in the SHIPPED index: the old
rendition carried B-frames, so packet (decode) order was not
presentation order, while ingest paired presentation-ordered
timestamps with decode-ordered packets. Decoded against the source,
lake/bench returned 42 of 64 sampled frames at >8 mean abs error
(mean 16.1, max 51.9) - visibly different scenes. The rebuilt store
returns 0 of 80 (mean 1.15). Search was never affected because
vectors are embedded from the source sequentially, but every decoded
surface (Desk playback, thumbnails, clip export) was serving wrong
pictures for roughly two thirds of episodes. Renditions now encode
with -bf 0, so the frame index means what it says.

Per-episode bytes read fell 323 KB -> 196 KB alongside, because the
keyframe now sits at the episode start rather than every second.
| 2026-07-28 21:49 | 80176c4 | 35/91 returned true | mean prec 0.38 | mean yield 0.36 | q00:2/10 q01:2/10 q02:3/6 q03:9/10 q04:8/10 q05:9/10 q07:1/5 q08:1/10 q09:0/10 q10:0/10 |
| 2026-07-29 00:23 | 0da8b4f | 35/91 returned true | mean prec 0.38 | mean yield 0.36 | q00:2/10 q01:2/10 q02:3/6 q03:9/10 q04:8/10 q05:9/10 q07:1/5 q08:1/10 q09:0/10 q10:0/10 |

### 2026-07-28 - Model bake-off: IV2 vs VideoPrism, and channel ablation

Two questions settled on the frozen truthset, not on public leaderboards.

**Which channels earn their ingest cost?** Leave-one-out through the
production path (scripts/ablate_channels.py). Baseline 35 true / 91
returned:

| dropped | true results lost |
|---|---|
| iv2 | 19 |
| mot | 7 |
| conj | 5 |
| act, vid, prf | 3 each |
| pe, obj | 1 each |
| sig2 | 0 |

The video-native contextual channels carry the product. The
appearance family (pe, sig2, obj) is worth ~2 of 35 results between
them while costing three vision passes per episode; sig2 contributes
nothing measurable and is deletable today.

**Can a smaller model replace IV2?** Head-to-head over one 681-episode
pool containing every graded-true episode, each model using its own
text tower, all at 16 frames (VideoPrism's native clip length):

| model | params | true/ret | precision |
|---|---|---|---|
| InternVideo2-Stage2 | 1B | 32/110 | **0.291** |
| VideoPrism LvT-B | 248M | 17/110 | 0.155 |
| VideoPrism LvT-L | 580M | 13/110 | 0.118 |

No. IV2 is ~1.9x better than the best VideoPrism here, and the LARGER
VideoPrism is worse than the smaller one - its edge on MSR-VTT does not
transfer to fixed-camera manipulation, where the signal is direction and
contact rather than scene variety. InternVideo3 was also considered and
rejected on inspection: it is an 8B generative agent with no dual
encoder, so it cannot do this job at all.

Conclusion: IV2 stays as the contextual anchor AND becomes the
distillation teacher for the single fast write-path channel. Speed was
deliberately not compared - JAX ran CPU-only here while IV2 ran on GPU,
so any such number would measure the backend, not the model.
| 2026-07-29 07:07 | b12fe2f | 22/100 returned true | mean prec 0.22 | mean yield 0.22 | q00:1/10 q01:1/10 q02:3/10 q03:7/10 q04:4/10 q05:4/10 q07:1/10 q08:1/10 q09:0/10 q10:0/10 |
| 2026-07-29 07:10 | b12fe2f | 35/100 returned true | mean prec 0.35 | mean yield 0.35 | q00:1/10 q01:2/10 q02:3/10 q03:9/10 q04:8/10 q05:10/10 q07:1/10 q08:1/10 q09:0/10 q10:0/10 |
| 2026-07-29 07:19 | b12fe2f | 35/100 returned true | mean prec 0.35 | mean yield 0.35 | q00:1/10 q01:1/10 q02:3/10 q03:9/10 q04:9/10 q05:10/10 q07:1/10 q08:1/10 q09:0/10 q10:0/10 |
| 2026-07-29 14:53 | f79439e | 155/776 returned true | mean prec 0.23 | mean yield 0.40 | q00:2/43 q01:4/75 q02:8/100 q03:64/100 q04:37/49 q05:22/41 q07:8/100 q08:9/100 q09:0/68 q10:1/100 |
| 2026-07-29 14:57 | f79439e | 194/1000 returned true | mean prec 0.19 | mean yield 0.66 | q00:5/100 q01:10/100 q02:8/100 q03:64/100 q04:40/100 q05:43/100 q07:10/100 q08:10/100 q09:2/100 q10:2/100 |
| 2026-07-29 15:26 | f79439e | 218/977 returned true | mean prec 0.22 | mean yield 0.67 | q00:4/100 q01:11/100 q02:6/91 q03:80/100 q04:45/100 q05:48/100 q07:11/100 q08:9/95 q09:2/91 q10:2/100 |
| 2026-07-29 15:39 | 98f22ac | k=1.5xsup | 267/629 returned true | mean yield 0.25 | mean prec 0.24 | q00:2/12 q01:4/26 q02:2/21 q03:93/119 q04:89/221 q05:69/179 q07:4/18 q08:4/27 q09:0/3 q10:0/3 |
| 2026-07-29 15:44 | 98f22ac | k=1.5xsup | 277/697 returned true | mean yield 0.28 | mean prec 0.24 | q00:2/12 q01:5/26 q02:5/21 q03:104/150 q04:78/213 q05:74/224 q07:6/18 q08:3/27 q09:0/3 q10:0/3 |
| 2026-07-29 15:50 | 98f22ac | k=1.5xsup | 295/760 returned true | mean yield 0.29 | mean prec 0.23 | q00:2/12 q01:5/26 q02:7/21 q03:122/213 q04:78/213 q05:74/224 q07:3/18 q08:4/27 q09:0/3 q10:0/3 |
| 2026-07-29 21:44 | 768574a | k=1.5xsup | 295/760 returned true | mean yield 0.29 | mean prec 0.23 | q00:2/12 q01:5/26 q02:7/21 q03:122/213 q04:78/213 q05:74/224 q07:3/18 q08:4/27 q09:0/3 q10:0/3 |
| 2026-07-30 02:41 | b30f6b4 | k=1.5xsup | 217/521 returned true | mean yield 0.32 | mean prec 0.30 | q00:3/12 q01:7/26 q02:7/21 q03:70/83 q04:93/248 q05:23/80 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |
| 2026-07-30 02:43 | b30f6b4 | k=1.5xsup | 341/768 returned true | mean yield 0.38 | mean prec 0.29 | q00:3/12 q01:7/26 q02:7/21 q03:142/221 q04:85/213 q05:83/224 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |
| 2026-07-30 03:57 | edf2e69 | k=1.5xsup | 305/805 returned true | mean yield 0.27 | mean prec 0.21 | q00:3/12 q01:4/26 q02:4/21 q03:150/272 q04:67/207 q05:70/216 q07:2/18 q08:5/27 q09:0/3 q10:0/3 |
| 2026-07-30 03:59 | edf2e69 | k=1.5xsup | 341/768 returned true | mean yield 0.38 | mean prec 0.29 | q00:3/12 q01:7/26 q02:7/21 q03:142/221 q04:85/213 q05:83/224 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |
| 2026-07-30 05:23 | 8bd9023 | k=1.5xsup | 341/768 returned true | mean yield 0.38 | mean prec 0.29 | q00:3/12 q01:7/26 q02:7/21 q03:142/221 q04:85/213 q05:83/224 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |
| 2026-07-30 05:50 | 8bd9023 | k=1.5xsup | 415/896 returned true | mean yield 0.42 | mean prec 0.30 | q00:3/12 q01:7/26 q02:7/21 q03:152/244 q04:119/248 q05:113/294 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |
| 2026-07-30 16:02 | 5340a6b | k=1.5xsup | 442/884 returned true | mean yield 0.36 | mean prec 0.26 | q00:2/12 q01:4/26 q02:7/21 q03:129/232 q04:148/248 q05:145/294 q07:3/18 q08:4/27 q09:0/3 q10:0/3 |
| 2026-07-30 16:04 | 5340a6b | k=1.5xsup | 361/884 returned true | mean yield 0.32 | mean prec 0.23 | q00:2/12 q01:5/26 q02:7/21 q03:129/232 q04:108/248 q05:103/294 q07:3/18 q08:4/27 q09:0/3 q10:0/3 |
| 2026-07-30 16:08 | 5340a6b | k=1.5xsup | 361/884 returned true | mean yield 0.32 | mean prec 0.23 | q00:2/12 q01:5/26 q02:7/21 q03:129/232 q04:108/248 q05:103/294 q07:3/18 q08:4/27 q09:0/3 q10:0/3 |
| 2026-07-30 16:12 | 5340a6b | k=1.5xsup | 415/896 returned true | mean yield 0.42 | mean prec 0.30 | q00:3/12 q01:7/26 q02:7/21 q03:152/244 q04:119/248 q05:113/294 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |
| 2026-07-30 16:14 | 5340a6b | k=1.5xsup | 483/896 returned true | mean yield 0.45 | mean prec 0.32 | q00:3/12 q01:7/26 q02:7/21 q03:152/244 q04:151/248 q05:149/294 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |
| 2026-07-30 16:52 | 22c54ad | k=1.5xsup [itm,anchor] | 481/896 returned true | mean yield 0.45 | mean prec 0.32 | q00:3/12 q01:7/26 q02:7/21 q03:152/244 q04:150/248 q05:148/294 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |
| 2026-07-30 17:22 | 054346c | k=1.5xsup [itm,anchor] | 483/896 returned true | mean yield 0.45 | mean prec 0.32 | q00:3/12 q01:7/26 q02:7/21 q03:152/244 q04:151/248 q05:149/294 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |
| 2026-07-30 17:25 | 054346c | k=1.5xsup [itm,anchor] | 361/704 returned true | mean yield 0.37 | mean prec 0.30 | q00:2/12 q01:4/26 q02:3/21 q03:134/223 q04:130/235 q05:75/136 q07:5/18 q08:7/27 q09:0/3 q10:1/3 |
| 2026-07-30 17:26 | 054346c | k=1.5xsup [itm,anchor] | 257/565 returned true | mean yield 0.28 | mean prec 0.25 | q00:0/12 q01:4/26 q02:2/21 q03:123/211 q04:99/200 q05:17/44 q07:4/18 q08:7/27 q09:0/3 q10:1/3 |
| 2026-07-30 17:28 | 054346c | k=1.5xsup [itm,anchor] | 193/467 returned true | mean yield 0.22 | mean prec 0.21 | q00:0/12 q01:3/26 q02:1/21 q03:103/180 q04:74/163 q05:3/14 q07:3/18 q08:5/27 q09:0/3 q10:1/3 |
| 2026-07-30 17:31 | 054346c | k=1.5xsup [itm,anchor] | 483/896 returned true | mean yield 0.45 | mean prec 0.32 | q00:3/12 q01:7/26 q02:7/21 q03:152/244 q04:151/248 q05:149/294 q07:7/18 q08:7/27 q09:0/3 q10:0/3 |


## QbE sprint, 2026-08-02 (autonomous overnight run)

Metric: 5 evenly-drawn support seeds per query, `k = ceil(1.5 x support)`,
`yield = true/support`, `prec = true/returned` (ungraded counts FALSE),
`prec_g = true/judged`. Truthset is pool-limited, so `prec` understates
and `prec_g` is the honest precision figure.

| | q03 stove | q04 close | q05 open |
|---|---|---|---|
| support | 247 | 165 | 196 |
| **yield** (was 0.64 / 0.76 / 0.83) | **0.75** | **0.94** | **0.90** |
| **prec_g** | 0.82 | 0.92 | 0.99 |
| yield == prec at k=support | 0.60 | 0.83 | 0.85 |
| yield, mean of 4 independent seed draws | 0.735 | 0.913 | 0.883 |

### What moved it, in order of size

| change | effect |
|---|---|
| LOO retrieval quality replaces Cohen's d as the channel weight | coherence names the wrong channel 4 times in 15 seed groups |
| z-score fusion replaces RRF | RRF discards the margin, the evidence of confidence |
| `otsu_cut` replaces `confidence_cut` | the knee fired at 16 returned of a 165 support |
| `drop_pc` 1 -> 0 | corrected for RRF's rank-blindness; distorts under z-fusion |
| pairwise instead of hold-one-out quality | 5 seeds give 5 samples to judge a channel on and it mis-selects ~1 group in 5; every ordered pair gives 20 |

### Measured dead ends (each cost a run, each stays recorded)

- **Fine-grained appearance channels** (per-frame DINOv3, per-track
  object sets, significant-participants-only, set-matched SigLIP2):
  0.34-0.41 on q03, no better than the pooled channel each replaces.
  This corpus is 2,097 episodes of one kitchen, so appearance
  similarity is near-constant and no appearance channel can separate it.
- **The answer join as a reranker** inside a high-recall channel pool:
  0.63 against 0.84. Precise but adds no ordering the channels lack.
- **PRF / Rocchio**: hurts at every expansion size and both fusions.
- **Query as a set** (max over seeds instead of centroid): 0.780 vs
  0.845. Seeds of one query type denoise by averaging; frames of one
  episode do not.

- **IV2 at sub-episode granularity** (3 overlapping windows, 6,291
  clips, 31 min): 0.72 against the whole-episode channel's 0.77. For a
  query about a whole demo's story, four frames SPANNING it beat four
  frames inside a third of it. Context beat density; retired.
- **IV2 at 8 frames over the same full span** (19 min; required
  interpolating two temporal position embeddings, since the checkpoint
  ships a 4-frame grid): **0.76 against 0.77**. Doubling the input
  changed nothing. Together with the windows result this settles it:
  IV2 is NOT frame-starved on this corpus, and q03's ceiling is the
  model's semantics, not how it is fed. `iv2.set_num_frames()` is kept
  as a capability; the channel is retired.

### The remaining gap is one channel on one query

`iv2` is the only channel that carries q03 (0.77 alone against 0.34-0.41
for every other). q04 and q05 are at target; q03 sits at 0.75 and its
own single-channel ceiling is 0.77, so lifting it further needs a
stronger video-semantic encoder, not another combiner.

### The holdout that exists: mechanism generalisation (2026-08-02)

The overnight mechanism choices were made by test-and-keep on q03/04/05
only — ordinary model selection, with its ordinary risk of tuning to
those three queries' quirks. The low-support queries were never
consulted in any decision, so they are a (noisy) blind set. Old
mechanism vs new, same harness, same seeds:

| mean over q00/01/02/07/08 | yield | prec_g |
|---|---|---|
| uniform RRF | 0.139 | 0.29 |
| coherence + weighted RRF (old) | 0.160 | 0.40 |
| pairwise-LOO + z-fusion (new) | **0.186-0.194** | **0.39-0.43** |

Better or equal on 4 of 5 held-out queries (q07 0.17→0.28, q08
0.11→0.17). The selection generalised; it did not merely memorise the
three queries it was tuned on. Caveat: supports of 8-18 make every one
of these numbers wide.

Full shipped-path table, all 11 queries: q00 0.00, q01 0.29, q02 0.23,
q03 0.74, q04 0.94, q05 0.91, q07 0.25, q08 0.19, q09/q10 0.00 (support
2 — no kind to learn from two examples; correct behaviour is the
no-match gate, not retrieval).

### Seed sensitivity, reported rather than exploited

q03 rises to 0.79 with 15 seeds while q04 and q05 peak at 5 and fall -
appearance denoises by averaging, direction blurs. Choosing per query
would be fitting the answer, so the bench uses 5 everywhere.


## Text vs QbE, all queries (2026-08-02, lake/fresh_bench)

Yield / prec_g per query. QbE = shipped `search_like` (5 seeds).
Text = `search_set` — DEGRADED RUN: its `obj` channel died on schema
drift (`Field "motion" does not exist`; the instances table predates
the DINOv3 rebuild), fix queued. Text elides 99.3% of corpus bytes at
~1s/query.

| q | support | text yield/prec_g | QbE yield/prec_g |
|---|---|---|---|
| q00 green | 8 | 0.12 / 0.25 | 0.00 / 0.00 |
| q01 yellow | 17 | 0.12 / 0.12 | 0.29 / 0.42 |
| q02 red | 14 | 0.00 / 0.00 | 0.23 / 0.97 |
| q03 vessel→stove | 247 | 0.38 / 0.79 | **0.74 / 0.82** |
| q04 close drawer | 165 | 0.39 / 0.89 | **0.94 / 0.92** |
| q05 open drawer | 196 | 0.29 / 0.86 | **0.91 / 0.99** |
| q06 no-match gate | 0 | PASS | — |
| q07 lid | 12 | 0.00 / — | 0.25 / 0.36 |
| q08 spoon | 18 | 0.11 / 0.25 | 0.19 / 0.48 |
| q09 eggplant | 2 | 0.00 | 0.00 |
| q10 banana | 2 | 0.00 | 0.00 |

## The low-support 0.60 target: status and the measured wall

Three query-time mechanisms built and measured today for the object
queries (all corpus/seed-derived, no content knowledge):

1. consensus x rarity over the object index, threshold = the store's
   fitted identity cut: DEAD (0.00-0.08). The cut is an INSTANCE bar;
   object KINDS in one kitchen live at 0.7-0.85 cosine, below it.
2. both sides restricted to EVENT-BOUND (manipulated) objects:
   separation gap improves to +0.07..+0.14 median - real but thin.
3. rank-based consensus (pairwise-LOO at the object level, no
   threshold): DEAD alone (0.01-0.08); and its degenerate scores
   POISONED pairwise channel selection in the lab (0.186 -> 0.051)
   while LOO-auc selection held (0.193) - recorded as a selection-
   robustness finding.

Root cause, measured: the DINOv3 track descriptors are instance-ReID
features on small crops; best cross-seed object match (med 0.75-0.85)
barely exceeds background episodes (med 0.70-0.72). No query-time
mechanism can retrieve at top-1% precision through a 0.1 gap. The
0.60 target on supports of 8-18 requires WRITE-PATH work: object
descriptors that separate kinds (native-resolution crops at track
close; a kind-contrastive head over them). Same wall the answer join
hit, now measured from a third direction.


## The return boundary: LOO-calibrated cut ships (2026-08-02)

User correction that drove it: k is a CEILING, not a target — the test
at k=1.5x support is whether the system returns close to SUPPORT. The
Otsu cut was failing open (q03 returned all 371). Shipped replacement:
hold each seed out, fuse with the rest, record where the held-out TRUE
item ranks; return down to the deepest such rank, corrected by the
maximum-of-uniforms factor (m+1)/m. Order statistics of the query's own
members — no truthset, no knob. Otsu remains the <3-seed fallback.

Shipped path, mean of 4 independent seed draws (ret vs support in
parens):

| | q03 (sup 247) | q04 (sup 165) | q05 (sup 196) |
|---|---|---|---|
| yield | 0.69 | 0.86 | 0.83 |
| prec | **0.52** (was 0.50) | **0.70** (was 0.67) | **0.74** (was 0.71) |
| prec_g | 0.83 | 0.93 | 0.99 |
| ret | 306-371 | **190-225** (was 232-248) | **220-251** (was 243-294) |

Precision and ret-to-support improved on every high-support query;
yield gave back 0.03-0.05 at the boundary (the ranking is unchanged —
the cut moves along its curve; both can only rise together with a
better ranking, whose ceilings are measured above).

Also recorded: the lab's consensus channels GAME pairwise selection by
construction (they optimise "rank sibling seeds high" — the selection
statistic itself) and collapsed fusion 0.85 -> 0.32 while enrolled.
Gated off. A channel must be built from evidence independent of the
selection statistic.

## 2026-08-02 — sim_chains store: second corpus, untouched mechanism

`lake/sim_chains` built from data/sim_chains mp4s, PIXELS ONLY (one
file = one demo; eval sidecars never opened). Same store-side builders
as the kitchen corpus, zero code changes: 150 episodes, 38,077 frames
(frame_vectors 1:1), 1,309 events (95% object-bound, 100% agent-bound,
14 transition types DISCOVERED from sim descriptors), 21,492 objects,
1.09M trajectory samples, identity cut FITTED from sim recurrence
(0.946 vs kitchen 0.894). Channels: scene/motion/iv2/sig2. 472MB,
~35 min wall.

QbE smoke — template identity (pure chain-shape, 5 seeds, k=1.5·sup,
chance prec ≈ 0.14):

| query               | yield | prec | top channel |
|---------------------|-------|------|-------------|
| swap                | 0.20  | 0.13 | motion 0.21 |
| precarious          | 0.55  | 0.37 | iv2 0.20    |
| push_then_build     | 0.10  | 0.07 | scene 0.25  |
| build_unstack_move  | 0.15  | 0.10 | iv2 0.23    |

Reading: only precarious (4 blocks, towers — a VISIBLE class) clears
chance; the rest sit at it. This is by construction: episodes share
one scene and differ only in event ORDER, and colors/zones are
deliberately decorrelated from templates. Appearance-era channels
cannot see chain structure — the measured gap the contextual-retrieval
work (events/trajectories joins over the same store) now has to close.

### Write-vs-truth audit (scripts/verify_sim_store.py, 2026-08-02)

| layer | verdict |
|---|---|
| A episodes | 150/150 frame-exact |
| B event timing | 0.95 of store events land on a truth primitive; 0.71 of primitives covered |
| C event types | purity 0.40 vs 0.42 majority baseline - types encode motion profiles, not primitives |
| D identity/binding | BROKEN here: 2.67 identities per true block; 8% of same-block event pairs share an id (fitted cut 0.946 oversegments) |
| E objkind | vectors DO see kind: 0.79 AUC vs pixel-colour labels; the 0.51 via truth labels was D's corruption |
| F channels | weak-real: iv2 same-n-blocks 0.615, same-template 0.559; motion/scene ~chance |

The chain reads bottom-up: everything above identity is healthy,
everything downstream of identity inherits its fragmentation. The
identity fit is the one write stage that did not transfer.

### Identity fix on sim (2026-08-02, commits 28abb3c, 66958fe)

Fixed: refit tool's 512/768 reshape scrambler (proven-same at 0.045
cosine was the tell); recurrence objective replaced (gameable by
fragmentation on lookalikes); identity now two-stage - handoff
continuity components + negative-calibrated component merge.

| metric | before | after |
|---|---|---|
| objects | 21,492 | 3,889 |
| false-merge (proven-diff pairs) | 0.452% | 0.24% |
| events object-bound | 95% | 100% |
| objkind labelled tracks | 2,781 | 17,410 |
| ids per true block (truth audit) | 2.67 | 2.65 |
| same-block event id consistency | 8% | 8% |

The consistency line is the honest one: consolidation succeeded but
binding across a CARRY did not move, and the reason is now measured,
not suspected - the same block's ids across a carry sit at 0.54 median
cosine (4.7% mergeable at the calibrated cut; a 0.60 cut merges 38%
while false-merging the corpus), and geometric agent-bridges fire 4
times in a 27,680-interval proposal soup. Identity-by-appearance is at
its ceiling on this corpus; the remaining link is EVENT-MEDIATED (the
event knows its mover before and after the carry) - the join layer's
next mechanism, with the sim truthset as its gradeable target.

### Binding + identity, final state (2026-08-02, commits 28abb3c..f0782ca)

| metric (truth audit) | start | final |
|---|---|---|
| objects | 21,492 | 3,889 |
| identities per true block | 2.67 | **1.76** |
| same-block event id consistency | 8% | **45%** |
| events object-bound | 95% | 100% |
| kind AUC via truth labels | 0.51 | 0.57 (0.79 via pixel-colour control) |

What moved it: two-stage identity (continuity components + calibrated
merge) x agent-appearance veto (4,197 arm-fragment tracks excluded from
mover candidacy) x contact-gated mover selection. What is measured and
still open: absolute mover correctness - the colour control puts the
bound object at the true mover's colour only ~13-17% (small-n; most
bound tracks are not colour-labelable crops), so binding is now far
more COHERENT (same id for the same block) while often still anchored
to a nearby proposal rather than the block crop itself. Next front:
candidate quality at the grip point, graded on the sim colour control.

The user's cosine challenge, answered with numbers: instance matching
of pooled crop descriptors across positions is genuinely weak (same
block across a carry 0.60 vs different blocks 0.40, AUC 0.72; pooled
interval descriptors invert entirely - same 0.54 vs different-tail
0.86, context domination). Cosine is not the sin; the pooled, context-
dominated representation is - and no cut on it can both bridge carries
and keep instances apart. Continuity and agent-mediated structure, not
appearance, carry within-episode identity.

### Mover-binding criterion ladder (2026-08-02, f0782ca..ceff4c6)

Each rung added against a filmed failure; graded on the sim colour
control (bound object's pixel colour vs true mover) + truth audit:

| criterion | mover acc | colourable n | same-block consistency |
|---|---|---|---|
| motion concentration | 17% | 53 | 8% |
| + contact gate | 5% | 20 | - |
| + agent-appearance veto | 13% | 39 | 45% |
| + rest-backed objects | 12% | 65 | 50% |
| + agent-size + held-inside gates | 15% | 101 | 45% |

Read: coherence is transformed (8% -> ~45-50%) and the bound objects
are increasingly real (colourable coverage x5), but exact mover
selection plateaus ~15% under criterion tweaks. Filmed junk classes
eliminated in order: arm shadows, arm fragments, transients,
table-sized proposals. The residual is WRITE-PATH work: proposer
quality around the gripper, and event-geometry (vanish/appear site)
disambiguation. Also on record: the criteria are corpus-agnostic
principles, but their selection was eval-guided against the sim
control - same standing as the QbE mechanism program.

## 2026-08-02 — Chain retrieval: mechanism validated, elements attributed

chain_qbe.py: episode = event sequence (instance identity abstracted
to SLOTS by first-appearance order), similarity = length-normalised
Needleman-Wunsch, same seeds/k-ceiling as the appearance smoke.

| configuration | swap | precarious | push_build | build_unstack |
|---|---|---|---|---|
| appearance fusion (baseline) | 0.20 | 0.55 | 0.10 | 0.15 |
| chain: store kind-only | 0.25 | 0.35 | **0.55** | 0.20 |
| chain: + slots + motion | 0.20 | 0.20 | 0.20 | 0.15 |
| **chain: ORACLE (truth prims)** | **1.00** | **1.00** | **1.00** | **1.00** |

(yield at k=1.5*sup; oracle prec 0.67 = every truth in the ceiling.)

The reading, end to end: contextual retrieval by chain alignment WORKS
- perfect recall with correct event tokens - and the entire remaining
gap is write-path element quality, now attributed per component:
event TYPING (types at majority-baseline purity vs primitives) is the
biggest lever; slots wait on binding coherence (45%); motion carries
nothing for chain shape. The write-path targets now have an
end-to-end consumer metric.

### Event-typing attempt 1: agent-kinematic types (2026-08-03) - negative

Descriptor: hold-state onset/offset + descent/rise shape + stop-height
percentile from the agent trajectory (geometry only, no text, types
discovered). Chain-benchmark verdict: 0.24-0.28 mean yield vs 0.34 for
the original screen-space types; store restored to the originals via
the events log (v12). Findings: 12-d kinematic manifold clusters as
100% noise (HDBSCAN); compact subsets type only ~30% dense cores;
boundary-kind alternation in the original tokens carries structure the
whole-event profile collapses. Next: token-ise hold-state ALTERNATION
(acquire/carry/release segments) instead of clustering event profiles.

### Event-typing attempt 2: hold-alternation tokens (2026-08-03) - negative

Mean segments/episode = 1.0: no alternation exists in the derivable
hold signal, because `contact` = overlap with the WHOLE-ARM agent box
(49% of participant samples fire) and no element records the gripper.
CONVERGENT CONCLUSION with the mover-binding plateau: the write path
needs a HAND sub-element (end-effector position + touch state) in
trajectories. One requirement now gates both open fronts and the
chain oracle gap (0.34 measured vs 1.00 ceiling).

## 2026-08-03 — Chain-QbE 0.90 run: single-view ceiling measured

Target: chain-template yield >= 0.90 (oracle: 1.00). Iteration ladder,
each rung audited against sim truth (film or truthset):

| configuration | dev mean yield | holdout |
|---|---|---|
| event kinds (best prior) | 0.337 | - |
| motion segments, first cut | 0.250 | 0.225 |
| + vetoes, per-track speeds | 0.175 | 0.200 |
| + windowed net displacement | 0.212 | 0.175 |
| + scale-anchored threshold | 0.175 | 0.250 |
| + rest-backed participants (otsu) | 0.213 | 0.150 |
| + ON-relation qualifiers | 0.312 | 0.200 |
| + colour slots (rest crops) | 0.262 | **0.350** |
| z-fusion (moves+kinds+appearance) | 0.337 | 0.300 |

Segmentation is structurally right (6.9 segs/ep vs 5-8 true, 91% of
segments on a real manipulation). The walls, each measured: carries go
BLIND inside the gripper (31% of manipulations fully covered; grasp +
release fragments rejoin by no available signal); single-view image y
irreducibly confounds tower height with table depth (ON-relations at
noise); identity across a carry 0.54 cosine; rest-crop colours
contaminated. Fusion does not lift (correlated failures).

VERDICT: ~0.34 is the single-view element ceiling; 0.90 requires
TWO-VIEW GEOMETRY. Every episode's second camera is already on disk;
cross-view time sync is free (shared clock). The build: ingest the
second stream, associate rest-crops across views (appearance, no
text), estimate per-episode table homography from static structure,
triangulate heights -> real ON-relations + end heights + repaired
slots -> re-run this ladder. Oracle says the mechanism is waiting.

## 2026-08-03 — Two-view program: 0.90 reduced to one discriminator

Two-view store rebuilt (76,154 frames, both cameras as streams on the
shared clock; nothing about poses/intrinsics anywhere). Findings:

| result | number |
|---|---|
| manipulation coverage (cross-view union) | 0.31 -> 0.49 |
| push-vs-carry flag (track-survival signature) | 0.95 |
| degraded oracle: slots + push flag ONLY | **0.992** |
| colour-slot per-position acc (under pairing) | 0.78 |
| best whole-pipeline yield so far | 0.34 |

The decisive discovery: ON-relations (place/stack/unstack geometry) -
which consumed most of the iteration budget - are UNNECESSARY. The
0.99-capable token is slots + push flag. Every component of it now
works except one discriminator: carry-fragment vs same-block
re-manipulation (colour pairing over-merges, anonymous gap-rest
under-merges; the fix is SAME-COLOUR gap-rest - one crop comparison
per candidate merge). That single mechanism stands between 0.34 and
the oracle-backed 0.99.

### Cap analysis (2026-08-03): unit structure is the wall

| configuration | dev | holdout |
|---|---|---|
| truth units + truth slots (oracle) | 0.99 | 0.99 |
| REAL units + truth slots | 0.40 | 0.43 |
| real units + real slots (best) | 0.29 | 0.35 |

Real-unit structure (6.1 noisy units/ep: carry fragments, collateral
topple units, boundary noise) caps the pipeline at 0.40 even with
PERFECT slot and push labels. All slot/colour polish sat above this
bottleneck. Segmentation quality is the singular next lever, graded
by re-running this cap. Gray edges closed: pairing window corpus-
fitted (P95 unit duration), no truth constants anywhere in output.

### Gap-pairing decidability panel (2026-08-03) - closed negative

Class-labelled audit of every available signal for "is this blind gap
one carry or two manipulations" (770 candidates, base rate 0.30):
ellipse excess via arm centroid (merged 484 but cap DROPPED 0.40 ->
0.29), via box-nearest-point (zero signal - the arm box covers both
foci), closure/rest-birth (precision 0.32 = base rate), rest-presence
at either anchor (~0.49), duration/distance/speed (0.35-0.65). The
pairing decision is NOT DECIDABLE from trajectory elements - a
systematic negative that redirects the 0.90 program to either a
boundary-free representation (slot-activity timelines, agent-colour-
suppressed crops) or a write-path tracker that holds the carried
block through the gripper.

### Route 1: slot-activity timeline (2026-08-03) - closed negative

Boundary-free representation (per-0.5s "which colour moves" timeline,
agent-colour-suppressed crops): benchmark 0.24 dev / 0.33 holdout,
coloured-bin coverage 0.63 - and the audit that explains it: per-bin
colour matches the TRUE mover 14% of the time. At-rest colour reads
measured 0.78 earlier; in-motion reads fail even with suppression,
and the strongest-mover-per-bin selection lands on proposal junk.
The representation dissolves the pairing problem as designed, but no
representation survives 14%-reliable primitives when the oracle needs
~95% sequence fidelity.

FOUR INDEPENDENT NEGATIVES now converge (mover binding plateau ~15%,
kinematic typing, gap-pairing decidability panel, in-motion colour):
the write path's per-moment mover identification is the entire
remaining gap. Route 2 - a tracker that HOLDS the carried block
through the gripper, and cleaner proposals around the arm - is the
one open road to chain-QbE 0.90. Gates unchanged: cap analysis
(units + oracle slots -> toward 0.99), then the benchmark.

## 2026-08-03 — Route 2 lab-side driven to bedrock

Rest-ledger (object permanence as bookkeeping): the design dissolves
the unpairable blind gap by construction, and each sub-iteration
audit named the next stratum down - junk entries (46 vs 8 real/ep-
view), an achromatic entry-colour space, lighting-broken within-
episode colour constancy (smooth pair-distance decay, no
bimodality; absolute classification survives what relative
clustering cannot), and finally ENTRY UNDER-COVERAGE: a block's
second resting spell usually produces no entry (transitions 0.3/ep
vs 3.5 true). Benchmarks throughout: 0.19-0.26.

FLOOR OF THE PROGRAM, now reached from every direction: per-frame
object detection quality at the write. The generic proposer's soup
(shadows, glare, arm fragments, missed blocks) caps every downstream
mechanism - binding, typing, segmentation, timelines, ledgers. Next
project: detection/tracking surgery in the write path (build_dinov3
proposer layer), then re-run this entire measured ladder, whose
instruments (oracle, cap analysis, decidability panels, colour
controls) are all built and waiting.

### Vision-native remediation (2026-08-03) - rule sharpened, floor confirmed

User sharpened the rule: hand-engineered appearance features (even
numeric - Lab, chroma) are non-compliant; only the learned encoder's
space is vision-native. Ledger rebuilt accordingly (encoder entry
signatures, agent+background gallery junk filter, discovered kind
codebook). Result replicates bedrock exactly: 4 kinds / 0.78
unclustered, 0.2 transitions/ep, benchmark 0.23/0.25. Compliance
sweep: no text or hand features remain in any path candidate;
Lab-based scripts marked as negative-result records; palette/name
anchors exist only in eval graders. The floor - write-path detection
quality - is unchanged and now measured rule-clean.

## 2026-08-03 — Route 3: persistent-change detection (chain_delta.py)

**Principle.** Every dead route watched the block WHILE IT MOVES
(14–40% reliable). A manipulation instead leaves a PERSISTENT mark:
guard-banded temporal-median states before/after each grid time,
illumination-invariant RGB-angle differencing (a shadow scales the
surface vector; a block replaces it), per-grid components associated
into events, bidirectional state-persistence validation (the arm's
poses pass any contrast bar but never persist). Direction from a
fitted dominant-surface model (3-means over the episode-median image
+ channel-ratio-uniformity for table-hued blocks); identity is pure
OBJECT-PERMANENCE BOOKKEEPING — per-(view,spot) LIFO stacks, exact for
stacking, union-find slots, serial dep→next-arr pairing. No appearance
clustering anywhere; every cut corpus-fitted.

**Measured en route (all recorded in the code):** |RGB| diff drowns in
the arm's table-wide shadow; a global change mask merges the arm's
park envelope with every spot it visits (one 45%-of-frame site);
per-grid components without persistence = 139 junk events/view; BOTH
store DINOv3 encoders are colour-blind on block crops (cyan–blue
0.85–0.90 ACROSS colours vs 0.73–0.84 same-object — DINO's
colour-jitter augmentation), MAE is colour-aware (same-red 0.78,
red–cyan −0.10) but absolute gallery/cluster machinery fails at scale
regardless (hand-picked 7/7 → 24/75); asymmetric after-guard: negative
(arr 439 vs 440, DEV 0.30 → 0.24).

**Extraction vs truth (eval-side graders):** manipulations 3.1/ep vs
3.5 true; cast 2.8 vs 3.0; slot pairwise agreement 0.88. BUT the
median episode has 2 manipulations (16 episodes have 0) — per-episode
variance, not the mean, is what retrieval feels.

| tokens (from one extraction) | DEV yield | HOLDOUT yield |
|---|---|---|
| slots w1.0 (main) | 0.300 | 0.200 |
| slots w0.5 | 0.312 | 0.200 |
| no slots (w0) | 0.363 | 0.150 |
| reuse-lag slots | 0.287 | 0.200 |
| + ratio-uniformity surface test | 0.275 | 0.150 |

**Verdict.** Best fully-compliant number to date (prior floor
0.225/0.250) and the first route whose aggregate extraction stats
match truth — but the 0.90 gate needs ≥95% per-episode event recall
and the arrival-side deficit (arr ~440 vs dep ~655 corpus-wide)
survived four candidate explanations (surface cut, shadow masks,
orange-hue collision, after-guard). Open front: event association /
persistence interplay at set-down moments, or a real segmenter-tracker
(SAM-3 video) at the write. Token design is NOT the bottleneck
(ablation flat); slot mechanism is sound (0.88 agreement).

## 2026-08-03 — The one-push battery: channels, selection, recall fix, cross-view

One comprehensive push per user directive (no more one-at-a-time):
stage-graded diagnosis, then every candidate signal built and measured
in a single sweep. chain_channels.py + chain_xview.py + chain_delta
bounded persistence.

**Stage diagnosis (measured, 140 truth set-downs):** raw components
1.00 → grouped 0.99 → persistence 0.70. Event-BOUNDED persistence
(state persists until the next event at that spot; additive OR so it
only adds) took set-down recall to **0.91**; extraction then reads
3.8/ep vs 3.5 true, cast 3.4 vs 3.0. But pairing is brittle BOTH ways
- the recovered events broke LIFO slots 0.88 → 0.84 and tokens fell to
0.20-0.25. Symbolic stage errors COMPOUND; the 0.992 oracle needs
every stage ≥0.95.

**Channel battery (11 channels, DEV / HOLDOUT yield):** iv2 0.31/0.28,
sig2 0.30-0.31/0.30-0.35, fdnnv 0.16-0.18/0.25-0.30, scene DTW
0.26/0.35, motion 0.20/0.33, novelty profile (change rhythm,
layout-invariant) **0.36/0.45** - best single, tokens 0.20-0.24/
0.15-0.25, raw event stream 0.24-0.25/0.23.

**Seed-LOO selection (the fresh_bench recipe):** DEV 0.36-0.41,
HOLDOUT 0.40-0.43. Different templates genuinely pick different
channels (swap→tokens, precarious→sig2, push_then_build→scene/
novelty) - plurality works, ceiling ~0.45.

**Cross-view contrastive (the last structural bet):** tiny conv+GRU,
InfoNCE simA↔simB, trains to 296/300 cross-view top-1 - and retrieves
at 0.25/0.38. Instance discrimination collapses to episode timing
fingerprints, not template structure, at 150 episodes.

**Where this leaves the 0.90 gate.** Best compliant numbers: ~0.42-
0.45 (selection / novelty), ≈3x chance, 2x the token route. Everything
cheap is now measured and banked. The remaining road is narrow and
named: pairing robustness (soft matching over the 0.91-recall event
sets instead of LIFO+serial), arrival/departure classifier balance
(523 vs 745), cross-view event fusion. fresh_bench's 0.90 was
category-shaped queries; chains demand sequence+identity fidelity that
no fixed channel measured today carries.

## 2026-08-03 — "Don't stop" push: the terminal measurement

Sequence of builds, each measured: seriality DP parse (max-weight
alternating dep→arr with skip costs — flat 0.10-0.30); fusion recipes
over 12 channels (RRF / softmax-LOO / PRF — best frozen DEV 0.36,
HOLDOUT 0.40; PRF hurts, weak channels poison pseudo-seeds);
episode-profile bag features (0.19-0.20); soft persistence + peak-
pixel states (pool recall of SOME event near every anchor: setdowns
0.98, picks 1.00 — at 42 events/ep, 10:1 junk).

**The grading lesson, learned four times then fixed:** every position-
blind grader (nearest-in-time, existence-in-window) counted co-timed
arm/shadow junk as recall. chain_grade.py grades with PALETTE-
GROUNDED position verification (does the event's clean crop contain
the truth block's colour on the correct side) — eval-side only.

**Terminal numbers (position-verified):**
- TRUE set-down recall of the detection layer: **0.48** (250/525).
- Feature AUCs vs verified labels: ratio_max 0.678, area 0.591,
  everything else ≤0.55 — no corpus-side feature separates junk.
- Same-block crop matching: DINOv3 0.538, MAE 0.547 (masked crops),
  MAE 0.522 (clean native crops) — identity unreadable at this crop
  quality regardless of encoder or preparation.
- **CEILING: verified events + TRUTH slots through the bench = DEV
  0.225 / HOLDOUT 0.350.** At 0.48 recall no downstream machinery can
  reach 0.90; the oracle needs ≥0.95 per stage.

**Best standing result: channel selection DEV 0.41 / HOLDOUT 0.43
(novelty single-channel 0.45)** — ~3x chance, 2x the token route.

**The one door left** (unchanged from the session's first diagnosis,
now proven at every layer): rebuild per-event detection at the write
to ≥0.95 position-verified recall — segmenter-tracker class (SAM-3
video) machinery, gated by chain_grade.py which now exists exactly
for that.

## 2026-08-04 — Pivot: domain-blind window sequences (native/seq*)

Owner-directed rethink: event grammars are TABLETOP grammar, not
physics — a product ingesting arbitrary robotics video (driving, ego,
warehouse) cannot ship hand-derived or sim-trained event parsers, and
two generations of fitted-cut cascades failed the same way. New write
path: frozen pretrained V-JEPA 2 ViT-L over 2s/1s-stride windows;
an episode IS its window-vector sequence; retrieval = temporal
alignment. No events, no elements, no fitted semantic cuts, nothing
domain-specific anywhere. Embed cost 12s/ep (sim), 3.1s/ep (kitchen
demos), MPS fp16.

**Sim (adversarial-structure corpus, the structural floor):**
pool 0.28/0.23, dtw 0.26/0.30, **delta-DTW 0.30/0.43**, novelty
0.24/0.25 (DEV/HOLDOUT yield). Matches the known ~0.45 ceiling for
every non-symbolic representation on this corpus — sim_chains was
BUILT appearance-confusable; it measures the extreme that belongs to
the cloud reranker tier, not the fast path.

**G1 tracker-substrate record (native/proto.py, banked):** two-stage
CoTracker3 (scout + segm_mask dense) 17s/view; recall 1.00/1.00
attainable unfiltered; best filtered 0.74/0.80 recall, 0.20 prec,
0.74 slots after 11 fitted-cut iterations — the hand-derived layer
oscillates exactly like its pixel-era predecessor. Event grammar
retired per owner directive; tracker survives as an optional generic
motion descriptor only.

Kitchen (real-video product truth) running: protocol-exact QbE mirror
(native/seqbench_kitchen.py) vs committed channel-fusion 0.94/0.90.

## 2026-08-04 — Kitchen (real video): window-sequence QbE measured

Protocol-exact mirror of bench_qbe (same truthset, seeds, k). Single
domain-blind channel, frozen V-JEPA 2 window sequences:

| variant | q03 | q04 | q05 | note |
|---|---|---|---|---|
| delta-DTW (temporal alignment) | 0.38 | **0.61** | **0.72** | best on motion-heavy queries |
| pooled (no temporal shape) | 0.39 | 0.37 | 0.38 | alignment DOUBLES q04/q05 |
| committed 8-channel fusion | — | 0.94 | 0.90 | the shipped system |

All-query mean (dtwd): 0.218 — low-support needle queries (q00,q01,
q08-q10) stay ~0, same as every single channel ever measured.

**Verdict.** Temporal alignment over pretrained window sequences is
REAL — it doubles pooled performance on manipulation queries (0.37→
0.61, 0.38→0.72) with zero domain knowledge, zero training, one
encoder, and it transfers unchanged between sim and kitchen (the same
two scripts produced both corpora's numbers). As a SINGLE channel it
does not replace the fused system (0.94/0.90) — it slots in as the
sequence-shape channel the fusion never had, and as the universal
default for corpora where no elements exist (any fresh upload). The
domain-blind write path is therefore: window sequences ALWAYS (works
on anything, day one), element channels only where a domain earns
them, cloud reranker for structure — which is the product architecture.

## 2026-08-04 — Memory index: domain-blind rebuild, measured to its ceiling

Goal: build the retrieval layer to working accuracy with pretrained
encoders only and zero domain knowledge (native/mem.py, memfuse.py).
Six frozen encoders in the store (DINOv3 33 win/ep, SigLIP2 8, V-JEPA
5 + our 2s/1s windows, motion 7, IV2 1) -> per-episode sequences ->
pooled + temporal alignment (batched DTW) + moment matching ->
all-channel fusion. No truth in any scoring path; FDNN student
excluded (retired by directive).

**Per-query yield, kitchen truthset, identical protocol:**

| query | support | SHIPPED 8-ch | domain-blind rebuild |
|---|---|---|---|
| q03 | 247 | 0.66 | 0.47-0.50 |
| q04 | 165 | 0.90 | **0.91** |
| q05 | 196 | 0.87 | 0.84-0.87 |
| q01 | 17 | 0.28 | 0.21 |
| q02 | 14 | 0.23 | **0.30** |
| q07 | 12 | 0.25 | **0.32** |
| q08 | 18 | 0.19 | 0.04-0.07 |
| q00 | 8 | 0.00 | 0.00-0.03 |
| MEAN | | **0.42** | 0.36 |
(q09/q10 support 2 are DEGENERATE - all positives become seeds, yield
is 0 by construction for every system. Not a failure mode.)

**21 fusion/aggregation variants measured over one score pass:**
all-channel fusion 0.35-0.36 >> seed-selected top-k 0.20-0.25 (the
inherited selection machinery is actively harmful at this scale);
consensus (mean over seeds) > max on large queries (q04 0.85->0.91);
temporal alignment > pooling (q04 0.59->0.91, q05 0.65->0.84);
moment-matching (max-sim / chamfer, aimed at needles) measured
NEGATIVE (0.27-0.32 vs 0.36) - needle failure is not a
temporal-averaging artifact.

**Conclusion (the ceiling of recombination).** Every strategy over
these channels lands in 0.20-0.36. Big-support queries are SOLVED by
domain-blind machinery (q04 0.91 matches the tuned 8-channel system,
q05 within 0.03). Needle queries (support 8-18) are NOT: 0.00-0.32
for every system measured, shipped included. The evidence itself is
the limit - all six channels score global scene/motion similarity, so
none can express "this specific thing happened". More fusion will not
fix it; different evidence must.

## 2026-08-04 — LONG-CONTEXT retrieval on sim: the wall, quantified

Sim is the long-context benchmark by construction: appearance,
placement and colour are randomised per episode, so two episodes of a
template share ONLY their chain of events. Every measurement below is
the frozen protocol (5 seeds, k=ceil(1.5*support), RandomState(0)).

| approach (all pretrained / domain-blind unless noted) | DEV | HOLDOUT |
|---|---|---|
| ORACLE event scripts through the aligner | — | **0.992** |
| novelty rhythm, single channel | 0.363 | **0.450** |
| channel selection top-2 (2026-08-03) | 0.413 | 0.425 |
| multi-encoder memory index (today, all-channel RRF) | 0.312 | 0.425 |
| V-JEPA window sequences, delta-DTW single channel | 0.300 | 0.425 |
| symbolic chain routes (delta / serial / ledger) | 0.13-0.30 | 0.13-0.25 |
| appearance baseline | ~chance | ~chance |

Bug found and fixed while measuring: the multi-encoder loader was
interleaving simA and simB vectors into one sequence by timestamp
(two views share the episode clock), scrambling the very structure the
index reads. Fixed to one sequence per stream; DEV moved 0.387->0.312,
HOLDOUT 0.350->0.425 (the pre-fix DEV was inflated by view-alternation
artefacts, i.e. it was measuring camera cadence, not chains).

**The quantified wall.** Fourteen distinct approaches - symbolic,
tracked, embedded, fused, selected - all land between 0.13 and 0.45,
while the ORACLE (correct event scripts + the same aligner) scores
0.992. Nothing perceptual has ever exceeded 0.45 on this corpus. The
gap is not fusion, not aggregation, not the encoder: on a corpus whose
episodes differ ONLY in event order and identity, retrieval requires
recovering the events, and no frozen global-similarity encoder
represents "which object moved when".

## 2026-08-04 — Motion-structure channel: probed, NOT built (negative)

Hypothesis: the oracle gap (0.992 vs <=0.45) is "which distinct thing
moved when", which pretrained tracking gives semantics-free - a bundle
is coherently-moving stuff, no agent/rest/pick vocabulary, as valid
for cars as blocks. Representation: per-window motion magnitude per
bundle-slot, slots ordered canonically by first motion; no thresholds
beyond fitted merges (native/motstruct.py).

Probe on 20 cached episodes, same-template vs different-template
separability (AUC), six similarity variants:

| variant | AUC |
|---|---|
| slot pattern, mean-removed | **0.631** |
| total motion profile | 0.625 |
| concurrent-count profile | 0.612 |
| slot activity binarised | 0.601 |
| number of active slots | 0.585 |
| DTW over full structure | 0.576 |

Reference, identical episodes and test: motion.dseq **0.675**,
scene.dseq 0.642, sig2.dseq 0.608, vjseq.dseq 0.521, all pooled
variants 0.48-0.56.

**Verdict: do not build.** The best motion-structure variant (0.631)
is BELOW an existing channel that costs nothing new (motion.dseq
0.675) and barely above scene.dseq. Tracking 130 more episodes (~40
min) to add a signal weaker than one already indexed is not
justified. The measured lesson is sharper than the negative: the
discriminative information on this corpus lives in the DELTA
(change-profile) view of channels we already have - every dseq
variant beats every pool variant - which is the same finding that
made alignment beat pooling on kitchen.

## 2026-08-04 — Two-tier architecture: precondition PROVEN, cheap Tier-1 REJECTED

**Tier-0 recall (the number that decides the architecture).** Kitchen,
2097 episodes, index = all-channel fusion of frozen encoders:

| query | sup | yield@1.5sup | recall@100 | @200 | @500 |
|---|---|---|---|---|---|
| q07 | 12 | 0.43 | 0.94 | 1.00 | 1.00 |
| q01 | 17 | 0.30 | 0.57 | 0.85 | 0.98 |
| q02 | 14 | 0.44 | 0.80 | 0.84 | 0.98 |
| q04 | 165 | 0.78 | 0.44 | 0.69 | 0.98 |
| q05 | 196 | 0.80 | 0.42 | 0.68 | 0.93 |
| q03 | 247 | 0.48 | 0.17 | 0.30 | 0.61 |
| q08 | 18 | 0.06 | 0.18 | 0.34 | 0.75 |
| q00 | 8 | 0.00 | 0.00 | 0.20 | 0.53 |
| MEAN | | **0.41** | 0.44 | **0.61** | **0.85** |

The answers ARE in the pool - on needle queries 84-100% of them sit in
the top 200 while final yield is 0.30-0.44. Ranking, not recall, is
the failure. Headroom for a reranker: 0.41 -> up to 0.85.

**Cheap Tier-1 (same encoders, no shortcuts) - NEGATIVE.** Full-
resolution sequences + symmetric late interaction + all-channel z
fusion over the top-200 (native/rerank.py):

| | q00 | q01 | q02 | q03 | q04 | q05 | q07 | q08 | MEAN |
|---|---|---|---|---|---|---|---|---|---|
| Tier-0 | 0.00 | 0.30 | 0.44 | 0.48 | 0.78 | 0.80 | 0.43 | 0.06 | 0.41 |
| +Tier-1 | 0.07 | 0.32 | 0.49 | 0.30 | 0.69 | 0.68 | 0.23 | 0.08 | 0.36 |

Helps needles (+0.02..+0.07), hurts large-support queries badly
(-0.09..-0.20). **Conclusion: spending more compute on the SAME
evidence cannot convert recall into precision.** Tier-1 must contribute
different evidence - a model that scores query and candidate JOINTLY
(cross-encoder / VLM), which is exactly the query-conditioned tier the
architecture reserves for read time. The index's job is settled and
measured: recall@200-500, cheap, universal, no domain knowledge.

## 2026-08-04 — TIER-1 cross-encoder (VLM) on needles: measured, NEGATIVE

Local judge: Qwen2.5-VL-7B-Instruct-4bit (MLX), 0.5 s/call, 3-6 frames
per clip, top-150 pool, kitchen needle queries (native/vlmrank.py).

**Scoring method matters more than the model.** Asked to rate 0-9 the
VLM answers "9" to almost everything: pairwise AUC **0.11**. Reading
the ANSWER TOKEN'S PROBABILITY instead (P(Yes) vs P(No)) recovers a
real signal: AUC **0.766** (q04), **0.562** (q01). Same model, same
frames - only the read-out changed.

**End-to-end, that signal is still too weak to rerank with:**

| variant | q00 | q01 | q02 | q07 | q08 | MEAN |
|---|---|---|---|---|---|---|
| Tier-0 index | 0.00 | 0.33 | 0.41 | 0.29 | 0.08 | 0.22 |
| VLM replaces ranking (3 grp) | 0.00 | 0.25 | 0.15 | 0.14 | 0.05 | 0.12 |
| VLM blended 50/50, 6 frames | - | 0.38 | 0.17 | 0.36 | 0.00 | 0.22 |

Replacing the index ordering costs -0.10; blending recovers to parity
(-0.02) with two real wins (q07 +0.14, q01 +0.04) and one real loss
(q02 -0.22). A single-seed-group run had suggested +0.25/+0.29 - it
did not survive averaging over 3 groups, and is recorded as noise.

**Reading.** The two-tier PRECONDITION holds (recall@200 0.84-1.00 on
these very queries, ranking headroom to 0.85), but a reranker only
helps if it is substantially better than the retriever it is
reordering. At pairwise AUC 0.56-0.77 this judge is not: six fused
encoders already order the pool better than it can. The tier is not
refuted - the LOCAL judge is. What it would take: a frontier video
model reading tens of frames (cloud-side, ~100 clips/query), not a
4-bit 7B image-VLM on 6 frames.

## 2026-08-04 — VLM descriptions at write: BOTH forms measured, both fail

Target restated (owner): yield AND precision > 0.90 at k = 1.5*support
where **k is a MAX BOUND, not a fixed return count** - abstention is
allowed and required, so both metrics can exceed 0.90 by returning
~support items that are nearly all true. (An earlier harness returned
exactly k, which forces prec = yield/1.5; that was a harness bug.)

**SIM (primary corpus, 6 templates, ~20 support each):**

| approach | yield | prec |
|---|---|---|
| ORACLE (true event scripts) | 0.99 | - |
| multi-encoder index, k=1.5 | 0.33 | 0.22 |
| VLM one description per episode, k=1.0 | 0.13 | 0.13 |
| VLM one description per episode, k=1.5 | 0.23 | 0.15 |
| VLM windowed descriptions + DTW, k=1.0 | 0.19 | 0.19 |
| VLM windowed descriptions + DTW, k=1.5 | 0.23 | 0.16 |
| abstention rules over the index (best) | 0.33 | 0.22 |

WHY the descriptions fail - inspected, not inferred. ep0 contains
orange/cyan/blue blocks; the local VLM reports "the yellow block is
lifted and placed on the brown block", calls the table a block, and
repeats the same sentence for different windows. The oracle's 0.99
needs the event script to be RIGHT; a 4-bit 7B VLM on 4x200px frames
per window cannot see which object is which.

**KITCHEN high-support with abstention (k=1.5 max bound):**

| rule | yield | prec |
|---|---|---|
| return full k | 0.676 | 0.450 |
| loo (seed-calibrated) | 0.664 | 0.458 |
| gap (largest score drop) | 0.334 | 0.728 |
| mad (median+3MAD) | 0.138 | 0.796 |

Abstention TRADES yield for precision along a curve; it cannot put
both above 0.90 because that needs the top ~support results to be
nearly all true, i.e. near-perfect RANKING. Best single-query result
today: q04 yield 0.78 prec 0.52 (returned 240 for support 160).

**Conclusion.** Sixteen approaches measured across two corpora. On
sim nothing exceeds 0.43 yield; on kitchen high-support nothing
exceeds 0.90 yield with precision above 0.52. The binding constraint
is perception quality: the retrieval half is proven (oracle 0.99), and
every local model available - frozen encoders, trackers, and a 4-bit
7B VLM - fails to recover what happened accurately enough to feed it.

## 2026-08-04 — SIM IS AT CHANCE: the number that reframes everything

Random-guess yield on sim at k=1.5*support is **0.207** (20 true among
145 candidates, k=30). Measured, every representation of every frozen
encoder:

| representation | yield | vs chance |
|---|---|---|
| random baseline | 0.207 | 1.00x |
| scene first-frame | 0.208 | 1.00x |
| scene evolution profile | 0.225 | 1.09x |
| scene end-minus-start | 0.250 | 1.21x |
| vjepa last-window | 0.258 | 1.25x |
| scene last / first+last | 0.283 | 1.37x |
| vjepa end-minus-start | 0.333 | 1.61x |
| full multi-encoder index | 0.333 | 1.61x |
| VLM descriptions (either form) | 0.19-0.23 | ~1.0x |
| ORACLE event scripts | 0.99 | 4.8x |

KITCHEN for contrast: random baseline 0.119, index q04 0.78-0.90 =
**6.5-7.5x chance**. Kitchen retrieval genuinely works; sim retrieval
has never engaged.

**Why.** sim_chains was built adversarially: colour, shape, placement
and camera are randomised per episode, so two episodes of one template
share NOTHING visual - only the order and identity of events. It is a
pure structural-reasoning benchmark, and structural reasoning at
database speed is exactly what no frozen encoder does. Real corpora
(driving, warehouse, drone) do not have this property: different
scenarios there differ in appearance too, which is why kitchen works.

**What would actually move it:** a general event-structure encoder
trained ONCE on diverse multi-domain robot video and shipped frozen -
zero-shot from the customer's point of view exactly as V-JEPA and
SigLIP are zero-shot, and distinct from training on a customer's
domain (which would not be zero-shot and would not transfer). Every
frozen encoder in this store exists because someone did that once.

## 2026-08-04 — DOMAIN-BLIND AUDIT: my earlier kitchen numbers were contaminated

Owner challenge: "just because you trained the model on domain data and
removed those domain info doesn't make it domain blind." Correct, and
the audit confirms it. Channel provenance in lake/fresh_bench:

| channel | meta | domain-blind? |
|---|---|---|
| scene_vectors | DINOv3, unit=frame | YES - every frame, uniform |
| sig2_vectors | SigLIP2 | YES - uniform windows |
| iv2_vectors | InternVideo2, 1/episode | YES |
| vjseq (native/seq.py) | V-JEPA2, 2s/1s windows | YES |
| **vjepa_part_vectors** | V-JEPA2, **unit=participant_track** | **NO** - tubelets from the domain object/event pipeline |
| **motion_vectors** | delta-appearance, 7.2/ep = per event | **NO** - events from the domain pipeline |
| frame_vectors | fdnnv (student trained on this corpus) | NO (already excluded) |

**Corrected numbers, pretrained-uniform channels only (k=1.5*support):**

| corpus / query | contaminated index | DOMAIN-BLIND |
|---|---|---|
| kitchen q03 | 0.47 | 0.50 / prec 0.34 |
| kitchen q04 | 0.91 | **0.68** / prec 0.45 |
| kitchen q05 | 0.84 | **0.78** / prec 0.52 |
| kitchen MEAN | 0.74 | **0.655** / prec 0.436 |
| sim MEAN | 0.333 | 0.333 / prec 0.222 (chance 0.207) |

So the honest zero-shot kitchen figure is **0.66 yield / 0.44 prec**,
not the 0.78-0.90 previously quoted: q04 loses 0.23 and q05 0.06 when
the participant-tubelet and per-event channels are removed. Sim is
unchanged (those channels never helped there).

CORRECTION OF RECORD: every earlier claim in this file that the index
"matches the shipped 8-channel system with zero domain knowledge"
overstated it - the match was partly carried by channels keyed to
domain-derived structure.

## 2026-08-04 — WRITE-PATH AUDIT: the retrieval unit itself is dataset metadata

Owner: "check the write path too, I don't think that's the only
contamination." Correct. Found in scripts/write_once.py:

```
def episode_spans(files):
    meta = pq.read_table(DATA / "meta/episodes/chunk-000/file-000.parquet",
                         columns=["episode_index", "length",
                                  ".../from_timestamp", ".../to_timestamp"])
```

**Every episode boundary in lake/fresh_bench is read from the Bridge
dataset's own metadata parquet** - from_timestamp / to_timestamp per
demo. The write path never segments video; it is TOLD where each demo
starts and ends, then renders one H.264 segment per demo with an IDR
each and separates them by a synthetic 60 s gap.

Consequences for every number this project has produced on kitchen:
1. The retrieval UNIT is oracle-segmented. A customer uploading a
   3-hour raw video has no such boundaries; the database would have to
   find them, and no measurement here has ever tested that.
2. Channels inherit it: iv2 is 1 vector per oracle demo; sig2 8 per
   oracle demo; native/seq.py windows are placed inside oracle spans.
   "Uniform windows" are uniform WITHIN a given segmentation.
3. The truthset is keyed to those same demo ids, so the benchmark can
   only ask "which demos match", never "where in the video".

sim_chains is cleaner on this axis - one generated mp4 per episode, so
the boundary is the file itself, which a customer upload also has - but
the corpus is synthetic and at chance for every method anyway.

**Honest status of the zero-shot claim.** Two contaminations now found
and recorded: (a) participant-tubelet and per-event channels [fixed:
kitchen 0.74 -> 0.655], (b) oracle episode segmentation [NOT fixable
by excluding a channel - it is the unit of evaluation itself]. The
kitchen figure of 0.655 yield / 0.436 prec is therefore still an
UPPER BOUND on true zero-shot performance, measured with segmentation
handed to the system for free.

## 2026-08-05 — FRESH SEGMENTATION-FREE STORES: the clean number

Both corpora re-written from RAW VIDEO with zero metadata
(native/rawwrite.py): uniform 2/4/8 s window grids, no boundaries
given, frozen V-JEPA2 per window. Retrieval returns TIME RANGES and is
graded by overlap with truth spans (native/rawbench.py). Metrics are
only yield and precision, k = 1.5*support as a max bound.

| corpus | yield | prec | chance |
|---|---|---|---|
| sim (long-context, 6 templates, support 20) | **0.208** | 0.194 | 0.207 |
| bench (high-support q3/q4/q5) | **0.296** | 0.303 | ~0.12 |

sim: 6,006 windows / 150 files. bench: 24,634 windows / 4 raw files
(235 min continuous, back-to-back demos).

**Sim is exactly at chance.** With segmentation removed, the last
apparent signal disappears: 0.208 vs 0.207 random. Everything the
segmented store showed on sim (0.33) was carried by knowing where
episodes began and ended.

**Bench is 2.5x chance but far from the target**: 0.296/0.303 versus
0.655/0.436 with oracle segmentation and 0.90 with the fully
contaminated pipeline. Segmentation was worth ~0.36 yield, i.e. MOST
of what looked like retrieval quality.

**Complete honest ladder for kitchen high-support QbE:**

| configuration | yield | what it assumed |
|---|---|---|
| shipped 8-channel | 0.90 | oracle segmentation + trained student + domain-derived channels |
| domain-blind channels | 0.655 | oracle segmentation |
| segmentation-free, this build | 0.296 | nothing |

Each removal of an assumption cost roughly half the score. The target
of yield AND precision > 0.90 zero-shot is not approached by any
configuration measured in this project.

## 2026-08-09 — World-model core v1: predictor-state memory (first measured round)

Embodiment-varied training corpus, self-generated, seed-disjoint from eval
(seeds 6000+ vs 5000-5149): panda 60 eps (0.93 events ok), xarm7 60 (0.89),
vx300s 40 (0.76) — three structurally different arms, 77 min of film in 13 min
wall. Predictor: 12.5M block-causal transformer over frozen vits16@320 global
latents, LeWM recipe (next-latent 1-cos + SIGReg Epps-Pulley), 12 epochs,
val pred-loss 0.017.

Gates, each vs its frozen-feature baseline (ground truth eval-only):

| gate | state | frozen baseline | verdict |
|---|---|---|---|
| a. primitive probe (frame->prim, 5-fold) | 0.504 | 0.510 (majority 0.442) | tie — no probe gain |
| b. surprise vs event boundaries (AUC) | 0.440 | 0.651 (latent derivative) | FAIL — anti-correlated |
| c. event QbE, same-prim P@10 (882 events) | **0.533** | 0.416 | PASS +0.117 |
| e. cross-embodiment xarm7->panda | **0.469** | 0.389 | PASS +0.080 |
| e. cross-embodiment vx300s->panda | **0.455** | 0.384 | PASS +0.071 |
| d. disguise battery (real clips, sim-trained) | 0.824/0.884, rho 0.486/0.292 | frozen 0.987/1.000, 0.499/0.423; pixel floor 0.553/0.390 | FAIL on struct-rho (below pixel floor) |

The retrieval claims (c, e) pass: predictor states beat frozen features on
same-kind event retrieval, including across embodiments. The battery says a
sim-only predictor DEGRADES real-footage discrimination — transfer needs
training on the target recording at ingest (the online design). Round-1
functional defect found by USING the QbE path: learned absolute positions
leaked into states and every query matched episode-START spans; round 2
retrains with NoPE (causal mask only) — a memory must be time-shift
invariant.

## 2026-08-09 — World-model rounds 2-5: the honest ladder after de-contamination

Random-retrieval prior for the 882-event mix is 0.341 — every number below
should be read against it as well as against frozen features.

| round | change | QbE P@10 | cross xarm7/vx300s | battery sev/struct |
|---|---|---|---|---|
| 1 | learned abs. positions | 0.533 (LEAKAGE) | 0.469/0.455 (LEAKAGE) | 0.486/0.292 |
| 2 | NoPE (honest baseline) | 0.352 | 0.373/0.359 | 0.385/0.256 |
| 3 | + horizons 0.1/0.5/1.5s | 0.361 | 0.364/0.363 | 0.407/0.257 |
| 4 | + grid input & targets | 0.359 | 0.388/0.340 | **0.507**/0.334 |
| 5 | + predicted-delta (read-time) | 0.359 | 0.382/0.349 | — |
| — | frozen fixed-mean | **0.416** | **0.389/0.384** | 0.499/**0.423** |
| — | random prior / pixel floor | 0.341 | 0.341 | 0.553/0.390 (floor) |

Verdict: with position leakage removed, no predictor-state variant beats the
frozen-feature mean on same-primitive retrieval; round 4's grid input put ONE
metric (battery sev-rho 0.507) past frozen and lifted the battery trend
monotonically, so spatial targets are the right direction at the wrong
granularity. Each null has a mechanism: h1 prediction is a copy task; global
pooling erases the arm; single-episode prediction never rewards cross-episode
motion abstraction; the delta head inherits the same coarse grid.

Next step recorded, not started: the true DINO-WM shape — per-PATCH token
prediction with factorized spatio-temporal attention, where the state must
model object-level motion rather than a 5x5 blur of it. The QbE SYSTEM
around the model (trajectory store, sequence search, Desk toggle, gates with
baselines) is built, functional, and model-agnostic: any better predictor
drops in behind vwm.states() with zero read-path changes.

Canonical yield/prec for WM-state event QbE on the sim truthset
(yield = true/support at k=1.5*support; prec = true/returned, 882 events):
wm-state 0.515/0.343 vs frozen 0.522/0.348 — a tie; both barely above the
0.341 prior at these support sizes. Desk verified live: mode:wm returns the
CLI's exact hits; the c2 path unaffected (86.6% elided).

## 2026-08-09 — Sweeps 2-5: the height-profile change operator, 0.416 -> 0.612

Five representation sweeps on the 882-event truthset settled the QbE scorer.
Three findings stacked, each with its control:

1. CHANGES beat appearances: ordered grid deltas 0.565 vs 0.416 pooled.
2. WHERE-invariance beats resolution: finer grids LOST (10x10 0.546,
   20x20 0.521) - they encode table position, and same experiences happen
   at different places.
3. HEIGHT is the one location axis worth keeping: row-marginal (image rows ~
   physical height) 0.600 vs column-marginal control 0.575.
4. Two temporal scales (3rds+5ths) compose: **0.612 / yield 0.562 / prec 0.374**.

Per primitive (P@10): pick 0.83, place 0.48, stack 0.38, unstack 0.27,
push 0.19. Trained-predictor arc closed for now: the delta-target round
could not predict frame-scale grid change at all (val 1-cos 0.94) and its
states stayed at 0.37 - a self-supervised state must be trained at a
granularity where prediction is possible; recorded next: per-patch tokens,
longer horizons, or contrastive-of-changes.

Shipped: vwm_qbe scores ordered height-profile change (dmulti, 3rds+5ths);
Desk wm mode serves it (verified live); trajectories stored per clip, no
summary at write time.

## 2026-08-09 — Atomic diagnosis of the 0.612 (the road to 0.9)

Confusion at top-10 (rows = query):

|        | pick | place | stack | unstack | push |
|--------|------|-------|-------|---------|------|
| pick   | 82.6 | 9.6   | 4.9   | 0.8     | 2.0  |
| place  | 35.3 | 47.9  | 13.2  | 1.4     | 2.2  |
| stack  | 32.0 | 27.0  | 37.7  | 1.7     | 1.6  |
| unstack| 30.2 | 24.8  | 16.4  | 26.7    | 1.9  |
| push   | 67.6 | 8.4   | 4.4   | 0.4     | 19.2 |

Three mechanistic atoms, each with a measured signature:

1. CAMERA BIAS (largest): 92.1% of retrieved spans share the query's camera
   vs 39.4% chance (+0.53). cam2 queries score 0.509 vs cam0's 0.665. Every
   episode has a second recorded view the store ignored. Fix: encode all
   views, score max over view pairs — raw data only.
2. HEIGHT RESOLUTION at one-block scale: 5 rows put "table level" and "one
   block up" in the same ~100px cell — precisely the stack->place/pick leak.
   Fix: 10/20-ROW marginals (finer height, columns still marginalized —
   full finer grids already measured as losses).
3. PUSH IS BLIND BY CONSTRUCTION: lateral motion at constant height is what
   the row marginal deletes (67.6% of push hits are picks). Needs a lateral
   term; smallest class (25), third priority.

Duration is a minor confound (within-primitive: pick 0.83 vs 0.79).

Can it reach 0.9? Honest bounds: pick plausibly yes (0.83 pre-fixes).
unstack has a SEMANTIC ceiling under this grading — its span IS a
grasp-from-tower + carry + release, so a large fraction of its visual
content is legitimately pick-then-place; 0.9 same-label retrieval for
unstack would require the representation to privilege the tower-origin
over everything else the clip shows. push at n=25 is statistically fragile.
A corpus-wide 0.9 P@10 therefore requires either (a) all three atoms fixed
AND the composite classes re-graded at sub-event level, or (b) accepting
that the honest corpus-wide number lands below the per-class ceiling of its
hardest class. Sweep 6 (running) isolates atoms 1+2.

## 2026-08-09 — Atoms fixed one by one: 0.612 -> 0.723

| step (each isolated, controls held) | P@10 | yield | prec |
|---|---|---|---|
| height-profile 3+5 (previous ship) | 0.612 | 0.562 | 0.374 |
| + BOTH camera views, max-matching (atom 1) | 0.668 | 0.578 | 0.385 |
| + 20-row height resolution (atom 2) | 0.676 | 0.577 | 0.385 |
| + CSLS hubness correction (leak-to-majority) | 0.695 | 0.578 | 0.385 |
| + third temporal scale (3+5+8) | **0.723** | **0.587** | **0.391** |

Per primitive at ship: pick 0.87, place 0.61, stack 0.56, unstack 0.61,
push 0.45. Lateral term measured NULL twice (push's remaining gap is not
coarse column deltas). Shipped in vwm_qbe + Desk (verified live): 20-row
profiles of every recorded view, 3+5+8 segment deltas, store-view max,
probe-sample CSLS at query time.

Toward 0.9, recorded next: (a) view-contrastive head - the dual-camera
recordings supervise view invariance for FREE (same moment, two views,
InfoNCE; no labels, self-generated data, allowed by every rule); train-
corpus v3 encode queued. (b) sub-event matching for composite classes.
(c) the semantic ceiling note stands: unstack IS pick+place on film.

## 2026-08-09 — Yield/prec is THE metric: decomposition + hypothesis ladder

Reframed per the product: the system must return a SET (own cut) with
yield = true/support and prec = true/returned, both 0.90. Harness v2
separates the two failure modes:

- ORACLE-CUT ceiling (best threshold given the ranking): **0.539/0.431**
  -> no cut can rescue the current ranking; DEPTH is the whole game.
- The sharpest symptom: pick has P@10 0.87 but deep AUC 0.527 - the scorer
  finds near-duplicates at the head and ranks the class tail randomly.
  Corpus-wide AP 0.442.

Hypotheses for the tail, each measured and KILLED:
1. colour-binding: correct hits share colour at 0.129 vs base 0.125 - the
   deltas are already colour-blind (shape likewise 0.515 vs 0.505).
2. row/depth conflation: pick-pick similarity vs row-of-change distance
   spearman -0.076; change-centered row alignment moved AP 0.442->0.444.
3. query expansion (alpha-QE): AP +0.02 but oracle YIELD -0.02 - sharpens
   the head, does not recover the tail.

Conclusion: the fixed operator's invariance budget is spent. In flight:
cross-view InfoNCE head (vwm_head.py) - two cameras filmed the same
moment, so agreement across views is free physical supervision; temporal
jitter positives add phase robustness; trained on the seed-disjoint
self-generated corpus only.

## 2026-08-09 — The supervised ceiling settles where 0.9 lives

An eval-only probe (labels, 5-fold by episode - an instrument, never
shippable) bounds what ANY scorer could do on the current features:

| class | ceiling AP | ceiling y/p | unsupervised y/p |
|---|---|---|---|
| pick | 0.917 | 0.934/0.906 | — |
| stack | 0.636 | 0.713/0.611 | — |
| place | 0.613 | 0.679/0.597 | — |
| unstack | 0.162 | 0.374/0.204 | — |
| push | 0.094 | 0.362/0.141 | — |
| ALL | 0.733 | **0.789/0.722** | 0.539/0.431 |

Three consequences:
1. 0.9/0.9 corpus-wide is NOT reachable on these features under this
   grading - even with labels the ceiling is 0.79/0.72. The blockers are
   the composite/rare classes (unstack IS pick+place on film; push n=25).
2. For PICK the target IS in the features (0.934/0.906 supervised) - the
   0.9 goal is realistic there, and the unsupervised-to-supervised gap
   (AP 0.44 -> 0.73) is the extractable-without-labels prize.
3. The cross-view head round: +0.02 AP on pick, minority classes dropped -
   window-level InfoNCE sharpens majority structure. Sixth attack on the
   ~0.52 oracle ceiling; the ceiling didn't move because the ceiling was
   never scoring - it is features + grading.

Decision fork recorded for the owner: (a) richer upstream features +
class-balanced generation (more unstack/push episodes - self-generated,
allowed) raises the ceiling itself; (b) sub-event units change what
"same experience" means for composites; (c) accept per-class targets
(pick to 0.9 first). These are product-semantics choices, not tuning.

## 2026-08-09 — Sub-event units v1: null, with the mechanism

Energy-valley segmentation produced a MEDIAN OF 1 unit per event: the arm
never pauses mid-event, so stillness valleys separate events, not
sub-events - the composite structure lives at kinematic phase changes
(grasp/lift/lower turning points), not motion gaps. Unit set-match scored
AP 0.428 vs 0.442 whole-span; push collapsed to 0.08 (its low-energy
motion falls under the recording's own Otsu threshold). vwm_units.py kept
as the instrument; next unit attempt must cut on change-direction
reversals, not energy.

## 2026-08-09 — Extended ruler + feature-ceiling flatness: the constraint is upstream vision

Balance batch (80+80 eps, 0.91 verified): push support 25->140, unstack
42->105. Extended-ruler (1352 events) supervised ceilings:

| class | old ceiling y/p | balanced ceiling y/p |
|---|---|---|
| push | 0.362/0.141 | **0.606/0.508** (~4x AP) |
| unstack | 0.374/0.204 | 0.571/0.452 |
| ALL | 0.789/0.722 | 0.777/0.713 |

Starvation PROVED and fixed. Then the feature-set ceiling sweep came back
FLAT: row-dmulti 0.741 AP, full-grid 0.717, +static context 0.735, all
combined 0.735 - no derived feature raises the bound, so the loss is in
the frozen per-frame encoding itself (320px, distant cams, one-block
height at the resolution edge). Sub-event units v1 also null (median 1
unit/event - stillness separates events, not phases).

In flight: identical operator at 448px/28-grid - the direct test of
"upstream vision is the constraint". If the ceiling rises, the road to
0.9 is resolution/encoder investment + closing the unsup gap (0.397 vs
0.736 AP); if it stays flat, the residual is scene/observation physics
and the honest targets are per-class.

## 2026-08-09 — Campaign terminal map (yield/prec toward 0.9)

Resolution test: 448px ceiling 0.792/0.732 vs 320px 0.783/0.719 - real but
SHALLOW (+0.01); no feasible resolution reaches 0.9. Cluster-assignment
scoring (kmeans k=8..64): purity maxes 0.60, precision drops - the
unsupervised structure of these features does not carve primitives.

THE MAP after ~20 measured experiments:
- Ceiling (supervised, any tested feature set/resolution): ~0.79/0.73
  corpus-wide; pick alone 0.93/0.90.
- Best unsupervised product configuration: 0.49/0.36 oracle-cut on the
  extended ruler (both views, r20 profile, 3+5+8 deltas, CSLS).
- What worked: camera views (+0.06), CSLS (+0.02), multi-scale time
  (+0.03), class-balanced generation (rare-class ceilings 2-4x).
- What nulled, each with mechanism: color/shape (already invariant),
  row-depth alignment, lateral term, QE, energy-based sub-units, view-
  contrastive head, delta-target predictor, cluster scoring, finer grids,
  static context.

REMAINING ROUTES, in order of expected value:
1. Kinematic sub-event units (cut at change-direction reversals) +
   sub-event grading - dissolves the composite-class ceiling, serves
   product case 4 directly.
2. A representation trained to agree across views AND time-jitter AND
   photometric disguise at the SUB-UNIT level (the window-level head
   sharpened the majority class; units change the positives).
3. Recording-side: more/closer cameras raise every ceiling at zero
   algorithmic cost - deployment guidance, not code.

## 2026-08-09 — Kinematic sub-event units: built, benchmarked, envelope conserved

Douglas-Peucker corner cuts on the latent trajectory (tolerance = alpha x
the clip's own radius of gyration; no labels, nothing fitted) find the
phases energy valleys could not: median 6 units/event vs 1 for the energy
cut. Extended ruler, 1352 events, oracle-cut protocol:

| scorer | AP | yield | prec | unstack AP | push AP |
|---|---|---|---|---|---|
| whole-span (base) | 0.394 | 0.474 | 0.386 | 0.347 | 0.307 |
| unit set-match | 0.339 | 0.533 | 0.341 | 0.158 | 0.196 |
| DTW ordered units | 0.356 | 0.498 | 0.356 | 0.229 | 0.224 |
| unit coverage (min) | 0.280 | 0.746 | 0.328 | 0.210 | 0.248 |
| span + DTW (w=0.5) | 0.396 | 0.483 | 0.386 | 0.365 | 0.310 |
| **span + DTW + cover** | 0.387 | **0.531** | 0.382 | **0.386** | **0.351** |
| adaptive per-query | 0.395 | 0.440 | 0.383 | 0.248 | 0.310 (pick AP 0.553) |

Findings:
1. Units do exactly what they were hypothesized to do: the COMPOSITE and
   rare classes gain (unstack AP 0.347->0.386, push 0.307->0.351) because
   an unstack shares its grasp-high phase with other unstacks even when
   the carry differs. The majority class pays for it.
2. The ENVELOPE IS CONSERVED: no configuration leaves ~0.39-0.40 AP /
   0.47-0.53 yield / 0.38 prec. Units redistribute across classes; they do
   not add information. This is the supervised-ceiling result again from
   the other side.
3. METRIC TRAP recorded: unit-coverage scores "pick yield 1.000 / prec
   0.442" while its pick AP (0.363) sits BELOW pick's 0.44 base rate -
   oracle-cut min(y,p) saturates at full recall for a majority class, so
   a degenerate ranking can look strong. Always read AP beside it.
4. Per-query adaptive weighting (informativeness from cross-view
   agreement) is a null overall; its first version had a zero-weight
   collapse (all-scorers-uninformative queries ranked by noise) worth
   remembering: normalize, but always fall back to uniform.

Best yield config is span+DTW+cover: +0.057 yield (+12% relative) at
-0.004 prec. Shippability check in flight: the benchmark cut units per
EVENT SPAN, but a live store must cut each episode once and clip - if the
numbers hold under episode-cut geometry the config ships, otherwise the
benchmark does not transfer.

### Shipping-geometry check (episode cut once, window inherits)

The benchmark above segments each EVENT SPAN; a live store must segment
each episode once at write time and let a window inherit the cuts inside
it. Re-run under that geometry (units/event median 5 vs 6):

| scorer | span-cut y/p | episode-cut y/p |
|---|---|---|
| whole-span | 0.474/0.386 | 0.474/0.386 |
| unit set-match | 0.533/0.341 | 0.564/0.333 |
| span+units w=0.3 | 0.483/0.385 | 0.482/0.385 |
| span+units w=0.5 | 0.498/0.381 | 0.500/0.379 |

The units TRANSFER: write-time segmentation loses nothing (set-match
yield is even slightly higher). So the shipping design is sound - cut
each episode once, store the unit reps, clip per candidate window at O(1).
Ordered-variant confirmation under the same geometry is running.

### Shipped: span + ordered-phase DTW (w=0.5), episode-cut geometry

Full confirmation under shipping geometry (1352 events):

| scorer | AP | yield | prec |
|---|---|---|---|
| whole-span (previous ship) | 0.394 | 0.474 | 0.386 |
| **span + DTW w=0.5 (SHIPPED)** | **0.396** | **0.494** | **0.388** |
| span + DTW w=0.7 | 0.391 | 0.506 | 0.384 |
| span + DTW + cover (yield-first option) | 0.385 | 0.542 | 0.379 |
| unit coverage alone | 0.285 | 0.831 | 0.334 |

span+DTW w=0.5 strictly dominates the previous ship on all three numbers,
so it is the default; cover buys +0.068 yield for -0.007 prec if a
deployment wants recall. Live in vwm_qbe + Desk (verified: same top hits
as the CLI, 3.7 s/query over the full store with no pruning yet).

## 2026-08-09 — THE EXTRACTION LADDER: the measurement that was never taken

Per-frame ground truth now dumps from the generator (SDX_TRACE: hand xyz,
grip ctrl, per-block xyz+quat, aligned to frames). 40-episode probe corpus,
ridge probes 5-fold by episode, R^2 per pipeline stage per quantity:

| stage | hand_x | hand_y | hand_z | grip | tower_z | nearblk_z | h-b dist | blk_speed |
|---|---|---|---|---|---|---|---|---|
| S0 pixels 32x24 | 0.370 | 0.393 | 0.243 | -0.00 | -0.00 | -0.00 | 0.121 | -0.01 |
| S1 grid 20x20 (JL4k) | 0.684 | 0.571 | 0.684 | 0.057 | 0.180 | 0.181 | 0.439 | 0.056 |
| S2 c 5x5 | 0.690 | 0.568 | 0.663 | 0.053 | 0.169 | 0.170 | 0.425 | 0.053 |
| S3 r20 row-marg | 0.537 | 0.308 | 0.668 | 0.055 | 0.154 | 0.159 | 0.448 | 0.058 |
| S4 g global | 0.548 | 0.218 | 0.523 | 0.042 | 0.126 | 0.132 | 0.353 | 0.039 |
| S5 rp768 (ship) | 0.575 | 0.333 | 0.663 | 0.051 | 0.161 | 0.165 | 0.454 | 0.058 |

READING: the quantities that define the task - nearest-block HEIGHT
(stack vs place), TOWER height (structure state), GRIPPER state (pick vs
release) - are essentially ABSENT already at the FULL frozen grid (R^2
0.18/0.18/0.06), before any pooling. Pooling losses are real but
secondary (row-marginal kills hand_y 0.57->0.31). Velocities are absent
from any single frame, as expected. This is why every matcher variant
conserved the envelope: the latent never contained the state.

Capacity check in flight (MLP + full-grid + frame-pair probes) to
separate "encoder lacks it" from "probe could not read it".

## 2026-08-09 — CAPACITY VERDICT + the atomic anatomy of the similarity lookup

Capacity check (MLP, full-grid, frame-pair) and the relational-confound
closer (non-relational aggregates):

| target | linear | MLP | verdict |
|---|---|---|---|
| hand_z | 0.66 | **0.84** | PRESENT, nonlinearly coded |
| nearblk_z | 0.17 | 0.26 | ABSENT |
| tower_top_z | 0.18 | 0.28 | ABSENT |
| mean_blk_z (non-relational) | 0.20 | 0.27 | ABSENT - closes the confound |
| n_lifted (non-relational) | 0.20 | 0.25 | ABSENT |
| grip | 0.06 | -0.03 | ABSENT |
| velocities (frame-pair) | 0.05-0.07 | 0.21 best | mostly absent |

MECHANISM: blocks are ~30px objects = ~2x2 patches at 16px stride. A 5cm
world height change is a sub-patch image shift, and DINOv3 patch features
are deliberately robust to sub-patch translation - so block VERTICAL STATE
lives below the representation's granularity. The arm spans hundreds of
pixels; its state is present (0.84). Global 448px barely helped because
the patch stride scaled with it.

### The similarity lookup, atomized and rated

| # | atomic stage | what it must preserve | measured | rating |
|---|---|---|---|---|
| A1 | camera/frames 480p@10fps | scene state visible | blocks ~30px | WEAK for small objects |
| A2 | frozen encoder (vits16 grid) | object + agent state | hand 0.84 / blocks 0.18-0.28 / grip 0.06 | **THE FAILURE POINT** |
| A3 | spatial pooling (row profile) | keep A2, drop nuisance | hand_y 0.57->0.31; killers already dead | secondary loss |
| A4 | JL projection | isometry | 0.166 vs 0.159 full-grid | LOSSLESS |
| A5 | temporal deltas (3+5+8) | ordered change | best hand-op family (0.416->0.612 arc) | OK given A2 |
| A6 | kinematic units + DTW | phase structure | +yield, envelope conserved | OK given A2 |
| A7 | view max-matching | cross-camera | +0.056, bias 0.92->absorbed | GOOD |
| A8 | CSLS hub correction | de-hub scores | +0.02, minority classes +0.1-0.15 | GOOD |
| A9 | cut/abstention | secondary per owner | not the constraint (oracle-cut used) | - |

Everything downstream of A2 is now measured to be either fine or a
secondary loss; SEVEN matcher variants conserved the envelope because A2
never handed them the state. The repair is the extraction stage: the
working representation must recover sub-patch object state - object
tracks (position/height per compact moving/foreground region, generic
detection, no names) as the state record, with the patch grid demoted to
an appearance descriptor per object. That is the tracks-first design the
project already holds, now with its quantitative justification.

## 2026-08-09 — Whole-problem gate + the no-training granularity candidates

Course correction (owner): a sim-trained encoder passes the sim ladder and
fails the product - it cannot generalize to unseen data, which is the
entire point. REQUIREMENTS restated as the gate for any repair:
 1. works on arbitrary unseen data (nothing fitted to any corpus, sim
    included; general substrate + per-recording self-calibration only)
 2. no labels / domain info / fixed axes anywhere in write/read/train
 3. preserves state at the granularity events happen (the measured A2
    failure)
 4. sim ground truth grades, never trains
Two-sided acceptance: raise block-state R^2 on the sim extraction ladder
AND hold the real-corpora disguise battery.

The corrected reading of the ladder: the GENERAL encoder's STRIDE failed,
not its generality - 640px frames were downscaled to 320, making a 30px
block a single 16px patch, and patch features are translation-robust
within themselves. Candidates that fix granularity with ZERO training and
ZERO domain knowledge (general by construction): tiled native-scale
encoding, earlier-layer taps (less invariance, more position), finer-
stride general backbones. Ladder run in flight.

Ladder result (20 eps, 5fps, 2513 frames, ridge R^2 — compare WITHIN run;
this run is lower-powered than the 40-ep ladder, hand_z 0.66->0.41):

| stage | nearblk_z | tower_top_z | grip | hand_z | hand_y |
|---|---|---|---|---|---|
| whole@320 L12 (ship) | 0.045 | 0.041 | 0.033 | 0.410 | 0.118 |
| whole@320 L6 | -0.000 | -0.003 | 0.006 | 0.207 | 0.061 |
| tiled native L12 | 0.061 | 0.060 | 0.039 | 0.452 | 0.197 |
| tiled native L6 | 0.004 | 0.003 | 0.004 | 0.174 | 0.119 |

Verdicts: early-layer taps strictly worse (L6 loses even hand_z 2x) —
less invariance is NOT more readable position. Native-scale tiling
(~2x patch granularity: a 30px block goes from ~1 patch to ~2) lifts
every target by a small consistent margin (hand_y +67% rel) but block
state stays absent-tier — 2x granularity is not a categorical fix.
Next zero-training candidates: (a) MLP readout on cached tiled feats
(nonlinearity + granularity jointly), (b) DINOv3 ConvNeXt-Tiny
stride-8 stage (finer stride from a general model already shipped in
the identity cut).

Follow-ups (same 20-ep protocol):

MLP readout (256 hidden, standardized targets) — the joint
nonlinearity+granularity test:
| stage | nearblk_z | tower_top_z | grip | hand_z |
|---|---|---|---|---|
| whole@320 MLP | 0.127 | 0.132 | -0.005 | 0.864 |
| tiled native MLP | 0.096 | -0.006 | -0.062 | 0.843 |
Readout at full power (hand_z 0.864 = the 40-ep reference despite half
the data); block state still absent; tiling adds NOTHING nonlinearly —
the small ridge lift was dimensional noise. GRANULARITY HYPOTHESIS DEAD:
2x patch resolution changes nothing under either readout.

ConvNeXt-Tiny stages (stride 16/8/4, per-cell l2 + JL): control never
validated — hand_z 0.090 at s16 vs ViT 0.410 on identical frames. Raw
conv spatial maps lose linear readability under JL for reasons
independent of stride (suspect common-component domination). Stride-8
door strictly untested but moot: the mechanism it targeted is dead
above. No more cycles here.

## 2026-08-09 — EXTRACTION REPAIR FOUND: per-recording background referencing

Referenced/temporal ladder, full 40-ep probe (9515 frames, ridge R^2):

| stage | nearblk_z | tower_top_z | grip | hand_z | hand_y |
|---|---|---|---|---|---|
| grid JL4096 (ctrl) | 0.181 | 0.180 | 0.057 | 0.684 | 0.571 |
| bg-subtracted JL4096 | 0.415 | 0.410 | 0.192 | 0.800 | 0.600 |
| foreground-mass 400 | 0.021 | 0.014 | 0.021 | 0.526 | 0.380 |
| referenced row profile | 0.469 | 0.464 | 0.230 | 0.827 | 0.518 |
| EMA 0.5/2/8s stack | 0.132 | 0.134 | 0.042 | 0.546 | 0.462 |

Block state was NEVER absent from the frozen features - it was drowned
by the static background in every unreferenced readout. Subtracting the
per-recording per-cell median (parameter-free self-calibration: no
training, no labels, generic on any data) lifts block quantities 2.6x
into clearly-present territory, grip 4x, hand_z to 0.827. Direction of
deviation carries it (mass-only is dead); temporal EMA is not the door.
This dissolves the granularity mechanism story: sub-patch objects ARE
encoded, as small deviations from the background cell feature.

Why retrieval behaved as it did: segment deltas r[t2]-r[t1] cancel the
background implicitly - change-matching worked, state-matching never
existed. The repair: referenced ENDPOINT STATES join the span
representation (state, not just change). Two-sided gate: (1) this
ladder PASSED; (2) retrieval A/B on the extended ruler + real-corpora
battery must hold.

Retrieval transfer of the repair, full extended ruler (1352 events;
sim_chains 882 + sim_eval_bal 470, all dual-view v3 cached):

| arm | AP | oracle y/p | fixed y/p |
|---|---|---|---|
| del alone (ship) | 0.397 | 0.474/0.388 | 0.524/0.349 |
| st (ref endpoint states) | 0.367 | 0.449/0.370 | 0.512/0.341 |
| sd (ref state diff) | 0.349 | 0.445/0.352 | 0.489/0.326 |
| z-fusion uniform | 0.385 | 0.445/0.381 | 0.522/0.348 |
| self-agree fusion | 0.389 | 0.454/0.384 | 0.524/0.349 |
| ORACLE channel pick (eval) | 0.417 | 0.438/0.408 | 0.552/0.367 |

Referenced states carry real standalone signal (0.367 AP from two
frames!) but re-rank what deltas already rank: even a PERFECT per-query
channel selector gains +0.020 AP. Naive concat null; states-only
actively hurts unstack (0.35->0.13). Diagnosis: cosine is variance-
weighted and the state vector is hand-dominated (hand_z 0.83 in the
same vector) - presence (ridge 0.47) != retrievability (cosine). The
del-alone 0.397 IS the recorded unsup-vs-supervised gap endpoint
(0.397 vs 0.736 on the same features): the gap is METRIC, not features.
Next attack: store-side PCA-whitening w/ eigenvalue floor (corpus
stats, label-free, same legal class as CSLS/hub correction) to undo
variance-weighting. Run in flight.

Whitening attack: NEGATIVE at every floor and rep. del 0.397 ->
0.345/0.290/0.319 (e=1.0/0.1/0.01); st 0.367 -> 0.351/0.322/0.292;
concat likewise. The high-variance directions cosine favors DO carry
the readable class signal; the state signal is a specific low-dim
subspace that variance statistics cannot locate. (Whitened arms trade
precision for spread: oracle yield up to 0.62 at prec 0.30 - not
useful.) Last label-free attack in flight: cross-view CCA - the
corpus's free supervision is that one event is seen by two cameras;
canonical dirs = view-invariant content. Closed-form, event-vector
level (distinct from the nulled InfoNCE encoder head).

CCA attack: FIRST POSITIVE after 5 metric nulls/negatives.

| arm | AP | fixed y/p |
|---|---|---|
| del raw (ship) | 0.397 | 0.524/0.349 |
| st raw | 0.367 | 0.512/0.341 |
| st cca k=64 r=1.0 | 0.407 | 0.555/0.370 |
| del cca k=64 r=1.0 | 0.404 | 0.539/0.359 |

Cross-view CCA on the referenced-state channel: +0.040 AP over st raw,
now ABOVE the whole delta machinery. place (the end-state class) gains
most (0.25->0.31). Trends coherent with a small real subspace: k=64 >
k=256, r=1.0 > 0.1. The corpus's free supervision (same event, two
cameras) locates class-relevant directions that raw cosine buries and
variance statistics (whitening) could not find. Pending before ship:
del+st-cca fusion, k/r sweep, and SPLIT-HALF holdout (fit on half the
episodes, benchmark inside the other half) - in flight.

CCA k/r sweep + fusion + holdout (event-fit): k=16 wins (subspace is
~16-dim). st cca16 0.433; fuse 0.5del+stcca16 AP 0.447 fixed
0.588/0.392 (ship 0.397/0.524/0.349); holdout fusion 0.432/0.438 -
generalizes across episodes. BUT the label-free fit CONTROL FAILED:
CCA fitted on stride-window pairs (correspondence-only supervision)
scores 0.350 < raw 0.367. The event-fit gain was carried by
event-SHAPED span boundaries, not multi-view correspondence alone -
shipping it would have been a hidden-label violation (the
contamination pattern, caught by the control this time). Legal route
in flight: fit on UNIT-ALIGNED spans from the shipped unsupervised
segmentation (vwm_kin.dp_cuts) - event-shaped by construction, no
labels anywhere.

Unit-aligned fit (dp_cuts spans, 9871 pairs): ALSO FAILS - 0.334 <
stride 0.350 < raw 0.367 < event-fit 0.433. The shipped segmentation
is not event-shaped enough to substitute for truth boundaries.

## 2026-08-09 — THE SIMILARITY LOOKUP, TRIANGULATED (session verdict)

The atoms, each measured on the 1352-event ruler:
 1. EXTRACTION: repaired. Per-recording background referencing lifts
    block-state R^2 0.18->0.47 (hand 0.83, grip 0.23); the granularity
    story died 3 ways first (tiling ridge+MLP, layers, stride).
 2. REPRESENTATION: referenced endpoint states carry standalone
    retrieval signal (0.367 AP from two frames vs deltas' 0.397).
 3. METRIC: a ~16-dim subspace of referenced-state space is worth
    +0.050 AP / +0.064 yield / +0.043 prec fused with deltas (0.447 /
    0.588/0.392 vs ship 0.397 / 0.524/0.349) - PROVEN to exist and
    holdout-stable (0.432/0.438 on unseen episodes).
 4. FIT SUPERVISION: the obstruction. Every label-free source fails
    to locate the subspace: variance (whitening, negative), cross-view
    correspondence + stride spans (0.350), + dp_cuts spans (0.334).
    Only truth-event-aligned spans work. The differentiator is SPAN
    QUALITY: segmentation is the binding constraint.

NEW GATE for segmentation work (label-free, per store): fit CCA on the
candidate segmentation's spans; the segmentation is good enough when
the event-fit gain is recovered. Segmentation now has a retrieval-
denominated acceptance test instead of an aesthetic one. Second
sanctioned route: Desk verdicts accumulate the span supervision
directly (post-query retraining is the allowed offline tier).

Gate applied to two more label-free candidates: refunits (dp_cuts on
the REFERENCED trajectory) 0.333, motion (Otsu energy valleys on ref)
0.342. All four label-free sources fail; notably STRIDE (0.350) beats
every smart segmentation - the unsupervised cutters align to
arm-motion structure, so their span pairs teach CCA the nuisance.
Event spans work because their boundaries mark manipulation
completion. The segmentation that passes this gate must localize
STATE-CHANGE boundaries, not motion boundaries.

Gate rounds 2-3: statechange boundaries (settled-state diff maxima)
0.342; change-SELECTED stride spans 0.354 (best label-free source,
confirming composition matters directionally). Final gate ladder:
units 0.334 ~ refunits 0.333 < motion 0.342 = statechange 0.342 <
stride 0.350 < stridesel 0.354 << raw 0.367 << EVENT-FIT 0.433.
Six label-free constructions fail; the event-supervision gap is
robust. Open routes (both substantive, recorded): a segmentation that
truly localizes manipulation completion (write-path proposer class),
or Desk-verdict span supervision via the sanctioned offline
post-query tier.

## 2026-08-09 (night 2) — the pipeline was not the constraint; the GRAPH was

Owner rejected the label explanation and the MLP "ceiling" (correctly:
an MLP probe on my own pooled vector bounds that vector's readout, not
the problem). Nine experiments, all on the 1352-event ruler.

WHAT IS NOT THE CONSTRAINT (each a measured null):
| test | result |
|---|---|
| spatial pooling: g(384) vs r20(7680) vs c5(9600) | 0.368 / 0.388 / 0.366 |
| full 20x20 grid (re-encoded, 17 GB) | 0.345 - no better than 5x5 |
| frame-DTW (the STRAP recipe) | 0.308 - WORSE than segment deltas |
| per-recording referencing on deltas | 0.000 change (deltas already cancel it) |
| persistent-effect field (settled medians) | 0.329-0.350 |
| change-site localization, 4 variants | site lands on the ARM (verified by eye) |

The arm defeats every "what changed" detector: it is not transient, it
is a large pedestal-mounted body always in frame at a different pose
each event, and it dwells exactly where the interesting thing happens.
Short medians keep it, long medians keep it, pixel medians keep it,
cross-event IDF keeps it.

THE ACTUAL DIAGNOSIS (diag_taxonomy + diag_support):
 - top-10 confusion: pick query returns 86.9% picks, push 77.4%,
   unstack 68.8%. The retriever is NOT returning junk.
 - P@1 0.872, P@5 0.801, P@10 0.742 (5 classes); merged into the 3
   genuinely distinct motions {grasp, release, push}: 0.943/0.884/0.836.
 - place<->stack confuse each other at 21% - exactly the pair that
   differs only by table-vs-tower, a ~30px relation.
 - yield is FLAT vs support (0.41 at S=5, 0.46 at S=80), so the low
   yield is not a support artifact.
P@1 0.87 with yield 0.46 means: local structure excellent, global
structure fragmented. A class is MANY small tight clusters (same
object, position, embodiment), not one region. Cosine sees only the
cluster the query lands in.

THE FIX - DIFFUSION ON THE AFFINITY GRAPH (label-free, closed form,
corpus statistics only, same legal class as the shipped CSLS):
| metric | cosine+CSLS | +diffusion |
|---|---|---|
| AP (5 cls) | 0.397 | **0.495** |
| P@10 | 0.742 | 0.786 |
| yield/prec | 0.524/0.349 | **0.606/0.404** |
| balanced-support20 y/p | 0.448/0.294 | **0.578/0.379** |
| 3-motion yield/prec | 0.683/0.455 | **0.722/0.481** |
Stable across k in {10,20,50} and alpha in {0.7,0.9,0.99} (AP
0.485-0.495) - robust, not fitted to the ruler.

Refinements on top of diffusion (rank affinity, post-CSLS, 2nd round,
alpha-QE): all within +-0.01; diffusion itself is the whole gain.
Chosen on dev (sim_chains 882), reported on holdout (sim_eval_bal 470,
never seen by the choice):

| holdout | AP | P@1 | P@10 | yield/prec | 3-motion y/p |
|---|---|---|---|---|---|
| baseline cosine+CSLS | 0.399 | 0.853 | 0.641 | 0.500/0.333 | 0.611/0.407 |
| + diffusion | **0.546** | 0.868 | 0.744 | **0.643/0.428** | 0.688/0.459 |

THE SCALING LAW (exp13) - the result that decides what to do next.
Same fixed config, corpus subsampled by EPISODE, 12 draws per size:

| episodes | events | baseline y/p | diffusion y/p | diffusion 3-motion |
|---|---|---|---|---|
| 20 | 116 | 0.535/0.354 | 0.445/0.295 | 0.645/0.428 |
| 40 | 233 | 0.525/0.349 | 0.465/0.309 | 0.641/0.427 |
| 80 | 475 | 0.521/0.347 | 0.530/0.352 | 0.662/0.441 |
| 150 | 880 | 0.523/0.349 | 0.586/0.391 | 0.702/0.467 |
| 230 | 1352 | 0.524/0.349 | **0.618/0.412** | **0.723/0.482** |

The baseline is FLAT in corpus size - cosine cannot use data it is not
being compared against. Diffusion rises monotonically and has not
saturated; below ~475 events it is WORSE than cosine because the graph
is too sparse to propagate. So the current number is CORPUS-limited,
not method-limited, and "more data" is a measurable path rather than a
hope. For a storage product this is the right shape: retrieval quality
improves as the store fills.

Two further attacks on the residual, both REJECTED by the holdout:
 - second-order similarity (neighbour-profile cosine, meant to bridge
   the clusters): dev 3cls 0.776 vs diffusion 0.769, but 5cls never
   beats it; selection kept plain diffusion.
 - arm-free clean scenes at 640px (pixel-median removes the arm; no
   site selection needed; 40x40 patches so a block is ~2x2 cells,
   aimed squarely at the place<->stack tower relation): helps on DEV
   (0.658/0.438 vs 0.638/0.425) and HURTS on HOLDOUT (0.625/0.416 vs
   0.643/0.428). The dev gain was selection overfitting; rejected.
   This is why the choose-on-dev / report-on-holdout split exists.

SETTLED, holdout: diffusion alone. AP 0.546, P@1 0.868, P@10 0.744,
yield/prec 0.643/0.428 (3-motion 0.688/0.459).

## 2026-08-09 — SCALING LAW CONFIRMED: +220 episodes moved the number

Prediction from exp13: adding fresh episodes raises diffusion's yield
and leaves the baseline where it is. Test: 220 new episodes (sim_grow,
seeds 9000+, disjoint from everything measured), corpus 1352 -> 2660
events over 449 episodes. Same fixed config (k=10, alpha=0.9).

| holdout (sim_eval_bal, never used for any choice) | AP | P@1 | P@10 | yield/prec | 3-motion |
|---|---|---|---|---|---|
| baseline @1352 | 0.399 | 0.853 | 0.641 | 0.500/0.333 | 0.611/0.407 |
| baseline @2660 | 0.396 | 0.851 | 0.636 | 0.499/0.332 | 0.610/0.406 |
| diffusion @1352 | 0.546 | 0.868 | 0.744 | 0.643/0.428 | 0.688/0.459 |
| **diffusion @2660** | **0.597** | 0.860 | 0.775 | **0.696/0.463** | **0.722/0.481** |

The baseline is flat to three decimals (0.500 -> 0.499) while diffusion
gains +0.053 yield from data alone. Full-corpus numbers: 5-class
0.668/0.445, 3-motion **0.768/0.512**, P@1 0.913, P@10 0.845.

Scaling curve, re-measured on the grown corpus (still not saturated):
| episodes | events | baseline | diffusion | diffusion 3-motion |
|---|---|---|---|---|
| 80 | 480 | 0.527 | 0.490 | 0.673 |
| 150 | 887 | 0.523 | 0.563 | 0.705 |
| 230 | 1339 | 0.524 | 0.620 | 0.734 |
| 300 | 1780 | 0.526 | 0.641 | 0.748 |
| 449 | 2660 | 0.526 | **0.668** | **0.768** |

So the honest position on the 0.80 target: 3-motion yield is at 0.768
and rising ~0.02 per +450 events; 0.80 is within reach of one more
growth batch. 5-class yield is at 0.668 and would need the place<->
stack tower relation, which four localization attempts and the
clean-scene channel all failed to supply.
