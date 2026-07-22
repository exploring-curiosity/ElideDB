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

## Reproduce

```bash
./build/sdx bench --store store --windows 20 --dur 2 --naive 2
./build/sdx query --store store --text "..." --eval-recall
```
