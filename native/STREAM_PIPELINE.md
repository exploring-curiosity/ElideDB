# STREAM_PIPELINE.md — continuous fetch → ingest → train → evict

Design only. NOT to be executed until the local SSL run (SSL_TRAINING.md)
reports its Stage-E numbers. This document is the plan for scaling that run
from 13 video-hours to thousands, without ever holding the video.

## 1. The one idea

Raw video is a **transient**. It exists only long enough to become a trace,
then it is deleted. Nothing downstream ever reads it again. So the disk
never holds the corpus — it holds a sliding window of it, and the model is
the only thing that accumulates.

Four daemons run concurrently and never block each other. They coordinate
through marker files in a work directory; there is no queue server, no RPC,
and any daemon can be killed and restarted at any moment because **state is
re-derived from the filesystem, never from a manifest** (a manifest here
once reported 128 episodes where 447 were on disk).

```
 fetcher ──► raw/<shard>/.ready ──► ingester ──► traces/<shard>/.ingested
    ▲                                   │                    │
    │ backpressure: sleeps while         │ deletes raw/       ▼
    └── uningested shards >= 3           │              compactor (tiering)
                                         │                    │
                          trainer ◄──────┴────────────────────┘
                             │  (CPU, continuous, replay-mixed)
                             ▼
                          evaluator  ── every 6 h, SEALED corpus, regression gate
```

## 2. Why the daemons genuinely overlap

Measured on this machine: ingest pins the **GPU at 98-100%** while leaving
**13 of 15 CPU cores idle**; one python process at 35% of a core drives it.
So the assignment is not arbitrary:

| daemon | resource | rate |
|---|---|---|
| fetcher | network + disk | ~45 GB/h @ 100 Mbit ⇒ 30-150 video-h/h |
| ingester | **GPU** | **4.17 video-h per wall-hour** ← the rate limiter |
| compactor | disk | seconds per shard |
| trainer | **CPU** | ~300 steps/min (measured 4.45 step/s on MPS; CPU is the fallback and is fast enough at 1.15M params) |
| evaluator | CPU, bursty | ~1.5 h per eval, every 6 h |

Trainer on CPU is the design decision that makes this a *cycle* rather than a
sequence. It costs the trainer some speed and buys 100% GPU utilisation for
ingest, which is the only stage that cannot be made faster without
distillation.

**Throughput: ~100 video-hours per day.** One week ≈ 700 h, one month ≈ 3,000 h.
That is the honest ceiling until the backbone is distilled (encoder is 63% of
write, predictor 22%; see the distillation notes in the ledger).

## 3. Storage — the constraint that shapes everything

Measured, not estimated: **traces cost 0.65-0.82 GB per video-hour**, and
**~82% of that is the v7 per-token array** (`b_tok`, T×256×96 fp16 ≈ 49 KB
per 0.25 s step). Traces are therefore *comparable to or larger than the
compressed video they came from*. Naive accumulation dies at ~300 video-hours
on a 244 GB disk.

Hence three tiers, enforced by the compactor:

| tier | contents | GB / video-h | policy |
|---|---|---|---|
| **A** | full: `b_tok` + gate + fix + sig + g | 0.80 | rolling reservoir, cap 120 GB (~150 h) |
| **B** | pooled only: fix + sig + g + where_map (no tokens) | 0.15 | cap 60 GB (~400 h) |
| **C** | evicted | 0 | gone |

A recording enters A, ages into B when A is over budget, and leaves. Tier B
still trains everything except the *learned token pooling*, which is the one
component that needs tokens — so the pooling head trains on a 150-hour
reservoir while the recurrence `f` trains on 550 hours. That asymmetry is
deliberate and should be stated in any result.

**Eviction is coverage-maximising, not FIFO.** Reservoir-sample per source,
and among candidates prefer the recording whose pooled `fix` is *farthest*
from existing reservoir centroids. Label-free, and it protects the tail —
FIFO would silently converge the reservoir onto whatever source is currently
downloading.

Steady state ≈ 22 GB raw (3-shard queue) + 180 GB traces ≈ **200 GB**, inside
the 244 GB free today.

## 4. The shard — unit of the cycle

A shard is **2-5 video-hours** from ONE source: small enough that an ingest
finishes in 30-70 min (so a crash costs ≤1 h), large enough that per-shard
overhead is noise. Layout:

```
stream/raw/<src>_<nnnn>/          videos + shard.json (+ .ready when complete)
stream/traces/<src>_<nnnn>/       vjrec6/ vjrec7/ vjsig6/ (+ .ingested)
```

Marker files are the state machine: `.ready` → `.claimed` (atomic rename by
the ingester) → `.ingested` → raw deleted. A `.claimed` older than 3 h is
treated as a dead worker and reset. Every transition is idempotent.

## 5. Sources and the diversity quota

**Direct-hosted or licence-gated datasets only. No YouTube scraping** — URL
lists rot, and bulk scraping is against terms; you accept each dataset's
licence yourself and the fetcher reads credentials from disk.

| source | hours available | why it is in the mix |
|---|---|---|
| Ego4D | 3,670 | egocentric, unscripted, closest to a robot's own view |
| Kinetics-700 (CVDF mirrors) | ~1,800 | broadest activity variety |
| EPIC-KITCHENS-100 | 100 | dense manipulation, long takes |
| Open-X-Embodiment / DROID | 1,000+ | real robots, many labs and embodiments |
| ActivityNet / Charades | ~900 | untrimmed, multi-event |
| SSv2 | ~200 | verb-centric — but clips are ~4 s, at our window's edge |

**Hard quota: no source may exceed 25% of the ingested pool.** This is the
bridge lesson made structural — bridge's 100 h would have been 94% of the
local corpus, so it was cut to 5 h and sampled one-episode-per-task. The
fetcher round-robins sources and refuses a shard that would breach the cap.

**Near-duplicate rejection** at ingest: pooled `fix` cosine > 0.98 against
anything already in the reservoir ⇒ drop. Label-free, catches re-uploads,
static cameras and near-identical robot retries.

## 6. Basis freeze — do this before shard 1

The token PCA and the predictor calibration are currently **fitted on rcasa**
and are baked into every stored trace. At internet scale that is
indefensible, and refitting mid-run silently makes early and late traces
incomparable.

So: **fit once, on a deliberately diverse bootstrap** (~200 recordings drawn
across every source), write it as `basis_v2`, and freeze it for the entire
life of the run. Every trace records the basis id it was written under; a
trace whose basis id differs is not loadable into the same pool. Any future
refit is a new lineage and a full re-ingest — priced accordingly.

## 7. Trainer — continuous, and guarded against forgetting

Same objectives as SSL_TRAINING.md §5 (span prediction + overlap InfoNCE +
VICReg). Two changes for the streaming setting:

- **Replay mixing:** each batch is 50% newest shard / 50% reservoir sample. A
  pure-stream diet causes catastrophic forgetting; the reservoir is the
  standard counter and we already maintain one for storage reasons.
- **EMA weights are what get evaluated and shipped.** The online weights chase
  the current shard's distribution; the EMA is the thing that has seen
  everything. Checkpoint both, ship the EMA.

Learning rate stays low and flat (no cosine schedule — there is no "end" to
anneal toward). Checkpoint every 2,000 steps, keep the last 5 plus every
evaluated one.

## 8. Evaluator — the regression gate

Every 6 h, freeze the current EMA weights and run the frozen protocol on
**`rcasa_composite_full`, which is sealed and never enters the pool**, plus
whatever held-out shard the compactor last sealed. Report seed-aggregated
prec@support with chance and lift, and PR(z).

Two automatic stops:

- **Collapse:** PR(z) falls below 15 ⇒ pause the trainer, the anti-collapse
  term has lost.
- **Regression:** eval precision drops more than 2 sd across three
  consecutive evals ⇒ pause and alert. Do not let a 3-day run quietly rot.

Every eval writes a row to the ledger with the shard count and video-hours
seen, so the result is always a *curve against data volume* — which is the
actual claim being tested: does this keep improving with more video?

## 9. Failure modes and their handling

| failure | handling |
|---|---|
| disk fills | fetcher backpressure (≤3 uningested shards) + compactor budget; both check free space and refuse rather than crash |
| corrupt download | checksum on `.ready`; failed shard moved to `raw/.bad/` and re-queued once, then skipped |
| ingester dies mid-shard | `.claimed` timeout 3 h → reset; `vjrec8` skips already-written files, so restart is cheap |
| trainer dies | resumes from last checkpoint; ingest is unaffected |
| one source dominates | 25% quota enforced at fetch time |
| link rot | source catalog records last-good date; a source failing 3 shards is disabled and logged |
| silent no-op ingest | verify **artifact counts on disk**, never exit codes (this has bitten before) |
| clips shorter than the 4 s window | counted and reported per shard; a source whose drop rate >50% is disabled (this is what makes movi_e useless) |

## 10. What this does not do

- No distillation. Ingest stays at 4.17× realtime; that is the ceiling.
- No labels, no VLM, no text supervision in the loop. Text layers are
  read-side and trained separately, per SSL_TRAINING.md §8.
- No cross-view positives, no tempo warping. Standing bars apply unchanged.
- No automatic promotion. A streamed checkpoint becomes the shipped head only
  when it beats the sealed-corpus bar in a snapshot you approve.

## 11. Build order, when approved

1. `stream/catalog.yaml` + fetcher for ONE source (EPIC, smallest real one)
2. ingester daemon wrapping `vjrec8` + verification + raw deletion
3. compactor with tiering and coverage-maximising eviction
4. trainer with replay mixing and EMA
5. evaluator + ledger curve
6. add sources one at a time, watching the quota and drop-rate reports

Steps 1-2 alone are a working cycle (fetch → ingest → delete) and are worth
running for a week before the rest exists, because they produce the trace
pool everything else consumes.
