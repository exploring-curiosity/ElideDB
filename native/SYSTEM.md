# SYSTEM.md — the working system, as of 2026-08-17

What ships, what it scores, how to run it, and what is known dead. Start here.
Companion docs: `EXPERIMENTS.md` (every measured dead end — grep it before
proposing anything), `SSL_TRAINING.md` (the self-supervised runbook, executed
and concluded), `STREAM_PIPELINE.md` (designed, not started).

## 1. What the system does

Answers "when did something like this happen?" over stored video, label-free
at write and at serve. Two products from one store:

| use case | metric that matters | status |
|---|---|---|
| **UC1** — robot memory layer beside a VLA | P@10, latency | **shipping** |
| **UC2** — mine past failures for retraining | P@support | **blocked at 0.43** |

## 2. The numbers (sealed corpus: `rcasa_composite_full`, 91% unseen tasks, chance 0.029)

| k | precision |
|---|---|
| P@1 | **0.950** |
| P@5 | **0.886** |
| P@10 | **0.798** |
| P@20 | 0.567 |
| P@support | 0.406 |

Read latency, unified `rcasa` store (3,556 recordings, 16–256 step episodes):

| config | p50 | p99 | P@10 |
|---|---|---|---|
| exact full scan | 20.2 s | 69 s | 0.765 |
| **M=100, band 0.25 (default)** | **554 ms** | **1.9 s** | **0.752** |
| M=50, band 0.25 | 286 ms | 947 ms | 0.728 |
| M=200, exact | 1.1 s | 3.8 s | 0.765 |

Write: **14.4 compute-min per video-hour** (4.17× real-time), one pass.
Of that, V-JEPA encoder 63%, predictor 22%, SigLIP 4%, decode 3%.

## 3. The shipped path, end to end

**Write** — `relmo.vjrec8`, one encoder pass, produces three artifacts per
recording (bit-identical to the older three-script path, verified over 5,688
arrays):

```
python3 -m relmo.vjrec8 --dataset <name> --fp16 --pca-suffix ""
```
- `vjrec6/<ds>_L6` — pooled `pred_change`/`obs_change` (fp32) + `where_map`
- `vjrec7/<ds>_L6` — per-token `b_tok` (fp16, PCA-96) + `gate`
- `vjsig6/<ds>` — SigLIP frame features (fp32)

**Read** — `relmo.vjstore`. A *store* is one deployment's memory. Five exist;
`rcasa` unifies all four rcasa-family datasets (owner ruling: the
train/eval split is experiment-side, not product-side).

```python
from relmo.vjstore import Store
st = Store("rcasa")                      # load once
hits = st.query(fix_trace, sig_trace, top_k=10)   # (id, score), ~554 ms
```

Representation is **frozen and untrained**: per-recording z-scored
`fix` (gate-pooled V-JEPA obs_change, 1024-d) ⧺ `sig` (SigLIP, 768-d),
matched by anchored symmetric2 DTW. Query encode is free for a robot
querying its own memory — the trace already exists.

**Benchmark** (always reports fidelity beside latency):
```
python3 -m relmo.vjstore --bench --store rcasa --queries 60
```

## 4. The load-bearing facts

- **Nothing is trained.** V-JEPA 2 and SigLIP 2 frozen; no head. Two full
  self-supervised training runs measured null (+0.006 against a ±0.021 CI).
  This is a feature for generalization: nothing to retrain or drift when
  pointed at a new corpus.
- **It is better on unseen tasks than seen ones** on the sealed corpus
  (0.404 vs 0.375) — no generalization gap.
- **Normalisation is the single biggest measured gain** in the read path
  (whitened-512d 0.376 → z-scored-1792d 0.401). Whitening helps a single
  channel's cosine and *hurts* a fused space.
- Corpus-fitted constants still baked into stored traces: token PCA and
  predictor calibration, both fitted on rcasa. Refitting = full re-ingest.

## 5. Known dead (see EXPERIMENTS.md for numbers)

Every label-free read-side mechanism has been measured: **graph diffusion**
(null), **mutual-verification re-ranking** (hurts), **PRF / query expansion**
(hurts — anchors are near-copies of the query), **where_map channel** (helps
seen, hurts unseen), **whitening a fusion** (hurts). Head training: null under
two very different objectives.

UC2's 0.43 → 0.7 gap therefore needs **features or data**, not read-side work.
There is no third door; all of them were checked.

## 6. Module map

| module | role |
|---|---|
| `vjrec8` | **write path** (one pass) |
| `vjstore` | **read path** (stores, prefilter, banded DTW) |
| `vjmatch` | DTW, arc-length resample. `band` is per-reference (v6 bug fixed 2026-08-17) |
| `vjrec6/7`, `vjsig6` | legacy per-artifact writers; superseded by vjrec8, kept for reference |
| `vjsnap` | pin a read path by sha256 (`v9` = the shipped one) |
| `vjtest`, `vjreps` | evaluation harnesses (chance + lift + seen/unseen breakdown) |
| `prep_data` | raw downloads → registry corpora |
| `vjssl`, `vjmine`, `vjdiffuse` | **experiments that concluded NULL.** Kept as the record; not in the serve path |

## 7. Evaluation discipline (non-negotiable, learned the hard way)

- **Grade `rcasa_composite_full` with `--event-key task`.** The default
  `group_key` parser cannot read compositional names and dumps 900/1152
  records into one bucket, driving chance to 0.543 and making prec
  uninterpretable.
- **Quote chance and lift** whenever pools differ; a bare precision is not
  comparable across corpora.
- **Never ship a speedup without a fidelity column.** Two defects were caught
  exactly this way this week.
- Seed-aggregate; a best checkpoint is not a result.
