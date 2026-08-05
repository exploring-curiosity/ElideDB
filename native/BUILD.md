# ElideDB memory layer — build plan, step by step, each one graded

Every step below gets its OWN acceptance measurement — a number that
says how well *that layer* did its job, independent of any downstream
QbE or query benchmark. A step is not "done" because it runs; it is
done when its own metric is measured and recorded here.

Rationale for the layering is in the session record: the architecture
is drawn from where this project measurably failed (segmentation was
worth ~0.36 yield and was being read from dataset metadata) plus
established findings on human memory (event segmentation by prediction
error; schema = store deviations; chunk vocabulary acquired by
exposure; consolidation by replay).

## Hard rules

- **No dataset metadata anywhere in the write path.** Boundaries,
  labels, counts, identities: none. The only structure allowed is what
  a customer upload carries by itself (files, timestamps).
- **Truth is EVAL-ONLY.** truth.parquet / graded.parquet / Bridge meta
  may be read by graders, never by a build step.
- **One encode function** shared by write and read. Divergence there
  invalidates every comparison.
- **Every threshold fitted from the data at hand**, never typed as a
  constant chosen by looking at results.
- **Each step's grade is recorded below before the next step starts.**

## Status ledger

| # | Step | Kind | Grade metric | Result |
|---|---|---|---|---|
| 1 | Ingest | keep | — | inherited, solid |
| 2 | Sample | keep | — | inherited |
| 3 | Encode (shared fn) | refactor | write/read vector identity | PENDING |
| 4 | Score surprise | NEW | boundary-signal AUC vs truth boundaries | PENDING |
| 5 | Segment | NEW | span F1 / boundary MAE vs truth spans | PENDING |
| 6 | Encode units | refactor | unit-vector stability across views | PENDING |
| 7 | Discover vocabulary | NEW | cluster purity vs truth labels (eval-only) | PENDING |
| 8 | Assign | NEW | assignment agreement across views | PENDING |
| 9 | Salience weighting | NEW | bytes vs information retained | PENDING |
| 10 | Index | extend | recall@k of the index vs exact scan | PENDING |
| 11 | Relate | NEW | containment/adjacency correctness | PENDING |
| 12 | Commit | keep | — | inherited |
| R3 | Segment query | NEW | same as step 5 on query clips | PENDING |
| R8 | Verify structure | wire-up | alignment score on known-same pairs | PENDING |
| R10 | Calibrate & cut | wire-up | cut quality vs oracle cut | PENDING |
| R12 | Log | NEW | write cost per query | PENDING |
| 13 | Consolidate | NEW | vocabulary improvement across replays | PENDING |

## Step records

(appended as each step is built and graded)

---

### STEP 3 — Encode (shared function) — **PASS**

`native/encode.py`. One `decode()` + `encode_spans()` used by both
write and read; nothing else may encode.

Grade (`--selftest`, sim media):
- decode deterministic across independent runs: **True** (byte-identical)
- write-path vs read-path encoding of the same spans: **max|diff| 0.00, min cos 1.000000**
- batch invariance (batch 4 vs batch 1): **max|diff| 0.00**

Verdict: the write/read seam is closed structurally, not by convention.
Any later comparison is now meaningful. No caveats.

---

### STEP 4 — Score surprise — **PARTIAL: passes on bench, weak on sim**

`native/surprise.py`. Per-time prediction residual over the latent
series at the 2 s scale. Eight estimators, none fitted to anything.
Truth boundaries used ONLY as the grader (±1.5 s tolerance).

AUC separating boundary-adjacent windows from interior windows:

| estimator | bench | sim |
|---|---|---|
| adjacent novelty | 0.792 | 0.603 |
| history-mean residual | 0.836 | 0.560 |
| **linear-extrapolation residual** | **0.844** | 0.589 |
| two-sided (past vs future block) | 0.769 | **0.616** |
| Foote checkerboard w=2 | 0.810 | 0.540 |
| Foote checkerboard w=4 | 0.810 | 0.445 |
| Foote checkerboard w=8 | 0.691 | 0.351 |
| Foote multi-scale | 0.800 | 0.429 |

Sample: bench 6,284 boundary / 1,859 interior windows; sim 678 / 323.

**Honest reading.** On bench the signal is real and strong (0.844) -
but bench boundaries are cuts between separately recorded demos, so
this is closer to shot-change detection than to semantic event
segmentation. Valuable (it is exactly the segmentation previously read
from dataset metadata) but the easy case.

On sim the best estimator reaches only 0.616. Sim boundaries are
action transitions *inside one continuous take* with no scene change,
which is the genuinely hard case, and the classic changepoint method
(Foote) performs WORSE than adjacent differencing there - at w=8 it is
0.351, i.e. actively anti-correlated, because the kernel's block
assumption is wrong when the visual scene is constant across the
boundary.

Consequence for step 5: segmentation will be usable on bench and poor
on sim. Recording the split rather than averaging it away; sim's
segmentation quality is the number to watch downstream.

---

### STEP 4 REVISION — the grade was inflated by a pooling confound

While extending step 4 to a third domain (Oxford driving), the same
estimator measured 0.734 one way and 0.611 another. Chasing that
discrepancy exposed a flaw in the metric, not in the code:

**Pooling window scores across media inflates AUC.** Media differ in
baseline surprise level AND in how many boundaries they contain. A
medium that is globally "hotter" and also boundary-dense contributes
mostly positives at high values, which the pooled AUC reads as
discrimination. The confound-free measure is AUC computed WITHIN each
medium, then averaged.

**Per-media AUC (mean ± sd across media), corrected:**

| scale / estimator | sim | oxford |
|---|---|---|
| 0.5 s two-sided | 0.583 ± 0.20 | 0.578 |
| 0.5 s **dir_change** | **0.642 ± 0.06** | **0.742** |
| 1.0 s extrap | 0.639 ± 0.11 | 0.650 |
| 2.0 s **extrap** | **0.705 ± 0.17** | 0.505 |
| 2.0 s two-sided | 0.419 ± 0.16 | 0.402 |

Every earlier step-4 number in this file (bench 0.844, sim 0.616 ->
0.747) was computed with the pooled metric and is therefore
OPTIMISTIC. The corrected sim figure is ~0.58-0.71 depending on
configuration, with a standard deviation across episodes of 0.06-0.20
- i.e. on some episodes the signal is near-random.

**Second finding: no configuration wins on both domains.** sim's best
is 2 s extrapolation residual (0.705) which is near-chance on driving
(0.505); driving's best is 0.5 s direction-change (0.742) which is
mid-tier on sim (0.642). Choosing per corpus would be exactly the
domain knowledge this build forbids.

**Status: STEP 4 FAILS its own bar.** The honest reading is that a
frozen encoder's latent trajectory carries a weak, domain-dependent
boundary signal (~0.6-0.7 per-media AUC, high variance), not a
reliable one. Steps 5+ stand on this, so it must be resolved before
they are built rather than after.
