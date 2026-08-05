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
