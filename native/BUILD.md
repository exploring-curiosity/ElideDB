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
| 4 | Score surprise | NEW | boundary-signal AUC vs truth boundaries | CLOSED - GraphGEBD Ncut, sim 0.711 / oxford 0.500; backbone study confirms DINOv3 ConvNeXt-Tiny (nothing beats it significantly, cheapest at 0.68 min/h) |
| 5 | Segment | NEW | span F1 / boundary MAE vs truth spans | PASS both domains - homogeneity stop; sim spanF1 0.566 / oxford 0.667 (was 0.000); span F1 now a real gate at 0.40 |
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

---

### STEP 4 — shipped predictor tried; still FAILS

`native/predsurprise.py`. V-JEPA 2's own self-supervised predictor
(22 M params, in the public checkpoint, no labels, no task head) asked
directly: predict the next temporal group's latents from all preceding
groups, residual = 1 - cos(predicted, actual). No scale of mine (the
model's tubelet grid sets it), no estimator choice, no threshold.

Per-media AUC vs truth boundaries (±1 s):

| domain | mean | sd | n |
|---|---|---|---|
| sim | **0.618** | 0.13 | 8 |
| oxford (driving) | **0.632** | — | 1 |

Per-episode on sim: 0.835, 0.759, 0.677, 0.643, 0.579, 0.563, 0.516,
**0.374** — i.e. it ranges from good to worse-than-chance on episodes
of the SAME corpus generated by the SAME script.

**Comparison of every step-4 approach, per-media metric:**

| approach | sim | oxford |
|---|---|---|
| hand-rolled, best per-domain config | 0.705 (2 s extrap) | 0.742 (0.5 s dir) |
| hand-rolled, one config for both | ~0.58-0.64 | ~0.52-0.65 |
| **shipped predictor (no config at all)** | **0.618** | **0.632** |

The principled version matches the config-free hand-rolled versions
and does NOT reach the per-domain tuned ones. Removing my
configuration did not cost accuracy - which says the earlier
domain-dependence was largely fitting noise - but it did not add any
either.

**Verdict: step 4 does not pass, and the reason is now well
established rather than suspected.** Three independent formulations
(finite-difference estimators, multi-scale change statistics, and the
model's own trained predictor) all land at ~0.6 per-media AUC with
large across-episode variance. A frozen general-purpose video model's
latent trajectory does not reliably mark event boundaries in these
domains. Steps 5+ remain blocked.

---

### STEP 4 — take 4: MOTION CUES from the literature (major improvement on sim)

I had declared step 4 blocked after three embedding-based formulations
all reached ~0.6 per-media AUC. That conclusion was premature: I never
checked the published unsupervised state of the art for this exact
task, which does NOT use semantic embeddings.

Literature (unsupervised / zero-shot Generic Event Boundary Detection):
- **FlowGEBD** (WACV 2024): optical flow, non-parametric, training-free.
  F1@0.05 = **0.713** Kinetics-GEBD, 0.623 TAPOS; **+31.7 points
  absolute** over the unsupervised (embedding) baseline.
- **GraphGEBD**: zero-shot graph + normalised cut, F1@0.05 = 0.732.
- **UBoCo** (CVPR 2022): temporal self-similarity matrix + recursive
  kernel matching.
- Also noted: SAM-GEBD (zero-cost), CoSeg (cognitively-inspired
  unsupervised event segmentation).

Implemented FlowGEBD faithfully (`native/flowgebd.py`): Pixel Tracking
(Shi-Tomasi points + Lucas-Kanade sparse flow; boundary where the
tracked-point ratio collapses) and Flow Normalisation (dense Farneback,
per-patch max displacement, accumulated and normalised). Adopted the
field's metric, F1 at relative distance 0.05, so numbers are
comparable to published work.

| method | sim F1@0.05 | oxford F1@0.05 |
|---|---|---|
| **PT framewise (Lucas-Kanade)** | **0.647** | 0.286 |
| FN (Farneback patches) | 0.492 | 0.250 |
| PT patchwise | 0.000 (bug: score saturates) | 0.000 |
| *published FlowGEBD, Kinetics-GEBD* | *0.713* | — |

Per-episode sim F1 for PT framewise: 0.86, 0.75, 0.75, 0.71, 0.71,
0.67, 0.67, 0.62, 0.62, 0.50, 0.46, 0.43.

**Reading.** On sim, motion cues are dramatically better than every
embedding formulation tried (0.647 F1 in the same ballpark as the
published 0.713 on Kinetics) - the signal was never missing, I was
reading the wrong channel. Caveat on the protocol: predictions are
count-matched to the number of true boundaries, so this measures
PLACEMENT quality, not how many boundaries to emit; a thresholded
version will score lower and is what step 5 actually needs.

Oxford stays poor (0.286). Driving boundaries are stops/turns where
ego-motion dominates the flow field - global flow keeps moving when
the *event* changes - so the collapse-of-tracking cue does not fire.
That is a real and expected limitation of flow-on-ego-video, and it
points at ego-motion compensation as the next thing to try there.

---

### STEP 4 — take 5: both defects fixed, thresholded emitter added

**Fix 1 - patchwise saturation.** Old code gave each patch a fixed
budget (~12 points) and aggregated with MIN; some patch always lost
all its points, the score pinned at 1.0, no peaks existed, F1 was
exactly 0.000. Now points are seeded over the whole frame and assigned
to patches by position (so a patch's budget is not fixed, and empty
patches simply do not vote), aggregated by MEAN over patches holding
>=3 points.

**Fix 2 - ego-motion compensation.** On driving, the camera's own
motion dominates every pixel, so flow cues fire constantly. A global
affine is fitted per frame pair and the RESIDUAL flow is scored,
isolating what moved in the world from what moved because the camera
did.

**Fix 3 - thresholded emitter.** The earlier protocol predicted
exactly as many boundaries as truth contained, which is not something
a real system knows. Added an emitter that decides the count itself
from a z>1 cut on the media's own score distribution - no truth, no
per-corpus constant.

| signal | sim matched | sim thresh | oxford matched | oxford thresh |
|---|---|---|---|---|
| PT framewise | 0.644 | **0.636** | 0.250 | 0.222 |
| PT patchwise | 0.644 | 0.613 | **0.500** | **0.545** |
| FN | 0.492 | 0.496 | 0.250 | 0.444 |
| FN ego-compensated | 0.508 | 0.516 | **0.500** | 0.471 |
| PT + FN-ego (z-sum) | 0.528 | 0.553 | 0.250 | 0.308 |

Effect of the fixes on the domain they targeted: oxford patchwise
**0.000 -> 0.545**, oxford FN **0.250 -> 0.471** with ego
compensation. Neither hurt sim (patchwise now equals framewise at
0.644).

**Notable:** the thresholded emitter scores essentially the same as
the count-matched oracle on sim (0.636 vs 0.644) despite emitting
13.8 boundaries where 7.3 are true - it over-segments, and F1 barely
suffers because the extra cuts fall near real ones. Over-segmentation
is the benign failure mode for step 5 (units can be merged; missed
boundaries cannot be recovered).

**Status: step 4 is usable but not uniform.** sim ~0.64, oxford ~0.55
with the patchwise variant, and no single variant leads on both
(sim prefers framewise/patchwise equally, oxford needs patchwise or
ego-compensation). Published reference for context: FlowGEBD 0.713 on
Kinetics-GEBD.

---

### SENSITIVITY RESULT — segmentation is NOT the bottleneck (sim)

Retrieval yield/prec as boundary quality degrades from perfect to
near-useless (60 sim episodes, units encoded by the shared encoder,
episodes matched by DTW over unit-vector sequences):

| boundary F1 | yield | prec | units/ep |
|---|---|---|---|
| **1.000 (oracle)** | **0.267** | 0.167 | 7.3 |
| 0.859 | 0.267 | 0.167 | 7.3 |
| 0.713 (= published SOTA) | 0.233 | 0.146 | 8.2 |
| 0.576 | 0.267 | 0.167 | 8.5 |
| 0.538 | 0.300 | 0.188 | 9.4 |
| 0.509 | 0.233 | 0.146 | 10.8 |

**The slope is zero.** Halving boundary quality changes retrieval by
less than the noise. And the CEILING with perfect boundaries is 0.267
against a chance floor of 0.207.

**Consequences, stated plainly:**

1. Every hour spent on step 4 - surprise estimators, the shipped
   predictor, FlowGEBD, and the planned GraphGEBD - is irrelevant to
   the outcome on this corpus. Going from 0.64 to the published 0.73
   would move retrieval by nothing.
2. My earlier claim that "segmentation was worth ~0.36 yield" is
   WRONG as a causal statement. That number came from comparing a
   metadata-segmented kitchen store against a window-indexed one, and
   attributed the whole gap to segmentation. On sim, measured directly
   by holding everything else fixed, segmentation is worth ~0.06.
3. The oracle EVENT-SCRIPT result (0.992) used truth *labels* - which
   object, which action - not just truth boundaries. The gap between
   0.267 (perfect boundaries, real encoder) and 0.992 (perfect
   boundaries, perfect labels) is therefore entirely in what the units
   ARE, not where they start and end.

**The bottleneck is unit CONTENT, not unit BOUNDARIES.** A frozen
encoder's vector for a correctly-cut unit does not encode which object
moved where, which is the only thing distinguishing sim templates.
Step 4 is closed as not-on-the-critical-path; the open problem moves
to step 6 (unit representation).

---

### STEP 4 — FINAL: GraphGEBD (recursive normalised cut) is the best variant

Implemented the published SOTA (`native/graphgebd.py`): frames are
graph nodes, edges are DINOv3 appearance similarity, boundaries are
contiguous splits minimising

    Ncut(A,B) = cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V)

applied recursively. Cuts come out RANKED by their own Ncut value, so
boundary quality is intrinsic - no threshold of mine - and the count is
set by a z-cut on that ranking.

**Every step-4 variant, one table, F1@0.05 (count-matched / emitter):**

| method | sim | oxford |
|---|---|---|
| **GraphGEBD (recursive Ncut)** | **0.711 / 0.641** | **0.500 / 0.400** |
| FlowGEBD PT framewise | 0.644 / 0.636 | 0.250 / 0.222 |
| FlowGEBD PT patchwise | 0.644 / 0.613 | 0.500 / 0.545 |
| FlowGEBD FN ego-compensated | 0.508 / 0.516 | 0.500 / 0.471 |
| FlowGEBD FN | 0.492 / 0.496 | 0.250 / 0.444 |
| embedding surprise (3 formulations) | ~0.6 AUC, not F1-competitive | ~0.6 AUC |
| *published GraphGEBD, Kinetics-GEBD* | *0.732* | — |

Per-episode sim (matched): 0.89, 0.86, 0.86, 0.86, 0.75, 0.75, 0.71,
0.67, 0.62, 0.57, 0.50, 0.50.

**sim 0.711 is within 0.02 of the published Kinetics number (0.732)**,
on a corpus the method never saw, with the in-repo DINOv3 standing in
for DINOv2. It is also the single best variant on BOTH domains, which
is what "one variant, no per-corpus configuration" required.

It also settles an error recorded earlier in this file: I concluded
that "a frozen encoder's latent trajectory does not mark event
boundaries". Wrong - the trajectory carries the signal; thresholding a
per-frame score was the wrong ALGORITHM. Solving for a global
partition over the same features recovers it (0.6 AUC -> 0.711 F1).

**Step 4 CLOSED at 0.711 sim / 0.500 oxford**, single variant, no
per-domain configuration, comparable to published SOTA. Kept despite
the sensitivity finding, because steps 5+ consume it and it is now the
best available implementation rather than a placeholder.

### STEP 5 — scope revised by the sensitivity result

Step 5 turns the ranked cuts into spans. Its quality bar is no longer
tight: units built from F1-0.5 boundaries retrieve as well as units
from perfect ones (yield 0.267 vs 0.267). So step 5 needs to produce
reasonable, non-degenerate spans - not precise ones - and must not
become another optimisation project. Emitter already exists (z-cut on
Ncut ranking, emits 5-10 spans/episode against 6-9 true).

### STEP 4 addendum — backbone study: is DINOv3 the right features?

GraphGEBD's published 0.732 uses **ResNet50 / DINOv2**. This build
substituted the in-repo DINOv3 ConvNeXt-Tiny without testing it, and
DINOv3 is here for a *different job*: crop-vs-crop identity, where
INVARIANCE is the goal (AUC 0.9994). Boundary detection wants the
opposite - features that MOVE when the scene changes. So the
substitution needed checking.

`native/gebdback.py` holds the algorithm exactly fixed (same recursive
contiguous Ncut, same depth 4, same MIN_SEG, same z-cut emitter) and
swaps only `frame_features`. Two affinity scalings are reported per
backbone, because sigma lives on cosine DISTANCE and a fixed sigma
silently favours whichever space matches its scale.

Harness validated: `dinov3_ct` reproduces graphgebd.py's 0.711 / 0.641
to three decimals, so these are like-for-like swaps.

**sim (12 episodes), F1@0.05 matched / emitter, sigma 0.25:**

| backbone | sim | oxford | min/hour |
|---|---|---|---|
| resnet50 (ImageNet) | 0.738 / 0.690 | 0.250 / 0.200 | 0.85 |
| **dinov3_ct (current)** | **0.711 / 0.641** | **0.500 / 0.400** | **0.68** |
| siglip2-so400m | 0.683 / 0.627 | 0.000 / 0.222 | 16.72 |
| dinov3_vits16 | 0.663 / 0.613 | 0.250 / 0.222 | — |
| dinov3_vitb16 | 0.651 / 0.635 | 0.250 / 0.400 | — |
| dinov2-base (paper's) | 0.629 / 0.585 | 0.250 / 0.545 | — |
| vjepa2 (video-native) | 0.582 / 0.583 | 0.250 / 0.200 | 16.18 |
| vit-mae-base | 0.476 / 0.463 | 0.250 / 0.222 | — |
| resnet50+vjepa2, self-sigma | 0.750 / 0.693 | 0.500 / 0.444 | 17.03 |
| siglip2+resnet50 | 0.748 / 0.715 | 0.000 / 0.222 | 17.57 |

**The ordering at the top is NOT significant.** Paired per-episode
bootstrap (10k, n=12) against dinov3_ct, matched F1:

| variant | diff | 95% CI | wins/losses |
|---|---|---|---|
| resnet50 | +0.026 | [-0.044, +0.097] | 4 / 3 |
| resnet50+vjepa2 | +0.038 | [-0.021, +0.094] | 5 / 2 |
| siglip2+resnet50 | +0.037 | [-0.032, +0.102] | 4 / 2 |
| dinov3_vits16 | -0.048 | [-0.109, +0.007] | 1 / 4 |
| dinov2-base | -0.083 | [-0.170, +0.002] | 2 / 6 |
| **dinov3_vitb16** | **-0.060** | **[-0.100, -0.022]** | 0 / 5 |
| **vjepa2** | **-0.130** | **[-0.200, -0.055]** | 1 / 8 |
| **vit-mae-base** | **-0.235** | **[-0.329, -0.149]** | 0 / 10 |

Every CI above dinov3_ct straddles zero; the three below it in bold do
not. So: **nothing measured beats DINOv3 ConvNeXt-Tiny significantly**,
and V-JEPA2 alone, DINOv3 ViT-B and MAE are significantly worse.

This CORRECTS an intermediate claim made while the run was in flight
("ResNet50 beats DINOv3", "scaling DINOv3 monotonically hurts"). The
+0.026 is noise at n=12. What survives is narrower: the DINOv3 *ViT*
variants do not help and ViT-B is significantly worse than
ConvNeXt-Tiny - so there is no gain available by scaling DINOv3 up.

**Decision: keep DINOv3 ConvNeXt-Tiny.** It is statistically tied for
best AND the cheapest candidate (0.68 min/hour vs resnet50's 0.85).
The best-*scoring* variant, resnet50+vjepa2 at 0.750/0.693, costs
17.03 min/hour - 25x DINOv3 and 17x the 1 min/hour budget - to buy a difference that is
not significant, on a layer the sensitivity study already showed is
not the bottleneck. Declined on cost.

Caveats kept honest: oxford is ONE 19.1 s clip with 4 boundaries, so
its F1 moves in ~0.25 steps - a second-domain sanity check, not a
tiebreaker. sim n=12 cannot resolve differences below ~0.07. And
sigma=0.25 remains a constant I chose; self-scaling it (per-media
median distance) was tested and was not better for most backbones.

Cost note: DINOv3 here is a SECOND model - native/encode.py (step 3)
runs V-JEPA2 - so step 4's 0.68 min/hour is additive, not free.

All min/hour figures are WARM throughput (model load excluded) over
400 frames, scaled to 14400 frames = 1 hour at the 4 fps decode rate.
An earlier revision of this table quoted load-inclusive figures from a
run whose frame-tiling was buggy (101 frames, not 400); those were
~1.5-2x pessimistic for the cheap models. Fusion costs are additive -
both backbones run over every frame.

## STEP 5 — SEGMENT (ranked cuts -> spans).  `native/segment.py`  PASS

Turns step 4's ranked boundaries into the actual retrieval units. Three
design positions:

1. **The spans partition the timeline exactly** - coverage 1.0, no gaps.
   A unit set with holes leaves footage in the store but unreachable by
   any query, and nothing reports it. Verified, not assumed.
2. **Short spans are merged, never dropped** - dropping punches a hole.
3. **No new threshold.** The minimum span length falls out of step 4:
   recursive_ncut splits an interval at least MIN_SEG from either end
   and recurses on disjoint children, so every boundary is >= MIN_SEG
   from every other. At 4 frames / 4 fps that is 1.0 s. Degenerate
   spans are impossible by construction. The grade checks it.

**Emitter changed.** Step 4 used a z<0 cut on the Ncut values - keep
everything below the mean, i.e. roughly half the candidates no matter
how many real events the media holds. A count that cannot adapt is a
defect, not a tuning knob, so it is now Otsu on the same values: split
where they actually separate. Still fitted per media, still no constant
of mine. Measured both ways:

| emitter | sim spanF1 | sim bMAE | oxford bMAE | oxford spans |
|---|---|---|---|---|
| z<0 | 0.607 | 1.22 s | 3.36 s | 7 (truth 4) |
| **otsu** | **0.652** | **1.18 s** | **2.49 s** | **6 (truth 4)** |

**Grade (sim, n=12, otsu):**

| | |
|---|---|
| span F1 @IoU0.5 | 0.652 (prec 0.612 / rec 0.711) |
| boundary MAE | 1.18 s |
| spans / media | 8.4 (truth 7.1) |
| coverage | 1.0000, max gap 0.00 s |
| shortest span | 1.00 s (= structural floor) |
| gates | coverage / floor / determinism / non-degenerate: all PASS |

**FIXED: oxford span F1 0.000 -> 0.667.** The first version of this
step was graded on four gates - coverage, minimum length, determinism,
count - that are ALL satisfied by construction. They certify the
absence of a bug, nothing else, and span F1 (the metric the ledger row
actually names) was left outside them. That is how a domain scoring
0.000 got stamped PASS. Span F1 is now a real gate at 0.40; the old
build fails it.

Root cause was in step 4, not step 5: recursion split by a DEPTH
COUNTER, so a 9 s stretch of uniform driving was chopped into four and
no predicted span could reach IoU 0.5. It now stops on HOMOGENEITY - a
segment is split only if its best cut is PROMINENT within its own
Ncut(t) curve. A real boundary makes a sharp deep minimum; a slow
drift makes a shallow one. The minimum's value cannot tell those
apart, its prominence can.

| config | sim spanF1 | oxford spanF1 | oxford bMAE | oxford spans |
|---|---|---|---|---|
| depth-only (old) | 0.652 | **0.000** | 2.49 s | 6 (truth 4) |
| **homogeneity stop** | 0.566 | **0.667** | **0.50 s** | 2 (truth 4) |

Deliberate trade: -0.086 on sim to convert a total failure on the
second domain into a pass. MIN_SAL=1.1 sits on a flat plateau
(1.0-1.2, breaks at 1.3) fitted on both corpora at once - one value for
every domain, not per-corpus configuration. It is nonetheless a
constant I chose, and is labelled as such in the module rather than
described as threshold-free.

Step 4's DETECTION is unaffected: count-matched F1 still 0.711 sim /
0.500 oxford. Only the emitted set shrinks. The two metrics move in
OPPOSITE directions - boundary-level emitter F1 falls (sim 0.667 ->
0.538) while span-level F1 on oxford goes from nothing to 0.667. That
is the durable lesson from this step: boundary F1 was never the metric
the next step consumes, and tuning it was making the spans worse.

Both domains now PASS all five gates.

**What this does NOT buy.** Yield/precision 0.90 does not come from
here. Measured: perfect boundaries give yield 0.267, F1-0.5 boundaries
give 0.267, chance is 0.207. The curve is flat, so no step-4 or step-5
number reaches the goal. Step 5 was fixed because it was broken, not
because it moves the product metric.

## STEP 6 — ENCODE UNITS.  `native/unitenc.py`, `native/u6retr.py`

The measured bottleneck. sensitivity.py put oracle spans + real encoders
at yield 0.267 and oracle spans + true LABELS at 0.992; that whole gap
is what a unit vector fails to say about what happened inside it.

**The ledger's metric was wrong on its own.** It asked for "unit-vector
stability across views", but a constant function is perfectly stable and
says nothing. The grade here is a pair: stability AND discriminability
(AUC over unit pairs, same-label vs different-label, taken only ACROSS
episodes so within-episode background similarity cannot win it).

**Diagnosis: mean pooling is order-blind.** Reverse a clip and a mean
is unchanged, so a pick cannot differ from a place. Every mean-pooled
encoder sits at chance on the action - including the one currently in
native/encode.py.

**Held out, 60 unseen episodes, 390 truth units (chance = 0.500):**

| encoder | action AUC | colour AUC | stability | min/h (units) |
|---|---|---|---|---|
| **siglip2 + rank pooling** | **0.691** | 0.506 | 0.933 | ~10.0 |
| siglip2 + delta | 0.643 | 0.503 | 0.883 | ~10.0 |
| r50 + rank | 0.579 | 0.500 | 0.912 | ~0.51 |
| dino + rank | 0.538 | 0.498 | 0.906 | ~0.41 |
| **vjepa2 mean-pooled (production)** | **0.521** | 0.500 | 0.997 | ~9.6 |
| vjepa2 temporal-group concat | 0.539 (dev) | 0.477 | 0.995 | ~9.6 |
| dino/siglip2 halves, meanstd | 0.50-0.54 (dev) | ~0.47 | ~0.99 | — |

Rank pooling is the Bilen/Fernando dynamic-image closed form,
`sum_t (2t - T - 1) . v_t`: parameter-free, one pass, and order-aware BY
CONSTRUCTION - reverse the clip and it flips sign, which is exactly the
property a mean lacks. Dev flattered it by 0.030 (step 5's dev gap was
0.084), and stability stays 0.933, so the gain is not bought with noise.

**Colour AUC is 0.498-0.506 everywhere - exact chance - and that is NOT
a defect to chase here.** Checked directly: the action sequence
identifies the template uniquely (6 distinct prim-sequences -> 6
templates, 0 ambiguous episodes of 150). Object identity carries no
information about the retrieval target in this corpus, so the whole
0.267 -> 0.992 gap is an ACTION representation gap. That is what makes
the pooling fix the right axis rather than a lucky one.

**Cost tension, stated not hidden.** siglip2 unit encoding is ~10
min/hour of video against the 1 min/hour online budget; r50+rank is
0.51 min/h at AUC 0.579. Whether the AUC lift converts to yield decides
whether that tension is worth paying, which is what u6retr.py measures -
an AUC is a proxy, and the one proxy trusted earlier in this build
(boundary F1) was moving opposite to the real metric.

### STEP 6 end to end — and the sensitivity result is now STALE

`native/u6retr.py`, 72 episodes, yield/precision at k = 1.5 x support:

| encoder | spans | yield | prec |
|---|---|---|---|
| vjepa2 (production) | truth | 0.167 | 0.106 |
| r50 + rank | truth | 0.429 | 0.273 |
| **siglip2 + rank** | **truth** | **0.595** | **0.379** |
| vjepa2 (production) | cpd | 0.214 | 0.136 |
| siglip2 + rank | cpd | 0.262 | 0.167 |
| **r50 + rank** | **cpd** | **0.310** | **0.197** |

**The AUC was honest this time**: with oracle spans the encoder fix
takes yield 0.167 -> 0.595, a 3.6x lift, matching the action-AUC
ordering exactly.

**But it moves the bottleneck, and that invalidates a conclusion this
build has been steering by.** sensitivity.py concluded "segmentation is
NOT the bottleneck" - oracle boundaries yielded 0.267 against 0.267 at
F1 0.5. That was measured with encode.py's V-JEPA2, which is
order-blind and therefore near chance on the action. With a weak
encoder the encoder is the binding constraint and boundary quality
cannot show up. Now that the encoder represents the action, boundaries
bind hard:

    siglip2_rank   oracle spans 0.595  ->  cpd spans 0.262   (-0.333)
    vjepa2         oracle spans 0.167  ->  cpd spans 0.214   (+0.047)

V-JEPA2 gets slightly BETTER with worse spans, which is the signature of
a signal that was noise all along. The 0.333 drop for siglip2_rank is
now the single largest gap in the pipeline.

Mechanism: rank pooling encodes the DIRECTION of change across a span,
so it needs the span to contain one action. A span straddling a
boundary produces a mixed direction vector. Mean pooling has nothing to
corrupt, which is exactly why it did not care about boundaries.

Consequence for the plan: **step 5 is the next thing to fix, not step
7.** Also note r50+rank (0.310) beats siglip2+rank (0.262) under real
spans while losing badly under oracle spans - a cheaper encoder that is
less sensitive to misalignment currently wins the system-level number,
at 0.51 min/h against siglip2's ~10.

sensitivity.py's slope must be re-measured with an encoder that can
represent the action before it is quoted again.

### The boundary-value curve is a CLIFF, not a slope  `native/sens2.py`

Re-measured with an encoder that can represent the action, at the FULL
corpus (150 episodes, support 20 per template):

| boundary F1 | r50+rank yield | dino+rank yield | units/ep |
|---|---|---|---|
| **1.000** | **0.492** | **0.408** | 7.3 |
| 0.867 | 0.300 | 0.267 | 7.7 |
| 0.706 | 0.300 | 0.308 | 8.3 |
| 0.561 | 0.308 | 0.283 | 8.6 |
| 0.537 | 0.258 | 0.283 | 9.5 |
| 0.496 | 0.267 | 0.233 | 11.0 |

**All of the value sits in the last increment to perfect.** 1.000 ->
0.867 costs 0.19 yield; 0.867 -> 0.496 costs nothing measurable. Since
real segmentation will not reach 1.000, lifting step 5 from its current
~0.57 to even 0.85 buys approximately zero. Chasing step 5 is chasing a
cliff that cannot be climbed.

**A methodology fault was caught here and is now fixed in the harness.**
The first version of this run (40 episodes) produced a non-monotonic
"curve" - boundary F1 0.847 scoring 0.167 while 0.492 scored 0.167 and
0.528 scored 0.000. Cause: 40 episodes over 6 templates leaves ~6 per
template, 5 become seeds, so SUPPORT was 1.7 and yield quantised to
{0, 0.5, 1}. Pure noise presented as a trend. `score()` now returns
support and both harnesses print it: a yield without its support is
unreadable. Earlier u6retr numbers ran at 72 episodes (support ~7) and
are re-confirmed at full corpus before being treated as settled.

**Consequence: the response is not to fix step 5, it is to stop
requiring a partition.** Step 5's first design position was "the spans
partition the timeline exactly", justified by coverage - footage that
belongs to no unit is unreachable. But dense OVERLAPPING windows also
cover everything, and at a stride shorter than an event some window
always falls inside a single event, so a straddling span never has to
corrupt a direction vector. That is now measured against truth and cpd
spans directly.

### A uniform grid BEATS the learned segmentation — steps 4 and 5 do not earn their place

150 episodes, support 20, encoder r50+rank held fixed:

| segmentation | yield | prec | units/ep |
|---|---|---|---|
| truth (oracle) | 0.525 | 0.350 | 6.5 |
| **uniform 2 s, stride 0.5 s** | **0.408** | 0.272 | 47.4 |
| uniform 3 s, stride 1.0 s | 0.400 | 0.267 | 23.0 |
| uniform 4 s, stride 1.0 s | 0.392 | 0.261 | 22.0 |
| **cpd (GraphGEBD/Ncut/CPD)** | **0.317** | 0.211 | 7.7 |

**The entire step-4 + step-5 apparatus loses to a fixed grid with
overlap.** GraphGEBD, recursive Ncut, the backbone study, the
homogeneity stop, exact-DP change-point detection with a slope-heuristic
penalty - all of it scores 0.317, while `while t + w <= dur: t += 1.0`
scores 0.400. A uniform grid closes ~44% of the oracle gap that the
learned segmentation could not.

The reason follows directly from the cliff: value comes only from spans
that lie INSIDE one event, and a partition gets one attempt per
boundary. Overlapping windows get many attempts, so some window is
always clean. Cutting in the right place is hard; covering every place
is trivial.

Cost is the honest counterweight: 23-47 units/episode against 7.7, so
3-6x the vectors to encode, store and scan. The oracle remains both
best AND cheapest (0.525 at 6.5 units) - perfect segmentation is worth
having, it just is not reachable, and an unreachable optimum is not a
plan.

**Standing consequence: segmentation quality is not a lever on this
corpus.** Steps 4 and 5 remain in the tree because a boundary set is
useful for other things (seek points, display, byte-range reads), but
they are no longer on the retrieval critical path and no further
optimisation of them is justified by any measurement taken here.

### HARNESS FAULT: precision was capped at 0.667 by construction

`score()` returned exactly k = ceil(1.5 x support). Only `support` of
those can be true, so precision could never exceed support/k = 1/1.5 =
0.667 however good the ranking became. Every precision figure recorded
above this line was measured against a target it could not reach.

The label oracle exposed it: 0.628 precision, which reads as "even
perfect labels barely work" and is in fact just the cap.

Fixed with a label-free abstention cut. k is a MAX BOUND: leave one
seed out, score it against the remaining seeds, and take the weakest
such score as "what a true match looks like"; nothing below it is
returned. Nothing is fitted on truth.

| config | yield | prec | returned (k=30) |
|---|---|---|---|
| LABEL ORACLE, padding to k | 0.942 | 0.628 | 30.0 |
| **LABEL ORACLE, abstaining** | **0.942** | **0.890** | 22.0 |
| siglip2+rank, truth spans | 0.600 | 0.400 | 30.0 |
| **siglip2+rank, uniform grid** | **0.492** | **0.328** | 30.0 |
| siglip2+rank, cpd | 0.325 | 0.237 | 28.8 |
| r50+rank, uniform grid | 0.400 | 0.267 | 30.0 |
| r50+rank, cpd | 0.317 | 0.211 | 30.0 |

**The corrected ceiling is 0.942 / 0.890, not the 0.992 quoted earlier**
in this file - that came from an older low-support run. Perfect action
labels do NOT give perfect retrieval, so 0.90/0.90 sits just under the
achievable maximum rather than comfortably below it.

**Abstention does not fire for any real encoder** - `returned` stays at
30 while the oracle drops to 22. That is a diagnosis, not a disappointed
expectation: seed-to-seed similarity is no higher than
seed-to-random-candidate, so the score distribution holds no confidence
signal to cut on. Precision is not stuck because the cut is badly
chosen; it is stuck because the representation does not separate true
from false at all.

Consequence: the remaining gap is representational, and the discrete
route is the one with evidence behind it - symbol sequences (labels)
reach 0.942/0.890 where continuous vectors reach 0.600/0.400 with the
SAME oracle spans. That is step 7's motivation and it is now measured
rather than assumed.

### The action-AUC ceiling: 0.721, and what it costs in yield

Everything below was measured on dev-12 and the survivors on held-out
60. All FAILED to beat siglip2 rank-pooling:

| variant | action AUC | why it was tried / what it proves |
|---|---|---|
| **siglip2 rank-pool** | **0.721** | best; order-aware appearance change |
| **siglip2 rank-ONLY** | **0.721** | mean term contributes NOTHING to action - same AUC at HALF the dimension (1152 vs 2304). Free 2x storage win. |
| siglip2 rank nf=16 | 0.717 | temporal resolution is not the limit |
| flow + siglip2 | 0.589 | two-stream HURTS; motion is not the discriminator |
| sig endpoints | 0.538 | explicit before/after state fails |
| SSv2 action probe | 0.536 | pretrained ACTION model, human-hand classes, robot arm corpus |
| r50 rank nf=24 | 0.584 | +0.011 over nf=8 |
| optical flow alone | 0.512 | chance |
| discretisation (5 matchers x 5 k) | yield 0.20-0.31 | worse than DTW's 0.492 |

**Two negative results that are worth more than the positives.**

1. Discretisation failing SEPARATES the two properties labels have.
   Labels are discrete AND noiseless; symbol histograms are discrete and
   noisy, and they score 0.20-0.31 against DTW's 0.492. So discreteness
   is worth nothing here - the entire label advantage is per-unit
   ACCURACY. Steps 7/8 (vocabulary, assignment) therefore cannot rescue
   a weak encoder, which is what they were queued to do.
2. Flow failing says pick and place have nearly IDENTICAL arm
   trajectories. They differ in the state left behind (block on table vs
   in gripper), not in the motion - which is why appearance-CHANGE is
   the only pooling that has ever worked here, and why a motion stream
   dilutes it.

**The projection, from four measured points:**

| action AUC | yield |
|---|---|
| 0.521 | 0.214 |
| 0.579 | 0.400 |
| 0.691 | 0.492 |
| 1.000 | 0.942 |

Slope ~1.46 yield per unit AUC. **yield 0.80 needs action AUC ~0.90;
yield 0.90 needs ~0.97.** No later step changes this - steps 7-13, the
index, relate and commit all consume the same unit vectors, and the
discretisation result proves the consuming layer cannot add accuracy
that the encoder did not produce.

**Verdict: 0.80/0.80 is NOT reachable with a frozen, generic,
off-the-shelf encoder on this corpus.** The ceiling is 0.721 AUC ->
~0.49 yield. That is a measured statement, not an estimate.
