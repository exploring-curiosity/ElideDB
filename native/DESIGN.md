# Vision-native QbE — design

**Contract.** The user hands the system one clip. The system returns every other
clip in the corpus that is *like that one*. Nothing but pixels enters the system,
at write time, at read time, or in the evaluation.

---

## 1. The rules, made concrete

Not a style preference — an admission test every component has to pass.

| Forbidden | Because | What that rules out here |
|---|---|---|
| Text of any kind | text is not pixels | SigLIP / CLIP / any image-text model, task strings, captions, prompts |
| Discrete codebooks | a codebook is a finite vocabulary, and a vocabulary is definite | LAPA / LAQ, Genie-style latent actions, VQ of any kind |
| Classifier heads | class list = vocabulary | ImageNet heads, SSv2 probes, action classifiers |
| Sensors / telemetry | not pixels | OXTS, GPS, IMU, proprioception, laser-tracker pose |
| Class-based grouping | imposes a taxonomy the data did not ask for | k-means labels, HDBSCAN labels, template names, "8 kinematic classes" |
| Fitting a constant on the evaluation corpus | that constant *is* the answer, smuggled | `MIN_SAL=1.1`, tuned thresholds, curated query sets |
| Training on customer data | the product cannot assume it | any head fit on the corpus being searched |

Allowed: self-supervised image/video encoders trained with no labels and no text
(DINOv3, DINOv2, V-JEPA2); parameter-free operators (mean, rank pooling,
self-similarity, PCA); per-video statistics computed from that video's own pixels;
per-query statistics computed from the query's own pixels.

One borderline case, stated up front rather than buried: **temporal alignment
between two files of the same take is used in EVALUATION only** (§6). It is a
property of the recording container, not a judgement about content, and it never
touches the write or read path. It is the only non-pixel bit anywhere in this
design, and if it is disallowed there is no vision-only ground truth at all.

---

## 2. What gets deleted from the current pipeline

Every deletion below is backed by a measurement already in `native/BUILD.md`.

| Dropped | Reason (measured) |
|---|---|
| Learned segmentation — `graphgebd`, `segment`, `cpd`, `flowgebd` | a fixed uniform overlapping grid beat it end to end, 0.408 vs 0.317. The partition constraint is harmful. |
| Event counting — `eventcount`, `countmatch` | the count bottleneck is an artifact of how `sim_chains` was generated (templates differ by 5/6/6/7/7/8 events). Not a property of real data. |
| Discretisation / matcher zoo — `vocab`, `match7`, `seq` | 5 matchers × 5 codebook sizes scored 0.20–0.31 against 0.492 for continuous DTW. Discreteness buys nothing, and it is a vocabulary. |
| SigLIP2 everywhere | text-aligned. It is the current best action encoder (0.691 vs DINOv3 0.538) and losing it will cost accuracy — that cost is accepted, not hidden. |
| `alldom`, `bridgeqbe`, `domains`, `crossdom`, `style`, `motionrep` truth | all four built ground truth from text or sensors. |
| LAPA — `laqenc` | codebook, and it scored 0.275. |

Kept: the frame index and byte-range decode (M1), `encode.py`, the DINOv3
backbone loader in `gebdback.py`, rank pooling from `unitenc.py`, and the
cross-view idea in `crossview.py`.

---

## 3. The problem this design actually attacks

Every high number this session came from **near-duplicate matching**: same
session, same room, same lighting, same camera pose. Each time a separation rule
was added the number collapsed (bridge 0.734 → 0.547 → 0.250 under adjacent-file
exclusion). Meanwhile the one honest invariance test — same moment, second camera,
genuinely different angle — sits near chance.

So the representation is dominated by *the static scene* and carries little about
*what changed*. Everything below follows from that single diagnosis.

---

## 4. WRITE PATH

One pass per uploaded video. Streaming. No training. No free parameters fitted on
anything but that video's own pixels.

```
video bytes
  │
  ├─► frame index  (pts, dts, byte_offset, is_key, gop_id)          [exists, M1]
  │
  └─► decode @ 4 Hz ──► DINOv3 ConvNeXt-Tiny (frozen, SSL, no text)
                          │  F ∈ R^{T×d}, L2-normed, CLS + patch tokens
                          ▼
              ┌─────  SCENE BASIS REMOVAL  ─────┐
              │  μ = mean_t F      ← the static scene: room, lighting,
              │  F̃ = norm(F − μ)     camera pose, the always-present objects
              └──────────────────────────────────┘
                          │
                          ▼
              uniform overlapping windows, multi-scale
              (1 s / 2 s / 4 s, stride = ½ scale — fixed a priori, never tuned)
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
   c1 CHANGE          c2 ORDER           c3 SHAPE
   mean(F̃)           rankpool(F̃)        SSM(F) → 12×12 → upper tri
   what changed       which direction     the temporal shape of it
```

**Why scene-basis removal is the core move.** Two clips of the same action in two
rooms differ enormously in μ and little in the residual. Cosine on raw features
therefore matches *the room*; cosine on residuals matches *the change*. This is
background subtraction performed in feature space, it costs one mean, it has no
tunable parameter, and it is precisely the mechanism that destroys the
session-leakage advantage that has inflated every number so far.

Default rank is 1 (the mean only) so there is nothing to tune. An optional
higher rank is chosen per video by the eigengap of that video's own temporal
covariance — parameter-free, and it never sees another video.

**The three channels and the invariance each buys.**

- **c1 CHANGE** — `mean(F̃)`, 128-d after PCA. The appearance *of the change*
  rather than of the scene. Order-blind by construction.
- **c2 ORDER** — `Σ_t (2t − T − 1)·f̃_t`, 128-d. Closed form, no parameters,
  antisymmetric under time reversal. This is the only channel that can tell
  a reach-in from a pull-out, and it lifted action AUC 0.521 → 0.691 when
  measured.
- **c3 SHAPE** — the window's own frame-to-frame similarity matrix, resampled to
  12×12, upper triangle, 66-d. Built entirely from *relative* similarities, so it
  is invariant to any global transform of the feature space — which means
  invariant to viewpoint, to lighting, and to what the objects are. It carries
  tempo, monotonic-progress vs return-to-start, and repetition, and it names
  nothing. This is the closest vision-native thing to "A acted upon B": the
  relation over time, with the identities stripped out. RepNet's result is the
  precedent — a self-similarity bottleneck is what buys class-agnostic
  generalisation.

**Channels are stored separately and never pre-fused.** Measured earlier: the
best single channel beats a fixed fusion. The weights belong to the query (§5).

**Residual energy is stored per window.** A window whose residual energy is below
the video's own noise floor is a window where nothing happened. That is a
vision-native "nothing here" signal, it feeds abstention, and windows below it
need not be indexed at all — which is elision at write time.

**Nothing to add.** No motion channel in v1. A relational motion field (DINOv3
patch correspondences, ego-motion removed by subtracting the field's own median)
is the first planned improvement, and it is deliberately held back until there is
a valid number to improve.

### Storage layout

| table | columns | purpose |
|---|---|---|
| `frames` | video_id, pts, byte_offset, is_key, gop_id | byte-range decode of returned spans only |
| `windows` | video_id, t0, t1, scale, residual_energy, **c1**, **c2**, **c3** | one row per window, three vector columns, chunked with zone maps on (video_id, t0) |
| `basis` | video_id, μ (and eigenbasis if rank > 1) | reconstruct residuals; tiny |
| `coarse` | centroids over c1 | **prune only** — an IVF speed structure with exact rerank inside probed cells, never a label. Its recall is verifiable against exact scan, and it is verified. |

At 4 Hz, three scales, half-scale stride: ~21.6k windows per hour of video;
322-d total across channels at fp16 ≈ 14 MB/h. The video is ~2 GB/h.

### Budget

DINOv3 ConvNeXt-Tiny measured at 0.68 min per hour of video. Mean, rank pooling,
a 12×12 SSM and one PCA are negligible beside it. Total ≲ 1 min/h, inside the
online rule; no offline tier exists.

---

## 5. READ PATH

Input: **one example clip**. Pixels. That is the entire query.

```
example clip
  │ decoded and encoded by the IDENTICAL write-path code
  ▼
Q = {q₁ … q_m}   the example's own overlapping sub-windows
  │
  ├── the example is not one vector, it is a SET, and the set is free supervision:
  │   its members are positives of each other by construction.
  │
  ├─► SELF-CONSISTENCY CUT
  │     cut_c = min_i ( max_{j≠i} sim_c(q_i, q_j) )
  │     leave-one-out over the query's own sub-windows. This is the abstention
  │     floor, derived from the query alone — no labels, no seeds, no corpus
  │     statistics, no fitted constant.
  │
  ├─► PER-CHANNEL INFORMATIVENESS
  │     w_c = (mean intra-Q sim in c) − (mean sim of Q to a random corpus sample in c)
  │     z-scored across channels, clipped at 0.
  │     A channel where the query is not self-consistent, or where everything in
  │     the corpus looks alike, earns no weight. Decided per query, from pixels.
  │
  ├─► COARSE PRUNE   probe c1 cells until the recall budget is met; exact rerank inside
  │
  ├─► SCORE          score(x) = Σ_c w_c · max_i sim_c(x, q_i)
  │
  ├─► ABSTAIN        keep x only if score(x) ≥ Σ_c w_c · cut_c
  │                  returning fewer than k is correct behaviour, not a failure
  │
  ├─► TEMPORAL NMS   collapse overlapping kept windows in one video into one span,
  │                  score = max. Five views of one moment are one answer.
  │
  └─► MATERIALISE    decode only the GOPs of the returned spans
```

Two properties worth defending out loud:

1. **One example becomes a calibrated multi-seed query.** The single-example
   problem — no leave-one-out, therefore no threshold, therefore precision capped
   at support/k by construction — is solved by treating the example's own
   sub-windows as the seed set. It costs nothing and it comes from the pixels the
   user already handed over.
2. **Query and corpus go through one code path.** A separate query encoder is
   where leakage lives. There is not one.

---

## 6. Evaluation — the only ground truth pixels can supply

Three gates. Two are cheap and run on every commit; one is the headline.

**N — nuisance invariance.** Re-encode a clip after a transform a QbE system must
be blind to: crop, rescale, brightness, JPEG. The transformed copy must come back
rank-1. Derived from pixels alone. Reports rank-1 rate.

**D — direction sensitivity.** Re-encode a clip time-reversed. The reversed copy
must fall *below* the query's own abstention cut. Derived from pixels alone.
Reports rejection rate. N and D pull against each other; a representation that
passes both is doing the right thing.

**V — cross-view retrieval. The headline.** Query a window from camera A; the
correct answers are camera B's windows overlapping it in time. Same physical
event, different viewpoint, no words, no sensors, no annotation. A system that
cannot recognise one event from a second angle cannot retrieve a *similar* event
from a new episode, so this is the necessary condition for the product.

Metrics, and only these: **yield = true/support**, **precision = true/returned**,
at **k = ceil(1.5 × support) as a MAX bound** with abstention on. Support and
returned printed on every row, always.

| corpus | views | character | weight |
|---|---|---|---|
| `sim_chains` | 2 of 4 cams, pair varies per episode | genuinely different angles, synthetic | **headline** |
| `bridge` | image_0 … image_3 | genuinely different angles, real, 50k episodes | **headline** — needs re-download (§8) |
| REIP lab | 4 sensor stations | genuinely different angles, real | secondary |
| KITTI, UZH-FPV | stereo pairs | short baseline, near-identical images — easy | reported, never headlined |

Contamination guards live in the harness, not in my discipline:

- candidates must come from a different file **and** not an adjacent file
  (adjacent bridge files are sequential in time — this collapsed one result from
  0.969 to 0.250);
- ≥ 60 s temporal separation inside a continuous stream;
- every query runs, or a deterministic uniform sample runs — no curation of which
  queries get reported;
- support, returned, query count and pool size printed on every row;
- any constant is either fixed a priori or derived per-video / per-query, and the
  file says which.

---

## 7. What this is expected to cost

Losing SigLIP2 costs accuracy today — it is the best measured action encoder and
it is disqualified. The bet is that scene-basis removal plus the shape channel
recovers more than the encoder swap loses, because the failure being fixed
(scene dominance) is larger than the gap between encoders. That bet is testable
by gate V and will be reported whichever way it lands.

There is no valid V number yet. The only run was `sim` at n=2 episodes because of
a hard-coded camera pair. Producing the first valid one is the point of the build.

---

## 8. Build order

Each step ends with a number, not a claim.

1. `vcore.py` — write path: decode → DINOv3 → basis removal → c1/c2/c3 → store.
2. `vstore.py` — window table, zone maps, coarse index, exact-rerank guarantee,
   bytes-read accounting on every read.
3. `vqbe.py` — read path: one clip in, ranked spans out, abstention, NMS.
4. `vgrade.py` — gates N, D, V with the guards above wired in.
5. **First valid V**, all 150 sim episodes, each episode's own camera pair.
6. Improve in this order, measuring each before the next: relational motion
   channel → multi-scale fusion → a frozen cross-view head trained on *public*
   multi-view video only.

**One decision that is the user's.** Bridge's other three camera streams were
deleted to reclaim 56 GB; the manifest still carries their timestamps, so
re-downloading them yields a real, large, genuinely-different-angle cross-view
corpus — by far the strongest evidence available for gate V. `sim` alone is
synthetic and 150 episodes deep. Cost: ~56 GB and an hour or two of download.

---

# AMENDMENT — what the build changed, and why

The design above was written before any of it ran. Five things in it were
wrong, and each was corrected by a measurement rather than a preference.
The corrections are recorded here rather than edited silently into the
text above, so the reasoning stays auditable.

### 1. Cross-view evaluation is gone. Single view, treated alone.

§6 made cross-view retrieval the headline. That is now out entirely, by
instruction: each media is treated alone and no comparison is made
between two views of the same moment. This removes the only ground truth
pixels could supply for "the same event", so the benchmark had to be
rebuilt around a different admissible question — see §4 below.

### 2. The scene basis is LOCAL and LEAVE-ONE-OUT, not a media mean.

The design said `mu = mean_t F` over the whole media. Measured, that is a
seam: an 8 s example encodes its own mean over 8 s while the stored media
took a mean over 41 minutes, and the same content compared at **cos
0.106**. Worse, a window spanning its whole context has a residual mean
of exactly zero, so its appearance channel is noise by construction.

The basis is now the mean of the frames around a window, EXCLUDING the
window's own frames, over a span of `CTX_MULT x scale` seconds. Scale-
relative, so a 2 s window needs 2 s either side whether it sits in an
8-second clip or a 45-minute flight — supplyable by both sides by
construction. A fixed wall-clock context was tried first and still
failed: seam on c1 against clip length was 0.936 at 20 s, 0.846 at 12 s,
0.798 at 8 s. Scale-relative closed it.

### 3. The write path was graded against a gate it could fail.

`c1 0.913 / c2 0.996 / c3 0.967` re-encoding agreement is meaningless on
its own — a threshold picked after seeing it is a fitted constant in a
test's clothing. The bar is now the channel's own **99th-percentile
similarity to unrelated windows**: re-encoding the same seconds must
agree more than different content ever does. Margins `+0.533 / +0.647 /
+0.064`. The thin one is informative: c3's unrelated-pair p99 is 0.898,
so the shape channel barely discriminates alone and needs the hubness
correction the read path applies.

### 4. The benchmark measures a necessary condition, and says so.

With no second view and no labels there is no answer key for "which
clips are LIKE this one". Nothing label-free can produce one. So the
harness re-films a sampled span — perspective warp, crop, photometric
shift, re-encode, resampled tempo — hands the altered clip back as the
example, and asks the store to find the original among the whole corpus
INCLUDING the seconds either side of the target. Passing does not show
QbE works; failing shows it cannot. Both halves print.

Truth is **full containment**: a store row is the queried moment when it
lies entirely inside the span shown. At the 0.5 originally used, rows
half outside the query counted as truth, the system correctly ranked
those marginal rows low, and the resulting ceiling was an artifact of the
rule — with the prune and the cut both disabled it still capped at 0.707,
while the median truth row sat at **rank 7.9 of 533**.

### 5. Defects the gates caught, which would otherwise have shipped

- **The coarse index probed the query's MEAN.** The whole read-path
  design says the example is a *set*; its sub-windows routinely fall in
  different cells and the mean sits in none of them. Recall against an
  exact scan was 0.533 — elision bought with answers. Probing by best
  match to any sub-window: 0.742.
- **The identity query failed.** Handed the exact seconds of the exact
  media, the store returned 0.292 of its own rows. Isolating the cause
  showed the abstention cost nothing, the prune cost 0.070, and the rest
  was §2 and §4 above. Now 0.879.
- **A transform did not model what it claimed.** "A different capture
  pipeline" was implemented as a 115-pixel JPEG at q25 and scored 0.061,
  dragging the mean down ~0.15 alone. Split into `codec` (480p at q50,
  what the description says) and `codec_hard` (the original), reported
  separately. Correcting a transform to match its own description is a
  fix; deleting it for scoring badly would not be, so it stays.

### Where it stands

Nothing here is a product number yet. On a 12-episode sim store the
harness reports yield 0.503 / prec 0.340 against chance 0.070, rank1
0.733, identity ceiling 0.879, direction margin 0.432, prune recall
0.742. The four stores are built and the harness runs on all of them; the
numbers that matter are the ones the full run produces.

The pipeline end to end: `vsrc` (one media abstraction, single view) →
`vcore` (DINOv3, local basis, c1/c2/c3) → `vstore` (columnar, counted
bytes, hubness, coarse cells) → `vqbe` (one example in, spans out) →
`vgrade` (one protocol, four stores). No text, no codebook, no class
list, no sensor, no label anywhere in any of them.

---

# RESULT — multi-resolution encoding, and the call it overturned

The nuisance battery split cleanly on all four corpora: photometric
(0.75-0.88) and temporal (0.60-0.81) invariance are fine; geometric and
resolution invariance are where the system loses. One failure mode, not
five. The standard label-free answer is to describe each frame at several
input resolutions and average the unit-normed descriptors.

Measured, two preserved artifacts per corpus:

| store | yield | prec | rank1 | warp (viewpoint proxy) |
|---|---|---|---|---|
| sim | 0.372 -> 0.437 | 0.242 -> 0.289 | 0.569 -> 0.653 | 0.390 -> 0.508 |
| bridge | 0.398 -> 0.460 | 0.261 -> 0.297 | 0.625 -> 0.708 | 0.386 -> 0.614 |
| car | 0.542 -> 0.564 | 0.374 -> 0.388 | 0.729 -> 0.708 | 0.580 -> 0.587 |
| drone | 0.378 -> 0.501 | 0.269 -> 0.368 | 0.514 -> 0.694 | 0.303 -> 0.568 |

**This reversed a decision.** On sim+car alone the reading was "helps the
synthetic corpus, leaves the real one flat, do not pay 3x encode cost" -
and that was stated as the rule before the data arrived. With all four
measured, car is the exception, not the verdict: it decodes to 256x78, so
there is almost no vertical resolution for extra scales to use, while the
other three are 144-192 high. Both REAL robot corpora gain the most.

Two of four corpora is not four corpora. The earlier call was made on a
sample chosen for convenience, and it was wrong.

**Cost, corrected.** The "3x encode, ~6 min/h" figure quoted while this
was in flight was wrong - extrapolated from sim, whose 25-second clips
make startup overhead dominant. Measured on the same media:

| | 1-res | 2-res | 3-res |
|---|---|---|---|
| bridge (video) | 1.41 | 1.63 | 2.49 min per hour |
| drone (image sequence) | 3.64 | 3.60 | 4.17 |

Three resolutions cost 1.8x on bridge and 1.15x on drone, because
DECODE dominates the write path and extra forward passes ride along
nearly free. That also settles the two-resolution compromise: it saves
almost nothing while giving up 13-24% of the yield gain and 83% of
bridge's rank1 gain.

The write path is still 2.5-4.2 min per hour against a stated
one-minute-per-hour budget. The lever for closing that is DECODE -
hardware decode, or a lower sample rate - not the encoder, which is not
where the time goes.

**Method failure worth keeping.** The first A/B copied the result file
AFTER the run and compared it against itself, printing a perfect null
down every column. A null is a decision - "do not pay 3x" - and it would
have been made on nothing. Runs now write under `--out` names, and the
manifest records `res` and `wmode` so settings travel with numbers.

---

# RESULT — backbone chosen by measurement, full run at <=1 h per store

## Backbone A/B (10 min video per store, 16 queries, identical stores)

| config | yield | prec | rank1 |
|---|---|---|---|
| **ViT-L/16 @320** | **0.718** | **0.512** | **0.880** |
| ViT-L/16 @224 | 0.681 | 0.486 | 0.870 |
| ViT-H+/16 @224 (841M) | 0.654 | 0.469 | 0.872 |
| ConvNeXt-L @448 | 0.637 | 0.433 | 0.841 |
| ConvNeXt-T @224 (incumbent) | 0.607 | 0.426 | 0.828 |

Two findings, both against expectation:

**Parameter scaling stops paying.** ViT-H+ (841M) LOSES to ViT-L (303M).
Bigger is not better past ViT-L, so the compute belongs in resolution
(224 -> 320 buys +0.037 yield on ViT-L) rather than in more weights.

**Architecture beats size within a family.** ConvNeXt-L @448 loses to
ConvNeXt-Tiny on sim and bridge; scaling ConvNeXt barely moves anything,
while switching to ViT does.

## Full run — DINOv3 ViT-L/16 @320, <=1 h per store, 40 queries

| store | yield | prec | chance | rank1 | ident | D-marg | prune | windows |
|---|---|---|---|---|---|---|---|---|
| sim | 0.597 | 0.387 | 0.004 | 0.854 | 0.848 | 0.623 | 0.967 | 6006 |
| bridge | 0.583 | 0.379 | 0.004 | 0.792 | 0.843 | 0.486 | 1.000 | 6120 |
| car | 0.774 | 0.537 | 0.017 | 0.883 | 0.936 | 0.541 | 0.950 | 1194 |
| drone | 0.702 | 0.604 | 0.007 | 0.879 | 0.982 | 0.786 | 0.958 | 4923 |
| **mean** | **0.664** | **0.477** | | | | | | |

Under the other truth rule (containment 0.5 rather than full), reported
so a definition changed after seeing results cannot be cherry-picked:
mean yield 0.543, prec 0.435.

**The one clean like-for-like.** car is the only store whose size did not
change between runs (all 22 drives, 1194 windows both times), so it is a
controlled comparison rather than a cross-run one:

    ConvNeXt-T, 256-wide decode   yield 0.542   prec 0.374
    ViT-L @320,  640-wide decode  yield 0.774   prec 0.537

sim, bridge and drone all grew as well as improved, so their gains are
not attributable to the encoder alone and are not claimed as such.

## The bottleneck moved

Geometric and resolution invariance was the whole failure mode. Then vs
now, per transform:

| | warp | crop | codec_hard |
|---|---|---|---|
| before | 0.30-0.58 | 0.17-0.55 | 0.01-0.18 |
| now | 0.58-0.86 | 0.39-0.83 | 0.23-0.49 |

`crop` on sim (0.389) and `codec_hard` on drone (0.232) are what is left.

## What it costs

ViT-L @320 runs at 17-21 min per hour of video, against a stated
one-minute-per-hour online budget - roughly 20x over. That is the honest
price of the accuracy and it is not solved. Decode still dominates on
image-sequence corpora, so the two levers are separate.

---

# IMPROVEMENT LOOP — iterations 1-3 (2026-08-07)

Each iteration: ask what the bottleneck is, research it, implement,
measure against a preserved artifact, repeat.

| iter | change | result (10 min/store A/B) |
|---|---|---|
| 1 | pooling: CLS -> GeM p=3 over patch tokens | yield 0.701 -> 0.750 |
| 2 | resolution: single 320 -> ladder 256/320/384 | yield 0.750 -> 0.793 |
| 3 | abstention cut: LOO floor -> otsu_auto | see below |

## Iteration 3 in the REAL benchmark (96 queries/store)

| store | yield | prec | returned | ret/sup |
|---|---|---|---|---|
| sim | 0.813 -> 0.698 | 0.526 -> 0.727 | 17.0 -> 11.8 | 1.07 |
| bridge | 0.777 -> 0.688 | 0.502 -> 0.666 | 17.0 -> 12.6 | 1.15 |
| car | 0.744 -> 0.673 | 0.488 -> 0.660 | 16.9 -> 12.9 | 1.17 |
| drone | 0.837 -> 0.793 | 0.681 -> 0.850 | 13.7 -> 10.7 | 0.97 |
| mean | 0.793 -> 0.713 | 0.549 -> 0.726 | | |

Precision +0.177 for -0.080 yield, and returned/support fell 1.55 -> 1.09:
the system now returns roughly as many things as there are, instead of
padding to the k bound. That convergence was the actual goal.

The diagnostic harness (`cutcmp`) predicted 0.898/0.781 for the same
change. The real vgrade says 0.713/0.726. The harness ranks rules
correctly but its absolute numbers are optimistic - which is why the
rule was validated here before anything was written down.

## Three cut rules rejected for contamination

Each BEAT the shipped rule. Each is recorded in vqbe.score_cut so it
cannot be reintroduced as an improvement.

| rule | score | contamination |
|---|---|---|
| otsu over top k*4, k from support | 0.873/0.815 | support IS the answer key |
| otsu over top 5% | 0.836/0.901 | 5% won a sweep on the eval corpora |
| recursive otsu, depth 3 | 0.820/0.866 | depth is that same fitted constant |

The pattern: the contamination moved up one level each time it was
removed, and every version looked clean in code and produced a
plausible number. Refusing to fit costs roughly 0.12 precision on the
harness. That is the accepted price.

## Also settled

- Published guidance that DINOv2/v3 favour MAX pooling did NOT transfer:
  max is worst on yield here and collapses on crop (0.498 vs 0.657).
  Only "pool the patches, not CLS" transferred.
- GeM was chosen on SPREAD, not mean: per-store yield spread was
  cls 0.206, max 0.218, gem 0.019. One approach across four corpora is
  what the spread measures; the mean hides it.
- Batch size buys nothing on MPS (plateau at 32; worse at 128). Free RAM
  converts to resolution and model size, never to throughput.

---

# ITERATION 4 — the three-channel architecture was wrong

Channel ablation (read-path only, each query encoded once and scored
five ways):

| variant | sim | bridge | car | drone | MEAN |
|---|---|---|---|---|---|
| all three | 0.822/0.774 | 0.904/0.771 | 0.909/0.812 | 1.000/0.790 | 0.909/0.787 |
| drop c1 | 0.830/0.748 | 0.921/0.777 | 0.908/0.811 | 1.000/0.790 | 0.915/0.782 |
| drop c2 | 0.765/0.799 | 0.880/0.695 | 0.699/0.556 | 0.536/0.736 | 0.720/0.697 |
| drop c3 | 0.905/0.875 | 0.907/0.858 | 0.981/0.861 | 1.000/0.777 | 0.948/0.843 |
| **c2 only** | 0.919/0.878 | 0.942/0.898 | 1.000/0.914 | 1.000/0.777 | **0.965/0.867** |

c2 alone beats all three together on every store. c1 and c3 are not dead
weight - they actively hurt. c2 is indispensable: without it the system
collapses to 0.720/0.697.

## Confirmed in the real benchmark (96 queries/store, identical stores)

| store | yield | prec | ret/sup |
|---|---|---|---|
| sim | 0.698 -> 0.753 | 0.727 -> 0.837 | 0.95 |
| bridge | 0.688 -> 0.801 | 0.666 -> 0.834 | 0.99 |
| car | 0.673 -> 0.849 | 0.660 -> 0.883 | 0.98 |
| drone | 0.793 -> 0.809 | 0.850 -> 0.829 | 1.02 |
| **mean** | 0.713 -> **0.803** | 0.726 -> **0.846** | |

Both metrics clear 0.80 with no fitted constant and no label leak, and
returned/support sits at 0.95-1.02: the three ratios have converged.
Store size fell 52% (car 10 MB -> 4.8 MB), 2114 -> 1024 dimensions.

## What the whole loop changed

| iter | change | yield | prec |
|---|---|---|---|
| 0 | ConvNeXt-T, 256 decode, CLS, LOO cut | 0.542* | 0.374* |
| 1 | GeM p=3 over patch tokens | | |
| 2 | resolution ladder 256/320/384 | | |
| 3 | otsu_auto abstention cut | 0.713 | 0.726 |
| 4 | c2 only | **0.803** | **0.846** |

*car, the only store whose size never changed, so the one honest
like-for-like across the whole loop: 0.542/0.374 -> 0.849/0.883.

## Two caveats the result does not settle

1. c3 was justified by Junejo et al. on VIEWPOINT stability. The
   benchmark's viewpoint test is a parallax-free homography. This shows
   c3 does not help against the proxy; it cannot show c3 would not help
   against a real second camera. Open, not settled.
2. This is the SECOND consecutive iteration whose fault was the
   per-query weighting rather than the representation. It gave c3 ~0.40
   weight while c3 was hurting, because its rule rewards self-consistency
   across adjacent windows rather than discrimination. Any future channel
   needs that rule rebuilt first.

---

# HELD-OUT VALIDATION (2026-08-08)

Every architectural choice in this system was selected by reading the
benchmark on four corpora: encoder (7 candidates), resolution ladder (4),
pooling (5), cut rule (9), channel set (5), weight mode (2), coarse cells
(2). No weights were ever trained or fine-tuned - every model is frozen
and pretrained - so no label ever entered the system. But seven
sequential selections on one set of corpora is model selection without a
held-out set, and that makes the headline a selected-best number.

Measured on media no A/B ever touched (sim episodes 24+, bridge files
15+, drone media 12+):

| store | held-out | selection set | gap |
|---|---|---|---|
| bridge | 0.778 / 0.708 | 0.801 / 0.834 | -0.023 / -0.126 |
| drone | 0.887 / 0.805 | 0.809 / 0.829 | +0.078 / -0.024 |
| sim | 0.688 / 0.825 | 0.753 / 0.837 | -0.065 / -0.012 |
| **mean** | **0.784 / 0.779** | 0.788 / 0.833 | -0.004 / -0.054 |

YIELD HELD (-0.004). PRECISION COST 0.054. drone scored HIGHER on unseen
media than on the media it was selected with, which is what a real gain
looks like. The precision gap is concentrated where the search was
heaviest - the cut rule - which is the expected place for it.

car has NO held-out set: all 22 of its drives were consumed by the A/Bs,
so the held-out mean is three corpora, not four. Stated rather than
quietly substituted.

The defensible headline is therefore **0.78 yield / 0.78 precision held
out**, with 0.80/0.85 as the development-set figure.
