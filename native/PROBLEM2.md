# THE PROBLEM — product statement (v2, 2026-08-10)

Supersedes PROBLEM.md's purity constraints. One rule survives from it:
**no ground-truth labels, ever** — no class names, no event annotations,
no task strings, at write or read, on any path. Everything else —
pretrained models, VLMs, self-supervised training, cosine — is allowed
wherever it buys results. Raw data stays immutable; reads stay counted.

## 1. The product

A memory for video. Given a moment — an uploaded clip or the live
now — return the like-moments from everything the store has ever seen:
any content, any camera, any corpus size, with latency fit for whoever
consumes the answer (a human browsing, or a machine deciding).

## 2. The four use cases, each with a hard budget

| # | use case | query | budget (p95) | config |
|---|---|---|---|---|
| U1 | **Search** — human asks "when did something like this happen?" | clip | ≤ 5 s | A+B, VLM in-loop on top-K |
| U2 | **Realtime decision-making** — an agent consults memory while acting | live window, 3–10 Hz | ≤ 300–500 ms per query | A+B + distilled student only; nothing slow in the hot path |
| U3 | **Show-then-do** — user demonstrates; robot retrieves matching experience and acts. A live product loop, not a demo | demo clip, cross-embodiment | ≤ 2 s demo-end → conditioning set | A+B + small-K VLM rerank (day 0) → student (steady state) |
| U4 | **Cold start** — a brand-new or tiny corpus, possibly a handful of examples | clip | U1/U3 budgets | A + VLM only; B stays OFF below a corpus-mass gate |

## 3. The architecture (three stages, no hand-built perception)

- **A — frozen video foundation encoder.** Sliding-window clip
  embeddings + vector kNN + corpus-graph geometry (CSLS/diffusion).
  Corpus-size independent. Always on.
- **B — per-corpus self-supervised adaptation.** A small head trained
  at ingest with positives = two sub-windows of the same continuous
  moment. Supervision is temporal continuity only — a fact of the
  recording, not an annotation. Auto-fits per corpus (minutes/GB);
  gated OFF below a data threshold so tiny corpora never overfit.
- **C — pairwise rerank, teacher–student.** A zero-shot VLM judges
  "is the same thing happening in these two clips" on shortlists.
  Every judgment is logged; a tiny cross-encoder student distills
  from the accumulated verdicts and replaces the VLM in all
  latency-bound paths (sub-ms per pair). The VLM never sits in a
  realtime loop; the student always does.

Explicitly deleted from the critical path: tracked entities, roles,
residue filters, hand-built relational facts. (Three weeks of
measurement: rule-based perception does not converge. It may return
someday as an explainability layer — never as the ranker.)

## 4. The day-0 promise vs steady state

- **Day 0, any corpus:** A (+B if the corpus clears the mass gate) +
  zero-shot VLM on shortlists. Full quality where the budget allows
  the VLM (U1, U3-at-2s, U4); embedding-only quality in U2 until a
  student exists. Nothing waits for training to be *correct*; only
  U2 waits (hours, not weeks) to be *fast and sharp*.
- **Steady state:** same quality, served by the student at
  milliseconds everywhere; B refreshed as the corpus grows; VLM
  spend drops to a background trickle (auditing + student refresh).

## 5. Acceptance gates (measured, per corpus, no labels leaked)

- U1/U4: precision-of-returned on human spot-verdicts (the existing
  Desk verdict flow) — target ≥ 0.9 on returned results, with
  abstention allowed; never graded against hidden class labels.
- U2: end-to-end query latency under sustained 5 Hz load within
  budget, quality within 5 points of U1 on the same queries.
- U3: cross-embodiment retrieval — a demonstration by one body
  retrieves the matching experience of another; graded by verdicts,
  ≤ 2 s. (The executing VLA/world model is a downstream consumer of
  this memory, out of this document's scope.)
- Every corpus ships with its own measured card: latency percentiles,
  verdict precision, bytes read vs stored. No number is quoted that
  was not produced by the running system.

## 5b. Design amendments from review (2026-08-10, accepted)

1. **Stage B objective is NNCLR-class, not plain InfoNCE.** Repetitive
   corpora (36 episodes of the same act) make naive negatives FALSE
   negatives - InfoNCE trains against retrieval. Nearest-neighbour
   positives convert them into extra positives. Collapse monitored by
   the recording-ID probe (provenance, not a label).
2. **Windows are motion-boundary-aware and multi-scale** (change-point
   detection on embedding deltas; GEBD machinery exists in-repo).
   Fixed strides cut mid-action and poison positives.
3. **The VLM teacher runs a change-decomposed BEFORE/AFTER protocol**,
   never free-form "same thing?": ordered keyframes, "what changed in
   A? in B? do the changes correspond?", direction asked both ways
   with a consistency check. (This repo already fixed open-vs-close
   this way, measurably.) VLMs are bag-of-frames by default; the
   protocol is what makes them usable as teachers.
4. **The student is a late-interaction (ColBERT-style) head over
   token-level + temporal-delta features** - never pooled vectors
   (pooled supervised ceiling measured 0.79 here; probe on richer
   features 0.86).
5. **Tiny corpora get a generic adapter**: the Stage-B head pretrained
   once, label-free, on unrelated public video, used frozen below the
   corpus-mass gate. U4 is never raw-A-alone.
6. **Anti-appearance augmentation in B**: same-scene hard negatives,
   disjoint-window positives, arrow-of-time and shuffle negatives,
   static-patch dropout.

## 5c. The go/no-go gate (runs BEFORE any build)

The 0.90 case rests arithmetically on ONE measurable: the VLM
teacher's pairwise accuracy on the corpus's confusable pairs, under
the before/after protocol, graded on a stratified sample (truth used
to GRADE the teacher only - never to train anything). Thresholds:
>= 0.90 -> build (0.90 yield reachable); 0.80-0.90 -> build, promise
0.8-class; < 0.80 -> STOP and report - no implementation happens
under a failed gate. Measured before Stage B exists, at a cost of
about one hour. Prior evidence: 7B VLM AUC 0.75-0.89 on a dirtier
corpus without the protocol.

**GATE RESULT (2026-08-10, measured): FAILED for the local teacher.**
Qwen2.5-VL-7B-4bit on 120 stratified pairs, three protocol variants:
1-token logprob 0.507 (all-Yes degenerate); generated before/after
reasoning 0.624 (best; place|stack 0.20, unstack-same 0.30);
full-res 6-frame grids 0.500 (all-Different degenerate). A small
quantized VLM is an unstable pairwise judge for fine-grained
temporal distinctions on 25px objects. Per the contract: STOP - no
implementation was built on top of the failed gate (total spend: the
gate itself, ~25 min).

**Gate re-run on Qwen3-VL-30B-A3B (harness fixed - v3/v4 had
produced EMPTY generations scored as verdicts; empty is now
INVALID): balanced 0.536 on 114 valid pairs - says SAME to nearly
everything; its own text collapses every clip to "picked up and
placed down". Two model generations measured: 0.54-0.62. The local
VLM teacher route is CLOSED.** Also measured: Stage A frozen
V-JEPA2 pooled = 0.309 (~chance) on this single-scene corpus; Stage
B (NNCLR+AoT+flow, all amendments) = 0.389 best - the fixes work as
mechanisms (unstack 0.16->0.30 from AoT negatives) but 324 events
cannot overcome frozen-feature poverty. Rank fusion with the
symbolic pipeline DILUTES it (0.51 -> 0.42). The best local system
remains the symbolic happened-facts pipeline at yield 0.510.

**EXACTLY WHY the VLM fails (field-level decomposition, 60 events
x 2 view conditions, single-clip forced choice on start/motion/end):
the model is a near-CONSTANT function on these clips.** It answers
END="held" on ~95% of all clips and MOTION="lifted" on ~97% - zero
clip information; START is "table" except a ~55% tower detection on
unstack (the one real perception it shows: a static high-contrast
configuration in frame 1). Its pick scores were the constant
canonical story (table -> lifted -> held) coinciding with pick.
Ruled out by measurement: resolution (2-3x zoom changed nothing),
pair-protocol contamination (fields already broken single-clip),
frame sampling (push shows no lift in ANY frame yet reads "lifted"
11/12 - pure language prior). Causal chain: contact/support states
at small scale on flat-shaded sim rendering produce weak visual
evidence -> the manipulation prior fills every gap with the
canonical grasp narrative -> near-constant outputs -> pairwise
verdicts are noise around a constant. NOTE the convergence: the
symbolic pipeline leads (0.510) precisely because it computes the
contact/support states no available component perceives - SAM2
tracks them noisily, monocular depth erases them (support prior),
VLMs narrate over them.

**Universal teacher pre-gate (corrected - the first draft asked
corpus-specific state questions, which is data-specific modelling;
the field probe stays a one-off INSTRUMENT, never a protocol).**
The content-free form, valid on any corpus with zero vocabulary:
  1. CONSTANCY - free-form "describe what changed" over N clips;
     if the description embeddings are near-identical across clips
     the model is a constant function (ours was: everything became
     "picked up and placed down"). Fail.
  2. REVERSAL - description(clip) must differ from description(
     time-reversed clip) on motion-bearing clips.
  3. REPEATABILITY - same clip twice -> same verdict.
All label-free, content-free, ~40 calls. The production teacher
protocol remains free-form change description + correspondence
judgment (gate v2's form - which also happened to score highest).

**KINEMATIC ROUTE (CoTracker3 dense point tracks, 2026-08-10):
tracks are GOOD - verified on film, points ride the manipulated
object through grip and carry, which is exactly what box tracking
lost. Content-free readouts measured: bundles+correspondence 0.327,
permutation-invariant field descriptor 0.351, fusion 0.347, +L8
0.335, cross-embodiment 0.30-0.32. THE DECISIVE NUMBER: a linear
probe on raw track descriptors scores 0.768 episode-disjoint
(chance 0.44). The signal IS in the tracks; the label-free readouts
recover about half of it. Diagnosed cause: 151/324 moments collapse
to a single common-fate bundle because arm and object genuinely
move together during a carry - the discriminating structure lives
in the APPROACH and RELEASE phases, not the transport. This is the
first route where the ceiling is demonstrably above the current
readout rather than at it.**

**THE CONVERGENT FINDING (all families, 2026-08-11).** Seven
label-free readouts over clean point tracks: bundles+correspondence
0.327, permutation-invariant field 0.351, phase-support 0.325,
trajectory curves 0.321, fusions 0.34-0.35, contrastive head
(disjoint-window positives, same-recording hard negatives, AoT
negatives) 0.375. Probe on the SAME features: 0.768. Across every
representation family now measured - frozen video (0.309),
adapted video (0.389), symbolic facts (0.510), kinematics (0.375) -
UNSUPERVISED similarity lands 0.31-0.51 while a SUPERVISED probe on
the identical inputs lands 0.77-0.86. The bottleneck is not
perception, not features, not tracking: it is the LABEL-FREE
METRIC. The information is present in every representation; no
unsupervised similarity function recovers more than about half of
it at this corpus size. That is the honest problem statement, and
it is a different problem from the one this document opened with.

**FRAMEWORK SWEEP (2026-08-11).** XIRL/TCC cycle-consistency
alignability implemented and measured on cached tracks: 0.310
(cross-embodiment 0.300-0.308) - the 8th label-free readout to land
in the 0.31-0.51 band. Reading the paper closely explains why: XIRL
trains TCC on video sets ALREADY KNOWN to show the same task, and
that grouping is exactly our retrieval question. VIP assumes
goal-conditioned progress; MOVE consumes annotated motion exemplars
(violates the no-label rule); MotionRAG is video generation, not a
retrieval metric. EVERY framework that works in this literature is
handed a grouping or goal signal. The binding constraint is the
no-label rule itself, not the choice of algorithm.

**What survives, honestly:** (a) a FRONTIER cloud VLM as teacher -
untested here, gate-able for pocket change (~120 API calls), the
remaining route to 0.90; per-corpus one-off distillation cost.
(b) No-VLM stack (A + NNCLR-adapted B + late-interaction readout):
honest ceiling ~0.75-0.80 on this ruler (pooled supervised ceiling
measured 0.79 here). The 0.90 promise CANNOT be made from local
compute alone; that is a measured statement, not an opinion.

## 6. What "sure-shot" means here, honestly

Stage A is deployed industry practice; Stage B is standard
self-supervised metric learning with a decade of evidence; Stage C is
the retrieval field's two-stage architecture (bi-encoder → cross-
encoder). None of it is novel — that is the point. The residual risks
are named: cross-embodiment likeness leans on the VLM (hence U3's
budget admits it), and tiny-corpus quality leans on the VLM entirely
(hence U4 keeps B off). If a target is missed, the miss will localize
to one stage with one knob — never to a pile of interacting rules.
