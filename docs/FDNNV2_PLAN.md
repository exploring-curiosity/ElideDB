# FDNN-V2 plan: a fast model that understands video context

Plan only. Nothing here is implemented yet. The constraint that governs every
choice: **if it is not fast enough to sit in a database's write path and keep
queries index-only, it is not proposed.**

---

## 1. What the research says, mapped to our measured failures

| finding | source | maps to |
|---|---|---|
| Predictive world-model self-supervision on raw video learns MOTION and action structure without any labels — 77.3% on Something-Something v2, the verb-heavy benchmark where appearance models fail. Encoder + predictor, trained by predicting future latents. | [V-JEPA 2, Meta 2025](https://arxiv.org/abs/2506.09985) | Our measured dead end: "distilling to an appearance teacher can never add verbs — the target space lacks them" (close/open separate at 0.60 in teacher space). Prediction across time is the training signal that CAN add them, and it is free on our own corpus. **This is also the biological line**: the brain understands by predicting what happens next (predictive coding); recurrence + prediction is the FDNN hunch made concrete. |
| Linear-time state-space (recurrent) backbones beat transformers on fine-grained motion (+5.9% SSv2) at 6× less memory; O(1) per step. | [VideoMamba, ECCV 2024](https://arxiv.org/abs/2403.06977) | Validates FDNN-V's architecture CLASS: a recurrent core carrying state per frame is the right shape for both action sensitivity and embed-on-write streaming. We keep the FDNN cell (its KAN-sum neurons are our rule 1), not swap it. |
| CLIP-family verb-blindness is fixable with Verb-Focused Contrastive training: hard negatives made by swapping verbs in captions + verb-phrase alignment. | [Verbs in Action, 2023](https://arxiv.org/abs/2304.06708) | Our probe 1: cos("close the drawer","open the drawer") = 0.951. The 7B captions (being generated now, once, offline) supply exactly the caption corpus VFC needs. |
| Forward-vs-reversed clip discrimination ("arrow of time") is a classic free self-supervision signal for temporal direction. | [Wei et al., CVPR 2018](https://donglaiw.github.io/paper/2018_cvpr_aot.pdf) | Open and close are literally time-reversals of each other. One auxiliary head, near-zero cost, directly forces them apart. |

## 2. The model: FDNN-V2

**Keep the skeleton** (it already meets the speed bar and the FDNN rules):
tiny stem → FDNN recurrent cell (KAN-sum FINER/Gabor/poly neurons, ω-banded,
gated) → heads. Causal, O(1)/frame, ≈2–3M params, measured 0.27 ms/frame.

**Change the training signal** (this is where V1 failed — not architecture):

```
                          ┌── appearance head (1152-d)
frames ─► stem ─► cell ─►─┤     distilled to SigLIP as today — keeps every
          (unchanged)     │     existing index and the text tower working
                          │
                          ├── context head (256-d)  ← NEW, the verb space
                          │     trained by:
                          │     L1 PREDICTIVE (V-JEPA-style): from state h_t,
                          │        predict the embedding of frame t+k.
                          │        Dynamics must live in h to predict.
                          │        Free labels; uses ALL 70k frames.
                          │     L2 ARROW OF TIME: classify forward vs
                          │        reversed clips. open ≠ close by
                          │        construction.
                          │     L3 VERB-FOCUSED CONTRASTIVE (VFC): sigmoid
                          │        contrastive against 7B captions, hard
                          │        negatives = verb/direction swaps
                          │        ("closes"→"opens", "into"→"out of",
                          │        clause order flips). Teaches the joint
                          │        space queries actually live in.
                          │
                          └── text adapter: 2-layer MLP on SigLIP text
                                embeddings → 256-d context space, trained
                                jointly in L3. Query cost ~0.1 ms.
```

Two columns per frame in `frame_vectors` (appearance 1152 + context 256).
Query = text through SigLIP tower once (~10 ms) + adapter (~0.1 ms) + two
matmuls + caption-word match, RRF-fused. **No VLM anywhere in the query
path.** The 7B exists in exactly one place: the offline captioner (running
now; once per corpus, background).

FDNN rules 2+3 (apoptosis → fine-tune → neurogenesis → fine-tune, PPO +
reverse attention) run post-training as before — and the fine-tune data is
now unlimited (self-supervised objectives), which is the regime where the
cycle measurably worked (context tower: fidelity IMPROVED while deleting 73%
of channels).

## 3. The speed budget (hard gates, not aspirations)

| operation | budget | current reference |
|---|---|---|
| embed, batched | ≤ 0.5 ms/frame | 0.27 (V1) |
| embed, streaming | ≤ 1.5 ms/frame | 0.71 (V1) |
| load+embed the 4 h dataset | ≤ 2 min | 79 s (V1 pipeline) |
| query, cold or warm | ≤ 50 ms | index-only; no model but the text tower |
| training a corpus's model | ≤ 1 h background | V1 trained in ~10 min |
| 7B captioner | once per corpus, background | ~2.5 h / 4 h video |

Anything that breaks a row gets cut, not excused.

## 4. Small dataset + evaluation gates

Dataset: **bridge4h** (already loaded; 70,436 frames of free self-supervision;
2,097 human-labelled episodes held OUTSIDE the store for eval only). Secondary
generality check: oxford (street domain, different everything) — same recipe,
retrained weights, no code changes allowed.

Corner-case matrix — each has a numeric probe, run every iteration:

| corner case | probe | signal that should fix it | gate |
|---|---|---|---|
| verb direction (open vs close) | nearest-centroid acc in context space | L1+L2 | ≥ 0.80 (pixels today: 0.60; corpus info bound from robot-state: 0.85) |
| time reversal sensitivity | AUC forward-vs-reversed embeddings | L2 | ≥ 0.90 |
| subject-of-action vs bystander | "green moved" vs "green present" ranking | L3 hard negatives | measured, reported |
| sequential compound ("X then close") | clause-order swap ranking | L3 order flips | measured, reported |
| appearance regression | SigLIP-head fidelity | (must not move) | ≥ 0.90 |
| query battery (8 complex queries) | label-verified top-3, index-only | all | ≥ 18/24 (today 13/24 WITH query-time VLMs; the gate is beating that WITHOUT them) |
| no-answer queries | margin calibration on 5-episode "sink" class | — | flagged low-confidence, not hallucinated |
| latency battery | every row of §3 | — | all pass |

## 5. Retrieval architecture: what the fast-retrieval literature adds

The 2024-2026 video-retrieval work converges on three ideas, two of which
this system already has, and one worth adopting:

| finding | source | status here |
|---|---|---|
| Hybrid two-stage: a two-tower (dual-encoder, indexable) proposes, a one-tower aligner reranks candidates only | [EDG, ICMR 2025](https://dl.acm.org/doi/10.1145/3731715.3733330); [CONE](https://arxiv.org/abs/2209.10918) coarse-to-fine | already our shape — except our reranker became a query-time VLM, which §6 bans; V2 replaces it with the verb-aware context head, moving that work to TRAIN time |
| Temporal redundancy should be MERGED, not sampled away: progressive token merging cuts tokens 95%, GFLOPs 51%, and IMPROVES retrieval (+4.4% R-Sum) | [TempMe, ICLR 2025](https://arxiv.org/pdf/2409.01156) | adopt as novelty-weighted pooling (below) — uniform mean-pool is precisely what erased verbs |
| Index EVENTS, not fixed windows: model each video as sparse discrete events; retrieval ranks events | [EDG](https://dl.acm.org/doi/10.1145/3731715.3733330); streaming fixed-budget memory banks ([2026 line](https://arxiv.org/abs/2606.25658v1), ~20x compression for hour-long streams) | **adopt — and it is nearly free**: see below |

**Event-first indexing, from the gate signal.** The FDNN cell's update gate
z_t measures how much of the scene model each frame is allowed to overwrite —
it is, by construction, a change detector. Sustained high gate activity = an
event boundary. So at ingest the encoder emits, at no extra model cost:

    per-frame vectors  (as today — the fine tier)
    event boundaries   (gate-activity peaks)
    one embedding per EVENT = novelty-weighted pool of its frames
                       (frames weighted by gate activity, so the moment the
                        drawer closes outweighs the seconds it sat still —
                        the TempMe lesson applied at pooling time)

Queries rank a few thousand events instead of tens of thousands of arbitrary
windows: a smaller index with boundaries that mean something, then refine to
frames within the winning events (CONE's coarse-to-fine, all index-only).
This replaces the fixed 4s/2s window plan as the primary retrieval unit;
windows remain only as a fallback for streams where the gate signal is flat.

Gate for this section: event segmentation F1 against episode boundaries
(bridge4h has 2,097 ground-truth boundaries held out) — report it; and
event-index queries must beat window-index queries on the battery at equal
latency.

## 6. Execution order (each phase ends with numbers)

1. **P0 baselines** — freeze today's numbers (done, in BENCHMARKS.md).
2. **P1 training data** — 7B captions (running); predictive pairs and
   reversal pairs are free transforms of the frame cache; verb-swap hard
   negatives from a small antonym/direction map applied to captions.
3. **P2 train FDNN-V2** — combined loss, time-split eval, keep-best guard;
   iterate against §4 gates exactly as V1 was iterated (every change
   justified by a measured delta, failures reported).
4. **P3 FDNN cycle** on the winner (rules 2+3, retrieval objective).
5. **P4 load/embed/read demo** — rebuild bridge4h with V2 columns inside the
   2-minute budget; run the battery index-only; ship only if ≥ 18/24 and
   ≤ 50 ms/query.
6. **P5 generality** — oxford, same recipe, report both numbers side by side.

## 7. Explicitly out

- Any model invocation at query time (2B, 7B, teacher — all of it).
- Dataset-specific inputs (robot state is an eval bound only).
- Frame skipping. Every frame is embedded, as established.
- Growing the model past the §3 budget to chase a gate — if a gate cannot be
  met inside the budget, that is reported as a boundary, not papered over.


---

## P2 execution log (honest, running)

| iter | change | probes (close/open · AoT · app_fid · ms/f) | verdict |
|---|---|---|---|
| baseline | V1 warm start, untrained heads | **0.70-0.72** · 0.49 · 0.995 · 0.17 | novelty-pooled state already beats the 0.60 pixel bound |
| A1 | L1 predict future frame, L2 AoT from final state | 0.70 · 0.47 · 0.992 | AoT at chance while 95% of loss: the gated cell is a leaky INTEGRATOR — an EMA of a smooth sequence is order-invariant; pred loss 0.008 = trivially solved from the present |
| A2 | motion pathway (signed Δg), L1 predicts change of V1 embeddings | 0.57 · 0.51 · 0.935 | worse: student-embedding differences are student NOISE (true change ~0.2 vs student error ~0.45); mean velocity cancels on reciprocal robot motion |
| A3 | own-latent change prediction (stop-grad), temporal-conv AoT | 0.42 · 0.42 · 0.964 | loss learns (1.97→1.54) but probes DEGRADE — optimizing prediction reshapes the ctx space away from retrieval structure |

| A4 | NORMALISED velocity (direction unit-norm, magnitude as bounded gain) | 0.47 · **0.64** · 0.964 | first real AoT signal — but trunk training still spends separability |
| A5 | trunk FROZEN, heads only | **0.70** · 0.55 · **0.995** | separability preserved exactly; AoT falls back — direction learning REQUIRED trunk movement |

**Stage-A conclusion after five iterations:** on this corpus the
self-supervised objectives buy arrow-of-time only by spending retrieval
structure; frozen-trunk keeps structure and loses AoT. Neither meets a gate.
The trunk already carries the separability signal (0.70 untrained vs 0.60
pixels); what remains is PROJECTING it into a queryable space, which is
stage B's job (caption contrastive trains head_ctx + text adapter directly,
on the frozen trunk A5 proved safe).

**Earlier boundary note (superseded by the above):** stage-A self-supervision
has not met any gate in three iterations. The consolidated suspect: the WARM START. V1's stem was trained so
its features barely move between near-identical frames (that is what made
appearance distillation easy) — a motion pathway fed by a motion-blind stem
has nothing to read. Testable next: probe ||Δg|| against pixel motion; if
confirmed, stage A requires stem training from a motion-preserving init,
which is a longer run than an iteration loop supports.

Next lever (unchanged from plan): stage B verb-focused contrastive, which
reads the state+glimpse directly and does not depend on Δg. Requires the 7B
captions (regenerating now, GPU-exclusive).

## Stage B execution log (2026-07-22)

| iter | change | offline text→ctx (close/open) | verdict |
|---|---|---|---|
| B1 | 1-pos + 1-swap-neg contrastive, 6 ep | AUC ~0.56, mean-cos gaps 0.002 | VISUAL COLLAPSE: every clip vector at pairwise cos 0.9993, centroids at 1.0000; adapter split verbs across two poles with zero grounding. Degenerate loss — a constant clip vector satisfies every positive |
| B2 | full in-batch sigmoid (SigLIP loss) + reset heads, 6 ep | put-in 0.83, close 0.65, open inverted | collapse escaping (clip-cos 0.94→0.67); everything pulled toward put-in — captions run 16:1 pick/place vs close |
| B3 | verb-balanced sampling (inverse-√df over generic swap vocab), 40 ep | put-in 0.88; centroid close/open **0.833 — first gate pass**; close/open text AUC still ~chance | visual ctx space now separates drawer polarity; the TEXT side does not |

**Root cause, measured to the source:** the 7B captions' "close" mentions are
ANTI-correlated with closing — 12.9% of captions inside open episodes mention
"close" vs 7.9% inside close episodes; 735/2348 captions are template
duplicates. Stage B's verb supervision was mislabeled at generation time.
Change-caption pilots confirm generation is the broken tier: 2B free-form
before/after 1/24 correct verbs, 7B 8/24 (close 2/12) — while yes/no
DISCRIMINATION margins work (2B AUC 0.75, 7B 0.82).

**Consequence (architecture, not tuning):** verb grounding must come from
discriminative verdicts. vlm_verdicts now stores query TEXT with each margin,
so ordinary queries accumulate (text, segment, ±margin) training triples —
database cracking extended to the model. Retraining the adapter on verdicts
is the next lever; captions remain useful for nouns/lexical recall only.

**Shipped to the query path meanwhile:** context_events index (1,295 events
from 70,436 frames in 19.5 s, event = gate-peak segment, novelty-pooled
256-d ctx), ctx as a third RRF channel beside appearance + caption-lexical,
TF-IDF fitted-model caching (50 ms → ~2 ms), one batched tower pass for all
query atoms (compound queries 106 → 43 ms). Warm index-only latency:
25/43/54 ms for 1/3/4-atom queries. Cold index-only battery 13/24 — equal to
the full sync VLM cascade's previous score, at query-time model cost ZERO.
