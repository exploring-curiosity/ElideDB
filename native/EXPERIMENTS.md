# EXPERIMENTS.md — what has been tried, and what it cost

The ledger that makes "do not repeat what failed" enforceable. Every entry is
MEASURED, not reasoned. Before proposing an idea, grep this file. Add a row
the moment a result lands, including — especially — the negative ones.

Columns: what was tried / the number / the reading.

## 0. The objectives, so iteration does not drift off them

1. **Generalize to data never seen.** The system answers "when did something
   like this happen?" on a corpus it was not trained on. Precision on the
   SEALED corpus is the score; in-domain numbers are diagnostics.
2. **Label-free at train and serve.** No task names, no VLM, no text
   supervision in the loop, no per-dataset priors.
3. **Real-time-able.** Write ≤ ~15 min/video-hour; read ≤ ~1 s/query.
4. **Precision is the deliverable.** Chance and lift are diagnostics for
   comparing across pools; the number that ships is prec.

## 1. Standing bars (violating these invalidates a result)

- error channel `pred - act` barred; at most one of a/b as a vector
- cross-view (same rollout, other camera) barred as positives AND negatives
- no tempo-invariance training — 7 s and 20 s events are DIFFERENT
- no UMAP for retrieval; no labels at serve; no hardwired vocabularies
- seed-aggregate, never a best checkpoint
- verify artifacts on disk, never exit codes

## 2. DEAD — measured, do not retry without a new reason

| tried | result | reading |
|---|---|---|
| clip-time descriptors (v4) | 0.533 prec but broken by construction | sampling rate was a function of clip length; `t` indexed position, not time. Worked by accident |
| per-step SNR weighting | null | no signal |
| extent normalisation | null | |
| differencing baseline k=1..8 | 0.375→0.323, peak k=6 0.395 | integrating a over k does not beat one step |
| encoder receptive-field sweep | null | |
| tempo-adaptive stride (`--tempo-ds`) | null | and conflicts with the no-tempo-invariance bar |
| 9 hand-built interleaved a/b/sig constructions | all ≤ baseline | interleaving by hand does not find what a learned gate finds |
| whitening/power-norm/smoothing **on v6 pooled** | null | NOTE: whitening DID work later on sig+fix at corpus scale (§3) — the earlier null was a different object |
| batching 1→8 at encode | 0.95-1.00x | GPU already saturated |
| crops, patch grids | dead at corpus scale | an 80-episode pool lied by 10x |
| learned scorer over latents | loses to plain cosine | rank with cosine, learn the space instead |
| 6 label-free state constructions | 0.333-0.354 | |
| pooling / DTW variants / grid / localization | null | |
| trained z (discriminative, rcasa labels) | in-domain 0.526→0.737, **out-of-domain 0.908→0.804** | THE central failure this program exists to fix |
| 192px encoder | test −0.034, ood −0.068 | 1.84x faster, not free |
| 64-frame window | test +0.041, **ood −0.114**, drops 13% of corpus | more context helps in-domain, hurts transfer |
| movi_e corpus | 0/1616 usable | 2.0 s clips vs a 4 s window |
| **graph diffusion at read** | **NULL: 0.404 -> 0.406 best (CI ±0.04); alpha 0.9 collapses to 0.28** | The +50% AP from an earlier system does NOT reproduce. Caveat on the implementation: the graph is MEAN-POOLED, and composite recordings are 208 steps (~52 s), so averaging destroys the moment structure the graph needs. A DTW-built graph costs n^2 alignments (hours) - not worth it on evidence this weak |
| **where_map as a retrieval channel** | seen 0.375→0.413, **unseen 0.404→0.394** | Spatial motion layout is a SCENE prior - where in frame motion happens depends on kitchen layout and camera pose. Helps in-domain, hurts transfer. Same failure shape as the trained head, different source |
| whitening a FUSION (truncates each channel to 256d) | 0.338 vs 0.404 z-scored | whitening helps a SINGLE channel, HURTS a fusion: 768d after truncation vs 2048d kept. Decorrelation does not pay for the dimensions lost |
| **grading composite by `group_key`** | **chance 0.543, lift 1.09-1.20x** | **INVALID METRIC.** The verb x object parser cannot read compositional names and dumps 900/1152 into `Other/Other`. Any prec measured this way is uninterpretable - it looked like 0.651 vs a 0.314 bar. Use `--event-key task`: 32 groups x 36, chance 0.031 |
| overlap-InfoNCE with cross-recording negatives only | nce -> 0.005 by epoch 2 | "which video is this" is trivial; the term stops contributing. Same failure as the earlier hard-negative lesson. Fix = within-recording disjoint spans (`--hard-neg`) |
| V-JEPA 2.1 ViT-B drop-in | unloadable | not in any released transformers; `encoder.layernorm` missing → random init, `predictor.proj` 1664 vs 768 |

## 3. ALIVE — measured wins, keep and build on

| tried | result | reading |
|---|---|---|
| stream-time records (v6/v8) | the shipped geometry | `t` is absolute time; windows tile exactly |
| gate-pooled `b` as primary | b alone 0.442 > a alone 0.420 | `a` is partly the model's prior |
| arc-length reparameterisation | largest single frozen gain | re-index by cumulative gate energy |
| transition anchor | q04 0.72→0.92 | direction from the trace, not from text |
| DINOv3 identity cut | AUC 0.9994 | identity rides on tracks |
| one-pass write (vjrec8) | 1.72x, bit-identical | 5688 arrays verified |
| **whitening on target corpus** | **sig unseen 0.258→0.293 (+14%)** | fits itself to any corpus; carries no rcasa |
| **fix+sig whitened** | **0.312/0.312/0.312** | ZERO seen-vs-unseen gap. Fully frozen |
| **head+fix+sig fused** | **unseen 0.248→0.314 (+27%)** | best overall 0.341; channels complementary off-domain |
| **frozen fix+sig, Z-SCORED, 1792d, SEALED corpus** | **0.401 ± 0.021 ALL / 0.404 unseen (240 q)** | **THE RESULT, and it is UNTRAINED.** 14x chance on a corpus 91% unseen-task, and BETTER on unseen (0.404) than seen (0.375) - no generalization gap at all |
| mined cross-video positives | 94.5% same-task, 1754 pairs | the frozen space is good enough to teach itself; channel consensus rejects look-alikes |

## 3a. THE DECISIVE CONTROL (2026-08-16) — the trained head is INERT

| representation | d | unseen prec |
|---|---|---|
| frozen fix+sig whitened | 512 | 0.376 |
| **frozen fix+sig z-scored, NO z** | 1792 | **0.400** |
| ssl_v1 z + fix + sig | 2048 | 0.406 |
| ssl_v2 z + fix + sig | 2048 | 0.404 |

**z contributes +0.006 against a CI of ±0.04.** Every "fusion win" reported
before this control was NORMALISATION and RETAINED DIMENSIONS, not the head.

Two objectives as different as SSL admits — v1 forecast-driven with a dead
contrastive term at PR 46.7, v2 contrastive-driven with a live one at PR 27.5
— produced 0.406 and 0.404. A result that insensitive to what the model
learned means the model is not the bottleneck.

**Retroactive caution:** the earlier atomic claim "head+fix+sig 0.341 beats
frozen 0.312, +27% from fusion" has the SAME confound — 0.312 was whitened,
0.341 z-scored, and the control was never run there. Treat it as unproven.

**Statistical caution:** at 60 queries CI is ±0.04, so 0.376 / 0.398 / 0.400 /
0.404 / 0.406 are ONE BAND, not a ranking. Only z-alone (0.202-0.218) is
clearly separated. Do not read rank order inside the noise.

## 3b. THE CENTRAL LESSON (2026-08-16, cost: one full training run)

**PR(z) is a DIAGNOSTIC, not a TARGET. Optimising it directly made retrieval
worse.**

| | effective dims | prec on sealed corpus (unseen) |
|---|---|---|
| shipped head | 3.8 | 0.248 |
| ssl_v1 | **46.7** | **0.218** |
| frozen fix+sig whitened | 28.3 | **0.376** |

The shipped head's 4 dimensions were 4 USEFUL dimensions. ssl_v1 has 47
dimensions that encode less of what matters. A VICReg variance floor spreads
a representation across axes; it does not make those axes semantic. Without a
strong invariance term, the extra capacity fills with nuisance — camera,
lighting, phase, background.

So the original diagnosis ("the head collapsed to 4 dims, force it wider")
was half right and half wrong. Collapse WAS the symptom. Width was not the
cure.

**Why ssl_v1 specifically failed:** every positive it ever saw was two crops
of the SAME recording, and its InfoNCE hit 0.005 by epoch 2 because "which
video is this" is trivial. That left span-prediction as the only live signal.
Forecasting your own future latents needs SPECIFICITY (encode exactly this
scene); retrieval needs INVARIANCE (two different videos of the same event
map close). Those objectives partly conflict. It learned within-video
consistency and was then measured on cross-video retrieval.

**Correction for the next run:** cross-video invariance must be the DOMINANT
term, not an absent one. Span prediction drops to a weak auxiliary. VICReg
becomes a floor against collapse, not a driver. And PR is read as "is it
collapsing" (< ~15 bad), never as "higher is better".

## 4. The measured diagnosis (2026-08-16)

Effective dimensionality (participation ratio of the covariance spectrum),
per representation, on the corpus it was trained on vs an unseen one:

| representation | rcasa | atomic_full | reading |
|---|---|---|---|
| head z (shipped, 128-d) | 3.9 | 3.8 | uses 4 of 128 dims; does NOT expand on wider data |
| fix (frozen b, 1024-d) | 16.6 | **26.2** | EXPANDS with corpus diversity |
| sig (frozen SigLIP, 768-d) | 25.2 | 14.5 | anisotropic: mean|cos| 0.80-0.86 |

Precision by seen/unseen task, matched pool (2250), same DTW:

| representation | ALL | task seen | task UNSEEN | gap |
|---|---|---|---|---|
| head z (shipped) | 0.317 | 0.437 | 0.248 | **−43%** |
| fix | 0.281 | 0.291 | 0.275 | −5% |
| sig | 0.256 | 0.254 | 0.258 | +2% |
| fix+sig raw | 0.296 | 0.312 | 0.286 | −8% |
| fix+sig whitened | 0.312 | 0.312 | 0.312 | **0%** |
| head+fix+sig | **0.341** | 0.389 | **0.314** | −19% |

**Conclusion:** V-JEPA generalizes; the head does not. The head's objective —
separate 12 rcasa tasks — is solvable in 4 dimensions, so 4 is what survived.
Corpus-fitted constants in the path: token PCA, predictor calibration
(both baked into stored traces), ARC_DS, and the head.

## 5. THE BARS to beat (all on the sealed corpus unless noted)

| bar | value | where measured |
|---|---|---|
| shipped head, unseen task | 0.248 | atomic_full |
| frozen fix+sig whitened | 0.312 | atomic_full |
| best fusion (head+fix+sig) | 0.314 | atomic_full |
| PR(z) target | ≥ 30 | training gate |
| **composite (sealed), frozen fix+sig whitened** | **0.376 unseen** | vjreps, task key, chance 0.029 |
| **composite (sealed), ssl_v1 z + fix + sig** | **0.406 unseen** | THE BAR TO BEAT |

## 6. RUNS

| id | change | PR(z) | prec (sealed) | verdict |
|---|---|---|---|---|
| ssl_v1_s{0,1,2} | span-pred + overlap-InfoNCE + VICReg, d=256 | 46.7/45.1/47.8 | alone 0.218, fused 0.406 | z adds +0.006 over the no-z control. NULL |
| vjmine (deduped pool) | mutual top-10 + fix∧sig consensus + DTW verify | — | 94.5% same-task @ 20% keep | mined set is sound; 1754 pairs kept |
| ssl_v2_s0 | + mined cross-video positives, hard negatives, span 0.5 | 27.5 | alone 0.202, fused 0.404 | nce ALIVE (0.42 vs v1's 0.005) - the training defect WAS fixed - and retrieval did not move. NULL |

## 6b. SATURATION REACHED on this pool (2026-08-16)

Everything cheap has been measured. On a 13-video-hour pool spanning two
domains, the following are ALL null against a ±0.02-0.04 CI:

  * the training objective — v1 (forecast) and v2 (contrastive, mined
    positives, hard negatives) land 0.002 apart
  * the head itself — +0.006 over a no-head control at 240 queries
  * graph diffusion — +0.002, collapses at high alpha
  * where_map as a channel — helps seen, hurts unseen
  * whitening a fusion — actively negative

What is NOT null: normalisation and retained dimensions (whitened-512d 0.376
-> z-scored-1792d 0.401), and the frozen V-JEPA + SigLIP channels themselves.

**Conclusion: the binding constraint is the POOL, not the model.** 13 hours
across sim-kitchen and one robot dataset. Every model-side and read-side
lever has now been tested and found flat. STREAM_PIPELINE.md is the designed
response and is on hold by owner decision.

Do not start a third training variant on this pool without a new mechanism —
two objectives as different as SSL admits produced the same number.

## 7. QUEUE — ranked, each must be self-supervised

1. **ssl_v2 = mined cross-video positives (dominant) + within-recording hard
   negatives + weak span + VICReg floor.** This is the direct correction of
   the v1 failure. `vjmine.py` bootstraps positives from the FROZEN space
   (0.371), not the trained one.
2. ~~Graph diffusion~~ — MEASURED NULL, see §2.
3. k-reciprocal re-ranking — deprioritised: it reads the same mean-pooled
   graph that made diffusion null, so it likely shares the defect.
4. Read-side whitening + fusion are now measured on every eval by `vjreps`.
5. Longer/harder spans; multi-horizon span targets — only after 1 lands.

**Do NOT queue:** wider d, higher VICReg weight, or anything else justified
by "raises PR". See §3b.

Do NOT queue: anything in §2 without a stated new reason.
