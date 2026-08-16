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
| V-JEPA 2.1 ViT-B drop-in | unloadable | not in any released transformers; `encoder.layernorm` missing → random init, `predictor.proj` 1664 vs 768 |

## 3. ALIVE — measured wins, keep and build on

| tried | result | reading |
|---|---|---|
| stream-time records (v6/v8) | the shipped geometry | `t` is absolute time; windows tile exactly |
| gate-pooled `b` as primary | b alone 0.442 > a alone 0.420 | `a` is partly the model's prior |
| arc-length reparameterisation | largest single frozen gain | re-index by cumulative gate energy |
| graph diffusion (earlier system) | **+50% AP holdout, 0.396→0.597** | **NOT in the current read path — untapped** |
| transition anchor | q04 0.72→0.92 | direction from the trace, not from text |
| DINOv3 identity cut | AUC 0.9994 | identity rides on tracks |
| one-pass write (vjrec8) | 1.72x, bit-identical | 5688 arrays verified |
| **whitening on target corpus** | **sig unseen 0.258→0.293 (+14%)** | fits itself to any corpus; carries no rcasa |
| **fix+sig whitened** | **0.312/0.312/0.312** | ZERO seen-vs-unseen gap. Fully frozen |
| **head+fix+sig fused** | **unseen 0.248→0.314 (+27%)** | best overall 0.341; channels complementary off-domain |

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
| composite (sealed) baselines | **TBD — first run pending** | |

## 6. RUNS

| id | change | PR(z) | prec (sealed) | verdict |
|---|---|---|---|---|
| ssl_v1_s{0,1,2} | span-pred + overlap-InfoNCE + VICReg, d=256, 13 video-h pool | ep0 13.2 | pending | running |

## 7. QUEUE — ranked, each must be self-supervised

1. **Apply the §3 read-side wins to the SSL z**: whitening + fusion with
   frozen channels. Both already measured on the old head; they are free and
   compose with any z. Test as part of every eval, not as a separate idea.
2. **Graph diffusion at read** — +50% AP measured on an earlier system,
   still absent from the read path. Highest untapped single win.
3. **Stage D mining** — mutual top-k across videos, fix∧sig consensus,
   DTW-verified, top 1% only.
4. Longer/harder spans; multi-horizon span targets.
5. Wider d if PR saturates at the ceiling of d.
6. k-reciprocal re-ranking (training-free, standard in re-ID).

Do NOT queue: anything in §2 without a stated new reason.
