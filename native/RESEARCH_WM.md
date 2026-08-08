# World models and latent-space design — research notes, 2026-08-08

Purpose: decide how ElideDB's latent memory should be built. The system
is latent-only end to end (read, write, training all label-free; ground
truth exists only inside evaluation instruments). The unit is the CLIP —
information extends across time and space.

## 1. Three families, three kinds of latent

**Generative / pixel-predictive** (Genie 3, NVIDIA Cosmos, Wayve GAIA-2).
Latents are optimized so a decoder can REDRAW the world. LeCun's central
objection applies: pixel prediction spends capacity on unpredictable
detail (leaf positions, sensor noise) that no query will ever need.
Wrong shape for a memory layer. Not our path.

**Recurrent stochastic state** (Dreamer v3 RSSM; TD-MPC2's implicit,
decoder-free variant). The latent is a FILTERED BELIEF updated causally
per observation: h_t accumulates everything so far. TD-MPC2 matters as
evidence that prediction-shaped latents with NO decoder are sufficient
for control across 100+ tasks. The lesson we keep: the temporal unit is
a recurrent state, not a pooled stack of frame vectors.

**Joint-embedding predictive (JEPA)** — I-JEPA, V-JEPA, V-JEPA 2,
DINO-WM, LeWorldModel. Predict the LATENT of the unseen part from the
latent of the seen part. No decoder. The representation keeps exactly
what is predictable and drops what is not — which is a definition of
"information worth remembering" that requires no vocabulary. This is
the family we build in.

## 2. The collapse problem is solved now, and cheaply

Historic JEPA training needed heuristics (EMA target encoders,
stop-gradients, multi-term losses) to stop everything mapping to one
point. Two 2025-26 results remove that:

- **LeJEPA** (arXiv 2511.08544, Balestriero & LeCun): proves the
  optimal embedding distribution for downstream use is an ISOTROPIC
  GAUSSIAN, and enforces it with SIGReg — random 1-D projections tested
  against N(0,I). One trade-off hyperparameter, no EMA, no stop-grad,
  ~50 lines, stable across scales.
- **LeWorldModel** (arXiv 2603.19312): end-to-end world model from raw
  pixels with exactly TWO losses — next-embedding prediction + the
  Gaussian regularizer. ~15M params, trains on a single GPU in hours.
  Evaluated by PROBING the latent for physical quantities and by
  SURPRISE (does prediction error spike at physically implausible
  events). Plans 48x faster than foundation-model world models.

This is the enabling result: a self-trained world model at a scale this
machine can afford, with a principled anti-collapse term instead of
folklore.

## 3. What the theory adds

LeCun-group preprints (May 2026) formalize WHEN a JEPA recovers real
world structure — and the paired benchmark finds current models
BRITTLE UNDER MINOR VISUAL SHIFTS. That is precisely what our disguise
battery measures (severity/structure rho), so the battery stays: it is
the brittleness test, and the pixel floor (0.553/0.390) is the bar.

## 4. DINO-WM — the closest blueprint to what we already hold

DINO-WM (arXiv 2411.04983, ICML 2025, NYU/LeCun group): FROZEN DINOv2
patch features + a causal ViT that predicts future patch latents;
planning = MPC with latent MSE as the cost. No pixel reconstruction, no
task supervision. It works for navigation and manipulation.

Direct read-across: we already cache frozen DINOv3 patch latents for
every corpus. What we never built is the DYNAMICS MODEL over them — and
our own measurements show the frozen features alone, however pooled or
projected, tie a 32x32 thumbnail on graded similarity. The temporal
information the user is asking for lives in the transition structure,
which only a trained predictor represents.

## 5. Design for ElideDB (the conclusions)

1. **The latent of a clip = the predictor's state trajectory over it**,
   not a pooled frame embedding. Write path: run the causal predictor
   over the stream, store its states (and prediction residuals).
   Read path: same operator on the query clip; similarity = sequence
   matching between state trajectories. No channels, no named axes —
   the state IS the record.
2. **Model**: DINO-WM-shaped — frozen DINOv3 ViT-S patch latents in, a
   small causal transformer/RSSM predicting next latents — trained with
   the LeWM recipe (prediction loss + isotropic-Gaussian regularizer).
   ~15M params is single-GPU-hours territory; over cached latents it is
   smaller still. No labels touch training.
3. **Data**: self-generated MuJoCo episodes (the generator exists;
   unlimited, disjoint from eval episodes by seed). Real-corpus
   training data stays out until the sim loop is proven.
4. **Measurement — sim ground truth, eval-only** (the user's rule):
   - PROBING: linear probes from frozen latents to ground-truth
     physical quantities (block positions, held/free, contact) — probe
     accuracy = "how much information did the latent extract". Probes
     are instruments; nothing they learn enters the system.
   - EVENT STRUCTURE: prediction-error peaks vs meta.json t0/t1
     boundaries (LeWM's surprise evaluation, with real ground truth).
   - RETRIEVAL: query one of 250 stack events, count how many of the
     other 249 return and how many of the 50 unstacks stay out — a
     precision/recall statement a human can read.
   - BRITTLENESS: the disguise battery; must clear the pixel floor.
5. **Gates**: (a) probes beat the same probes on raw frozen features;
   (b) surprise aligns with event boundaries; (c) sim retrieval beats
   the frozen-feature mean baseline; (d) disguise battery >= pixel
   floor. Fail any gate -> the recipe changes, not the ruler.

## Sources

- DINO-WM: https://arxiv.org/abs/2411.04983 , https://dino-wm.github.io/
- LeJEPA: arXiv 2511.08544
- LeWorldModel: https://arxiv.org/abs/2603.19312
- V-JEPA 2 / V-JEPA 2-AC: arXiv 2506.09985
- JEPA theory + brittleness benchmark: LeCun-group preprints, May 2026
  (via techtimes summary; fetch exact ids before citing formally)
- TD-MPC2, Dreamer v3: decoder-free / RSSM background
