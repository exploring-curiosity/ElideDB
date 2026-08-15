# ElideDB / RelMo — V-JEPA 2 span retrieval

State as of 2026-08-14. Branch `relmo-vjepa-span`. The V-JEPA 2 **encoder and
predictor remain frozen** — no gradient ever reaches them. A small recurrence
`f` on top of their frozen output is trained (333k params, `relmo/vjz.py`);
everything below the "frozen encoder" heading predates it and is kept as the
zero-training floor.

## What the system is

Given a query clip, return **located segments** — `(recording, t0, t1)` — from
anywhere in a corpus, where the segment boundaries come from the match and not
from any fixed chunking of the input.

```
video ──► V-JEPA 2 encoder (frozen)
            │
            ├─ layer 6   ‖h(t+k) − h(t)‖         WHERE  the gate: what changed
            └─ layer 24  predictor
                          pred(t+k) − actual(t)   WHAT   the content: what the
                                                         model expected to change
                     ↓ pool WHAT over the frame, weighted by WHERE
              per-timestep descriptor sequence
                     ↓
              subsequence DTW, free start and end
                     ↓
              (recording, t0, t1)
```

Nothing is labelled at write time. The corpus family names are parsed in
`vjeval.py` only, for scoring.

## Numbers - learned ranker on latents (2026-08-14, current)

No hand-written event channels anywhere. Inputs are V-JEPA `a_t`, its realised
change as two scalars, and SigLIP `sig_t`. Labels supervise a head over those
latents as a graded RANKING (relmo/vjrel.py); the model never predicts a verb,
object, scene or camera - given two clips it emits one number. 337k params,
3 seeds, precision@support under group_key.

| stage | val | test | ood_val (unseen tasks) |
|---|---|---|---|
| frozen encoder | 0.463 | 0.526 | 0.518 |
| physics-supervised recurrence | 0.732 | 0.703 | 0.694 |
| **learned ranker** | **0.924 +/-0.011** | **0.894 +/-0.015** | **0.730 +/-0.021** |
| chance | 0.205 | 0.216 | 0.214 |

### The limit, from family holdout

Whole task families removed from training labels, then queried:

| held out | prec on held-out families | chance |
|---|---|---|
| Microwave (cabinet remains - same event group) | **0.816** | 0.194 |
| Microwave + StackBowls | 0.648 | 0.194 |
| Drawer (nothing sliding remains) | **0.215** | **0.194** |

**An unseen OBJECT inside a trained event type generalises (0.816). An unseen
event type that is CONFUSABLE with a trained one collapses to chance (0.215).**
A held-out drawer opening gets absorbed into the "hinged open" category the
model did learn, because nothing ever taught it that sliding is a different
event.

Novelty alone is not the problem. On ood_val, `ArrangeTea` - a wholly novel
multi-step task - still self-clusters at 0.569 against 0.067 chance, 8.5x,
because it resembles nothing in training. `OpenFridge` reaches 0.776 because
hinged doors are all over rcasa. The failure mode is confusability, not
unfamiliarity.

### The learned scorer is refuted

The owner asked for a similarity scorer rather than cosine, so the ranker
scores the full 24x24 matrix of step-to-step similarities with a small CNN - a
learned generalisation of DTW. At identical parameter count, encoder and loss
it is WORSE everywhere: val 0.836 vs 0.924, ood_val 0.607 vs 0.730. The
learning belongs in the encoder; the comparison should stay cheap. That also
makes the later approximate-index stage tractable.

Size vs value: d64 148k -> 0.834, **d128 337k -> 0.924**, d256 862k -> 0.856.
Bigger is worse, not merely wasteful.

## Numbers — trained recurrence (2026-08-14, superseded)

The frozen numbers below are the ceiling of a frozen encoder. A trained
`z_t = f(z_{t-1}, a_t, g_t)`, supervised only on **relational physics** read
from sim state at training time, moves the shipped read path a long way past
it. Splits are scene-grouped (188 scenes, no scene or camera variant straddles
a split), train 65 / val 15 / test 20. Mean ± sd over 5 seeds.

| split | frozen | learned `z` | chance |
|---|---|---|---|
| test, train-disjoint pool | 0.526 | **0.737 ± 0.003** | 0.216 |
| test, deployment index | 0.534 | **0.739 ± 0.006** | 0.222 |
| ood_val — unseen tasks (`ArrangeTea`, `OpenFridge`) | 0.518 | **0.653 ± 0.008** | 0.214 |

Per query class, test / train-disjoint pool:

| class | chance | frozen | learned | ≥0.70 |
|---|---|---|---|---|
| Open/sliding | 0.095 | 0.753 | **0.938 ± 0.027** | yes |
| PickPlace | 0.271 | 0.611 | **0.848 ± 0.012** | yes |
| Open/hinged | 0.211 | 0.438 | 0.637 ± 0.022 | −0.063 |
| Close/hinged | 0.169 | 0.433 | 0.626 ± 0.008 | −0.074 |
| Close/sliding | 0.095 | 0.329 | 0.471 ± 0.014 | −0.229 |

**The controls carry the claim.** Reconstruction-only (no physics loss) scores
0.458 and permuted physics targets 0.325 — both *below* the 0.526 frozen
baseline. The recurrence architecture alone loses; the physics supervision is
what works.

**Why physics and not labels.** Targets are openness rate, contact, speed,
rotation and motion in the gripper frame (`relmo/vjphys.py`), read from sim
state at TRAINING time only — never at serve, never indexed. Task-family labels
are used solely to grade. The transfer claim rests on the targets being
physical: an `OpenFridge` query, an object never trained on, goes 0.481 → 0.662
against in-domain distractors.

**Feasibility gate first.** Before any training, the frozen descriptor was
probed for these targets: opening-vs-closing reads at 0.923 per step and 0.974
per episode on held-out scenes, against a 0.505 shuffled control. The
information was already present and the cosine metric was discarding it — which
is the whole diagnosis in one number.

**Seed noise nearly fooled the selection.** One checkpoint read 0.678 on
ood_val; five seeds of that config average 0.603 ± 0.049. Selecting on
seed-mean moved the choice to `wrec=1.0` (ood_val 0.653 ± **0.008**). Every
number above is a seed aggregate, never a best checkpoint.

### ood_test on real robot video - the trained model LOSES

bridge, 503 episodes, real cameras, real kitchens, a WidowX arm, no sim state.
Two event types ("sweep into pile" vs "put X in pot/pan and pot/pan on stove"),
balanced, chance 0.499. Read once.

| arm | ood_test | 95% CI |
|---|---|---|
| **frozen `pred_change`** | **0.908** | [0.899, 0.917] |
| `wrec=0.1` + SigLIP | 0.875 | |
| CTRL no-physics | 0.839 | |
| trained `z` (`wrec=1.0`, 5 seeds) | 0.804 +/- 0.030 | |
| `wrec=0.1` | 0.692 | |
| CTRL shuffled targets | 0.748 | [0.739, 0.757] |

**Training loses to not training, out of domain.** The more appearance an arm
preserves the better it does here, and the no-physics control beats every
physics arm.

The reason is specific, not general. Both bridge classes are free-body
transport: in our target space both have `has_art = 0` and `d_open = 0`, so the
physics targets cannot separate them at all, while appearance separates them
trivially. The target set is ARTICULATION-CENTRIC and this corpus has no
articulation, so the appearance `z` gave up is never repaid.

So the transfer claim has to be stated narrowly and honestly:

  transfer across TASKS within a domain   HOLDS   0.518 -> 0.653 (ood_val)
  transfer across DOMAINS                 FAILS   0.908 -> 0.804 (ood_test)

A caveat that cuts the other way: this is a two-class, appearance-separable
problem, which is the setting least able to show what the representation is
for. It is evidence that training costs appearance sensitivity; it is not
evidence about fine-grained event matching on real video, which bridge's
long-episode subset cannot test.

NOTE ON SELECTION. The SigLIP arm dominates on val (0.787) and ood_test (0.875)
and ties on ood_val, so it looks like the better arm - but ood_test was
designated read-once and switching arms on it would consume it as a selection
set. Recorded, not acted on. A future selection needs a fresh OOD domain.

### What the ceiling actually is

Retrieving with the **true** physics sequences as the descriptor, under the
identical protocol, sets the ceiling of the physics-supervision route. A
label-supervised probe (fit on `group_key` on purpose, never shipped) sets what
supervision itself is worth. Val:

| descriptor | overall | PickPlace | Open/hin | Close/hin | Open/sli | Close/sli |
|---|---|---|---|---|---|---|
| frozen `pred_change` | 0.463 | | | | | |
| GT-physics oracle | 0.709 | 0.727 | 0.634 | 0.740 | **1.000** | **1.000** |
| GT-physics + scale invariance | 0.743 | | | | | |
| GT-physics, val-pruned channels | 0.799 | | | | | |
| **trained `z`** | **0.765** | 0.848 | 0.637 | 0.626 | 0.950 | 0.471 |
| label-supervised probe *(ceiling)* | 0.862 | **0.949** | **0.882** | 0.717 | 0.650 | 0.550 |

Three readings:

1. **The trained model already beats the physics oracle** (0.765 vs 0.709). It
   is not merely recovering the physics - the reconstruction term carries
   appearance structure the targets do not have.
2. **Label supervision is not a uniform ceiling.** It reaches 0.862 overall yet
   is *worse* than the label-free model on Open/sliding (0.650 vs 0.950).
3. **Each source is strong where the others are weak.** Physics owns the
   sliding classes; labels own hinged and transport. A per-class best-of is
   ~0.90, so that target is reachable only by a representation carrying physics
   AND appearance structure at once - not by adding categories.

The largest single gap in the system is `Close/sliding`: oracle **1.000**,
model **0.471**. Perfect physics separates closing drawers flawlessly, so this
is estimation error, not a missing concept.

Channel pruning must be derived on val. A test-side pass called `grip_dist` and
`speed` nuisance; val leave-one-out says they are among the most valuable
channels (-0.051, -0.048 to drop) and only `d_rel_x` (+0.026) and `open`
(+0.021) hurt.

### Why the remaining classes are short

`Close/sliding` is the smallest physical event in the corpus. `CloseDrawer`
episodes begin from a half-open drawer and traverse **0.450** of the joint
range, against 0.740 for `OpenDrawer` and 0.958–0.998 for cabinets, and a
drawer rotates least of anything measured. Least translation *and* least
rotation — it is the worst class for the frozen model (0.329) and the trained
one (0.471) alike. That is a property of the data, not a bug in the metric.

Adding rotation targets (`d_rot`, `rot_cum`) was aimed at hinged-vs-sliding
confusion and **worked on its mechanism** — Open/hinged wrong-kinematics errors
halved, 23/186 → 12/186, precision 0.726 → 0.796 on val — but did **not** move
the aggregate (ood_val 0.656 ± 0.019 vs 0.653 ± 0.008). Recorded as a mechanism
win and an aggregate null.

## Numbers — frozen encoder (superseded as the shipped path)

447 episodes, k = support, group-aware grading, chance 0.222.
95% intervals bootstrapped over **queries** (2000 resamples); paired
differences use shared resample indices.

| arm | true/returned | precision | 95% CI | lift |
|---|---|---|---|---|
| **pred_change / span** | 19423/36990 | **0.525** | [0.508, 0.540] | **2.37×** |
| pred_change / cosine | 19093/36990 | 0.516 | [0.499, 0.532] | 2.33× |
| obs_change / span | 18128/36990 | 0.490 | [0.471, 0.508] | 2.21× |
| obs_change / cosine | 16634/36990 | 0.450 | [0.430, 0.468] | 2.03× |
| scene-only baseline | 11162/36990 | 0.302 | [0.294, 0.309] | 1.36× |

Paired: `span − cosine = +0.0089 [+0.0046, +0.0132]`, excludes zero. The span
matcher genuinely beats whole-sequence cosine rather than merely tying it.
`pred_change` beats `obs_change`, so *what the model expected to change*
carries more than *what actually changed*.

Time-warp invariance — median rank of the true original among 447, query is the
same episode re-rendered under a warp (correct answer known without annotation):

| channel | matcher | identity | ease_in | ease_out | sigmoid | zoom |
|---|---|---|---|---|---|---|
| **pred_change** | **span** | **1.0** | **1.0** | **4.5** | **1.0** | **1.0** |
| pred_change | cosine | 1.0 | 18.0 | 55.0 | 1.0 | 1.0 |
| obs_change | span | 1.0 | 2.5 | 4.5 | 1.0 | 1.5 |
| obs_change | cosine | 1.0 | 306.0 | 212.5 | 3.0 | 7.5 |

### The barred channel

`error = pred(t+k) − actual(t+k)` is **forbidden** by standing owner rule and is
not cached, benchmarked or reported. It scored 0.584 [0.566, 0.602] and was
rank 1.0 on every warp — better on both. That is not a reason to use it.

It is a function of *(event, model prior)*, not of the event: its magnitude
reports what the model happened to guess, so the same event yields different
descriptors as the prior shifts and an index built from it drifts against
itself under any adaptation. It is structurally worst on exactly the open/close
distinction the product exists to make. Its advantage is evidence about one
frozen checkpoint, not about the design. The ~0.06 is the accepted cost of a
signal that generalises.

I argued for keeping it three times on benchmark grounds. That was converting a
design constraint into a metric question, and the justification I gave for it
("error is a contrast operation") was constructed after seeing it win.

## Things that were refuted, and by what

**Appending predicted physics to the descriptor** (2026-08-14). The diagnosis
said the discriminative directions are low-variance inside `z`, so giving the
physics head's standardised output unit weight in the metric should help. It
does, in domain: val 0.765 -> 0.783. It **hurts out of domain**: ood_val
0.653 +/-0.008 -> 0.616 +/-0.020. The head was fit on rcasa; on unseen tasks its
predictions are less reliable and weighting them up amplifies that error.
Rejected by the ood_val selection rule. Selecting on val would have shipped it.

**Pairwise term from channel agreement** (2026-08-14). The loss had never seen
two clips together, so nothing optimised the geometry cosine+DTW reads - the
gap between a regressive z (0.73) and a discriminative probe on the SAME frozen
features (0.862). Positives from two independent channels agreeing are 97.5%
same-group against 0.189 chance, with no label and no declared category. It
helps in domain and HURTS out of it: val 0.732 -> 0.754, ood_val 0.694 ->
0.667. Rejected by the ood_val selection rule - the third time in this session
that a change helping val hurt OOD.

**MaxSim as a DTW surrogate** (2026-08-14). Theory: mean-pooling optimises a
different geometry than the DTW the query path uses. Measured WORSE at every
weight - 0.678-0.714 against mean-pooling's 0.754. MaxSim lets any step match
any step, so it is a LOOSER surrogate than DTW, not a tighter one; it discards
the ordering DTW enforces. Batch size was ruled out separately (bs 96 vs 291:
0.697 / 0.702), so "positives invisible in a batch" was also wrong.

**Between/within-episode scatter subspace** (2026-08-14). Directions where
episodes differ more than moments within an episode differ, as a label-free
route to the discriminative subspace. 0.29-0.35 against the frozen 0.463. It is
instance discrimination in disguise: it rewards separating two OpenCabinet
clips, which is exactly backwards.

**Uncurated physical targets** (2026-08-14). Raw pose trajectories of the top-2
movers, to remove the hand-design in picking 11 scalars: 0.455 vs 0.709. The
curated channels encode invariances (normalised openness, unit-free rotation)
that raw world pose lacks. Less hand-designed is not automatically better.

**Motion-weighted physics loss** (2026-08-14). Most trace steps of a
Close/Drawer episode carry no event - 1.6% of its articulation is in the first
six steps - so weighting the loss by per-step motion should concentrate
capacity where events happen. Val over 4 seeds: 0.7528 +/-0.0062 unweighted,
0.7612 +/-0.0213 at w=2, 0.7605 +/-0.0298 at w=5. The mean rises 0.008 and the
spread triples. When the sd exceeds the gap the arms are indistinguishable, and
the instability is specific to the weighted arms. Null.

**Rotation targets as an aggregate win** (2026-08-14). `d_rot`/`rot_cum` were
added to fix hinged-vs-sliding confusion and did fix it - Open/hinged
wrong-kinematics errors 23/186 -> 12/186, precision 0.726 -> 0.796 on val - but
ood_val was unchanged (0.656 +/-0.019 vs 0.653 +/-0.008). A mechanism win is not
an aggregate win.



Each of these was believed, then measured, then abandoned.

| claim | outcome |
|---|---|
| error-based representations shed the scene confound | **false** — static patches 83.3 vs moving 88.3, ratio 1.07 at layer 24 |
| …though at layer 3 the ratio is 6.64, so the refutation was depth-specific | retracted and re-recorded |
| ordering vs warp-robustness is a fundamental tradeoff | **false** — an artefact of the context-length ramp; span is both ordered and warp-robust |
| a time warp changes the residual's content irreducibly | **false** — rank 32.5 → 1.0–4.5 once context length is fixed |
| the error channel was worth keeping because it scored higher | **false** — a design constraint is not a metric question; barred |
| half of every clip is wasted | **false** — the discarded half is the setup, 0.461 alone |
| a short prediction horizon is sharper | **false** — long horizon 0.586 vs short 0.574 |
| subtracting appearance removes nuisance | **false** — 0.472 → 0.335 |
| fusing channels helps | **false** — every combination ≤ best alone |
| the sweep geometry carries the event | **false** — most object-biased feature measured, lift 1.55 on the wrong axis |
| drawers are a failure case | **false** — a grading error; they are the best family once grouped correctly |

## Open, in the order they bite

1. **Index cost — 6.5× real-time.** An hour of recording takes ~6.5 h to index.
   This is what blocks the real use case. Engineering, not research.
2. **`ease_out` warps sit at rank 4.5, not 1.0.** Fast-start/slow-end is the one
   differential profile the alignment does not fully absorb. Still top 1% of
   447, so it degrades gracefully rather than failing.
3. **Close is systematically weaker than Open.** `CloseCabinet` 0.339 is the
   weakest real family. Closing ends in contact and stillness, which is how many
   things end; opening ends in a revealed interior, which is more distinctive.
4. **Boundary accuracy is unvalidated.** Span containment is 8/8, but that only
   proves the right episode. The clean label-free test is to query with a
   sub-span of a known episode, where true boundaries are exact.
5. **Layer 6 was chosen by peeking at the graded score.** Layers 3/6/9 came out
   1.92/1.93/1.91, so it is inconsequential, but it is the one hyperparameter
   that saw labels.

## Reproduce

```bash
python -m relmo.vjrec4  --dataset rcasa      # build records  (~45 min, resumable)
python -m relmo.vjeval4 --dataset rcasa      # table above, with intervals
python -m relmo.vjtime  --episodes 10        # warp invariance
python -m relmo.vjspan  --episodes 8         # span localisation, stitched
python -m relmo.vjview  --queries 18         # viewer -> data/relmo/viewer
```

---

# v6 — stream time: `t` is absolute time, not a position in a clip

The owner's correction: *"t doesnt make sense with this logic at all. t means
time. this doesnt signify any history. just an variable length encoder of
smaller vs bigger window. I am saying t should come from the fixed rate
windows. 0-8, 4-12 etc."*

That was right, and the defect was structural. v4 sampled a fixed number of
frames across the WHOLE episode, so the sample rate was a function of duration
(4.6 fps for a 7 s clip, 0.53 fps for a 60 s one), and `z` was re-initialised
to zero for every clip — so the recurrence `z_t = f(z_{t-1}, …)` had no history
to carry.

## The redefinition

    dt = TUBELET / STREAM_FPS = 0.25 s        t is an absolute stream index
    window   32 frames @ 8 fps = 4.0 s        context 2.0 s, descriptors 2.0 s
    hop      2.0 s  ==  descriptor span       so windows TILE: no gap, no overlap

    a_t = x^_t - x^_{t-1}    expected one-step change, forecast vs forecast
    b_t = x_t  - x_{t-1}     the change that occurred
    g_t = [cos(a_t,b_t), log(|b_t|/|a_t|)]    reality enters as 2 scalars
    z_t = (1-g)*z_{t-1} + g*c_t               initialised ONCE PER RECORDING

Tiling is what makes concatenation legal, and it is asserted at runtime, not
assumed. `j(T) = T//8 - 1` is the unique window describing step T.

Geometry chosen against the corpus, not by taste: at 8 s / 4 s (the literal
"0-8, 4-12"), 58 of 447 rcasa episodes are shorter than one window and would be
silently dropped, and 242 of 447 would yield a single window — so the one thing
v6 exists to test, carrying `z` across a boundary, would never happen for most
of the corpus. At 4 s / 2 s nothing is dropped and the median recording spans
4 windows. Cost: 2× the windows per second of video.

## Two defects found by measurement, not inspection

**Mixed differences.** The first draft referenced each block's opening step to
the observed anchor, since no forecast of the preceding step existed. |a| then
spiked ~2× at every block start — a period-4 sawtooth locked to the grid, the
exact artefact the rewrite existed to remove. Forecast and observation are not
interchangeable. Fixed by predicting one extra step per block so every `a_t` is
forecast-minus-forecast. Costs nothing.

**Window phase.** With a cubic trend in episode time removed and phase permuted
*within* each recording as the null, window phase explains **67.7%** of `a_t`'s
direction variance and 27.4% of `b_t`'s (null 0.1%, p<0.005). Two sources, both
structural: V-JEPA's temporal position embeddings (which is why `b_t`, pure
observation, shows it at all), and two prediction blocks per window at horizons
1–5. Under v4 this cancelled as common mode because every clip sat on an
identical grid.

`relmo/vjphase` removes it — per-phase mean of the unit descriptor, fitted on
TRAIN only, label-free, same category as the affine predictor calibration.
Held-out phase R² **0.696 → 0.023** (val), **0.724 → 0.019** (test).

## Numbers — precision@support, k=support, group_key grading, 3 seeds

| arm | val | test | ood_val |
|---|---|---|---|
| v4 clip-time, frozen `a_t` + DTW | 0.463 | — | 0.518 |
| v6 stream, frozen `a_t` + DTW | 0.314 | — | 0.542 |
| v6 stream, frozen, phase removed | 0.289 | — | 0.516 |
| v4 clip-time, trained `z` | **0.924** ± 0.014 | **0.894** ± 0.018 | **0.730** ± 0.026 |
| v6 stream, trained `z` | 0.793 ± 0.092 | 0.763 ± 0.041 | 0.502 ± 0.067 |
| v6 stream, trained `z`, phase removed | 0.738 ± 0.083 | 0.752 ± 0.066 | 0.610 ± 0.108 |
| v6 stream, trained `z` + CNN scorer (1 seed) | 0.768 | 0.736 | 0.655 |

chance: val 0.205, test 0.216, ood_val 0.214. Same evaluator, same splits, same
pools throughout — the only change is which records the arm reads.

**Stream time costs 0.13 on test and 0.12–0.23 on ood_val.** It is worse, and
the reason is visible in the frozen row: v4's whole-episode normalisation gave
duration invariance for free — a 7 s and a 20 s instance of one event landed on
the same 24 steps. A fixed rate does not. v6 wins only where v4's sampling was
degenerate: `ood_val`'s recordings have a median of 41.9 s, which v4 sampled at
1.5 fps, and the frozen row moves 0.518 → 0.542 there.

That trade is not optional. v4's advantage comes from being handed episode
boundaries, and the production input has none — *"all the frames will be
timestamp attached sitting in a parquet file only with time differences to show
episodic different"*. v4 cannot run on that input at all; v6 can. The 0.13 is
the price of the missing piece, and the missing piece is named: **duration
invariance, which has to come back from encoding at several rates** and letting
the matcher align across them. Not built.

## The phase correction is neutral in-domain and helps out of domain

test 0.763 → 0.752 (inside seed noise), ood_val 0.502 → **0.610** (+0.108, ~1
SD). Direction is as expected: the grid signature is corpus-specific, so
removing it costs nothing at home and helps transfer. An earlier single-seed
read said the correction *hurt*; that was seed noise, which is why the
seed-aggregate rule exists.

## Grid-alignment control — the models learned the event, not the grid

Every episode file starts at its demonstration's onset, so the window grid is
locked to event onset — a coincidence production does not supply. Re-ingesting
the 65 val queries with the grid shifted 1 s and scoring against the ordinary
pool:

| arm | aligned | shifted | delta |
|---|---|---|---|
| frozen `a_t` | 0.300 | 0.313 | +0.013 |
| frozen, phase removed | 0.298 | 0.271 | −0.027 |
| trained `z` × 3 seeds | 0.804 / 0.766 / 0.741 | 0.797 / 0.759 / 0.756 | −0.006 / −0.006 / +0.015 |
| trained, phase removed × 3 | 0.737 / 0.656 / 0.801 | 0.751 / 0.587 / 0.796 | +0.014 / −0.069 / −0.006 |

Worst case −0.069, typical ±0.015. The 68% phase variance is a nuisance the
matcher was already robust to, which is also why removing it changes so little.

## Cost, fp16, end to end including ffmpeg decode

| stage | rate |
|---|---|
| `vjrec6` V-JEPA records | 3.58× real-time |
| `vjsig6` SigLIP | 47.1× real-time |
| **combined, one stream** | **3.33× real-time** |

447 + 27 recordings, 23,744 stream steps, 98.9 video-minutes described. Nothing
dropped: 0 too-short, 0 failed.

## Open

1. **Duration invariance is gone and nothing replaces it.** This is the whole
   0.13. Multi-rate encoding is the named fix and is not built.
2. **Seed variance tripled** — ±0.09 on val against ±0.014 under v4. Ragged
   batches shrink to fit the pair budget, so effective batch size varies; that
   is the first thing to check.
3. **`z` has no leak term.** The recurrence now runs a whole recording, but
   nothing bounds how far back it reaches. On 5–56 s recordings this never
   binds; on an hour it will.
4. **Only rcasa.** bridge (`ood_test`) untouched, and the 63-task / 26,674-
   episode ingest remains held pending confirmation.

## Reproduce

```bash
python -m relmo.vjrec6 --dataset rcasa --fp16          # stream records (~24 min)
python -m relmo.vjsig6 --dataset rcasa                 # aligned SigLIP (~2 min)
python -m relmo.vjphase --fit                          # phase model + held-out R2
python -m relmo.vjood                                  # frozen arms
python -m relmo.vjrank --no-scorer --tag rk6_ns_s0     # train (~2.5 min)
python -m relmo.vjrankeval --tags rk6_ns_s0            # val / test / ood_val
python -m relmo.vjrec6 --dataset rcasa --fp16 --splits val --offset 1.0 --suffix _off1
python -m relmo.vjgrid --tags rk6_ns_s0                # grid-alignment control
```
