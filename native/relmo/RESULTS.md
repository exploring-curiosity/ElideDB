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

---

# v7 — duration invariance: a 7 s and a 20 s instance are the same event

Owner: *"a 7 s and 20s instance of one event is the same and it should be
graded true."*

## What was actually wrong, found before building anything

The v6 read path was not failing at invariance first. It was ranking by length.

| measurement, v6 frozen, val | value |
|---|---|
| Spearman(DTW score, **candidate trace length**) | **+0.710**, positive on **65/65** queries |
| a **duration-only** ranker — no pixels at all | **0.315** |
| the v6 frozen **content** matcher | **0.314** |
| chance | 0.215 |

`vjeval5.dtw_from_cost` has free endpoints and divides by the QUERY length only,
so a longer candidate offers more sub-spans to find a cheap match in at no extra
cost. Under v4 every trace was 24 steps and the bias cancelled; stream time
exposed it. A no-pixel baseline tied the content channel.

Duration mismatch was *also* real — missed true positives sit further apart in
duration than retrieved ones, median ratio 1.33x vs 1.25x, 8.4% vs 3.2% beyond
2x, Mann-Whitney p = 2e-28 — but second in line.

## Why precision@support cannot be the target

Duration is a group CUE on rcasa: Close/sliding runs 8-16 steps, Open/hinged
32-80, Stack/hinged 112-160. Making the system duration-INVARIANT deletes a
shortcut the aggregate rewards, so the aggregate can fall while the system gets
more correct. `relmo/vjwarp` is the metric that isn't contaminated: replay a
recording's own footage at another speed and ask whether it still retrieves the
ORIGINAL out of 447. The answer is known without any label.

The first run of that test scored 1.000 everywhere and was **wrong** — the warp
factors (0.5, 2, 3) sat exactly on the rate ladder, so a 2x-slow clip encoded at
4 fps resamples the identical source frames as the original at 8 fps. Numerical
identity, not invariance. Every number below uses off-ladder factors
(1.25, 1.75, 2.5), which no rate pair can hit exactly.

## Three fixes, in the order they were found

**1. Anchored symmetric2 DTW** (`relmo/vjmatch`). Step weights 2/1/1 make every
complete path accumulate exactly Q+R, so dividing by Q+R is an exact
normalisation with no `pen` to tune, and anchoring both ends removes the
cherry-picked sub-span. Verified against brute force on 60 random cases. Free —
no re-ingest.

**2. Arc-length reparameterisation** (`relmo/vjmatch.arc_resample`). Re-index the
trace by cumulative layer-6 gate energy instead of by time: emit a descriptor
every fixed increment of *how much happened*. A fast and a slow execution then
emit the same number of steps, idle stretches compress, and order is preserved —
which matters, since `vjtime` had already established that surviving a warp by
discarding order solves nothing. `ds` was selected on TRAIN only (0.8/1.0/1.3/
1.7/2.2x of the train median gave 0.388/0.401/0.392/0.382/0.361 vs 0.349 for
none). Free — no re-ingest.

**3. Multi-rate encoding.** Rates 8 / 4 / 8-3 fps, windows 4 / 8 / 12 s, each
tiling. A recording only exists at a rate whose window fits inside it (447 /
389 / 205 for rcasa), which is not a gap: coarse rates exist exactly for the
long recordings a short query needs to match. Candidates are scored under the
best rate pair. Costs an ingest.

Multi-rate is not cosmetic. Arc-length alone cannot fix this because the
descriptor IS a rate: at 8 fps a 2.5x-slow event moves 40% as far per step, so
V-JEPA's one-step forecast is a different quantity, not the same one sampled
differently.

## Duration invariance — the primary metric

Warp factors 1.25 / 1.75 / 2.5x, query = the recording's own footage replayed,
pool = all 447, correct answer = itself. Chance MRR 0.0149. Factor 1.0 is the
harness control and is rank-1 1.000 in every arm.

| arm | rank-1 | top-5 | MRR | rank-1 at 1.25 / 1.75 / 2.5x |
|---|---|---|---|---|
| frozen, time-indexed | 0.083 | 0.229 | 0.151 | 0.250 / 0.000 / 0.000 |
| frozen, + arc | 0.292 | 0.500 | 0.384 | 0.750 / 0.125 / 0.000 |
| frozen, + multi-rate | 0.521 | 0.667 | 0.596 | 0.562 / 0.625 / 0.375 |
| frozen, + multi-rate + arc | 0.688 | 0.854 | 0.759 | 0.688 / 0.688 / 0.688 |
| **trained, + multi-rate + arc** | **0.750** | **0.938** | **0.838** | 0.812 / 0.750 / 0.688 |

**MRR 0.151 -> 0.838, and flat in the warp factor** — the degradation stops
growing with how much the duration changed, which is what invariance means.
Selecting the length-matched rate pair instead of the minimum was tried and
lost on both metrics (MRR 0.737, aggregate 0.371 vs 0.389).

## Aggregate precision@support — the contaminated metric, reported anyway

FROZEN:

| config | val | test | ood_val |
|---|---|---|---|
| v6 as shipped (legacy matcher, time) | 0.314 | — | 0.542 |
| + anchored symmetric2 | 0.354 | — | 0.523 |
| + arc-length | **0.414** | **0.419** | **0.554** |
| + multi-rate | 0.391 | 0.388 | 0.536 |
| v4 clip-time reference | 0.463 | — | 0.518 |

TRAINED, 3 seeds:

| config | val | test | ood_val |
|---|---|---|---|
| v6 time-indexed | 0.793 ± 0.092 | 0.763 ± 0.041 | 0.502 ± 0.067 |
| **v6 + arc, own scorer** | 0.758 ± 0.029 | **0.787 ± 0.019** | **0.651 ± 0.022** |
| v6 + arc + multi-rate DTW | 0.670 ± 0.066 | 0.662 ± 0.005 | 0.533 ± 0.061 |
| v4 clip-time reference | 0.924 ± 0.014 | 0.894 ± 0.018 | 0.730 ± 0.026 |

Training on arc-resampled traces moved test 0.763 -> 0.787 and ood_val
0.502 -> 0.651, and cut seed variance roughly in half on test and by two thirds
on ood. The gap to v4 clip-time narrowed from 0.13 / 0.23 to 0.11 / 0.08.

## Where the trade actually lands

Recall of true positives, split by how far apart the two durations are
(val+test queries, k = support):

| duration ratio | n pairs | arc only | + multi-rate |
|---|---|---|---|
| < 1.25x | 1180 | 0.704 | 0.571 |
| 1.25 - 1.75x | 1870 | 0.433 | 0.394 |
| **> 1.75x** | 1132 | **0.091** | **0.192** |
| all | 4182 | 0.417 | 0.389 |

Multi-rate **doubles** recall on the far-duration pairs — the case the owner
named — and pays for it on the near-duration pairs, where matching lengths were
doing work that content now has to do alone. 0.091 is a system that does not
answer the question at all; 0.192 is one that sometimes does. That is the trade,
and it is a monotone one: pushing invariance harder (length-matched rate
selection) took >1.75x to 0.231 and the aggregate down to 0.371.

## Cost

| stage | rate |
|---|---|
| 8 fps records (base) | 3.58x real-time |
| + 4 fps records | 9.5 min for 389 recordings |
| + 8/3 fps records | 5.3 min for 205 recordings |
| all three rates, one stream, fp16, incl. decode | **~1.8x real-time** |

Windows per second of video go 0.5 -> 0.917, so multi-rate is 1.83x the base
ingest. Still faster than real time on one stream.

## Open

1. **The two matchers disagree about which is better.** The trained model's own
   pooled-cosine scorer beats anchored DTW on z (test 0.787 vs 0.662), but only
   DTW currently supports multi-rate. A multi-rate-aware learned scorer is not
   built, and it is the obvious next gain.
2. **Near-duration recall dropped 0.133** under multi-rate. Some of that is the
   deleted shortcut and some is real loss from comparing at a coarser rate than
   necessary; those have not been separated.
3. **The rate ladder is hand-set** at 8 / 4 / 8-3. It covers ratios up to 3x.
   Beyond that there is nothing, and the warp curve is flat only inside the
   ladder's span.
4. **`ds` is one number for the whole corpus.** A per-recording arc increment
   would be adaptive and is untested.

## Reproduce

```bash
python -m relmo.vjrec6 --dataset rcasa --fp16 --stream-fps 4 --hop 4 --suffix _f4
python -m relmo.vjsig6 --dataset rcasa --suffix _f4
python -m relmo.vjrec6 --dataset rcasa --fp16 --stream-fps 2.667 --hop 6 --suffix _f3
python -m relmo.vjsig6 --dataset rcasa --suffix _f3
python -m relmo.vjwarp --ingest --episodes 16 --max-steps 40   # off-ladder warps
python -m relmo.vjwarp --arms time,arc,multi,multi_arc         # the primary metric
python -m relmo.vjwarp --arms multi_arc --tag rk7_arc_s2       # trained arm
python -m relmo.vjrank --no-scorer --tag rk7_arc_s0            # trains on arc traces
python -m relmo.vjrankeval --tags rk7_arc_s0,rk7_arc_s1,rk7_arc_s2
```

---

# v8 — duration is content: marginal invariance, and the GT that says so

Owner, 2026-08-15: *"dont make the warps too strong. I want 7s and 10s episode
to be linked but not ranked 1. time invariance is a moment. for example a
speeding car is not the same as a driving car. make the moment invariances
marginal. a 7s and 10s can be the same but a 7s and 20 are different."*

v7 optimised for total duration invariance. That was wrong. Under this
definition a duration ratio is not a nuisance — past a point it is a DIFFERENT
EVENT, and a system that retrieves a 2.5x-slower replay at rank 1 has failed.

## The GT

The cut is not chosen by taste. It is the geometric mean of the owner's own two
examples, 10/7 = 1.429 (same) and 20/7 = 2.857 (different):

    R_CUT = sqrt(1.429 * 2.857) = 2.02  ->  2.0

    weight(ratio) = max(0.25, 1 - log(ratio)/log(2.0))   for ratio < 2.0
                  = 0                                     for ratio >= 2.0

    1.00x -> 1.000   identical duration
    1.43x -> 0.484   7 v 10 s: linked, and ranked BELOW a duration match
    1.80x -> 0.250   still the same moment, at the floor
    2.86x -> 0.000   7 v 20 s: a different moment

It multiplies the WHOLE graded relevance, not just the event floor — past the
cut, nothing about a shared object or kitchen brings it back. The binary
positive class used by precision@support is gated the same way: same event AND
same moment. Duration comes from the manifest, never from the trace, so the GT
cannot move when the encoder does.

On rcasa this is a light touch: 4056 of 4232 positives survive (95.8%), 176 are
demoted. The corpus rarely stretches an event past 2x — which is why this had to
be specified rather than discovered.

## The warp metric becomes two-sided

Same warped clips, reinterpreted — no re-ingest. 1.25x and 1.75x are INSIDE the
band and must still retrieve the source; 2.5x is BEYOND it and must NOT.

    MARGIN = in-band rank-1  -  beyond-band rank-1

| arm | in-band rank-1 | beyond rank-1 | **margin** |
|---|---|---|---|
| frozen, time-indexed | 0.125 | 0.000 | +0.125 |
| frozen, **+ arc-length** | **0.438** | **0.000** | **+0.438** |
| frozen, + multi-rate + arc | 0.688 | 0.688 | **+0.000** |

**Multi-rate scores zero.** It retrieves a 2.5x replay exactly as readily as a
1.25x one — perfectly invariant, and therefore unable to tell a moment from a
different moment. v7 shipped it as the headline; under the corrected definition
it is the worst arm, and the owner's instinct that the warps were "too strong"
is the whole finding. **Arc-length alone is the right amount of invariance**: it
recovers in-band retrieval 0.125 -> 0.438 while still rejecting every
beyond-band replay.

## Frozen encoder, time-indexed, under the new GT

v6 records, no arc, no multi-rate. Anchored symmetric2 matcher.

| split | frozen content | duration-only control | chance |
|---|---|---|---|
| val | **0.354** [0.31, 0.39] | 0.338 | 0.212 |
| test PRIMARY | **0.388** [0.36, 0.42] | **0.396** | 0.208 |
| test DEPLOYMENT | **0.403** [0.38, 0.43] | 0.395 | 0.208 |
| ood_val | **0.581** [0.48, 0.68] | 0.500 | 0.177 |

With the pre-v7 legacy matcher, val is 0.297.

**On test, the frozen time-indexed encoder does not beat a ranker that looks at
no pixels at all** — 0.388 against 0.396 for duration-only. Gating positives on
duration necessarily strengthens that control, so it has to be quoted next to
every number or the content figure cannot be read. Only `ood_val` shows a clear
content margin (0.581 vs 0.500), and that is the split where durations are long
and varied enough that length alone stops being informative.

Per-group, val: PickPlace 0.465, Open/hinged 0.327, Close/sliding 0.275,
Open/sliding 0.235, Close/hinged 0.195 (chance 0.095-0.270).

## Open

1. **Frozen + time-indexed is at the duration-only baseline on test.** That is
   the honest state of the frozen encoder under this GT.
2. **Every v7 number needs re-reading**, since v7 selected arms on total
   invariance. arc-length survives the correction; multi-rate does not.
3. **The trained arms have not been re-scored under the new GT or the two-sided
   warp metric.**

---

# The two metrics, and why these two

Owner, 2026-08-15: *"if there are 10 supports and 10 non-supports in a corpus of
20, then I want something that ranks all the 20 in the best similarity ranking
and gives the 10 alone returned. So prec@support and one more metric for the
overall ranking."*

## The pair

    prec@support   rank everything, return exactly `support` items, count how
                   many are the same event. Binary; ignores order and grade.
                   "Of what I hand back, how much belongs."

    NDCG@support   the SAME cut. sum(rel / log2(rank+1)) over those `support`
                   items, divided by the best achievable for that query.
                   Graded and position-discounted, normalised per query.
                   "Are the best ones at the top, and did near-relevant beat
                   far-relevant."

They are cut at the same k, so they describe the same returned list from two
angles. Nothing else is needed.

## Why not the others, measured on the whole corpus

Random-ranking floors, 474 recordings, same GT:

| metric | random floor | best arm | headroom | spread across the 3 arms | % of headroom used |
|---|---|---|---|---|---|
| prec@support | 0.179 | 0.519 | 0.340 | 0.153 | **45%** |
| **NDCG@support** | **0.150** | 0.606 | 0.456 | 0.089 | **20%** |
| NDCG full-list | 0.528 | 0.807 | 0.279 | 0.043 | 15% |
| concord | 0.503 | 0.751 | 0.248 | 0.029 | 12% |

(macro-averaged for this table so the four are compared on one basis)

**Full-list NDCG is the trap.** Its random floor is 0.528, because in a
474-long list a random ranking still accumulates most of the achievable
discounted gain. Reporting "0.781 vs 0.807" makes a real 0.026 difference read
as rounding; against the floor it is 0.253 vs 0.279, a 10% relative gap. Cutting
at support drops the floor to 0.150 and the same comparison becomes 0.517 vs
0.606.

**concord is position-blind.** An inversion between ranks 1 and 2 costs exactly
what one between ranks 400 and 401 costs. Its floor is 0.503 and it separates
the arms least of the four.

**rel@10** has a fixed k that does not track support, which is the wrong shape
for "rank till the support".

**dur-rho** is a diagnostic, not a quality metric. It holds the event constant
and asks only whether duration-matched instances outrank stretched ones. Keep
it for answering "is the passive decline honoured", never as a headline.

## The headline table, on these two metrics

Whole corpus, 474 recordings, frozen, anchored symmetric2 matcher.
prec@support is micro-averaged, matching every earlier number in this file.

| arm | prec@support | NDCG@support |
|---|---|---|
| random ranking | 0.217 | 0.150 |
| **v4 clip-time** | **0.533** | **0.606** |
| v6 stream-time | 0.375 | 0.517 |
| v6 stream-time + arc | 0.424 | 0.554 |

`relmo/vjzeval.evaluate` now returns `ndcg_sup` and `report()` prints it, so
every future number carries both.

---

# wAUC, and why clip-time "beats" stream-time

Owner: *"NDCG just focusses on getting top ranks correct... but the overall till
support is messed up. and after support ranking too. and why is clip time vs
stream time fail. try a v7 with both cliptime and stream time."*

## The metric NDCG was hiding

NDCG's `1/log2(rank+1)` discount means rank 1 is worth 6.6x rank 100. It reads
the head of the list and almost nothing else. **wAUC** is the same graded
relevance with a UNIFORM position weight:

    wAUC = ( sum_c rel(c) * pct(c)  -  worst ) / ( best - worst )

where pct(c) = 1 - rank(c)/(N-1). 1.0 = every item above every less-relevant
one; **0.502 = random**. It reads the whole ranking - through the support and
past it - which is exactly what NDCG does not.

Whole corpus, 474 recordings, frozen, anchored matcher:

| arm | prec@sup | NDCG@sup | **wAUC** | order-in-k | miss depth |
|---|---|---|---|---|---|
| random floor | 0.217 | 0.150 | **0.502** | 0.009 | 0.587 |
| v4 clip-time | **0.533** | **0.606** | 0.790 | **0.414** | 0.492 |
| v6 stream-time | 0.375 | 0.517 | **0.791** | 0.284 | **0.457** |
| v6 stream-time + arc | 0.424 | 0.554 | **0.796** | 0.316 | 0.461 |

`order-in-k` = Spearman(rank, relevance) INSIDE the returned k.
`miss depth` = mean rank percentile of positives that missed the cut; lower is
better.

**On the whole ranking the two are indistinguishable** (0.790 vs 0.791/0.796),
and stream-time buries its misses less deeply. Clip-time's entire advantage is
in the head of the list. Every earlier conclusion in this file that
stream-time "regressed" was reading a head-of-list metric.

## Why clip-time wins the aggregate: the corpus duration histogram

Queries split by their own duration, whole corpus as the pool:

| query duration | n | clip prec | strm prec | clip wAUC | strm wAUC |
|---|---|---|---|---|---|
| < 8 s | 58 | **0.503** | 0.375 | 0.693 | **0.733** |
| 8-12 s | 187 | **0.570** | 0.410 | **0.830** | 0.775 |
| 12-20 s | 144 | **0.525** | 0.442 | 0.783 | **0.801** |
| > 20 s | 85 | 0.447 | **0.457** | 0.781 | **0.878** |

Stream-time improves monotonically with length - wAUC 0.733 / 0.775 / 0.801 /
**0.878**. Clip-time peaks at 8-12 s and decays. 187 of 474 recordings sit in
8-12 s, which is clip-time's sweet spot, so **the aggregate favours clip-time
because of the corpus's duration distribution, not because the formulation is
better.** Past 20 s stream-time wins on both metrics.

The cause is trace LENGTH, not sampling rate. Re-scoring the same 229 long
recordings at 8 / 4 / 2.7 fps (steps of 0.25 / 0.50 / 0.75 s) gave prec 0.527 /
0.520 / 0.491 and wAUC 0.826 / 0.814 / 0.802 - coarsening the rate makes it
WORSE, refuting the per-step-SNR hypothesis. A 5 s recording yields 8 descriptor
steps under stream time where clip-time always emits 24.

## v7 — fuse them

Per query, z-score each arm's distances over its own candidate set, then
combine. Whole corpus, identical pools:

| arm | prec@sup | NDCG@sup | wAUC | NDCG full | order-in-k | miss depth |
|---|---|---|---|---|---|---|
| clip-time only | 0.533 | 0.606 | 0.790 | 0.807 | 0.414 | 0.492 |
| stream+arc only | 0.424 | 0.554 | 0.796 | 0.781 | 0.316 | 0.461 |
| z-fuse, clip 0.25 | 0.489 | 0.634 | 0.819 | 0.829 | 0.419 | 0.466 |
| **z-fuse, clip 0.50** | **0.533** | **0.659** | **0.825** | **0.840** | **0.460** | 0.475 |
| z-fuse, clip 0.75 | **0.547** | 0.645 | 0.813 | 0.829 | 0.441 | 0.487 |
| RRF (rank fusion) | 0.514 | 0.644 | 0.819 | 0.835 | 0.439 | 0.471 |

**Every fusion weight in [0.25, 0.75] beats BOTH single arms on wAUC, NDCG@sup
and NDCG-full.** At 0.50 the fusion matches clip-time's precision exactly while
adding +0.053 NDCG@support and +0.035 wAUC; at 0.75 it also beats clip-time's
precision (0.547 vs 0.533). Parameter-free RRF lands in the same place, so the
gain is not an artefact of the weighting.

That is the expected result given the duration table: the two representations
fail on opposite ends of the duration axis, so they are complementary rather
than redundant. Cost is one extra encode per recording.

Not tested: a duration-AWARE fusion weight. It would likely help, and it would
also be a per-corpus tuned parameter, so it needs a held-out corpus before it
can be believed.

## Standing metrics, final

    prec@support   membership of the returned set        floor 0.217
    NDCG@support   order at the HEAD of that set         floor 0.150
    wAUC           the whole ranking, position-uniform   floor 0.502

`vjzeval.evaluate` returns all three and `report()` prints them.

---

# Why clip-time beats stream-time: four hypotheses, one survivor

All measured frozen, whole corpus, anchored symmetric2 matcher, updated GT.

| hypothesis | test | verdict |
|---|---|---|
| per-step SNR / sampling rate | re-score the same 229 recordings at 0.25 / 0.50 / 0.75 s steps | **refuted** - prec 0.527 / 0.520 / 0.491, coarsening HURTS |
| extent normalisation (event phase) | resample stream traces to a fixed 12 / 24 / 48 steps | **refuted** - prec 0.375 -> 0.318 / 0.338 / 0.368, all worse |
| differencing baseline (derivative vs displacement) | integrate a_t over k steps to recover v4's anchored form | **refuted** - 0.375 -> 0.395 peak at k=6, collapse to 0.323 at k=8; wAUC falls monotonically 0.791 -> 0.632 |
| encoder receptive field | v6 at 64 frames / 8 s window = v4's structure exactly | **largely refuted** - prec +0.035, NDCG@sup -0.019, wAUC -0.013 |

Receptive-field arm, on the 416 recordings every arm covers:

| arm | steps | prec@sup | NDCG@sup | wAUC |
|---|---|---|---|---|
| v4 clip-time (whole episode) | 24 | **0.546** | **0.620** | **0.800** |
| v6 32f / 4 s window | 40 | 0.386 | 0.523 | 0.785 |
| **v6 32f / 4 s + arc** | 44 | **0.440** | **0.561** | **0.797** |
| v6 64f / 8 s window | 24 | 0.421 | 0.504 | 0.772 |
| v6 64f / 8 s + arc | 31 | 0.433 | 0.531 | 0.790 |

## The survivor: the PREDICTION HORIZON must scale with the event

v4 spreads 64 frames over the whole episode, so its tubelet - and therefore its
prediction horizon - is a FRACTION of the event, not a fixed number of seconds:

    11 s episode -> 5.8 fps -> tubelet 0.34 s -> context 2.8 s, horizon 0.3-1.4 s
    30 s episode -> 2.1 fps -> tubelet 0.94 s -> context 7.5 s, horizon 0.9-3.8 s

v6 asks "what happens in the next 0.25 s" of every event regardless of its
timescale. v4 asks "what happens next, at this event's own tempo". That is a
different and better-posed question, and it is the one thing none of the four
tests above changes: resampling the TRACE cannot alter a horizon already baked
into the descriptor, and a longer fixed window keeps the horizon fixed.

The supporting evidence is already in the table. Arc-length reparameterisation -
re-indexing by cumulative change instead of by time - is the post-hoc, trace-
level approximation of exactly this, and it is the single largest v6 gain
anywhere in this file: +0.054 prec, +0.038 NDCG@support, +0.012 wAUC. It closes
roughly a third of the gap using only the descriptors that already exist.

## What this means for continuous video

Clip-time is undeployable on a stream: it needs the episode extent. But the
survivor hypothesis says the extent is not what matters - the TEMPO is, and
tempo is estimable from content without any boundary:

    adapt the encoder's frame stride per window so each window spans a roughly
    constant amount of CHANGE rather than a constant amount of TIME

That is arc-length applied at the encoder INPUT rather than to the output trace.
It is boundary-free, label-free, and streaming-compatible. Untested.

The alternatives, if that fails: multi-span hypothesis indexing (emit traces
over several candidate extents at each position, index all) or label-free
change-point segmentation. Both re-introduce an extent decision that the tempo
route avoids.

---

# v8b — raising the FROZEN stream-time number, no training, no re-encoding

Owner: *"dont think of what v4 brings with clip time. thats a broken method
which worked somehow. Just think what can be done to bring the frozen vjepa-2
stream time result higher."*

Every arm below is built from channels already on disk. Whole corpus, 474
recordings, anchored symmetric2 matcher, arc-length reparameterised, PCA-256 per
channel fitted on TRAIN only.

| arm | dim | prec@sup | NDCG@sup | wAUC |
|---|---|---|---|---|
| `a` pred_change — v6 as shipped | 256 | 0.420 | 0.556 | 0.803 |
| `b` obs_change alone | 256 | 0.442 | 0.572 | 0.796 |
| `[a;sig]` | 512 | 0.480 | 0.570 | 0.823 |
| `[b;sig]` | 512 | 0.487 | 0.563 | 0.812 |
| `[a;sig]` centred | 512 | 0.461 | 0.612 | 0.812 |
| **`[a;b;sig]`** | 768 | **0.495** | **0.614** | **0.833** |
| `[a;b;sig]` centred | 768 | 0.466 | **0.626** | 0.820 |

**+0.075 prec@support, +0.058 NDCG@support, +0.030 wAUC over the shipped
baseline, for zero extra compute at write time.**

## Two findings worth keeping

**obs_change beats pred_change as a descriptor** (0.442 vs 0.420 prec, 0.572 vs
0.556 NDCG@sup). This is the owner's own argument about the barred error
channel, applied one step further: `error` was rejected because it is "a
function of (event, MODEL PRIOR), not of the event". But `pred_change` =
pred(t+1) - act(t) is ALSO partly a function of the prior - it is what the model
guessed. `obs_change` = act(t+1) - act(t) is the only channel that is purely a
function of what happened. It is not the error channel and never can be: no
prediction enters it, so no residual can be formed from it.

**SigLIP is weak alone and strong fused.** On its own it is 0.410 prec / 0.390
NDCG@sup - the worst arm measured. Concatenated with `a` it adds +0.060 prec.
Appearance and predicted-dynamics fail on different queries, which is the same
complementarity STRAP relies on; the mistake was reading its solo score and
setting it aside.

**Centring** (subtracting each recording's own mean descriptor) trades head for
body: NDCG@support 0.614 -> 0.626 while prec 0.495 -> 0.466 and wAUC
0.833 -> 0.820. Use it if the head of the list is what matters.

## THE ERROR BAR STILL APPLIES TO TRAINING

Concatenating `a` and `b` is safe for the FROZEN matcher: each part is
L2-normalised before concatenation, so the cosine of the whole is the mean of
the two part-cosines and `a - b` is never formed. A TRAINED head over `[a;b]`
would form it in its first linear layer, which is exactly what
relmo/vjrank.py's structural bar exists to prevent. If this descriptor ever
feeds training, `b` must go back to entering as two scalars.

## Evaluation cost, and why it was 45 minutes

Not fp16 and not the encoder - fp16 encoding runs at 3.58x real-time. The
evaluation was slow, and profiling (rather than guessing) found the cost matrix
einsum at **93-98%** of it, scaling linearly in descriptor dimension:

| Q x R | dim | einsum | DP | einsum share |
|---|---|---|---|---|
| 44 x 44 | 1024 | 107 ms | 8 ms | 93% |
| 44 x 44 | 2816 | 417 ms | 8 ms | 98% |
| 216 x 216 | 1024 | 2541 ms | 176 ms | 94% |
| 216 x 216 | 256 | 398 ms | 176 ms | 69% |

A Sakoe-Chiba band on the DP was tried first and bought only 1.5-2.3x while
damaging the ranking below band=0.25 - because the DP was never the bottleneck.
PCA to 256 dims per channel is the fix: 3.9x end to end, 99.1 / 93.8 / 97.4 %
variance kept for a / b / sig, and it reproduces the full-dimension baseline to
within 0.004 prec. It also shrinks whatever index this eventually feeds.

Also fixed: Accelerate's BLAS raises spurious divide-by-zero / overflow FP flags
on Apple Silicon. Verified against a float64 einsum reference (max abs diff
1e-5, relative 8e-7) and silenced with a SCOPED errstate, so a genuine
non-finite value elsewhere still surfaces.
