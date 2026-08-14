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

## Numbers — trained recurrence (2026-08-14)

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
