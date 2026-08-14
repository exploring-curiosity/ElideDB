# Learned experience recurrence `z_t`, trained on relational physics

Date: 2026-08-14
Status: approved, in implementation
Supersedes the fixed-`f` recurrence in `2026-08-14-experience-recurrence-design.md`
(that one was tested and refuted: no arm beat the `pred_change` baseline).

## Why

The frozen system tops out at **0.525** overall precision at k=support, and only
one of twelve task families (`OpenDrawer`, 0.712) reaches the owner's 0.70 bar.
Three measurements say the frozen path cannot be pushed further:

| measurement | value | what it forecloses |
|---|---|---|
| union of V-JEPA and SigLIP hits | **0.652** | no score-combination of the two channels reaches 0.70 |
| oracle true-object gate vs shipped gate | 0.566 vs 0.545 | a perfect gate is worth +0.021 |
| linear probe reads event group off the same descriptor | **92.4%** | the information is present; cosine + DTW extracts about half of it |

The last row is the whole argument. A descriptor a linear map can classify at
92.4% would retrieve at roughly 0.85 if the metric matched the structure. It
does not, because cosine weights all 1024 directions equally and the
discriminative subspace is low-variance — which is also why whitening (0.380)
and PCA (0.484) both *lost*: unsupervised transforms select by variance, and
variance is the wrong criterion here.

So the lever is the metric, and closing that gap needs a trained function.

## What is trained

The owner's recurrence:

```
z_t = f(z_{t−1}, act(t), pred(t+1), act(t+1))
```

### The barred term, and how the bar is enforced

`pred(t+1) − act(t+1)` is the **error** channel, barred absolutely. It lies in
the linear span of the arguments above, so any MLP over their concatenation can
form it in its first layer. Policy cannot enforce the bar; architecture must.

Expectation enters as a **vector**, reality enters only as **scalars**:

```
a_t = pred(t+1) − act(t)                        expectation      (1024-d)
b_t = act(t+1) − act(t)                         realised change  (never passed to f)
g_t = [ <â_t, b̂_t> , log(‖b_t‖ / (‖a_t‖+ε)) ]   reality          (2 scalars)
z_t = f(z_{t−1}, act(t), a_t, g_t)
```

Two scalars cannot reconstruct a 1024-d residual, so the bar holds
structurally. This also matches the audit: gating is worth +0.12 over no gate
while a *perfect* gate is worth only +0.02 more — reality's job is to weight,
not to carry.

`act(t)` is the appearance/context term and the main out-of-domain overfitting
risk. It is an **arm**, not an assumption:

- **arm A1** `z_t = f(z_{t−1}, a_t, g_t)` — no appearance. Uses only records
  already cached in `vjrec4`; zero rebuild.
- **arm A2** `z_t = f(z_{t−1}, act(t), a_t, g_t)` — the owner's formula
  literally. Requires caching `act(t)`, which `vjrec4` does not store (it keeps
  only deltas relative to it).

Which one ships is decided by `ood_val`, not by argument.

## Supervision: relational physics, never labels

Training targets are read from `state.npz` and are used **at training time
only** — never at serve, never in an index, exactly the posture already used
for the oracle gate. The task-family labels used for grading never touch
training.

Per trace step, for the manipulated body:

| target | source | why it is universal |
|---|---|---|
| `d_art` signed articulation rate | joint of a target body via `jnt_bodyid`/`jnt_qposadr` | "hinge opening" is the same fact in a real kitchen |
| `art_pos` normalised joint position | `jnt_range` | how far through the motion the clip is |
| `contact_grip` | `contact_pairs` ∩ gripper bodies | grasp/release is embodiment-independent |
| `d_rel` displacement of target **in the gripper frame** | `xpos`, `xquat` | relational, not world coordinates |
| `speed` ‖target velocity‖ | `xvel` | scale of the event |

The claim these buy transfer: the targets are **physical, not lexical**.
`OpenFridge` and `OpenCabinet` share "revolute joint angle increasing while
contact holds", so a model fit to that has no reason to require having seen a
fridge. `ood_val` and `ood_test` exist to falsify that claim.

### Feasibility gate, run before any training

Probe the frozen cached descriptors → these targets, held out **by episode**.
A prior audit got R² ≈ 0.000 for content → *world-metre* displacement; this asks
an easier and camera-invariant question (sign-level, relational). If the gate
fails, approach C is dead and the honest conclusion is that 0.70 is only
reachable as a label-fitted number.

### Reference ceiling (arm R)

Contrastive training on `group_key` — the eval labels. Never shipped. Run only
to report what label supervision reaches in-domain, so the physics number has a
denominator.

## Splits

**The grouping unit is the scene** (`layout_id`_`style_id`, 188 of them), not
the episode. Two reasons the existing id-hash split in `relmo/splits.py` is
wrong here:

1. episodes ship as camera variants of one rollout
   (`..._agentview_left` / `_right`) — near-duplicates that an id hash puts on
   opposite sides;
2. scenes recur across episodes, so a shared kitchen lets appearance
   memorisation inflate the held-out number.

Scene assignment is stratified by task-family composition so the small families
survive. **train 65 / val 15 / test 20**, under the owner's "train < 70" bar.

| split | n (scenes → eps, approx) | use |
|---|---|---|
| `train` | ~122 → ~290 | gradient updates |
| `val` | ~28 → ~68 | early stopping only |
| `test` | ~38 → ~89 | touched once |
| `ood_val` | `rcasa_eval`, 27 eps, `ArrangeTea` + `OpenFridge` | **unseen task**; arm selection |
| `ood_test` | `bridge`, real robot video, ~400 sampled eps ≥64 frames | **unseen domain**; touched once |

Model selection uses `val` for early stopping and `ood_val` for choosing the
arm. `test` and `ood_test` are each read once, at the end.

`arctic_v1` was considered and dropped: the manifest is present but the RGB was
never downloaded (only MoCap and backgrounds are on disk).

## Evaluation protocol

Unchanged from the frozen system so the numbers are comparable: k = support,
group-aware grading via `group_key`, chance = support/pool, bootstrap CIs over
queries (2000 resamples), paired differences sharing resample indices.

Two pools are reported and both are always quoted:

- **primary** — query ∈ `test`, pool ∈ `val ∪ test`. Fully train-disjoint.
- **deployment** — query ∈ `test`, pool ∈ all. The index holds history; the
  query is new.

**Stated limitation.** At a 20% test slice, `LoadDishwasher` (11 eps),
`StackBowlsCabinet` (14) and `PrepareCoffee` (16) land at support 2–6. The eval
already skips support < 5, and at support 4 one item moves precision by 0.25.
"0.70 on every class" is not measurable to better than about ±0.25 for those
three, on any split of this corpus. That is a property of the corpus, not of
the method, and it is reported rather than hidden.

## Order of work

1. `vjsplit.py` — scene-grouped stratified split, frozen to disk.
2. `vjphys.py` — extract relational targets for `rcasa` + `rcasa_eval`.
3. **feasibility gate** — probe frozen descriptors → targets.
4. `vjz.py` — the recurrence, arm A1 (no rebuild needed).
5. eval on `val`; arm A2 once `act(t)` finishes caching in the background.
6. arm R ceiling; `ood_val`.
7. `bridge` records; `ood_test` read once.

Steps 1–4 need no new forward passes; every input is already cached.
