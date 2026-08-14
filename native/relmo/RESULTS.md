# ElideDB / RelMo — V-JEPA 2 span retrieval

State as of 2026-08-14. Branch `relmo-vjepa-span`. Frozen model, no gradient
steps taken anywhere.

## What the system is

Given a query clip, return **located segments** — `(recording, t0, t1)` — from
anywhere in a corpus, where the segment boundaries come from the match and not
from any fixed chunking of the input.

```
video ──► V-JEPA 2 encoder (frozen)
            │
            ├─ layer 6   ‖h(t+k) − h(t)‖         WHERE  the gate: what changed
            └─ layer 24  predictor
                          pred(t+k) − actual(t+k) WHAT   the content: what the
                                                         model could not predict
                     ↓ pool WHAT over the frame, weighted by WHERE
              per-timestep descriptor sequence
                     ↓
              subsequence DTW, free start and end
                     ↓
              (recording, t0, t1)
```

Nothing is labelled at write time. The corpus family names are parsed in
`vjeval.py` only, for scoring.

## Numbers

447 episodes, k = support, group-aware grading, chance 0.222.
95% intervals bootstrapped over **queries** (2000 resamples); paired
differences use shared resample indices.

| arm | true/returned | precision | 95% CI | lift |
|---|---|---|---|---|
| **error / span** | 21612/36990 | **0.584** | [0.566, 0.602] | **2.63×** |
| error / cosine | 21288/36990 | 0.576 | [0.557, 0.593] | 2.59× |
| pred_change / span | 19423/36990 | 0.525 | [0.508, 0.540] | 2.37× |
| pred_change / cosine | 19093/36990 | 0.516 | [0.499, 0.532] | 2.33× |
| obs_change / span | 18128/36990 | 0.490 | [0.471, 0.508] | 2.21× |
| obs_change / cosine | 16634/36990 | 0.450 | [0.430, 0.468] | 2.03× |
| scene-only baseline | 11162/36990 | 0.302 | [0.294, 0.309] | 1.36× |

Time-warp invariance — median rank of the true original among 447, where the
query is the same episode re-rendered under a warp (correct answer known
without annotation):

| channel | matcher | identity | ease_in | ease_out | sigmoid | zoom |
|---|---|---|---|---|---|---|
| error | cosine | 1.0 | 15.5 | 85.5 | 1.0 | 1.0 |
| **error** | **span** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| pred_change | span | 1.0 | 1.0 | 4.5 | 1.0 | 1.0 |
| obs_change | cosine | 1.0 | 306.0 | 212.5 | 3.0 | 7.5 |

## Things that were refuted, and by what

Each of these was believed, then measured, then abandoned.

| claim | outcome |
|---|---|
| error-based representations shed the scene confound | **false** — static patches 83.3 vs moving 88.3, ratio 1.07 at layer 24 |
| …though at layer 3 the ratio is 6.64, so the refutation was depth-specific | retracted and re-recorded |
| ordering vs warp-robustness is a fundamental tradeoff | **false** — an artefact of the context-length ramp; error/span is both |
| a time warp changes the residual's content irreducibly | **false** — rank 32.5 → 1.0 once context length is fixed |
| half of every clip is wasted | **false** — the discarded half is the setup, 0.461 alone |
| a short prediction horizon is sharper | **false** — long horizon 0.586 vs short 0.574 |
| subtracting appearance removes nuisance | **false** — 0.472 → 0.335 |
| fusing channels helps | **false** — every combination ≤ best alone |
| the sweep geometry carries the event | **false** — most object-biased feature measured, lift 1.55 on the wrong axis |
| drawers are a failure case | **false** — a grading error; they are the best family once grouped correctly |

## Open, in the order they bite

1. **Index cost — 6.5× real-time.** An hour of recording takes ~6.5 h to index.
   This is what blocks the real use case. Engineering, not research.
2. **The content channel depends on the model being wrong.** `error` is a
   function of *(event, model prior)*, not of the event. Structurally worst on
   the open/close pair — `CloseCabinet` 0.381 is the weakest real family — and
   it would drift under online adaptation. Retained only because it beat
   `pred_change` by −0.050 [−0.059, −0.042], and only while the model is frozen.
3. **Boundary accuracy is unvalidated.** Span containment is 8/8, but that only
   proves the right episode. The clean label-free test is to query with a
   sub-span of a known episode, where true boundaries are exact.
4. **Layer 6 was chosen by peeking at the graded score.** Layers 3/6/9 came out
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
