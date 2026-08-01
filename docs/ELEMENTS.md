# Elements and channels: what each is for, and what joins to what

User-set spec, 2026-08-01. This supersedes every earlier description.
Measurements in this document are from `lake/fresh_bench` (2,097 episodes,
3.91 h of video) on the same date.

---

## The five elements

| element | is | must link to |
|---|---|---|
| **scene** | the setting. **Standalone** — but where the scene CHANGES (self-driving), a **time series of scene→scene linkage**, not one static vector | — |
| **agent** | the self-moving thing | `object_id`; its **pose** (only the agent is broken into poses); its **trajectory** in `mot` |
| **participants** | **ALL objects in the scene**, not only what the agent contacted. Relations *between* participants need not be stored | `object_id`; a **trajectory each** in `mot` |
| **events** | typed, timestamped transitions | agent + participants, **joined with `mot` per participant** |
| **answer** | the join of scene + agent + participant + event, telling a **collective story** | all four |

## The seven channels

| channel | is for | granularity today | measured |
|---|---|---|---|
| **mot** | **the trajectory of motion** — 3D where affordable, else 2.5D. **Per participant AND agent.** NOT a direction of state change | per **event** | only element-level channel; 0.959 AUC on open/close but wrong by construction |
| **vjepa** | **physics and dynamics** of the agent *and* every participant. Per participant, **never whole-frame**. Seeded from the ELEMENTS, not re-detected. Only participants that **moved significantly**, plus the agent | episode | flat 0.56–0.59; predictor not stored, so its actual task has never been evaluated |
| **pe, sig2, iv2, xclip** | sit **on top of `answer`** — consume the structured element output **and** join their clip embeddings, augmenting each other | episode (pe/sig2: 8 frames) | see redundancy below |
| **act** | purpose not established. No naming of actions, no summarising trajectories | episode | candidate for removal |
| **frame_vectors** | per-frame appearance; the substrate `mot` is a delta of | frame | healthy, cheap |

### The granularity mismatch — the core problem

Elements are **sub-episode entities**: a track, an interval, a span.
Channels are **one vector per episode**. So no channel describes any element,
and *"object1 (and similar) undergoing action1 (and similar)"* cannot execute —
nothing is a vector of object1, nothing is a vector of action1.

`motion_vectors` has 15,175 rows = exactly the event count. **It is the only
channel stored per element, and it is the best channel on direction.** That is
not a coincidence; it is the argument for moving every channel to element
granularity.

### Redundancy, measured (pairwise rank correlation of episode orderings)

```
        pe   sig2   iv2  xclip  vjepa   act  motion
pe    1.00   0.89  0.54  0.70   0.65  -0.18   0.02
sig2  0.89   1.00  0.57  0.71   0.67  -0.13   0.01
iv2   0.54   0.57  1.00  0.54   0.35  -0.05   0.01
```

- **pe ~ sig2 = 0.89** — near-duplicates. Drop one (sig2 scores marginally
  better; pe costs 36 min/build vs sig2's 23).
- **xclip ~ pe/sig2 = 0.70** — mostly redundant, and its cross-frame attention
  is discarded at write anyway.
- **iv2** is the most independent (0.54–0.57) and best on appearance — keep.
- **motion ~ 0.00 with everything** — measures what nothing else does.
- **act** anti-correlates with the appearance group.

---

## What is built, and what is missing

```
BUILT
  scene         frame_vectors                              70,436
  agent         events role=agent                            2,097
  events        events kind=t0..t14/contact/release          15,175
  participants  presence.object_id                           73,521 intervals
  object index  object_vectors (one row per interval)        NEW

MISSING — the joins the architecture needs
  agent        -> object_id                     events name no object
  events       -> agent/participant object_id   events name no object
  events       -> scene position                a box is computed, not stored
  participants -> relation to scene object      the "in/on what" of the answer
  mot          -> per participant trajectory    currently one delta per event
  agent        -> pose                          no pose model
  vjepa        -> per participant tubelets      currently whole-frame, episode
  act          -> a purpose                     or removal
```

`answer` needs all of them at once. **It is a join, not a table.**

---

## Ordering: identity is core

Everything joins on `object_id`, so joins built on unreliable ids inherit the
unreliability. Current state: **73,521 intervals resolve to 32,756 objects,
78% seen exactly once**, at a match cut of **0.739**.

The cut is the 99.5th percentile of *proven-different* pairs (co-existing
tracks — 200k+ of them), clipped to [0.50, 0.95]. It is fitted, not
hand-picked, exactly as the no-hardwire rule requires. Two readings remain and
they were not separable until now:

- **cut too strict** — genuine same-object pairs fall below 0.739
- **descriptor too weak** — same-object pairs are *also* ~0.74, so no threshold
  separates them and the percentile is doing its job on a signal with none

`object_vectors` makes this decidable: intervals sharing an `object_id` are the
**positive** distribution the free negatives never had a counterpart for. Sweep
`ELIDEDB_OBJ_CAL_Q`, measure singleton rate and cross-episode reuse, and if no
cut produces a plausible object count, the descriptor is at fault and must be
replaced.

## Two write-path bugs worth remembering

**Fixed-size batches defeat the store's own random-access design.** Every media
segment carries exactly one IDR at its first frame — that is what makes a 2 s
read a byte range. A 64-frame batch always straddles an episode boundary and
silently returns only what it could reach from the last keyframe: measured 29.3%
of the corpus. Batch by segment.

**Descriptors were computed and discarded.** The same shape as `events.role`
hardcoded empty, the ITM head never called, and the V-JEPA predictor not
stored — the pipeline repeatedly computes something useful and throws it away
before it reaches disk. Check for that pattern before adding a model.
