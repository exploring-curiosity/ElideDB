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
unreliability. Settled 2026-08-01, once `object_vectors` persisted the
descriptors and made all of this measurable.

### The negatives were contaminated

The cut is the 99.5th percentile of *proven-different* pairs. `free_negatives`
proves difference with two tests — the tracks **co-exist** and their boxes are
**disjoint** — and says the second is load-bearing, because a detector that
puts two boxes on one object makes two tracks that co-exist and look identical.
Both writers tested co-existence only. Those double detections are **0.5% of
co-existing pairs** and `q=99.5` reads the top 0.5%, so the cut was very nearly
a readout of the contamination.

```
cut on contaminated negatives   0.748
cut on clean negatives          0.692
```

### The mirror argument is free supervision

Two tracks in *disjoint* regions of one frame are two objects. Two tracks in
the *same* region are one object. Same geometry, no annotation — so the double
detections are not waste, they are the **proven-same** set the calibration
never had (`free_positives`, `interval_pairs`).

### But the cut is not the blocker — the descriptor is

Fitted honestly against those two sets, the descriptor's separability is
**AUC 0.90**, not the near-perfect number an earlier circular sample suggested
(that sample drew "positives" from pairs already *assigned* the same id, which
they could only be by passing the cut). Even at IoU>0.95 — same object, same
frame, same instant, nearly the same box — **5% of pairs score below 0.465**.
The easiest possible case fails one time in twenty.

### The singleton rate is the wrong target

Sweeping the cut against cross-episode recurrence — the thing a persistent id
exists to produce — gives an interior maximum:

| cut | objects | singletons | **recur** | false-merge |
|---|---|---|---|---|
| 0.843 | 49,944 | 89.5% | 5,070 | 0.101% |
| 0.749 | 34,468 | 79.7% | 6,695 | 0.224% |
| **0.692** | **25,629** | **71.8%** | **6,901** | 0.278% |
| 0.603 | 14,754 | 56.1% | 6,215 | 0.481% |
| 0.408 | 2,345 | 16.7% | 1,927 | 1.709% |

Driving singletons from 80% to 17% costs **two thirds of the recurrence** and
multiplies proven-wrong merges by eight. Fewer, bigger, wronger objects. The
store now sits at the peak (`scripts/refit_identity.py`).

`fit_cut` replaces the percentile because both free samples are *same-frame*
pairs, and the decision the gallery actually makes — "is this the object from a
**different episode**?" — has no same-frame evidence at all. An objective that
measures the target quantity beats a statistic of a proxy.

### Assignment order matters more than the cut

`Gallery.assign` is greedy, so arrival order changes the result. At a **fixed**
cut of 0.739, order alone moves recurrence from 6,484 to 6,821 — larger than
the entire gain from re-fitting the cut. Globally-ts-sorted order is the worst
measured, because it interleaves all four cameras; the writer's track-close
order is among the best. Any tool that re-assigns ids must reproduce it.

### Next, and it is a descriptor question

`yolo26n-reid.onnx` is a nano model, and AUC 0.90 with a 5% floor failure on
trivial pairs is its ceiling, not the threshold's. Improving identity means a
stronger ReID encoder (vision-only, per [[no-text-identity]] — no CLIP,
SigLIP or DINOv2), not further tuning of the cut.

## The consolidated model roster (2026-08-01)

One model per task; a backbone earns its place by serving more than one.
Redundancy measured above, not assumed.

| task | model | note |
|---|---|---|
| frame embedding, every frame | **FDNN-V** (ours, 1.95M) | substrate for scene + all distillation |
| scene element (scene→scene series) | FDNN-V vectors | no new model — a time series over what exists |
| region proposal (write) | **YOLO11n-seg** single_cls | unchanged |
| association → presence | **BoT-SORT** geometry | no weights at all |
| identity descriptor | **DINOv3-S** teacher → distilled student | replaces yolo26n-reid (AUC 0.90 ceiling); student trained on the corpus's own free geometric pairs |
| trajectory channel (replaces mot) | tracker boxes + **Depth Pro** z | geometry, not an encoder; Depth Pro benchmark is user-gated |
| physics/dynamics per participant | **V-JEPA2 ViT-L** | tubelets seeded from elements, never whole-frame |
| act | same V-JEPA2 backbone + SSv2 probe | kept ONLY until trajectory lands; re-measure, delete if covered |
| story on top of answer | **SigLIP2** + **InternVideo2** | the two non-redundant survivors |
| text→video (later) | SigLIP2 / IV2 text towers | the reason they stay: DINOv3 has no text tower |

**Dropped:** PE-Core (0.89 rank-dup of sig2, 36 vs 23 min build), X-CLIP
(0.70 dup, its one distinct ability — cross-frame attention — was discarded
at write anyway), delta-appearance mot (wrong by design), yolo26n-reid
(replaced), FastSAM stays teacher-only.

Backbone count: 10 → 6. "Find this object elsewhere" is served by the
identity index (`object_vectors` + joins), not by episode-level fusion —
which is why `spaces()` excludes it.

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
