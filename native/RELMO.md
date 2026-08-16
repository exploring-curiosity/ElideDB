# RelMo — a relational motion encoder for ElideDB

**One line:** a small model that turns any video into *what physically
happened* — which things there are, how they moved, and how they
acted on each other — trained on physics data other people already
generated and released, and deployed on unseen domains with zero
labels, zero setup, zero per-corpus configuration.

Status: SPEC. Nothing built yet. Written 2026-08-11 after the
measurement ladder in PROBLEM2.md closed.

---

## 1. Why this exists (the measured chain, not an argument)

Four weeks of measurement on a clean 324-event corpus produced one
number that matters and one that explains it:

| what | yield@support |
|---|---|
| every label-free readout tried (8 of them, 4 representation families) | 0.31 – 0.51 |
| a supervised probe on the *same inputs* | 0.77 – 0.86 |

The information is present in the data. No hand-built or
unsupervised similarity function recovers more than about half of
it. The families tried and their measured verdicts:

- frozen video foundation features (V-JEPA 2): 0.309, at chance -
  appearance dominates, the relational difference is a rounding error
  in the pooled vector;
- adapted video features (contrastive head, all modern amendments):
  0.389 - the mechanisms work but 324 events cannot teach a metric;
- hand-built symbolic facts over SAM2 tracks: 0.510 - the best, and
  it plateaus because the rules never converge (every threshold fix
  destabilised a neighbour);
- kinematics over CoTracker3 point tracks: 0.375, and the two
  specific failures were diagnosed exactly - (a) grouping points
  into objects by common fate collapses during transport, because a
  carried object and the arm genuinely move as one (151/324 moments
  became a single blob); (b) contact/support state is not reliably
  computable from image-plane geometry (measured blind: median box
  gap 0.00 for both touching and separated pairs);
- alignment-based (XIRL/TCC cycle-consistency): 0.310, and its
  published form needs same-task video groupings, which is the
  retrieval question itself;
- local VLM judges (two generations): 0.54 - 0.62, and the
  decomposition showed why: near-constant outputs, "held" on 95% of
  clips regardless of content.

**The two things that broke are both learnable perception
problems, and both have massive open training data with exact
ground truth already generated.** That is the entire thesis:
stop hand-coding grouping and contact; learn them from physics
someone else already simulated; keep everything downstream
relational and label-free.

## 2. What RelMo is

```
video ──► CoTracker3 (frozen, open) ──► point tracks (N x T x 2 + vis)
                                              │
                                              ▼
                            ┌─────────────  RelMo  ─────────────┐
                            │  set-transformer over trajectories │
                            │  ~10-30M params, permutation-      │
                            │  equivariant over points           │
                            └────────────────────────────────────┘
                                              │
        ┌──────────────────┬──────────────────┼──────────────────┐
        ▼                  ▼                  ▼                  ▼
   object slots      contact/support     relative-motion     moment
   (which points     events (who         structure           embedding
    are one thing)   touched whom,       (per-pair over      (retrieval
                     who supports whom)  time)               unit)
```

Inputs are motion, not pixels. That is deliberate: point tracks are
already domain-invariant (a track is a track whether it is on a
gripper, a rotor or a fish), which is the cheapest available hedge
against the sim-to-real appearance gap. Appearance (DINOv3) enters
only as an optional side-channel for object identity, never for
deciding what happened.

Outputs are *structure*, not a class. There is no vocabulary in the
model - no "pick", no "flip", no noun list. Two moments are compared
by correspondence between their structures, which is the same
operation whether the domain is an arm, a drone or a submarine.

## 3. What it is trained on — all pre-generated, all open

Nothing is simulated by us. These datasets exist, are released, and
carry exact physical state:

| source | what it gives | why it matters here |
|---|---|---|
| **Kubric / MOVi-A..F** (Google, Apache-2.0, pre-generated on GCS) | per-object segmentation, 3D poses and velocities, **collision events**, depth, flow, up to 23 objects, moving cameras | the exact two supervisions we need: *which points are one object* and *when things touch* |
| **Physion / Physion++** | 8 physical scenarios - support, containment, collision, dropping, rolling, draping - with relational variables | supervision for support/containment, the relation the depth route failed on |
| **PhysInOne** (2026) | ~2M videos, 153k scenes, 71 physical phenomena | scale and phenomenon coverage if more is needed |
| **Open-X / Bridge / DROID** (real robot data, no annotation used) | real footage with proprioception | sim-to-real adaptation without labels |

Supervision extracted from these (labels ARE allowed at training
time — the owner's rule binds demo and eval, not training):

1. **grouping**: predict which tracked points belong to one rigid
   body (Kubric instance masks). Replaces the common-fate clustering
   that measurably collapsed.
2. **contact/support**: predict make/break and force-bearing
   support (Kubric collision events, Physion support scenarios).
   Replaces the image-plane box-gap that measured blind.
3. **correspondence metric**: for pairs of clips, the target is the
   cost of the best correspondence between their interaction-graph
   trajectories, computed exactly from simulator state, allowing
   entity permutation, time warp and rigid transform. Trained as a
   ranking loss, never a classification.

Note carefully what is NOT trained on: any class name, any task
name, any human annotation, and any of ElideDB's own corpora.

**Why this transfers where class labels cannot.** A model trained on
`{pick, stack}` cannot say anything about `{flip, 360}` - the label
space does not exist there. But *contact*, *support*, *rigid group*,
*relative motion* exist in every physical domain, and they are what
this model predicts. Falling and colliding blocks teach the same
alphabet that an arm, a drone and a submarine are written in.

## 4. What deployment looks like

```
ingest:  video → tracks → RelMo → structure + embedding → store
query:   clip  → tracks → RelMo → structure + embedding → kNN → 
                                   correspondence rerank → moments
```

- No labels, no vocabulary, no per-corpus training, no configuration.
- Nothing is asked of the customer, ever.
- Latency: tracks are the cost, and THE COST IS THE PROBLEM. Measured,
  not estimated (tracks2.log, `tracks2/rcasa 100/100 [1:57:19, 70.40s/ep]`,
  100 episodes / 33,110 frames at 320x240 on MPS-with-CPU-fallback):

      212.6 ms per frame        5.1 s per 24-frame window
      70.4  s  per episode      4.7 fps  =  0.16x realtime

  So ONE HOUR of 30 fps video costs **6.4 hours** of tracking (108,000
  frames x 212.6 ms). At 320x240; higher resolution is worse.

  This supersedes an earlier claim here of "~0.1-0.3 s/window", which
  was 17-50x optimistic and was never measured. The owner's standing
  constraint is "high speed read and write are non-compromisable... in
  order of ms not minutes", budget < 1 s/episode. The current tracker
  is ~70x over that budget, and no downstream number changes it: RelMo
  itself is sub-10 ms and kNN is sub-ms, so 99.9% of serve cost is the
  tracker. Closing this is a tracker problem (the MPS
  grid_sampler_3d CPU fallback, resolution, point count, or a
  different tracker), not a modelling one — and it is open.
- The per-corpus adaptation (Stage B) becomes *optional* rather than
  load-bearing: RelMo is corpus-independent by construction.

## 5. Use cases it serves

| # | use case | how RelMo serves it | budget |
|---|---|---|---|
| U1 | **Search** — "when did something like this happen?" over an archive | embedding kNN + correspondence rerank on the top-K | ≤ 5 s |
| U2 | **Realtime decision-making** — an agent consults memory while acting | tracks on a rolling window + embedding + kNN; no VLM, nothing heavy in the loop | ≤ 300-500 ms, 3-10 Hz |
| U3 | **Show-then-do** — a human demonstrates, the robot retrieves its matching experience | cross-embodiment by construction: hands and grippers produce the same contact/support structure, and appearance is never consulted for what happened | ≤ 2 s demo-end → set |
| U4 | **Cold start / tiny corpus** — five clips, brand new domain | RelMo is corpus-independent, so quality does not depend on corpus size; nothing is fitted per store | U1/U3 budgets |

Beyond retrieval, the same structure output is directly useful as:
- an **event index** (contact events are boundaries — segmentation
  for free, no thresholds);
- a **world-model interface** (structure is what a planner needs to
  roll out, per docs/MPC_FEASIBILITY.md);
- an **explanation** ("these matched because a thing left support
  and became carried") — grounded, not narrated.

## 6. Evaluation — unseen domains only, no ground truth of any form

Hard rule from the owner: the demo and the evaluation must use
**no labels, no simulator state, nothing**. Therefore:

- **Never evaluated on anything in training.** No MOVi, no Physion,
  no PhysInOne at eval.
- **Domains at eval must be new in form, content and rendering**:
  the ElideDB robot corpora (a different simulator, different
  content), real kitchen manipulation, real driving, drone footage,
  underwater if available.
- **Grading is by human verdict** through the existing Desk verdict
  UI: precision-of-returned on what the system actually returns,
  abstention permitted. Where an internal ruler exists, its truth is
  used to *grade only*, never inside the pipeline, and it is
  reported separately from the human numbers so nothing is confused.
- **The demo is an upload**: a clip from a domain nobody prepared,
  answered in seconds, with the retrieved moments shown.

## 7. Cost, honestly

The expensive things — simulating physics, rendering, generating
millions of annotated frames, training a point tracker — have all
been done and released. What remains:

| step | cost |
|---|---|
| download a MOVi subset (+Physion) | hours of bandwidth, free |
| run frozen CoTracker3 over ~10-30k training clips | ~5-15 h on this Mac, $0 |
| train RelMo (10-30M params over cached tracks) | hours per run on this Mac, $0 |
| eval on 4-5 unseen corpora | hours, $0 |

No cloud spend is required for a first credible model. Scale-up
(PhysInOne, larger encoder) is optional and only justified if the
gates below pass.

## 8. Gates — each kills the program cheaply if it fails

**G1 (days, laptop): does supervised grouping+contact beat the hand
rules?** Train on MOVi only, apply to the ElideDB robot corpus, and
measure the two things that were diagnosed as broken: object
grouping quality and contact-event detection, against that corpus's
own ruler used for grading only. Pass = grouping collapse rate falls
well below the measured 151/324 and contact detection beats the
measured-blind box gap. Fail = the whole thesis is wrong, at a cost
of days.

**G2 (days): does it move retrieval on a domain it never saw?**
Same corpus, full pipeline, compare against the standing 0.510.
Pass = clearly above; borderline = 0.5-0.6, treat as unproven.

**G3 (weeks): does it hold on real, unprepared footage?** Kitchen,
driving, drone — graded by human verdicts only. This is the real
test of the whole claim.

**G4: cross-domain, the drone/arm question asked directly.** Train
with drone-like phenomena excluded; test on drone footage. This is
the question that killed the label route, asked of this one.

## 9. What would make me abandon it

- G1 fails: grouping and contact do not transfer from MOVi-style
  physics to robot manipulation. Then the "universal physical
  alphabet" premise is false and no amount of scale fixes it.
- G3 fails while G2 passes: it works in simulation-like renderings
  and dies on real footage. Then the honest product is the coarse
  tier plus a usage-driven sharpening loop, and this document is a
  research note rather than a plan.

Both failures are cheap and early by design. That is the whole
difference from how the last three weeks went.
