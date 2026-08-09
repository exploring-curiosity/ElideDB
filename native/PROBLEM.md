# THE PROBLEM — definitive statement

This document is the check every implementation must pass. Before any
code ships, ask: does this address the problem below, or is it a patch
for the corpus in front of us? The pattern this document exists to
break: every time a general mechanism hit a wall, a piece of the
current dataset leaked into the design (height profiles = image rows
as world height; "the arm"; "the tower relation"). Each hardwire was
a manual stand-in for a missing general layer.

## 1. The product problem

A user hands the system a clip — a moment from some recording — and
asks: **"when did something like this happen?"** The system must
return those moments, from raw video it has never been told anything
about. Any customer, any camera, any content. No labels, no text, no
task knowledge, no per-dataset anything. Ground truth exists only to
grade us, never to train or configure.

Everything hard lives inside the words "like this":

1. **"Like this" is an intent the example only partly reveals.** A
   clip is simultaneously an instance of this action, this object,
   this place, this actor, this outcome. All are legitimate readings
   of "like this." A system that bakes ONE similarity into its
   embedding has answered the question before the user asked it.
2. **The salient content of a moment is almost never the dominant
   signal.** What makes two moments alike is usually small — a small
   object, a brief contact, a change of relation — inside a signal
   dominated by the constant scene and the recurring actor that every
   recording has (an arm, a hand, a car hood, a gimbal).
3. **The constraints are the product.** Arbitrary unseen data;
   nothing per-dataset; per-recording self-calibration only; bounded
   online compute; raw bytes immutable; reads elided.

## 2. The atomic decomposition

Any video gives exactly three primitive signals; everything "like
this" can mean must be built from them:

- **(a) spatial coincidence** — which measurements occur together in
  a frame (appearance);
- **(b) temporal persistence** — which measurements recur across
  frames (identity, constancy);
- **(c) co-variation** — which measurements change together
  (structure-in-time).

A frozen encoder supplies (a). (b) and (c) exist only in the
recording itself; no pretrained model hands them over.

Every worldly concept the system needs is a composition:

| worldly thing | atomic composition |
|---|---|
| entity | spatial coincidence that persists (appearance that travels) |
| scene | persistence with no co-variation (never changes) |
| recurring agent | persistence with co-variation everywhere (in every change) |
| state | an entity's persistence path over time |
| interaction | co-variation between entities' paths |
| outcome | difference in persistent configuration across a moment |
| moment | the sub-graph of entities/interactions active in a window |

**Alikeness, atomically:** two moments are alike under some intent
when *there exists a mapping between their entities that preserves
the queried part of this structure*. Same action = a role assignment
(mover→mover, affected→affected) under which interaction pattern and
outcome correspond while appearance may differ. Same object = the
assignment preserving appearance/identity while all else may differ.

**The core gap:** likeness of moments is a CORRESPONDENCE; cosine
over any pooled descriptor computes an OVERLAP OF DESCRIPTIONS.
Overlap cannot express "there exists a role assignment." This is why
every pooled representation showed P@1 high / yield low: nearest
neighbours share everything (description finds them); distant true
matches share only role structure (which pooled vectors do not
contain). Classes shatter into clusters along nuisance factors —
the thing diffusion partially patched at the graph level.

## 3. The general solution (layers)

Write path, per recording, label-free, self-calibrating:

- **L0 appearance** — frozen general encoder. The only pretrained
  knowledge in the system.
- **L1 entities** — persistence linking of appearance through time →
  tracks. Converts fields into things. (V1 concession from the owner:
  capture can be arranged so the manipulated entity stays visible —
  removes occlusion, the hardest sub-problem, for V1.)
- **L2 roles by statistics** — per-recording persistence/co-variation
  profile per track → scene / recurring agent / episodic entities.
  Figure-ground is COMPUTED, never asserted.
- **L3 entity records** — see §4. NOT descriptions.
- **L4 interactions** — pairwise co-variation between tracks: contact
  form/break, common-fate spans, topology events (split/merge,
  birth/death), initiation order. Roles indexed by POSITION in the
  pattern, never named.
- **L5 moments** — a window's content = its sub-graph (entities,
  roles, relational state paths, before/after difference). Boundaries
  fall out of interaction-topology changes.

Read path:

- **L6 query** — identical operators on the uploaded clip; it
  decomposes into its own episodes.
- **L7 matching** — best PARTIAL role correspondence between query
  graph and stored graphs, scored per component (interaction
  structure, outcome, patient appearance, agent appearance, scene),
  query-implied weighting. Graded, never exact-isomorphism.
- **L8 corpus geometry** — hubness correction + diffusion on whatever
  similarity L7 emits. Axis-agnostic; already measured (+50% AP
  holdout; gain grows with corpus size).

Order matters absolutely: POOLING BEFORE SEPARATING DESTROYS THE
STRUCTURE PERMANENTLY. Every experiment of 2026-08-09 pooled at L0
and tried to recover L1–L5 with vector arithmetic; each recovery
degenerated into a dataset-specific patch.

## 4. The entity store: particulars, not kinds

A *description* maps a thing into a space organized by kinds — where
a pen and a pen-shaped cosmetic collapse, because resemblance is all
a description knows. The entity store instead holds **particulars**:

- An entity is THIS specific tracked thing, individuated by L1
  continuity. Resemblance cannot merge entries; only physical
  continuity creates identity. Two lookalikes are two entries because
  they are two instances — for no other reason.
- An entry = three parts, none a description:
  1. **fingerprint** — raw appearance vector at the entity's own
     scale. Job: comparison at read time. Never thresholded into a
     category, never clustered at write.
  2. **state series** — see §5.
  3. **biography** — links to every L4 interaction it participated
     in. Two lookalikes with different histories are distinguishable
     by what happened to them, with zero semantics.
- Identity claims are graded: same-track = certain; strong
  fine-grained instance match = probable same instance (identity-cut
  machinery, threshold fitted per store from the recording's own
  negatives); below that = merely similar-looking, a ranking signal,
  never an identity.
- **No partition at write, ever.** Write-time clustering into "kinds"
  — even unsupervised, even unnamed — is labelling with the serial
  numbers filed off. Kind-ness exists only at read time, as a
  neighbourhood around a query, relative to that query's intent.

## 5. State: stored absolute-private, compared relational-only

The state series is RECORDED raw in the recording's own image frame
(O(n), lossless, every relation derivable later; the reverse is not).
That frame is PRIVATE: a property of the observation, not the world.
**Absolute coordinates never cross a recording boundary.** Conversion
happens at moment-graph assembly; only relations enter comparison.

The three generic reference frames (zero domain knowledge):

1. **the entity's own past** — displacement, move/hold timing, order;
2. **co-present entities, pairwise** — approach, contact (relative
   distance → ~0 with correlated motion), separation. Distance-to-
   agent is just one pair; the agent is not special-cased;
3. **the scene** — L2's persistent remainder is the recording's own
   reference structure (at rest on it / departed / returned). Also
   absorbs CAMERA MOTION: scene-relative state survives drone or
   handheld footage where image coordinates are garbage.

Scale: distances normalized by the participating entities' own
extents — "near"/"contact" in units of the things themselves;
self-calibrating across zoom, resolution, viewpoint.

The shipped r20 representation fails precisely this test: image rows
used as a world axis — an absolute-frame commitment that worked only
because sim cameras are fixed, level, and shared between query and
corpus. No uploaded video honors those assumptions.

## 6. The acceptance example (pen test)

Corpus: robot arms in trials — picks a pen up, flips it, rolls it,
pushes it, opens its cap. Query: a HUMAN with two hands picks a pen
up and opens the cap, one clip. Required: return the pick trial and
the cap trial.

Why the general solution passes structurally: hands land in the
recurring-agent role by the same per-recording statistic that put the
arms there (roles are positional, never appearance) — cross-
embodiment costs nothing. The query splits into two episodes at
interaction-topology changes (second contact; entity SPLIT when the
cap separates — entity birth by split is native to a persistence
graph). Pick beats flip/roll/push by atomic facts only: patient left
the resting surface AND agent-patient correlation persisted after
contact (vs ceased). Cap trial is the only entity-split in the
corpus. The current shipped scorer fails this test outright (a human
demo's pooled frame deltas share nothing with a robot's).

This is a complete, label-free acceptance test: the corpus trials are
their own truth. The in-house proxy until a human-video corpus
exists: query panda episodes against a store restricted to
xarm7/vx300s episodes (cross-embodiment slice of the sim ruler).

## 7. Current system vs the general solution

| layer | requirement | shipped system | status |
|---|---|---|---|
| L0 | frozen appearance | DINOv3 vits; identity cut AUC 0.9994 | have, validated |
| L1 | entity tracks | none in shipped path; prior lineage recall 0.48 | missing / failed old gate |
| L2 | derived roles | fragments in wrong space (median bg = scene part) | partial |
| L3 | entity records | none — r20 pools everything | missing |
| L4 | interactions/topology | none in any lineage | missing |
| L5 | moment = graph | moment = pooled vector | missing |
| L6 | symmetric query | yes | have |
| L7 | role correspondence | cosine + DTW (time-correspondence only) | 1 of 2 dimensions |
| L8 | graph geometry | CSLS + diffusion, +50% AP holdout | have, carries unchanged |

The current system is the general solution with its middle removed.
Both validated ends (L0, L8) carry unchanged. The prior tracks-first
lineage failed from (a) tracking below its gate and (b) hand-coded
named axes on top — the general design differs in exactly those two
places: roles are positional statistics, and matching is
correspondence, not fixed-axis channels.

## 8. Gates (all pre-existing; build must pass them, not redefine them)

- L1: entity recall / id-consistency vs sim truth (chain_grade class,
  position-verified; truth is EVAL-ONLY, always).
- L4/L5 boundaries: episode spans vs true event spans; plus the
  segmentation gate (CCA-on-spans must recover the event-fit gain).
- End-to-end: the sim ruler (leave-episode-out yield/prec at
  k=1.5·support + AP + P@k), config chosen on sim_chains, reported on
  sim_eval_bal. Standing baseline to beat on the same holdout at the
  same corpus size: diffusion-current 0.643/0.428 @1352 events
  (0.713/0.475 @4902).
- Cross-embodiment slice: the in-house pen test (§6).
- Real corpora: the disguise battery floors must hold.
- The extraction ladder (probe R^2 vs sim truth) for any new
  representation stage.

## 9. What the problem is NOT

Not action classification. Not 0.80 on one ruler (the 5-primitive
ruler grades ONE axis of likeness at one granularity — optimizing it
directly rewards hardwiring). Not the arm, not towers, not pens. The
problem passes only when the same store, unchanged, answers "moments
like this" for an action-shaped, an object-shaped, and an
outcome-shaped example, on a corpus nobody prepared.
