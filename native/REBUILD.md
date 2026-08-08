# REBUILD v2 — the complete plan, 2026-08-08

v1 of this file hand-coded three similarity axes and put distillation
before the representation was proven. Both were wrong and both are fixed
here. v1's evidence section stands.

## 0. What is established by LOOKING, not by any score

Two contact sheets, reproducible from the Desk, saved in eval/:

- car 0039_sync 28-36s (narrow street): top hit is a wide-open road.
- bridge file-050 20-28s (towel folding): every top hit is the same
  towel in the same session. file-054 — same cloth, but the action is
  placing cups ON it — passed the cut. file-047 — different cloth, arm
  picking at a bowl — passed the cut.

Diagnosis: the descriptor matches THE APPEARANCE OF THE MOVING REGION.
No actor state, no action structure, no goal, no memory of anything
before the window.

Corrections on the record:
- "car has almost no structure" was measured in c2's view of car, not in
  the pixels. There is structure and patterns in everything; extraction
  is the product being built.
- vgrade is re-identification. Demoted to a sanity gate; banned from
  summaries. c2 is renamed what it is: the near-duplicate channel (real,
  useful for dedup/"seen this exact moment"; not retrieval).
- No number claimed about retrieval quality survives from before Phase E.

## 1. The organizing principle (replaces the hand-coded axes)

v1 defined similarity as actor state + action + goal. The user's
correction is accepted: those are EXAMPLES of kinds of information, and
enumerating kinds is a vocabulary one level up. Sometimes the query is
about the background scene; sometimes an element in front; sometimes the
maneuver; sometimes a chain of events. No fixed list covers it.

The replacement comes from how world models work (JEPA/AMI, V-JEPA 2-AC,
Dreamer/RSSM, Genie, Marble):

    A representation is sufficient when it can PREDICT. State is
    whatever the future depends on. Information is kept or discarded by
    predictive relevance, not by a schema — the tight road constrains
    future motion, the pedestrian will move, the parked corridor implies
    parallax, the folded towel stays folded. All of it enters the state
    UNNAMED, because prediction needs it, and sensor noise leaves for
    the same reason.

Four architectural facts follow, each stolen from a running system:

  P1  PREDICTION DEFINES STATE (JEPA; the AMI thesis). Train nothing to
      classify, caption or match — train only to predict future latents.
  P2  MEMORY IS A FILTERED RECURRENT STATE (RSSM; V-JEPA 2-AC's
      block-causal predictor). The state at t accumulates everything so
      far. Amnesiac per-window encoding — what c2 does — is the gap.
  P3  PLACE STATE IS SEPARATE FROM MOMENT STATE (Marble: the persistent
      scene is one artifact, moments happen inside it). A media owns one
      slow accumulated scene state; moments are stored as deviations and
      transitions against it.
  P4  SIMILARITY IS FUNCTIONAL (V-JEPA 2-AC plans by comparing predicted
      future latents to a goal latent). Two moments are alike if they
      imply alike — alike states, alike transitions, alike futures. The
      operator inside can still be a distance; the object it runs on is
      structured state, never one flat appearance vector.

And the derived signal that unlocks long chains:

  P5  SURPRISE SEGMENTS TIME. Prediction error spikes at event
      boundaries, label-free. window -> event (surprise-bounded) ->
      chain (sequence of transitions) -> episode. Chain retrieval is
      sequence alignment over transitions, not pooling minutes into one
      vector.

The judge axes (actor / action / goal) SURVIVE ONLY IN THE EVAL as
questions a human answers about a pair. They are probes, not schema.

## 2. Rules (standing, restated)

- Vision only. No text, labels, class lists, sensors, codebooks, or any
  encoder shaped by a vocabulary. Single view. One approach, all corpora.
- NO eval/truth media in any training input — prediction training and
  distillation both draw from a written, committed input manifest that
  excludes every eval and held-out medium. Fresh sim generator episodes,
  the ~90 unused bridge hours, oxford/nuscenes/lab footage, augmentation.
  (KITTI has no disjoint day on disk; the driving domain is covered by
  nuscenes/oxford. car is eval-only.)
- There is no fine-tuning step anywhere. Model surgery recovery is
  re-distillation against the teacher. Self-supervised prediction trains
  on non-eval media only.
- Judging is human eyes; the user is the gate; Claude's eyes are a dev
  instrument with provenance recorded. No VLM on any shipping path, no
  VLM truthsets. eval/ is gitignored and unreachable from the query path.
- Proxy screens must sample negatives at the operating point (the
  hard-negative lesson). Same-session spans are the hard negatives.
- Online budget stands: the write path target remains ~1 min per media
  hour after distillation; the world-model core is sized for it.

## 3. Phase E — evaluation first (unchanged from v1, plus chains)

E1  Query battery: 8-12 per store, picked by LOOKING, spanning kinds —
    including scene-focused and element-focused queries on purpose, and
    2-3 CHAIN queries per store (multi-event spans in sim, multi-step
    manipulations in bridge). Frozen as (media, t0, t1) in
    eval/battery.json with judge-only intent notes.
E2  Confusers by construction: per query, a known same-scene/
    different-action span, and a different-scene/same-action span where
    the corpus has one. Today's failure is the first cell.
E3  Desk judging: per-axis verdicts (actor / action / goal, yes/no)
    per (query, hit), source player pinned, provenance user|dev,
    appended to verdicts.jsonl with span, score, build.
E4  vjudge.py: scores ANY system against the pool. Strict + per-axis
    precision-of-judged, ACTION-OVER-SCENE (of judged retrieved hits,
    action-matches vs scene-only matches), coverage on every row,
    unjudged never counted as wrong. Verdicts accumulate across builds.
E5  Baseline row: the current system judged first — the honest zero.

## 3b. W0 — experimentation config (speed before anything else)

User directive 2026-08-08: fast iteration, no more multi-hour loops.
Of the two options (distill the frozen teacher first, or use a smaller
frozen model), the second is faster BY A TRAINING RUN: distilling ViT-L
into an FDNN student is itself days of work, a small frozen sibling
costs zero. So:

    experimentation encoder   DINOv3 ViT-S/16 (frozen)   SDX_ENC=vits
    resolution                single 320                 SDX_RES=320
    measured                  vits 97.6 fps vs vitl 12.5 fps  (7.8x);
                              8 s query encodes in ~0.4 s; write path
                              becomes decode-bound

FDNN distillation stays in Phase D (last) as COMPRESSION of whatever
stack the eval validates. Headline numbers, when they exist, get
re-measured on the final stack; experimentation numbers are labelled as
such.

## 3c. The psychological frame (how human retrieval says to build this)

- GIST + VERBATIM in parallel (fuzzy-trace theory): people keep both,
  retrieve gist-first. The old system stored ONLY verbatim - which is
  exactly why the orange towel matched the orange towel. c2 is the
  verbatim trace; the world-model state is the gist.
- COMPLEMENTARY LEARNING SYSTEMS (hippocampus/neocortex): episodes
  written once, pattern-separated (the store's rows); regularities
  extracted slowly across episodes (the predictor). "Stacking" exists in
  the slow system as a regularity with no name.
- ENCODING SPECIFICITY (Tulving): a cue retrieves what was encoded by
  the same process - one encoder for write and query, already honored,
  now load-bearing. Retrieval is PATTERN COMPLETION: a fragment brings
  back the whole episode span.
- STRUCTURE-MAPPING (Gentner): surface similarity dominates retrieval
  UNLESS encoding is relational. Cylinder-on-block and block-on-cylinder
  share no surface, only the relation and the transition (two stable
  things -> one stable composite). Hence: state keeps parts and
  arrangement (spatial grid), and TRANSITIONS are the primary key for
  action-like queries.
- EVENT SEGMENTATION (Zacks): humans segment where their predictive
  model errs, and remember event units, not fixed windows. That is the
  surprise trace, with human evidence behind it.

The pipeline this dictates: cue -> gist match (state bands) -> episode
retrieval -> verbatim rerank. THE FIRST JUDGED TARGET is the user's own
example: a stacking query must return the OTHER stacking episodes
(different objects, colors, positions) and must NOT return unstacking
(same surface, opposite goal) or same-scene hovering. Seeded in
eval/battery.json (sim: q_stack_a/b, c_hover_a, c_unstack_a/b).

## 4. Phase W — the world-model memory layer (the core build)

W1  MODEL. Perception stays frozen DINOv3 patch features. On top, a
    small causal predictor — RSSM-style recurrent state or block-causal
    transformer, tens of M params (V-JEPA 2-AC's 300M shape at ~1/10
    scale) — consuming the feature stream at 4 Hz and predicting future
    latents at three horizons (~0.5 s / ~2 s / ~8 s). Multi-horizon
    heads give fast / mid / slow state bands: transitions vs object
    motion vs scene identity. Feasibility precedent: FDNN-V2's
    predictive training ran on this machine.

W2  TRAINING. Objective: predict future DINOv3 latents (cosine + scale
    match), teacher-forced one-step plus short rollouts (the V-JEPA
    2-AC recipe against error accumulation). Data: the non-eval
    manifest only, committed before the run. No labels, no actions
    (our streams have none at write time), no eval media, no other
    objective of any kind.

W3  WHAT THE STORE HOLDS, per stream:
      state    filtered state at surprise boundaries + fixed stride,
               kept as a SMALL SPATIAL GRID plus a global vector — one
               flat vector would erase element-level facets
      trans    the delta between settled states around each event: the
               action/effect signature, unnamed
      surprise the prediction-error trace: event boundaries + the
               "something happened" signal (replaces energy)
      place    one slow accumulated scene state per media (P3); moment
               records store deviations against it
    All columnar, all counted by the byte ledger, elision measured as
    ever.

W4  RETRIEVAL. The example is encoded by the SAME filter (its own
    frames as context — short context is a real limitation, measured
    not assumed). Then:
      intent    what is invariant across the example's own sub-windows
                and (when given) across a SET of examples — facet
                selection with no labels: if the examples share the
                scene band, the query is about place; if they share the
                transition, it is about the action
      match     candidate prune on state bands -> exact rerank on the
                intent-weighted state + transition
      rollout   optional tier: roll both states forward with the
                predictor, compare predicted trajectories (functional
                similarity; expensive; rerank-only)
      chains    sequence alignment (DTW/edit) over each media's
                transition sequence for chain queries
      abstain   the self-derived cut, unchanged in spirit, recomputed
                for the new score space.
    The old per-query channel weighting is dead; intent-from-invariance
    replaces it and must be validated on E2's confuser pairs before it
    ships.

W5  GATES (in order):
      a. surprise boundaries land where a human puts event boundaries
         (checked by eyes on contact sheets, all four corpora);
      b. state bands separate E2's confuser cells — same-scene/
         different-action must score LOW on the transition band and
         high on the scene band, visibly, per corpus;
      c. E4 on the battery beats the c2 baseline on ACTION-OVER-SCENE
         and on strict precision-of-judged;
      d. write-path cost measured, with the path to the online budget
         stated (the predictor is small; decode still dominates).

## 5. Phase B — hand-built baselines (kept, demoted)

Flow figure/ground, actor-centric pooling, persistent start->end change:
built SMALL, not as the product but as (a) baselines Phase W must beat
on E4 and (b) diagnostic probes for what W's state actually captures.
If a hand primitive beats the learned state on some axis, that is a
finding about W's training, not a shipping decision.

## 6. Phase D — distillation, LAST

Moved from before-R (v1) to after-W, because distillation is a
one-to-one FUNCTION COPY: probe inputs in, teacher outputs matched. It
preserves capacity, it cannot add information, and a copy of an unproven
teacher reproduces its faults at high fidelity. So it runs only on the
teacher stack that E4 has validated:

  - DINOv3 perception -> FDNN student (teacher structure retained, MLP
    neurons -> rule-1 heterogeneous units; apoptosis -> RE-DISTILL ->
    neurogenesis -> RE-DISTILL; every gradient matches teacher outputs
    on the non-eval manifest).
  - The W predictor is already student-sized; it ships as-is.
  Gates: token fidelity >= 0.95 on held-out disjoint inputs; store-swap
  rank correlation >= 0.9 on the battery; measured wall-clock >= 5x.

## 7. Sequence, budgets, kill criteria

    E   harness + baseline judged pool         ~1-2 days, first
    W1-2 predictor training                    starts once E1 exists;
         multi-day on this machine; tracked stages, real ETAs, tqdm
    B   baselines at tiny scale                while W trains
    W3-5 store + retrieval + gates             after training lands
    D   distillation                           only after W passes E4

Kill criteria:
- W fails gate (a) after two training recipes -> the predictor is
  re-scoped (bigger context, different horizon set) once; a second
  failure reopens the architecture question rather than tuning on.
- Anything that cannot beat the c2 baseline on ACTION-OVER-SCENE at
  equal scale dies, whatever its other numbers.
- A D student missing fidelity at target size gets one capacity
  increase, then the architecture is reconsidered.
- No step ships on a number the eval harness cannot produce.

References that shaped Phase W: V-JEPA 2 / V-JEPA 2-AC (arXiv
2506.09985) — frozen encoder + block-causal latent predictor, rollout
loss, planning by predicted-latent distance; LeCun's JEPA/AMI program —
prediction in latent space as the definition of understanding; Dreamer
RSSM — filtered recurrent state as memory; Marble (World Labs) —
persistent place state separate from moments; Genie — long-horizon
consistency from causal latent context.
