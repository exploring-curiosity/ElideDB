# REBUILD — the complete plan after 2026-08-08

## 0. What is established by LOOKING, not by any score

Two contact sheets, both reproducible from the Desk:

**car 0039_sync 28-36s** (narrow street, parked cars both sides): the top
hit is a wide-open street; the list is generic driving. Judged bad.

**bridge file-050 20-28s** (towel being folded): the top hits are all the
same orange towel — but files 050/051/054 are the SAME session. The tell
is file-054 at +0.114, above the cut: same towel, but the action there is
placing cups ON it. Actor-scene matches, action does not, goal does not —
retrieved anyway. file-047: different cloth, arm picking at a bowl —
retrieved anyway.

Diagnosis, stated once and built on everywhere below:

    the current descriptor matches THE APPEARANCE OF THE MOVING REGION.
    It has no representation of actor state, no representation of the
    action's structure, and no representation of the goal/effect.

Corrections to the record:
- "car has almost no structure to retrieve" was WRONG. What was measured
  was separation in c2's view of car. The pixels contain the structure
  (tight vs open, uphill, corridors of parked cars); the extractor cannot
  see it. Extraction is the product being built.
- vgrade's 0.751/0.757 is re-identification (a distorted span finds its
  own original). It was quoted as retrieval quality. It is not. vgrade is
  demoted to a sanity gate and never again appears as a headline.
- The system as it stands does not answer the product question anywhere.
  Re-id works; kind-retrieval does not exist yet.

## 1. The definition of similarity (the missing specification)

Two clips are the same kind iff they agree on:

    ACTOR STATE   the agent and its configuration relative to its
                  surroundings. Manipulation: arm + what it holds/faces.
                  Ego corpora (car, drone): the ego camera IS the actor;
                  its state is its geometric relation to the scene —
                  clearance, layout, free space. "Tight" is a state.
    ACTION        the structure of the motion being executed — how things
                  move, independent of what color they are. Folding is a
                  motion topology, not an orange blob.
    GOAL/EFFECT   the persistent change the window produces: spread ->
                  folded, gap ahead -> gap behind, door open -> shut.

A query matches a result when all three agree (strict) — reported beside
per-axis agreement, because WHICH axis fails is the diagnostic.

## 2. Rules (unchanged, restated so this file stands alone)

- Vision only. No text, labels, class lists, sensors, or any encoder
  whose space is shaped by a vocabulary. Single view. One approach for
  all corpora.
- No eval/truth media in ANY training or distillation input, ever. The
  student is a one-to-one function copy of its teacher; every gradient
  step matches teacher outputs on non-eval inputs. There is no
  "fine-tuning" step anywhere in this plan — post-surgery recovery is
  RE-DISTILLATION against the teacher, same data rule.
- Judging is human eyes (the user as gate; Claude's eyes as a dev
  instrument, recorded with provenance). No VLM in any shipping path and
  no VLM-generated truthset. Eval materials live under eval/ (gitignored,
  unreachable from the query path).
- Every proxy screen must sample negatives at the operating point where
  ranking is decided (the hard-negative lesson: near-zero negatives rank
  self-consistency, and self-consistency is not retrieval).

## 3. Phase E — evaluation first (nothing else is trusted until this exists)

E1. QUERY BATTERY. 8-12 queries per store, chosen by LOOKING at contact
    sheets of the corpus, spanning distinct kinds (bridge: fold, pick,
    place, drawer, door, wipe; car: tight passage, open road, turn, stop,
    oncoming; drone: ascend, traverse, hover, turn; sim: per event type).
    Frozen as (media, t0, t1) in eval/battery.json with a one-line
    intent note FOR JUDGES ONLY (eval/ never touches the system).

E2. CONFUSERS BY CONSTRUCTION. For each query, the battery records at
    least one known SAME-SCENE-DIFFERENT-ACTION span (judged not-match)
    and, where the corpus contains one, a DIFFERENT-SCENE-SAME-ACTION
    span (judged match). Today's failure is exactly the first cell of
    that 2x2, so the metric is sensitive to it by design.

E3. JUDGING. Desk verdicts become three-axis: actor / action / goal,
    each yes/no, per (query, hit), appended to verdicts.jsonl with span,
    score, build, provenance (user | dev). The source player stays
    pinned so judging is a comparison, not a memory test.

E4. SCORING HARNESS vjudge.py. Scores ANY system against the accumulated
    pool: per-query precision-of-judged at the returned list, strict and
    per-axis, plus ACTION-OVER-SCENE — of the retrieved judged hits, the
    fraction that are action-matches vs scene-only matches. Coverage
    (judged fraction of returned) prints on every row; unjudged is never
    counted as wrong (the pool-limited lesson). Verdicts accumulate
    across builds so later systems are scored against judgements made
    before they existed.

E5. RETIREMENT. vgrade stays runnable as the re-id sanity gate. Its
    numbers are banned from summaries. The eval that gets quoted is E4.

Deliverables: eval/battery.json, Desk 3-axis judging, vjudge.py, an
initial judged pool (dev-provenance) over the current system as the
baseline row — the honest zero point.

## 4. Phase D — capacity: FDNN distillation of the frozen backbone

Why now: every representation experiment in Phase R pays the encoder
cost. 6-8 s per query and 17-21 min per store-hour makes iteration the
bottleneck. Distillation is also the ONLY step that is fully measurable
without any truth set: the target is the teacher's own output.

D1. TEACHER: DINOv3 ViT-L/16 (frozen). Later, any adopted flow/depth
    model gets the same treatment (distill-every-channel rule).

D2. STUDENT: the teacher's macro-structure retained — patch embed,
    attention blocks, residual layout — with MLP neurons replaced by
    FDNN rule-1 heterogeneous units (FINER oscillator / Gabor / poly
    sub-functions, as in fdnnvideo.py). Start at ~ViT-S capacity;
    grow only if the fidelity gate fails at target size.

D3. DATA RULE (the one that was violated in spirit before): inputs are
    disjoint from every eval medium, including held-out ranges —
    fresh sim episodes from the generator, bridge chunk files outside
    the 60 eval + held-out lists (~90 h unused), other on-disk imagery
    (oxford, nuscenes-mini, lab), heavy pixel augmentation. The input
    manifest is written to disk BEFORE training and committed.

D4. LOSS: token-level match to the teacher (patch tokens + CLS) at the
    output and at 3-4 intermediate depths, cosine + scale. Then rule 2
    (apoptosis by measured contribution) -> re-distill -> rule 3
    (neurogenesis where residual error concentrates) -> re-distill.
    No step ever sees eval media. No step optimizes anything but
    teacher agreement.

D5. GATES, all three required:
    a. fidelity: mean token cosine to teacher >= 0.95 on held-out
       DISJOINT inputs, and measured (inference only) on eval corpora;
    b. store-swap: rebuild one store with the student, Spearman rank
       correlation of search results vs the teacher store >= 0.9 on the
       battery;
    c. wall-clock: >= 5x encoder speedup at the shipping resolution
       ladder on this machine, measured not estimated.

## 5. Phase R — representation: extract actor / action / goal from pixels

Principle: DECOMPOSE FIRST, THEN POOL TIME. One global vector cannot
carry three kinds of information; three channels can. All primitives are
functions of pixels only. Each candidate is tested at the operating
point — full store as distractors, same-session spans as the hard
negatives — and judged by eyes via E. No other acceptance exists.

R1. FIGURE/GROUND WITHOUT LABELS. The actor is what moves coherently
    (common fate): optical flow + DINOv3 patch affinity -> actor mask
    per frame. Ego corpora: everything moves, so figure/ground becomes
    the FLOW FIELD itself (divergence = passing through; asymmetry =
    turning; fast near-border flow = close structure). Verified by
    looking at masks/fields on all four corpora before anything is
    built on top.

R2. STATE channel. Manipulation: actor-centric pooled features (what
    the arm holds / faces), separated from scene. Ego: clearance
    profile from the flow/depth field — the geometry of free space,
    which is what "tight" is. Appearance enters only inside the actor
    region, never the whole frame.

R3. ACTION channel. The time pattern of the ACTOR-CENTRIC signal, not
    the global frame: rank pooling and self-similarity applied to
    actor-local features and to mask/flow SHAPE (area, contour,
    spread) so folding matches folding in any color. Direction
    sensitivity (open vs close) is preserved by construction — rank
    pooling stays antisymmetric.

R4. GOAL/EFFECT channel. Persistent change: the difference between the
    window's settled start and settled end states, computed on the
    region the action touched (spread towel -> folded towel is an
    area/shape change; the prior delta-appearance result, AUC 0.98,
    says this signal is strong). Symmetric actions with different ends
    are separated exactly here.

R5. Each of R1-R4 is a small store build (10 min/corpus) -> battery ->
    contact sheets -> judged. Keep/kill per channel per corpus. A
    channel that helps one corpus and hurts another is not shipped
    with a per-corpus switch — it is redesigned until one rule works,
    or killed (one-approach rule).

## 6. Phase I — integrate

- Store schema: state / action / goal channels per window (+ energy).
- Query planning: per-query channel weighting REBUILT from scratch. The
  old one rewarded self-consistency across adjacent windows and was the
  fault in two consecutive iterations; the new one must be validated on
  the battery's confuser pairs before it ships. Abstention cut is
  unchanged (self-derived, no fitted constants).
- Full rebuild, all four stores, student encoder. Quoted results: E4
  (strict + per-axis + action-over-scene + coverage), re-id sanity,
  elision, latency. In that order.

## 7. Sequence, budgets, kill criteria

    E   eval harness + baseline judged pool          first, ~1-2 days
    D   distillation                                  starts in parallel
        (independent of E's design; gate is teacher fidelity)
        training on this machine is honestly multi-day; plan runs
        overnight with tracked stages and real ETAs
    R   primitives on the teacher at tiny scale while D trains;
        full iteration speed once D lands
    I   only after >= 2 R channels beat the baseline on E4

Kill criteria:
- an R candidate that does not beat c2 on ACTION-OVER-SCENE at equal
  scale dies, whatever its other numbers;
- a D student that misses the fidelity gate at target size gets one
  capacity increase, then the architecture is reconsidered;
- any step whose acceptance depends on a number the eval harness cannot
  produce does not run.

What does NOT survive from before: c2-as-the-product (it remains as the
re-id/near-duplicate channel, which is real and useful — dedup, "have I
seen this exact moment" — but is no longer called retrieval), vgrade as
a headline, and every claim of retrieval quality made before E exists.
