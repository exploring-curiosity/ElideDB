# Plan B — the Event Teacher

Objective unchanged: **yield = true/support at k = ceil(1.5×support),
prec = true/returned, target ≥ 0.90**, measured by `bench_truth` on
lake/bench (now demo-separated, A complete at 452ff9a). The teacher is
offline and may be expensive; the FDNN student comes after the teacher
earns its number. Everything below derives from pixels; task strings
and dataset labels stay out (no-metadata rule); the truthset is
eval-only.

The organizing principle (user-set): a clip is CLASSIFIED, not scored —

    scene        ← any one frame
    agent        ← the self-moving thing
    participants ← what the agent contacts, in order
    events       ← state transitions (contact, release, containment,
                   articulation, appear/disappear), timestamped
    answer       ← initial→final diff: [agent][verb≈transition]
                   [participant][relation to scene object]

This is the semantic-role frame the literature converged on
independently: VidSRL/VidSitu (verb + roles per event), Action Genome
(spatio-temporal scene graph per action). We adopt the SCHEMA, not
their models.

## 1. Design laws — one per measured failure

| # | law | the measurement that forced it |
|---|-----|-------------------------------|
| L1 | Never pool away order. Appearance embeddings are a recall floor, never the verdict. | Reversal test: frame-reversed clip flips meaning; pe/sig2/obj scores identical. Median true rank 224–276 under the pooled stack. |
| L2 | Presence ≠ participation. A participant is a region that CO-MOVES with the agent after contact. | GDINO presence separated (0.689/0.526) but an eggplant on the table scores like one being put away; 7 negatives above best true. |
| L3 | No isolated-crop identity. Identity is read at full-frame context; a crop exists only to FOCUS the namer after structure picked the moment and region. | Crops dead twice at corpus scale (ranks 280/534; 0.08 mean yield); resolution was not the cause — context loss was. |
| L4 | No extreme-value pooling over many draws. Aggregate evidence by counts/means inside matched structure; max only over a track's few candidate moments. | Patch MaxSim 0.08 (last of ten); 3-window max WORSE than one window (0.40→0.36) — negatives get extra draws. |
| L5 | VLM = generator, never judge. One constrained NAMING question per participant; no yes/no relevance, no A/B, no menus at rank time. | Yes/no saturates (10 negs above best true); pairwise: 826 comparisons, zero movement; forced-choice twitchy (menu aliasing q01 0.47→0.35; gate failed to protect). |
| L6 | Detector over judge for "what is here". Detection confidence is evidence with a calibrated-ish scale. | GDINO 2.7 s/clip, separates where the 4-bit VLM churned. |
| L7 | ITM = evidence, not authority. Tie-break inside candidate sets only. | Corpus AUC 0.85–0.97 still leaves a true episode near rank 95; needle queries (sup 2) need rank ≤ 3. |
| L8 | Corpus-scale acceptance only. Small probes exist to REJECT fast. | The 80-episode pool reported 1.00 that was 0.00 on 1,121 (14× easier). |
| L9 | One code path for fit and live; k-invariant artifacts (k-ladder); weak search as regularizer; rotate a predecessor before every lossy artifact write; channel death must be loud. | Fit-at-K=10-serve-at-100 cost 0.26; random restarts raised in-sample and dropped LOQO 0.265→0.151; the unrotated `_set_weights` overwrite cost a HF recovery; transformers-5 silently killed iv2 (0.38→0.13). |
| L10 | Temporal logic within demo walls only. | 2,097 stitched recordings; store now enforces 60 s gaps (A). |
| L11 | Fidelity is judged against RAW only. | The old rendition sat one frame off its own timestamps (~7 MAE). |

## 2. Transferables from industry and literature

**Twelve Labs Marengo 2.7 — multi-vector decomposition.** They dropped
the single pooled clip embedding for multiple SPECIALIZED vectors per
clip (appearance / motion / OCR / speech), each answering a different
aspect, routed at query time. That is our schema in embedding form:
our event record's fields (participant name-vec, verb class,
articulation sign, trajectory) are typed aspect vectors. Adopt: store
per-FIELD representations, match per-field, fuse with per-field
confidences — never one fused score.
([Marengo 2.7](https://www.twelvelabs.io/blog/introducing-marengo-2-7))

**Marengo indexing order — segment FIRST, embed second.** Their unit
is a short clip cut before any embedding. Ours after A: demo → event
sub-spans (B). The retrieval unit becomes the EVENT SPAN; the demo is
the graded container so the truthset keeps working via containment.
([Marengo docs](https://docs.twelvelabs.io/docs/concepts/models/marengo))

**Marengo 3.0 — temporal dynamics as first-class.** Their 3.0 pitch is
exactly the reversal-test argument: retrieval fails "where temporal
dynamics dominate" unless the representation is temporal. Validates
transitions-first. ([Marengo 3.0](https://www.twelvelabs.io/blog/marengo-3-0))

**Search-response design.** Their hits carry confidence + modality
attribution. Ours: every answer field carries (confidence, evidence
tag: which instrument produced it, at which timestamp). This feeds
the existing confidence cut and makes every hit explainable.

**Daft — one mixed CPU/GPU pass.** Their engine's win is decode +
filter on CPU concurrent with GPU inference in one pipeline, batched.
The teacher extraction adopts the shape: per demo, decode ONCE; flow,
tracks, contacts on CPU; naming batched on GPU across demos. (Their
engine-level ideas — pushdown, late materialization — are already in
our Rust core; nothing new to adopt there.)
([Daft](https://github.com/Eventual-Inc/Daft),
[multimodal embeddings](https://www.daft.ai/blog/multimodal-embeddings))

**VidSRL / VidSitu / Action Genome — the event frame.** Event = verb +
roles (agent, patient, location/target) with temporal extent; scene
graph relations per action. Our answer record IS this frame; matching
is frame unification, not similarity search.
([VidSitu](https://arxiv.org/pdf/2104.00990),
[HostSG](https://arxiv.org/abs/2308.05081))

## 3. The pipeline (write-time, per demo, offline)

    P0 scene         mid-frame + existing gist vector (subjects/pe)
    P1 agent         flow-asymmetry track (validated: 40–100%
                     persistence, probe_actors)
    P2 participants  contact = agent track meets an object region;
                     participation = the region CO-MOVES with the
                     agent between contact and release (L2). Regions
                     from motion CCs + GDINO boxes at contact moments.
    P3 naming        best moment per participant (detector conf ×
                     visibility) → ONE constrained VLM naming
                     ("what object is this? 1-3 words") + SigLIP text
                     embedding of the name + detector confidence (L5,
                     L6, L3: full-frame context, crop only as focus).
    P4 events        contact/release (P2 boundaries); containment
                     (participant box enters a static container
                     region and track ends there); articulation
                     (large planar region displacement, signed —
                     open vs close); appear/disappear (track
                     birth/death away from frame edge). Camera-shake
                     guard: all motion measured against the frame's
                     own flow median. SAM3.1 masklets only if the
                     cost gate passes (throughput on MPS unmeasured).
    P5 answer        per-event rows: (verb_class, participant_name,
                     name_vec fp16, relation, container_name, t0, t1,
                     per-field confidence, evidence tag) + one
                     initial→final rollup per demo.

Tables: `events`, `answers` — additive; nothing existing is touched.

## 4. Query-time matching

    parse   atoms_of (exists) + closed-class verb→transition lexicon
            (generic English, applied uniformly — allowed)
    match   verb_class compatible AND cosine(name_vec, query noun) AND
            relation compatible — soft-AND weighted by stored per-field
            confidences (Marengo-style per-aspect fusion)
    floor   existing fused channels as the recall floor (L1);
            ITM tie-break inside the candidate set only (L7)
    gate    no compatible verb_class in corpus, or every name cosine
            below a corpus-derived floor → empty set (q06 stays PASS)

Fitted matching weights, if any: same fitter discipline (L9) — k-ladder,
LOQO reporting, warm-start rotation, one code path.

## 5. Validation ladder (reject fast, accept only at corpus scale)

    V0  det×motion participation probe (the interrupted measurement) on
        q09/q10/q00 bands. Gate: trues inside top-5 by participation.
        RESULT (2026-07-29): rejects the QUERY-TIME shortcut, not the
        design. q09 improved (ranks 7/11 -> 5/9) but q10's noun
        saturates the detector (true 0.92 / false 0.87) and q00's
        attribute phrase means nothing to it (0.68 / 0.69). Query-
        conditioned detection is dead as a reranker; the write-time
        pipeline never uses it - regions come from structure, names
        from generation, matching from name-vec space.
    V1  P1–P4 on 20 random demos: printed event scripts, cost/demo.
        Gate: agent found ≥90%, ≥1 participant with a plausible name
        ≥80%, cost ≤ 8 s/demo.
    V2  full corpus (1,122 demos) extraction, then the ledger metrics
        on all 10 queries — THE acceptance number (L8).
    V3  field ablations: name-only / verb-only / +relation — which
        field earns yield.
    V4  honesty: LOQO for any fitted weights; k-ladder; ledger row per
        change.

## 6. Risks and non-goals

Risks: SAM3-on-MPS throughput (fallback: GDINO boxes + motion masks);
namer vocabulary drift (mitigate: name embedding matching, not string
match); articulation false positives from camera shake (median-flow
guard); multi-event demos vs single-label truthset (rollup answers).

Non-goals in B: no student training, no new global encoders, no
prompt-engineering iterations on judges (L5 closes that door), no
metadata of any kind.
