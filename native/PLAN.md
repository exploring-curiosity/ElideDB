# ElideDB Native Pipeline v2 — "TRACKS FIRST"

**Status: approved plan, not yet built. This document is the handoff.**
Backup of the entire prior system: branch `backup/pre-native-2026-08-04`
/ tag `backup-pre-native-20260804` (HEAD 075f96d). All new work lives in
`native/`. Nothing outside `native/` may be modified except additive
store tables and BENCHMARKS.md / docs/memory entries.

---

## 1. What is being built (product truth, from the owner)

A **database for physical AI**. Two workloads, one store format:

1. **Cloud / bulk**: customers upload huge videos and retrieve specific
   scenarios by long-contextual queries (to retrain or analyse their
   AI systems). Heavy compute allowed at READ time on a pruned
   candidate set (approved 2026-08-04).
2. **Robot / edge**: the store is a live memory for a standalone robot
   (no cloud). Millisecond read/write with small models only. The
   robot keeps a **cheap append-only log of queries and results** so
   that, when idle/powered, the on-device student can be **slowly
   trained from that log** (experience replay). Design the log schema
   from day one; the trainer must never block the read path.

Everything must be **zero-shot on unseen datasets**: the same binary,
zero config changes, runs on any customer's video. Per-corpus
calibration happens only through automatic corpus-fitted statistics at
ingest.

## 2. Doctrine — DO / DON'T (the constitution)

### DON'T
- **No VLM / video-LLM / text encoder in any online path.** There is
  no 2M-param VLM; VLM capability does not compact three orders of
  magnitude. VLMs are allowed OFFLINE only (labeling, evaluation,
  cloud-side read-time rerank — see DO list).
- **No captions/text scripts as an index.** A describer commits to ITS
  salient distinctions at write time; the query distribution is open
  (owner's "vibration in shoulder pad" argument). Language may
  supervise; it may never be the stored representation.
- **No hand-built perception.** No background subtraction, no
  hand-made change detectors, no hand colour/appearance features (Lab,
  chroma, palette-as-feature). This is the measured failure of the
  entire prior chain program (see §6). Perception comes from
  pretrained vision models.
- **No FDNN-V as a foundation.** Owner's verdict: bad experiment,
  retired. Do not build on it or cite it as precedent.
- **No text identity / naming at write. No camera meta, intrinsics,
  poses. No per-dataset hardwiring** (no dataset priors in code, no
  magic constants tuned to a corpus — thresholds are corpus-fitted
  automatically, and scale-anchored where distributions are
  imbalanced).
- **No training on truthsets.** Truth sidecars (truth.parquet,
  meta.json, palettes) are EVAL-ONLY, forever.
- **No position-blind grading.** Time-window graders reported 0.91–
  0.98 recall where position-verified truth was 0.48. Every detection
  claim goes through the chain_grade.py pattern (palette/position-
  grounded verification). No exceptions.
- **No raw Otsu on unimlanced/multi-modal mass** without inspecting
  the distribution first (misfitted ≥4 times, measured). Prefer
  1-D k-means midpoints, percentiles, scale-anchored cuts.
- No UMAP for retrieval decisions. No whole-corpus caching that fakes
  the elision metric. No fabricated benchmark numbers — every number
  from a runnable script, written to BENCHMARKS.md.

### DO
- **Pretrained, frozen, vision-native perception.** Point tracking is
  the primary primitive; segmentation is prompted by tracks;
  appearance comes from a small self-supervised encoder. No language
  anywhere in these paths.
- **Teacher = the same architecture, larger; compaction preserves
  structure.** Every representation-path model must have an in-family
  compaction story (smaller same-family variant or within-family
  distillation). Exploratory teachers must be reasonable-sized (run on
  this M-series MacBook, MPS) and themselves compactable.
- **Derived logic is code, not weights** (events, contact, support,
  slots — pure kinematics over tracks; zero parameters on the robot).
- **Eval graders may use truth** (they ARE the truth side): palette
  anchors, position verification, truth prims. Never in a path.
- **Cloud read path may run a query-conditioned heavy model over the
  pruned top-k** (approved). The index's job is recall; understanding
  may be late. On the robot: students only + the query/result log.
- **Preserve the storage core**: Parquet tables via the existing store
  API, immutable raw media, byte-elision accounting on every read,
  snapshots/manifest discipline, `set_layout` cluster keys.
- Progress bars on every bulk job (wrap the iterable); resumable
  multi-stage jobs; artifact-is-the-test asserts; per-stage wall-time
  prints; memory files updated as work happens.

## 3. Architecture — the tracks-first substrate

```
raw video (immutable)
   │  decode once per episode-view (existing FrameSet path)
   ▼
[T] DENSE POINT TRACKS        pretrained tracker (CoTracker3-class,
   │                          ~25M; candidates: CoTracker3, LocoTrack,
   │                          TAPIR/BootsTAP — pick by measured MPS
   │                          throughput + spot-check quality)
   ▼
[O] OBJECT BUNDLES            common-fate grouping of tracks (rigid-
   │                          consistency over windows); life cycle
   │                          rest/move segments from kinematics;
   │                          identity RIDES ON TRACKS through carries
   ▼
[A] AGENT                     self-identified by CAUSALITY: the bundle
   │                          whose motion precedes+accompanies other
   │                          bundles' onsets. No appearance prior —
   │                          works for arm, hand, forklift.
   ▼
[E] EVENTS (code, no weights) rest→move→rest per bundle = one
   │                          manipulation; contact = proximity + co-
   │                          motion onset; support/ON = settled
   │                          static contact; push vs carry from
   │                          displacement + occlusion profile;
   │                          slots = first-appearance ordinals
   ▼
[S] MASKS + APPEARANCE        SAM-2/3-class masks prompted BY bundle
   │                          points at rest moments; appearance vec
   │                          from small SSL encoder over mask crops
   │                          (DINOv3-S already in store; test MAE-S
   │                          where colour identity matters — DINOv3
   │                          is colour-blind, measured)
   ▼
[K] KIND (cloud-side, opt.)   V-JEPA-2-class video embedding per event
                              window (vision tower only, no text)
```

**Write records** (all via store API, ms budget, per-stage timing):
objects (id, life cycle, appearance vec), events (actor, object, span,
geometry, kind vec, slot), per-segment chain-token sequences, frame
embeddings, raw media untouched. **Read/retrieval** reuses the already
committed and proven machinery unchanged: `chain_qbe.align` (0.992 on
correct scripts), channel pool + seed-LOO selection
(`chain_channels.py`, `chain_fuse.py`), bench harness
(`chain_moves.bench`). Cloud adds the approved top-k heavy rerank.
Robot adds the query log:

**`query_log` table (design now, build at G5):** append-only rows
(ts, query representation, channel scores, returned spans, accepted /
rejected / ignored signal if any, compute budget). Cheap to write,
local-only, consumed by an idle-time trainer that fine-tunes the
on-device student slowly. Never blocks reads.

## 4. Gates — strict order, none skippable

- **G0** Backup (done: `backup/pre-native-2026-08-04`) + `native/`
  skeleton + this plan committed.
- **G1** Tracker substrate on sim_chains. GATE: **position-verified
  event recall ≥ 0.95 on `scripts/chain_grade.py`** (the standing
  instrument). Until G1 passes, nothing else is worked on.
- **G2** Chain-QbE **≥ 0.90** via the existing bench harness (recipe
  frozen on DEV, reported on HOLDOUT).
- **G3** Zero-shot proof: the SAME binary, zero config, on the kitchen
  corpus (fresh_bench / bridge stores with their existing truthsets).
  No code changes permitted between G2 and G3 runs. Report both.
- **G4** Throughput + compaction: measure write-path ms/frame;
  compact tracker/SAM/encoder in-family until the edge budget holds;
  re-run G1–G3 with the compacted students (numbers may drop —
  report the compaction cost honestly).
- **G5** Robot tier: query_log schema + idle-time trainer skeleton +
  a measured cheap-write demo.

## 5. Repository plan

```
native/
  PLAN.md          this document
  track.py         [T] tracker wrapper (batched, MPS, cached npz)
  objects.py       [O]+[A] grouping, life cycles, agent causality
  events.py        [E] event derivation (pure code) + chain tokens
  appearance.py    [S] mask-prompted crops + SSL vectors
  write.py         store writer (tables above, timing, elision)
  grade.py         thin wrapper calling scripts/chain_grade.py
  bench.py         thin wrapper calling the existing harness
  qlog.py          [G5] robot query/result log + idle trainer
```
Caches in the session scratchpad, keyed by store+stage. Old scripts
stay untouched as instruments/records.

## 6. Measured lessons the next session must NOT relearn

(All numbers from BENCHMARKS.md 2026-08-03 entries; scripts committed.)

1. Hand-built detection true recall was **0.48** (position-verified);
   four time-window graders reported 0.91–0.98. Grader first.
2. Oracle event scripts through the existing aligner: **0.992** yield;
   degraded oracle (slots + push flag only) still **0.992** — the
   retrieval side is solved; ONLY perception was ever missing.
3. Ceiling proof: position-verified events + truth slots benched at
   **0.23/0.35** — at 0.48 recall nothing downstream can reach 0.90.
4. Encoder identity on detector-quality crops is dead in BOTH DINOv3
   and MAE (~0.53 AUC, masked AND clean). DINOv3 is colour-blind
   (cross-colour 0.85–0.90 > same-object 0.73–0.84); MAE is
   colour-aware only on good crops. Identity must ride on TRACKS.
5. Channel plurality without good events ceilings at **0.41/0.43**
   (novelty single 0.45). Fusion recipes, PRF, profile features,
   seriality DP parse, cross-view InfoNCE (150 eps → instance
   fingerprints): all measured flat. Do not retry them on bad events.
6. Junk floods beat every decoder: with 10:1 junk, per-event scalar
   features max at 0.72 AUC and global parses stay flat. High-recall,
   high-precision primitives FIRST; cleverness above them second.
7. Otsu misfits on imbalanced mass (≥4 occurrences). Inspect, then
   fit; prefer k-means-1d midpoints / percentiles / scale anchors.
8. Hard per-stage gates compound errors; prefer soft evidence with a
   global decision — but no decoder rescues low-recall primitives.

## 7. Instruments inventory (already committed, reuse as-is)

- `scripts/chain_grade.py` — THE gate: palette/position-verified event
  grading. `scripts/chain_qbe.py` — alignment. `scripts/chain_moves.py`
  — bench harness + otsu. `scripts/chain_channels.py` — channel pool +
  seed-LOO selection. `scripts/chain_fuse.py` — fusion recipes.
- `scripts/chain_delta.py`, `chain_serial.py`, `chain_ledger.py`,
  `chain_2v.py`, `chain_slotline.py` — superseded detectors; negative-
  result records; reference only.
- Corpus: `data/sim_chains` (150 eps, truth.parquet EVAL-ONLY),
  `lake/sim_chains` (two-view store). Kitchen: `lake/fresh_bench`,
  bridge stores + truthsets for G3.
- `scripts/sim_chains.py` / `sim_stack.py` — corpus generator (more
  episodes can be minted if needed; validated).

## 8. Compute envelope

Apple Silicon MacBook (MPS), single machine. Decode ~5 episode-views/s
warm. Tracker/SAM/encoder batch sizes must be measured before full
runs; every bulk loop gets a tqdm bar and a resumable checkpoint.
Models must download and run locally (HF gates permitting — flag any
gated repo to the owner instead of working around it).

## 9. Open questions for the owner (non-blocking)

1. Edge budget: target ms/frame and model-size ceiling on the robot?
   (G4 needs a number; propose ≤10ms/frame write, ≤30M params total.)
2. G3 acceptance: which kitchen metrics constitute "works zero-shot"
   (propose: existing q04/q05 QbE ≥ prior committed numbers).
3. query_log: what user feedback signal exists on the robot
   (accept/reject? task success?) to label replay data.
