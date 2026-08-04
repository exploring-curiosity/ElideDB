# ElideDB Native Pipeline v2 — "TRACKS FIRST" — EXECUTION PLAYBOOK

**Read this once, fully, before running anything. This plan is meant to
be EXECUTED, not re-planned.** Strategy questions are settled; the
owner has approved this document. If a step fails, follow its written
failure branch. Stop and ask the owner ONLY at the conditions in §12.

Backup of the entire prior system: branch `backup/pre-native-2026-08-04`,
tag `backup-pre-native-20260804` (commit 075f96d). Never touch that
branch. All new code lives in `native/`. Files outside `native/` may
not be modified, except: appending to `BENCHMARKS.md`, appending to
`docs/memory/*` and `docs/logs/*`, and creating NEW store tables via
the store API. Never delete or rewrite existing tables of
`lake/sim_chains`, `lake/fresh_bench`, or any other store.

---

## 0. Mission in one paragraph

Replace the failed hand-built perception with a substrate built on
**pretrained dense point tracking**. Objects are bundles of tracks
that move together; identity rides on tracks (through carries and
occlusion); events are rest→move→rest transitions derived in pure
code; the agent identifies itself by causality. Gate G1: **≥0.95
position-verified event recall AND ≥0.80 verified precision AND ≥0.90
slot consistency** on `scripts/chain_grade.py`'s grading machinery
(sim_chains). Then G2 chain-QbE ≥0.90 with the already-proven
alignment. Then G3: the same binary, zero changes, on the kitchen
corpus. Then G4 compaction, G5 robot tier. Work the gates strictly in
order. Do not optimize anything ahead of its gate.

## 1. Product truth (context, not tasks)

A database for physical AI. (a) Cloud: customers upload bulk video,
retrieve scenarios by long-contextual queries; heavy query-conditioned
compute is allowed at read time on a pruned top-k (owner approved).
(b) Robot: standalone live memory, ms read/write, small models only;
keeps a cheap append-only log of queries+results so the on-device
student can be slowly trained when idle (G5). Everything must be
zero-shot on unseen datasets: same binary, zero config; per-corpus
calibration only via automatic corpus-fitted statistics at ingest.

## 2. Constitution — DO / DON'T (violations = failed work)

DON'T:
1. No VLM / video-LLM / text encoder in ANY online path. (No 2M-param
   VLM exists; VLM capability does not compact. VLMs allowed offline
   only.) No captions or text as an index, ever.
2. No hand-built perception: no background subtraction, no hand change
   detectors, no hand colour/appearance features (Lab, chroma,
   palettes-as-features). Perception = pretrained vision models +
   pure-code geometry over their outputs.
3. No FDNN-V as a foundation (owner: retired, bad experiment).
4. No text identity/naming at write. No camera meta/intrinsics/poses.
   No per-dataset constants in code — every threshold corpus-fitted
   automatically (see §3.4 for the approved fitting methods).
5. Truth sidecars (`data/*/truth.parquet`, meta.json, palettes) are
   EVAL-ONLY. Never read them in any write/read path. Never train on
   them.
6. No position-blind grading. Any recall/precision claim about
   detection MUST use the position+palette verification of
   `scripts/chain_grade.py` (§5.6). Time-window-only graders reported
   0.91–0.98 where truth was 0.48.
7. No raw Otsu on imbalanced/multi-modal distributions. Print the
   histogram first; use `chain_delta.k_means_1d` midpoints,
   percentiles, or scale anchors (all importable).
8. Never reuse a cache across a code change that affects its content
   (a whole bug class last session). Cache filenames must embed a
   content-affecting version string you bump manually.
9. No whole-corpus caching that fakes elision; no fabricated numbers —
   every number comes from a runnable script and is appended to
   BENCHMARKS.md.

DO:
1. Teacher = same architecture, larger; compaction preserves structure
   (fewer params, same family). Every representation-path model must
   have an in-family smaller variant or within-family distillation
   story. Exploratory teachers must run on this machine (Apple
   Silicon, MPS) and be compactable.
2. Derived logic (events, contact, support, slots) is code, not
   weights.
3. tqdm bar on every loop over episodes/frames (wrap the iterable;
   model loads BEFORE the bar, announced). Checkpoint long jobs every
   ~10 units so an interrupt costs minutes.
4. Commit after every working stage and every gate, message states the
   DESIGN decision. Append every measurement to BENCHMARKS.md.
5. Keep docs/memory/todo.md, decisions.md and docs/logs/ updated as
   you go (project-memory discipline).

## 3. Environment facts + API cheat sheet (verified this week)

### 3.1 Paths and data
- Repo: `/Users/sudharshanramesh/Studies/MyProjects/StreetDex`.
- Python deps importable after:
  `sys.path.insert(0, ROOT/"python"); sys.path.insert(0, ROOT/"scripts")`.
- Sim corpus: `data/sim_chains/` (150 episodes, `ep*/cam*.mp4`,
  `truth.parquet` EVAL-ONLY). Store: `lake/sim_chains` — 150 episodes,
  2 views (`simA`,`simB`) on a SHARED clock, 76,154 frames total,
  640×480 @ 10 fps, ~25–40 s/episode.
- Kitchen (for G3): `lake/fresh_bench` (+ bridge stores). Committed
  QbE-by-example results to beat: q04 0.94 / q05 0.90 (BENCHMARKS
  2026-08-02; locate the runner via `grep -rl "q04" scripts/`).
- Scratchpad for caches:
  `/private/tmp/claude-501/-Users-sudharshanramesh-Studies-MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/scratchpad`
  (or the current session's scratchpad; NOT /tmp).

### 3.2 Store API
```python
from elidedb import Store
db = Store.open(str(ROOT/"lake/sim_chains"))
ep = db.table("episodes").scan().to_pydict()
# episodes: ts,t1 (ns), episode_index, stream; truth t0/t1 are SECONDS
# from episode start: te_s = (t_ns - ep_ts[e]) / 1e9
```
Decode one episode-view (frames table is grouped by source; one source
= one episode-view):
```python
import chain_delta as cd            # scripts/chain_delta.py
views = cd.episode_views(db)        # [(episode, stream, frame-rows)]
ts, F = cd.decode_view(db, rows)    # ts int64 ns, F uint8 (T,480,640,3) RGB
```
Write new tables with `db.table(name).replace(arrow_table, kind=...,
meta={...})` and `set_layout` (see build_dinov3.py for the pattern).

### 3.3 Grading + benchmark machinery (REUSE, do not rewrite)
- `scripts/chain_grade.py`: `PAL` (name→RGB), `frac_color(crop, rgb)`
  (fraction of crop pixels within 60 milli-angle of a palette colour,
  magnitude>60), `THR = 0.08`, and `label_all(db, events, cols_of)`
  which needs event rows `[ep, sv, t_ns, cx, cy, ...]` (only indices
  0–4 are used) and returns per-event `{block: colour-fraction}` for
  a before-crop and an after-crop (±30 px at (cx,cy), before/after
  windows from `chain_delta.grid_params`). **Native grading (§5.6)
  imports these; it must use its OWN label cache file.**
- `scripts/chain_moves.py`: `bench(seqs, tmpl, label, targets,
  w_slot)` — the frozen protocol (RandomState(0), 5 seeds/template,
  k=ceil(1.5·support)). Needs all 150 episodes present in `seqs`.
- `scripts/chain_qbe.py`: `align(seq_a, seq_b)`; set
  `chain_qbe.W_KIND = 0.5` before benching. Token format (exact):
  `((kind_str, q1:int, q2:int), slot:int, dur_s:float, None)`;
  gap tokens `("G", gap_q, 0)` with slot −1. See
  `chain_serial.tokenise` for a working reference implementation.
- Templates: DEV = ("swap","precarious","push_then_build",
  "build_unstack_move"); HOLDOUT = ("relocate_build",
  "two_sites_merge"). `tmpl = {episode:int -> template}` from
  truth.parquet (EVAL side).
- `truth.parquet` columns: episode, template, i, prim
  (pick/place/stack/push/…), block, shape, color, ok, state_ok,
  achieved, violations, t0, t1 (seconds). A "manipulation" = one
  non-pick prim; ~3.5/episode; 525 set-downs, 450 picks corpus-wide.

### 3.4 Approved fitting methods (importable from chain_delta)
`k_means_1d(vals, k)` → sorted centroids; `two_means(vals)` →
(midpoint, c0, c1). Fit cuts on log-transformed values when the mass
spans decades. Always print the fitted value + the mode locations.

## 4. Architecture (what you are building)

```
[T] tracks    pretrained point tracker, grid-seeded, per episode-view
[O] objects   co-movement clustering of tracks -> bundles; life cycle
              rest/move segments; identity = the bundle, forever
[A] agent     the bundle that moves in (nearly) every motion burst and
              whose onset precedes the others' (causality)
[E] events    per non-agent bundle: rest->move->rest = one
              manipulation (dep at burst start, arr at burst end);
              push = burst with small net displacement; slots = first-
              appearance order of bundles     [pure code]
[S] masks+appearance (NOT needed for G1/G2; cloud/G3+): SAM-family
              masks prompted by bundle points at rest; appearance vec
              from a small SSL encoder over mask crops
[K] kind      (cloud-only, optional) V-JEPA-class event embedding
```
Files: `native/track.py` [T], `native/objects.py` [O+A],
`native/events.py` [E + tokens], `native/grade.py` (adapter to
chain_grade), `native/bench.py` (adapter to bench), `native/write.py`
(store tables), later `native/appearance.py` [S], `native/qlog.py`
(G5). Each file runnable standalone with `--store` and `--limit`.

## 5. G1 — the tracker substrate (do nothing else until it passes)

### 5.1 Load the tracker
Primary: CoTracker3 offline via torch.hub:
```python
import torch
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
# if that entrypoint name errors: print(torch.hub.list("facebookresearch/co-tracker"))
# and use the offline cotracker3 entry listed there. Fallback: "cotracker2".
```
Device: try MPS; set `PYTORCH_ENABLE_MPS_FALLBACK=1`. If MPS produces
NaNs or unsupported-op crashes, run CPU — G1 is offline, correctness
beats speed. fp32 only (fp16-on-MPS NaN trap is a banked lesson).

### 5.2 Smoke test (episode 0, simA) — before any loop
Track with `grid_size=30` on the full 253-frame clip (offline mode
tracks the whole clip; API: `pred_tracks, pred_visibility =
model(video, grid_size=30)` with `video` float tensor (1,T,3,H,W),
values 0–255). Print shapes and wall time. Overlay 50 random tracks
on 3 frames, save PNG to scratchpad, LOOK at it (Read the image):
tracks on blocks must follow the blocks through a pick-place. If the
tracker is unusably slow (>120 s/view on this machine), retry at
half resolution (320×240) and/or temporal stride 2 (5 fps); timing
precision needed is only ~0.3 s. Record the chosen setting; it is
then FROZEN for the whole corpus.

### 5.3 Full-corpus tracking (`native/track.py`)
All 300 episode-views, tqdm, checkpoint every 10 views to
`scratch/native_tracks_v1.npz` parts (version string in name). Store
per view: tracks (N,T,2) float16, visibility (N,T) bool, ts (T)
int64. Projected cost printed at start from the smoke timing.

### 5.4 Objects + agent (`native/objects.py`)
Per view, from tracks alone:
1. Per-track windowed displacement (0.5 s windows). Pool corpus-wide,
   fit move/rest cut: `two_means(log10(disp+eps))` midpoint — print
   histogram + modes first (Constitution DON'T-7).
2. Motion bursts: per track, moving-mask; per view, a burst = a
   connected time span where ≥ some tracks move (span merging with a
   fitted gap, reference `chain_moves` gap logic).
3. Bundling: for each burst, cluster the moving tracks by velocity
   coherence (correlation of per-window velocity vectors > fitted
   cut) + spatial adjacency (distance < fitted radius from the
   corpus's spatial scale); connected components = per-burst groups.
   Merge groups ACROSS bursts by shared track ids → bundles. A
   track's bundle is its identity for the whole episode.
4. Agent: the bundle moving in the largest fraction of bursts; verify
   with causality (its onset time ≤ other movers' onsets in ≥80% of
   bursts). Print per-episode agent stats. If two bundles tie,
   merge them (an arm is articulated).
5. Diagnostics per episode (print + save): #bundles, #bursts,
   bundle sizes, agent burst-fraction. Expect ~3–6 non-agent bundles
   (the blocks) on sim_chains. If you see 20+, the velocity-coherence
   cut is too loose — refit, do not hand-tune.

### 5.5 Events (`native/events.py`)
Per non-agent bundle: each of its motion bursts = one manipulation:
`(t_dep, t_arr, bundle_id, pos_dep, pos_arr, net_disp)` with
positions = median track position over the 1 s of rest before/after
the burst. Push = net_disp below a fitted cut (two_means on log
net_disp: carries are long, pushes short) — token kind "P". Slots =
first-appearance order of bundle ids (identity is free from tracks —
this is the structural win over everything that failed). Cross-view:
the two views share the clock; merge manipulations whose spans
overlap ≥50% (union span, keep both views' positions). Tokens exactly
per §3.3, gaps per `chain_serial.tokenise`.

### 5.6 The gate (`native/grade.py`) — position-verified, non-negotiable
Build grading rows for every claimed event: departures
`[ep, sv, t_dep, x_dep, y_dep]`, arrivals `[ep, sv, t_arr, x_arr,
y_arr]`. Call `chain_grade.label_all` (own cache file:
`native_labels_v1.npz`). Verification rule (from chain_grade, THR
0.08): claimed ARRIVAL verified iff ∃ block b: after-frac[b] > THR
and before-frac[b] < THR/2; DEPARTURE symmetric. Then:
- **Recall**: fraction of truth set-downs (episode e, block b, anchor
  a1=t1 seconds) with a verified arrival OF THAT b within
  |te − a1| < 3.0 s in ≥1 view. Same for picks/departures.
- **Precision**: fraction of claimed events that verify (any block).
- **Slot consistency**: among verified events, the fraction of
  same-bundle pairs that verify to the same truth block.
**G1 passes iff set-down recall ≥0.95 AND pick recall ≥0.95 AND
precision ≥0.80 AND slot consistency ≥0.90.** Report all four to
BENCHMARKS.md with the fitted cuts. Iterate ONLY the fitted-cut
methods and bundling logic to get there; if stuck below 0.90 recall
after refits, produce a per-failure dump (which truth prims lack
verified events; look at 5 with saved crops) before changing design.

Failure branches: tracker misses blocks entirely in the overlay
(5.2) → try grid_size 40 / full res; tracks drift off carried blocks
→ use the tracker's occlusion-aware predictions (do not gate on
visibility); MPS OOM → chunk the clip in overlapping windows only if
unavoidable (offline whole-clip is strongly preferred; reduce
resolution first).

## 6. G2 — chain-QbE ≥ 0.90
Run `chain_moves.bench` on the native tokens (chain_qbe.W_KIND=0.5,
w_slot=1.0), DEV then HOLDOUT. **Gate: HOLDOUT mean yield ≥0.90 with
everything frozen before looking at HOLDOUT.** If G1 passed but G2 <
0.90, localize with the proven instrument: substitute oracle slots
(truth blocks) into your manipulation spans → if that scores ≥0.95,
the gap is slots (bundling); if not, the gap is spans/segmentation.
Fix the located stage only. The aligner itself is proven (0.992 on
oracle scripts) — do not modify chain_qbe.

## 7. G3 — zero-shot on the kitchen corpus
Same binary, ZERO code/config changes: run [T]→[O]→[E] on
`lake/fresh_bench` (its frames table uses the same store API). Write
the native objects/events tables (new names, e.g. `nat_objects`,
`nat_events`). Then re-run the committed QbE-by-example benchmarks
(q04/q05; find the runner via BENCHMARKS 2026-08-02 + grep) with the
native-event channels replacing the old event channel. Acceptance:
q04 ≥ 0.89, q05 ≥ 0.85 (committed − 0.05 tolerance). Report deltas
honestly. Any code change needed to make kitchen run = a zero-shot
failure; log it, fix it generically, re-run BOTH corpora.

## 8. G4 — throughput + compaction
Measure ms/frame per stage on both corpora. Edge targets (owner to
confirm, work to these until then): write ≤10 ms/frame, total
representation-path params ≤30M. Compact in-family: tracker → smaller
same-family variant or within-family distillation; encoder ViT-S→
tiny. Re-run G1+G2+G3 with the compacted students; report the
compaction cost as its own BENCHMARKS row (numbers may drop — that is
information, not failure).

## 9. G5 — robot tier
`native/qlog.py`: append-only `query_log` table (ts, query
representation, channel scores, returned spans, feedback signal if
any, compute budget). Cheap write (<1 ms), local-only. Idle-time
trainer skeleton that consumes the log to fine-tune the on-device
student slowly; must never block reads. Demo: log 100 synthetic
queries, run one training pass, show the student changed and reads
stayed fast.

## 10. Measured lessons — do NOT relearn (numbers are committed)
| Lesson | Number |
|---|---|
| Hand-built detection true recall (position-verified) | 0.48 |
| Time-window graders' false recall | 0.91–0.98 |
| Oracle scripts → aligner (retrieval side is SOLVED) | 0.992 |
| Degraded oracle: slots+push only (ON-relations unneeded) | 0.992 |
| Ceiling with truth slots at 0.48 recall | 0.23/0.35 |
| Encoder identity on detector-quality crops (DINOv3 & MAE) | ~0.53 AUC |
| DINOv3 colour-blindness (cross-colour vs same-object cos) | 0.85–0.90 vs 0.73–0.84 |
| Channel plurality ceiling on bad events | 0.41/0.43 |
| Cross-view InfoNCE at 150 eps → instance fingerprints | 296/300 train, 0.25–0.38 retrieve |
| Per-event scalar junk features max | 0.72 AUC |
Every clever decoder (seriality DP, fusion, PRF, profiles) is flat on
bad primitives. Primitives first; the gates enforce this.

## 11. Inventory (reuse; never rewrite)
`scripts/chain_grade.py` (gate), `chain_qbe.py` (aligner),
`chain_moves.py` (bench+otsu), `chain_channels.py` (channel pool +
seed-LOO selection), `chain_fuse.py` (fusion recipes),
`chain_delta.py` (decode/episode_views/k_means_1d + superseded
detector), `chain_serial.py` (tokenise reference), `sim_chains.py` /
`sim_stack.py` (corpus generator). Negative-result records:
`chain_ledger.py`, `chain_2v.py`, `chain_slotline.py`, `chain_hand.py`.

## 12. STOP and ask the owner when (and only when)
1. A required model repo is gated/needs login (never work around it).
2. G1 still fails after the per-failure dump of §5.6 and one
   full refit cycle — present the dump, do not thrash.
3. Any step would delete/modify data outside `native/` + the allowed
   appends.
4. Edge budgets (G4) need confirmation before compaction targets are
   locked.
5. Disk pressure: corpus caches exceeding ~20 GB.

## 13. Open questions already posed to the owner (non-blocking)
1. Edge budget numbers (proposed: ≤10 ms/frame, ≤30M params).
2. G3 acceptance thresholds (proposed: q04 ≥0.89, q05 ≥0.85).
3. Robot feedback signal available for labeling replay data.
