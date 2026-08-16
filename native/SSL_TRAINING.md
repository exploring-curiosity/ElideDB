# SSL_TRAINING.md — self-supervised training of `f` (the moment head)

Runbook. Written so any agent or person can execute or resume it without this
conversation. Every stage states its command, its artifact, its verification,
and its abort condition. Check off stages in the STATUS table as they land.

## 0. What is being trained, and why

The shipped head (`p1_reg_low_*`) was trained discriminatively on 291 labeled
rcasa records. Measured consequence: its 128-d output collapses to ~3.8
effective dimensions (participation ratio of the covariance spectrum) and its
precision drops 43% on task families it never saw, while the frozen channels
drop only 5-8%. The encoder generalizes; the head is the bottleneck, and its
objective — separating 12 tasks — is why.

This run replaces the objective. `f` is retrained on a broad unlabeled corpus
with restoration/correspondence objectives that manufacture supervision from
the video itself. No labels, no manifests read at train time, no VLM, ever.
The bet, falsifiable: *a restoration objective on broad video will not fit in
4 dimensions.*

## 1. Standing rules (owner-set, non-negotiable)

- **No labels at train time.** Task names, scene ids, camera ids: never read
  by the training loop. (Eval may read them to GRADE, never to retrieve.)
- **No VLM anywhere.** No error channel (`pred - act`): at most one of a/b
  enters any layer as a vector. **No cross-view positives** (same rollout,
  different camera). **No tempo-invariance training** — a 7s and a 20s event
  are different events; do not warp time as augmentation.
- **No hardwired domain priors** (verb maps, canonical vocabularies).
- **Seed rule:** report mean±sd over 3 seeds. A best checkpoint is not a result.
- **Ingest rule:** every bulk job prints an ETA before starting, gets a
  tqdm bar, a 20-minute watcher, and is verified by counting ARTIFACTS on
  disk, never by exit code.
- **Eval seals:** `rcasa_composite_full` is THE eval and is never trained on,
  never mined, never PCA-fitted. `prim_actions_v2` stays sealed (older eval).

## 2. Corpora

| dataset (registry name) | source | hours | role |
|---|---|---|---|
| rcasa | already ingested | 1.78 | train |
| rcasa_eval | already ingested | 0.26 | train |
| rcasa_atomic_full | already ingested | 5.68 | train |
| bridge_wide | sampled from data/bridge (LeRobot v3) | 5.0 | train + text layer |
| movi_e | data/relmo/datasets/movi_e (1,616 mp4) | ~1 | train |
| kitti_seq | data/kitti PNG drives → mp4 @10fps | ~0.8 | train |
| oxford_seq | data/oxfordDataset PNGs → mp4 | ~0.03 | train |
| drone_fpv | data/drone zips (uzhfpv ×2, rpg_race) → mp4 | ~1.5 | train |
| **rcasa_composite_full** | already ingested | 11.9 | **EVAL ONLY** |
| arctic | NOT ON DISK (manifest only) | — | later, needs download |

Bridge sampling law: group 50,415 episodes by task string (22,199 distinct),
take ONE episode per task (camera image_0 only), round-robin until 5.0 h.
Wide beats deep. The (instruction, episode) pairs are saved to the manifest
for the text layer — they are used by the TEXT adapter only, never by `f`.

Total training pool ≈ 16 video-hours across sim-kitchen, real-robot, synthetic
objects (movi), driving (kitti), egocentric-drone. rcasa_composite is the only
graded eval and is 91% unseen-task relative to rcasa.

## 3. Stage A — data prep (CPU, ~45 min)

Script: `native/relmo/prep_data.py` (subcommands: bridge, kitti, oxford,
drone). Each writes `data/relmo/datasets/<name>/` with per-episode
`frames.mp4` + `manifest.json` in registry format
(`{"fps": F, "episodes": [{"id", "video", "fps", "T"}]}`).

```
python3 -m relmo.prep_data bridge   # cut 5h wide sample, re-encode x264
python3 -m relmo.prep_data kitti    # PNG drives -> mp4 @10fps
python3 -m relmo.prep_data oxford   # PNGs -> one mp4 @16fps
python3 -m relmo.prep_data drone    # unzip + seqs -> mp4 @30fps
```

VERIFY: each manifest's episode count == mp4 files on disk; total duration
printed and within 10% of the table above. movi_e needs no prep (check its
manifest loads via `R.read_manifest("movi_e")`).
ABORT a corpus (do not block the run) if its source is malformed; note it here.

## 4. Stage B — ingest (GPU, ~2.2 h)

One-pass writer, per corpus. Resumable: reruns skip files already on disk.

```
python3 -m relmo.vjrec8 --dataset bridge_wide --fp16 --pca-suffix ""
python3 -m relmo.vjrec8 --dataset movi_e      --fp16 --pca-suffix ""
python3 -m relmo.vjrec8 --dataset kitti_seq   --fp16 --pca-suffix ""
python3 -m relmo.vjrec8 --dataset oxford_seq  --fp16 --pca-suffix ""
python3 -m relmo.vjrec8 --dataset drone_fpv   --fp16 --pca-suffix ""
```

Print total ETA first (8.6 vh x 14.4 min/vh ≈ 2.1 h), arm a 20-min watcher.
Known decisions: token PCA and predictor calibration stay rcasa-fitted for v1
(fixed linear maps, identical everywhere; refit = v2 + full re-ingest).
Recordings shorter than one 4 s window are dropped by design; expect drops in
bridge (some episodes < 4 s at 5 fps).
VERIFY: `vjrec6/<ds>_L6`, `vjrec7/<ds>_L6`, `vjsig6/<ds>` counts match the
manifest minus too-short.

## 5. Stage C — train `f` (GPU/MPS, ~2.5 h for 3 seeds)

Module to write: `native/relmo/vjssl.py`. Inputs per recording, straight from
the store: `tok (T,256,96)`, `gate (T,256)`, `g (T,2)`, `sig (T,768)`,
`fix (T,1024)`. Architecture = vjrank2.Ranker with `d=256` (z widened from
128; the anti-collapse loss is what makes the width real).

Objective, three terms, weights in brackets:

1. **Span prediction [1.0]** — mask a contiguous span of 8-24 steps (2-6 s);
   predict the span's pooled representation from the unmasked prefix state
   z_t, against a stop-gradient EMA target encoder (momentum 0.996). Loss:
   negative cosine (BYOL-style). Single-step gaps are banned as spans —
   interpolation must not solve the task.
2. **Overlap correspondence [1.0]** — two crops of the SAME recording's trace
   that overlap by 25-75% must map to nearby z at the shared steps; crops
   from different recordings are negatives (InfoNCE, temp 0.1). Same-rollout
   pairs (other cameras of one demo) are EXCLUDED from positives AND from
   negatives (cross-view bar).
3. **VICReg anti-collapse [0.5 var / 0.1 cov]** — variance floor 1.0 per
   z-dimension across the batch; covariance off-diagonals penalized.

Training-time GATE, checked every epoch on a held-out 10% of the pool:
participation ratio of z covariance. Target ≥ 30 by epoch 5; if it plateaus
< 15, stop — the objective is not escaping collapse and the eval would be
noise. (The shipped head sits at 3.8; frozen fix reaches 26.)

Seeds 0/1/2, tags `ssl_v1_s{0,1,2}`, saved to `models/vjrank/`. Log every
epoch: three losses, PR(z), lr. AdamW 3e-4, cosine to 3e-5, batch 24 crops,
~40 epochs or 45 min per seed, whichever first.

## 6. Stage D — mining round (optional, +~2 h, only if Stage E shows signal)

Using seed-0 z: candidate pairs across DIFFERENT videos = mutual top-10 under
pooled cosine. Keep a pair only if (a) fix-channel AND sig-channel cosine both
above their corpus 99th percentile, (b) DTW alignment cost in the top 1% of
candidates. Add as positives to term 2, retrain 3 seeds as `ssl_v1m_s*`.
Wrong positives poison; abstention is free.

## 7. Stage E — eval (frozen, ~1.5 h)

```
python3 -m relmo.vjtest --snapshot v10 --dataset rcasa_composite_full \
    --queries 150 --breakdown rcasa
```

First cut snapshot v10 (vjsnap) pinning the ssl checkpoints. Protocol is
vjtest's: full pool, sampled queries, chance+lift reported, 3 seeds.
Bars, all on rcasa_composite_full, measured 2026-08-16 (single seed s0,
120 q, atomic — treat as context, not gospel; composite bars TBD this run):
shipped head unseen-task prec 0.248; frozen fix+sig 0.286; whitened fusion
0.312; three-way fusion 0.314. Success = ssl z beats the 0.31 band on
composite AND PR(z) ≥ 30. Also run rcasa in-domain to confirm nothing
regressed below the frozen floor there.

## 8. Text retrieval layers (read-side, after Stage E)

- L1 state match (zero train): SigLIP text tower vs stored per-step `sig`.
- L2 transition match (zero train): query phrased before→after; score spans
  by cos(Δsig_t, embed(after)−embed(before)). Direction comes from the trace.
- L3 verb adapter (deferred): only if L1+L2 verb recall is short. Trained on
  noun-stripped instruction strings from ≥2 sources (bridge_wide manifest +
  SSv2 templates), output = unsupervised motion prototypes of the union
  store. Never single-source.
- Query decomposition: UI asks for before→after phrasing (owner choice (a)).

## 9. Failure modes already known

- Background wrapper starting in repo root → `No module named relmo`; always
  `cd native` first. Exit 0 is not success; count artifacts.
- zsh glob on empty dir kills watchers; count with `find`.
- MPS fp16: ViT NaN trap seen before with fp16 ViT-S — if NaNs, first check
  dtype boundaries, not the loss.
- DTW eval cost scales with sequence length SQUARED: composite ≈ 12 s/query.
  Budget queries accordingly (150, not 400).
- A state-carrying variant once "won" via leakage: any surprise win must
  survive a reseed and a shuffled-corpus control before being believed.

## 10. Artifacts

| artifact | path |
|---|---|
| this runbook | native/SSL_TRAINING.md |
| prep script | native/relmo/prep_data.py |
| ssl trainer | native/relmo/vjssl.py (Stage C writes it) |
| new corpora | data/relmo/datasets/{bridge_wide,kitti_seq,oxford_seq,drone_fpv} |
| records | data/relmo/{vjrec6,vjrec7,vjsig6}/<ds>* |
| checkpoints | data/relmo/models/vjrank/ssl_v1_s{0,1,2}.pt |
| snapshot | data/relmo/snapshots/v10.json |
| text pairs | bridge_wide/manifest.json episodes[].task |

## STATUS

- [ ] A: prep (bridge_wide / kitti_seq / oxford_seq / drone_fpv verified)
- [ ] B: ingest (5 corpora verified on disk)
- [ ] C: train 3 seeds (PR gate passed)
- [ ] D: mining round (skipped unless E shows signal)
- [ ] E: eval vs bars, snapshot v10 cut, numbers attached
- [ ] text L1/L2 measured
