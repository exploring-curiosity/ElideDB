# The development of ElideDB

**2026-07-17 → 2026-08-22. 489 commits, five weeks, one question.**

The question never changed: *when did something like this happen?* Everything
else did — the storage format, the language, the encoder, the query modality,
and eventually the belief that any of it needed to be trained.

This document is the honest version. Numbers here come from `BENCHMARKS.md`,
`native/EXPERIMENTS.md` and the commit log, not from memory. Where a claim was
later retracted, the retraction is written next to it.

---

## Act 0 — StreetDex: the best read is the read elided

**2026-07-17. C++20, and a thesis about bytes.**

The project started as a storage engine, not a retrieval system. The founding
premise: multimodal data (video + sensors) is stored so that *every query reads
the minimum possible bytes*. Semantic search was milestone four of five.

What got built:

- **SDX**, a hand-rolled columnar format — header, column chunks, per-chunk
  zone maps (min/max ts, min/max value), footer with the chunk directory at the
  end of the file. A conscious simplification of Parquet, built to prove the
  author understood what Parquet was doing.
- **A frame index** — per-file table of `(pts, dts, byte_offset, is_keyframe,
  gop_id)`, built once at ingest by scanning packets with libavformat. This is
  what makes "decode only the GOPs overlapping `[t0,t1]`" expressible at all.
- **`metrics::CountingReader`** on every I/O path. If a byte was read, it was
  counted. The headline metric was *% of bytes elided*, and it had to be honest.
- **Query-time alignment.** Raw data immutable; rate and interpolation are
  query parameters, not storage commitments.

The first real bug set the tone for everything after it. Video segments ≥1 in
the lab dataset turned out to be **headerless raw MJPEG**, which libavformat
probes by reading ~60 MB — silently destroying the elision number the whole
project existed to report. The fix was to construct the decoder from the codec
id stored in the frame index and never open the container at all. *Read
discipline is not a property you assume; it is a property you measure.*

A second dataset (Oxford RobotCar) was brought in specifically to prove the
engine was not shaped around one rig.

---

## Act I — The Parquet pivot, and SigLIP arrives

**2026-07-18. The owner overrules the design doc.**

> "no custom datatypes, parquet is everything, big-data design"

The custom SDX format — the intellectual centrepiece of Act 0 — was retired in
a day. What replaced it was a Parquet lake with Delta-style commits: tables are
Parquet files, the log is JSON, and `O_EXCL` on the log file *is* the
transaction. File-level zone maps live in the log, row-group zone maps come free
from the Parquet footer. The frame index became a table; media stayed untouched
with byte ranges recorded as columns.

The engine was renamed **ElideDB**, packaged for strangers (`pip install -e .`,
an `elidedb` CLI, a **Desk** browser and a macOS app), and the git history was
rewritten with `filter-repo` to purge 1.5 GB of committed store data.

Then the semantic layer landed, and this is where the retrieval story actually
begins:

| what | number |
|---|---|
| encoder | **SigLIP so400m-patch14-384**, MLX, local, mean-pooled per window |
| clustering | HDBSCAN → 12 clusters + 3.0% noise, used as **IVF cells** |
| text query, k=8 | 3/12 clusters probed, **26.9% of vectors scanned, recall 1.000** |
| query-by-clip, k=5 | 17.1% scanned, **recall 1.000** |

Noise was always scanned, so recall was never sacrificed to the prune. The
mechanism — *learned cells as a coarse quantizer, exact cosine inside* — is the
same two-stage shape the system still uses today, five encoder generations
later.

Secondary indexes followed: an immutable bulk-loaded B+ tree (C++20 and a numpy
twin producing **identical on-disk bytes**), HNSW, and IVF-PQ, all at recall@10
= 1.00. One subtle lesson: an index must **not** bump the data version, because
it is a derived sidecar keyed to the data — the first build self-invalidated by
committing, then never matching.

---

## Act II — Text search, and the ceiling it hit

**2026-07-18 → 07-23. Everything text can do, and then what it can't.**

Compositional queries came first, because plain cosine was too flat: `a AND b`
as a min-pool (a clip missing `b` is *rejected*, not merely scored lower), `NOT
c` as vector subtraction, percentile floors for precision. Then hybrid retrieval
(RRF + rerank), BridgeData2 ingest, and a VLM reranker.

Then the result that shaped the next month — **context retrieval**, measured on
66 held-out windows with an independent VLM judge:

| method | ms/query | judge margin |
|---|---|---|
| appearance (plain semantic search) | 12.9 | +0.276 |
| **caption TEXT, materialised at ingest** | **0.6** | **+0.347** |
| caption **embedding** (SigLIP text tower) — *oracle captions* | — | +0.218 |
| VLM rerank at query time | 2508.5 | +0.306 |

Two findings, both durable:

1. **Materialised context is 4,000× faster than query-time VLM reranking and
   scores higher.** Distil the expensive judge into the index; never call it per
   query.
2. **Caption *text* beats caption *embedding* — even with perfect captions.**
   SigLIP's text tower is trained to sit near images, not near other text. The
   embedding route loses information the raw string still has.

Then the write path got fast. **FDNN-V** distilled SigLIP into a 1.95M-parameter
recurrent video encoder so that *every frame* could be embedded at write time:

| | SigLIP-384 | SigLIP-224 | **FDNN-V** |
|---|---|---|---|
| params | 428M | 428M | **1.95M** |
| ms/frame | 90.3 | 27.7 | **0.270** (~103×) |
| 4 h of video, every frame | ~87 min | ~27 min | **79.1 s (178× real time)** |
| held-out fidelity vs teacher | 1.0 | 0.921 | 0.902 |
| **text top-10 agreement** | 10/10 | 4.3/10 | **0.5/10** |

That last row is the whole act in one line. The student reproduced the teacher's
*visual* space almost perfectly and its *text-alignment* not at all. And the
honest boundary underneath it, graded on held-out labels:

> Singleton instruction retrieval among 2,097 near-identical clips: **0.000
> R@10** for the student, 0.030 for the teacher. Even teacher-everywhere
> shortlist recall@48 was 0.17.

Verified search (recall proposes, a frame-sequence VLM disposes, verdicts cached
into the store — *database cracking*: 15% of the corpus was teacher-embedded
after 100 queries) pushed quality where users looked. But the diagnosis kept
returning: **verbs are erased at both embedding ends.** "open" and "close" sat
at cosine 0.957 in text space, and the visual encoders were no better.

The counter-discovery was the first genuinely video-native channel: a **motion
vector channel** built from delta-appearance, which reached **AUC 0.98** on
direction where video-native encoders failed. It carried *direction, not
content* — which is exactly the half text could not express.

---

## Act III — De-hardwiring: deleting everything I taught it

**2026-07-24 → 07-31. The rules get written down.**

Two owner rulings reshaped the system:

- **"No VLM judges, ever."** A retrieval system that phones a generative model
  at query time is not a retrieval system.
- **NO-HARDWIRE.** Dataset and task priors in code are forbidden. The `CANON`
  list, the hyponym dictionary and the verb maps were all violations — mine, not
  the corpus's.

Deleting them cost real points, publicly: the ledger fell from a **0.41
hand-prior peak to 0.21** the day the priors came out. It was recorded rather
than hidden. The replacements had to be self-recognised: **fitted roles**
(LOQO), **corpus-attested vocabulary** (WordNet × SigLIP2 over the store's own
frames — a kitchen attests *pot*, a street corpus would attest *boat*),
per-query channel informativeness, and PRF anchors.

The climb back was measured a step at a time:

| change | ledger prec |
|---|---|
| de-hardwired baseline | 0.21 |
| SigLIP2 channel + mechanical conjunctive atoms | 0.25 |
| **fitted filter membership** — veto authority learned per query type | — |
| fitted confidence cut (the returned set ends where confidence does) | 0.28 |
| **InternVideo2-Stage2 1B as the `iv2` channel** | **0.39** |
| temporal event NMS + clean object phrases | 0.37–0.38 (35/91 true) |

The single most important *mechanism* insight of this act: **fusion authority is
the binding gap.** Consensus fusion drowns a minority channel — a conjunctive
constraint that votes gets outvoted. *Filter, don't vote.* Binding queries went
from 0 to working the moment the conjunction became a veto instead of a weight.

Also in this act: the **transition anchor** (the corpus supplies the direction
the query cannot — q04 0.72 → 0.92), **DINOv3** for identity (ConvNeXt-Tiny, AUC
**0.9994**, and a fp16 ViT NaN trap caught loudly), a **teacher → student**
distillation (student 88× faster at 64% of quality, then rebuilt to produce all
five elements itself: 4.1× faster write, *identical* quality, q05 0.89 vs the
teacher's 0.58), and a one-pass write that took **535 ms → 26.8 ms per episode**.

And a run of bugs whose shape recurs so often it became a named pattern:

- **The dead branch.** The correct cut was computed, then thrown away by a
  condition that was never true. Three instances in one night.
- **The channel with no caller** — the motion channel had never existed in any
  store.
- **An inverted channel voting at full strength.**
- **A metrics bug where nested Parquet leaf names meant vector bytes were never
  charged** — invalidating every older elision number.
- **A manifest reporting 128 episodes where 447 were on disk.** Rule adopted:
  *verify counts against the filesystem, never a manifest.*

---

## Act IV — Chains, and the wall

**2026-08-01 → 08-06. Building a corpus honest enough to fail against.**

To test compositional retrieval without dataset labels, a simulator corpus was
built where episodes are **programs**: a Panda arm executing 5–8 chain-verified
spatial events, with settled-contact verdicts and expected-vs-actual world state
as ground truth.

Chain retrieval by event-sequence alignment scored **oracle 1.00** — given
correct event units, the matcher is perfect. With *real* units it scored
0.43–0.45, with true recall 0.48. Every attempt to close that gap was measured
and closed:

| route | verdict |
|---|---|
| agent-kinematic event typing | negative |
| hold-alternation tokens | negative |
| motion-segment token programs | ceiling ~0.34 |
| two-view cross-view units | reduces 0.90 to **one** discriminator |
| gap-pairing decidability panel | closed negative |
| slot-activity timeline (route 1) | 14% primitive reliability |
| rest-ledger / object permanence (route 2) | detection layer is the floor |
| persistent-change detection (route 3) | best compliant number, arrival deficit open |

The cap analysis said it plainly: **unit structure is the wall, and per-frame
detection quality is the program's floor.**

Then two audits reframed everything.

**Sim was at chance.** Random baseline 0.207; every encoder 0.21–0.33. The
kitchen corpus was 6–7× chance. Months of sim-side mechanism work had been
tuned against a corpus that carried no signal.

**The retrieval unit was oracle-segmented.** A write-path audit found episode
boundaries were being read from the Bridge dataset's own metadata. The system
had been handed the answer to the hardest part of the problem.

Fresh segmentation-free stores were built, and a full unsupervised
boundary-detection study followed (surprise scoring, V-JEPA prediction error,
optical-flow GEBD cues, GraphGEBD recursive Ncut, exact-DP change-point
detection). It ended with two results that saved months:

- **Segmentation is not the bottleneck.** Oracle boundaries yield 0.267 against
  a 0.207 chance floor. Boundary value is a *cliff at perfect*, not a slope.
- **Uniform overlapping windows beat the learned segmentation** (0.408 vs
  0.317).

Finally, contamination: bridge lacked the cross-session guard, and the
celebrated "close the drawer" result was **retracted** — adjacent-file leakage.
An honest four-dataset position replaced it.

---

## Act V — The reckoning, and the world model

**2026-08-08 → 08-13. If the features are wrong, learn better ones.**

The reset (`native/REBUILD.md`) redefined similarity itself: **actor / action /
goal**. Appearance of a moving region is a near-duplicate channel, nothing more.
No fine-tuning of the backbone, ever.

The hypothesis was a **world model**: train a causal predictor over frozen
latents, and use its *state* — what it takes to predict the next moment — as the
retrieval representation. If the model must know what is happening to predict
it, the state is the description.

Five rounds, honestly laddered:

| round | result |
|---|---|
| v1 causal predictor over frozen latents | QbE +0.117, cross-embodiment +0.07 |
| NoPE retrain | exposes **position leakage** in round 1 |
| multi-horizon targets | attacks the copy task |
| grid input/targets | first metric past frozen |
| **rounds 2–5 aggregate** | **no state variant beats frozen after de-contamination** |

A "surprise win" that did not survive reseeding taught the standing rule: *any
surprise win must survive a reseed and a shuffled-corpus control before being
believed.*

Meanwhile the read-side scorer was pushed hard by atomic diagnosis — camera bias
+0.53, one-block height cells, row-marginal push blindness — each fixed
separately: **P@10 0.416 → 0.565 → 0.612 → 0.723**. The supervised ceiling was
measured at **0.789 / 0.722**, which located the 0.9 blocker in *features and
grading*, not in scoring.

Then the graph: **diffusion re-ranking on the affinity graph gave +37% AP on
holdout (0.396 → 0.546)**, label-free and closed-form, with every refinement on
top of it worth ±0.01 — the diffusion itself was the whole gain. Growing the
corpus took it to 0.597 at 2,660 events and 0.619 at 4,902 while the baseline
stayed flat to three decimals, confirming a scaling law that was nonetheless
**decelerating** (+0.053, then +0.017 per step). (This result matters because it
did **not** reproduce later; see Act VI.)

And then the data-first reset. The tracker everything was built on had a **75%
jump rate on textureless renders** — the sim renders were the noise. A pixel-GT
engine and a tracker-validation gate were made a precondition for any further
training.

The physics route was driven to its end, and every terminal finding was a
refutation:

- **G4 gate unreachable**: an omniscient rigid-body oracle *loses* to
  constant-velocity at short horizons. 0/8 is not proof a model failed.
- **Relational, not trajectory**: on trajectory-matched pairs, REL AUC 0.997 vs
  TRAJ 0.771 — motion-R² is invariant to the exact distinction the world model
  exists to make.
- **Metric, not encoder**: a trained `z` took test 0.526 → 0.737, and lost
  out-of-*domain* (0.908 → 0.804). The physics route was exhausted at ~0.77.
- **Ranker on latents**: latents-only + graded relevance hit test 0.894, a
  *learned scorer lost to plain cosine*, and unseen event **types** collapsed to
  chance.

The shape of the failure was now unmistakable and would repeat once more: **what
is learned in-domain is paid for out-of-domain.**

---

## Act VI — V-JEPA, QbE, and the answer nobody wanted

**2026-08-14 → 08-17. The version ladder that ends in "don't train it".**

A clean rebuild on **V-JEPA 2** (`vitl-fpc64-256`, frozen) as the change channel
and **SigLIP 2** as the appearance channel, versioned v0 → v10:

| version | what it settled |
|---|---|
| v1 | fixed-length context, all three content channels cached |
| v2 | bootstrap CIs and a per-channel time-axis test |
| **v3** | **time axis solved** — warp-invariant at rank 1.0 on every warp |
| v5 | **error channel barred** by owner ruling; content is `pred_change` |
| v6 | span localisation; index cost halved to 3.33× real time |
| v7 | SigLIP2 channel alongside V-JEPA |
| **v8** | **full ground-truth audit: the stated mechanism was false** |
| v9 | "experience as recurrence over expectation and outcome" — **thesis not supported** |
| v10 | learned ranker on latents only — test 0.894, collapses on unseen event types |

Then the geometry was fixed properly. **`t` became stream time** — absolute, so
windows tile exactly and `z` carries across them. **Duration became content**
rather than noise to be normalised away (7 s and 20 s events are *different* —
no tempo-invariance training). `prec@support` and `NDCG@support` became the
standing metric pair, with **chance and lift always quoted**, because a bare
precision is not comparable across pools.

`vjrank2` (learned 256-token pooling + `f`) **met every target** on 3-seed
aggregates. Snapshot `v9` pinned the read path by sha256 — every weight, every
fitted artifact, and, critically, **every flag**, because a snapshot that omitted
its flags had once made 0.42 read as 0.32 with nothing in the record to explain
it.

Then the self-supervised program (`SSL_TRAINING.md`) ran on 5,047 recordings /
13 video-hours across five corpora, with `rcasa_composite_full` sealed as the
eval:

| run | objective | PR(z) | prec (sealed) |
|---|---|---|---|
| ssl_v1 ×3 seeds | span prediction + overlap-InfoNCE + VICReg | 46.7 / 45.1 / 47.8 | fused **0.406** |
| ssl_v2 | + mined cross-video positives, hard negatives | 27.5 | fused **0.404** |

Two objectives as different as self-supervision admits landed **0.002 apart**.
So the control that should have been run first was finally run:

| representation | d | unseen prec |
|---|---|---|
| frozen fix+sig **whitened** | 512 | 0.376 |
| **frozen fix+sig z-scored, NO head at all** | 1792 | **0.400** |
| ssl_v1 `z` + fix + sig | 2048 | 0.406 |
| ssl_v2 `z` + fix + sig | 2048 | 0.404 |

**The trained head contributes +0.006 against a CI of ±0.04. It is inert.**

Every "fusion win" reported before that control was **normalisation and retained
dimensions**, not the model. The earlier "+27% from fusion" claim carries the
same confound and was marked *unproven* rather than quietly kept.

Three more corrections landed at once:

- **PR(z) is a diagnostic, not a target.** The shipped head's 3.8 effective
  dimensions were 3.8 *useful* ones; ssl_v1's 46.7 dimensions encoded less of
  what mattered. A variance floor spreads a representation across axes; it does
  not make those axes semantic.
- **Graph diffusion does not reproduce.** The +50% AP of Act V measured
  **+0.002** here and collapsed at high alpha.
- **PRF / query expansion hurts** — P@10 0.820 → 0.778. The owner called it
  before it ran: *"pointing hands at itself."* Anchors are near-copies of the
  query's own view.

With diffusion, verification re-ranking, PRF, the `where_map` channel and
whitening-a-fusion all measured dead, **every label-free read-side mechanism was
exhausted.** The conclusion written into the ledger: *the binding constraint is
the pool, not the model.*

So what shipped is what was there all along:

> **Frozen V-JEPA 2 + frozen SigLIP 2, per-recording z-scored, 1792-d, matched by
> anchored symmetric2 DTW. Nothing trained.**

| sealed corpus, 91% unseen tasks, chance 0.029 | |
|---|---|
| P@1 | **0.950** |
| P@5 | **0.886** |
| P@10 | **0.798** |
| P@20 | 0.567 |
| P@support | **0.401 ± 0.021** |
| unseen tasks / seen tasks | **0.404 / 0.375** — no generalization gap |

Then the read path was made a product. A **store** became one deployment's
memory (the owner ruled that the train/eval split is experiment-side, not
product-side), and the query got a pooled prefilter plus a Sakoe-Chiba band:

| config | p50 | p99 | P@10 |
|---|---|---|---|
| exact full scan | 20.2 s | 69 s | 0.765 |
| **M=100, band 0.25 (shipped)** | **554 ms** | **1.9 s** | **0.752** |

**36× faster for 1.3 points.** The fidelity column — mandatory beside every
speedup — caught two defects on the way: a prefilter pooled from a
per-recording-z-scored representation (whose mean is the zero vector *by
construction*: P@10 0.765 → 0.007), and a band computed on the **padded** width
rather than per-reference (0.765 → 0.203, latent since v6).

The write path was merged into one encoder pass — **1.72×, bit-identical over
5,688 arrays** — landing at **14.4 compute-minutes per video-hour**.

Finally the surface: a `Memory` / `Hit` Python API, an `elidedb` CLI, and a
README that states where the system is weak. Three internals leaks were closed
in the process — negated DTW cost shown as a score, every result labelled
`frames.mp4`, and internal dataset splits listed as stores.

---

## Act VII — The memory inside an agent

**2026-08-16 → 08-22. Proving it by using it.**

A retrieval number is an argument. An agent that fails without memory and
succeeds with it is a demonstration.

**Brigade** put a persistent simulated kitchen behind a VLA and gave it ElideDB
as its memory layer. LIBERO ran fully autonomous on Apple Silicon (400-episode
benchmark, **98.0%**; pi0.5 4× faster than SmolVLA). Then the measurement that
mattered:

| condition | result |
|---|---|
| memory-driven control, **with** memory | **6/6** |
| same task, **without** memory | **0/6** |
| spatial memory, with / without | **7/7 vs 1/7** |

Two constraints emerged from it. **Persistence costs about half** — a continuous
kitchen works and carries state across 7 tasks, but halves pi0.5's success rate;
carrying the *robot pose* breaks everything. And a **video-only ruling**: nothing
textual is ever stored. That forced the most interesting result of the act —
**latent instruction**, where pi0.5 is commanded by a tensor rather than words.
The memory worked once RelMo was queried for **spans, not tiles**.

**Precedent** then took the same memory into a deployable shape: four tables on
CockroachDB, corpus on S3, a live agent console. Engineering findings, each paid
for:

- 49 serial S3 GETs = **37.3 s/query**; prefetch → **9.1 s**. An N+1 is invisible
  on local disk.
- Repointing the corpus to S3 silently broke the local-only run. *Verify with the
  cloud env unset.*
- `ca-certificates` and `sslrootcert=system` both fail under `psycopg2-binary`;
  ship the cluster root by path.
- **The agent's value is abstention, not tuning.** It ties a hand-tuned script on
  tuning, and four measured-negative actions were deleted. It declines 40% of
  queries with zero junk returned.

On real bridge footage the POC held: direction twins **0.951 / 0.012**, QbE
**0.96 vs 0.80**, agent **0.79**.

The act closed with the open-source cut: MIT, public, history rewritten to LFS,
and **two Hugging Face Spaces** — a text demo, and a **query-by-example Space
that loads no model at query time**, because everything it ranks was computed at
ingest. Cold start: seconds.

---

## What five weeks actually taught

The technical arc is easy to state: *a hand-built columnar format became a
Parquet lake; text retrieval hit a verb-shaped ceiling; a world model was built
to break it and did not; and the thing that finally worked was two frozen
encoders, correctly normalised, matched in time.*

The methodological arc is the more useful one.

**On measurement**

- A speedup without a fidelity column is a regression wearing a speedup's
  clothes. Two real defects were caught exactly this way in one week.
- The pool must always be the whole corpus. An 80-episode pool once flattered a
  result by **10×**.
- Quote chance and lift whenever pools differ. 0.355 over 2,250 and 0.900 over
  447 are not the same distance above chance.
- Never report a best checkpoint. One run read 0.678 where its config averages
  0.603 ± 0.049.
- Verify the **artifact**, never the exit code. An ingest can exit 0 having
  written nothing.
- At 60 queries the CI is ±0.04, so 0.376 / 0.400 / 0.406 are **one band**, not
  a ranking.

**On mechanism**

- Distil the expensive judge into the index; never call it per query (4,000×).
- Filter, don't vote — consensus fusion drowns the minority channel that is
  actually right.
- Normalisation and retained dimensions beat every learned head tried here.
- Whitening helps a single channel and *hurts* a fusion; decorrelation does not
  pay for the dimensions it costs.
- What is learned in-domain is paid for out-of-domain. Measured four separate
  times, on four different architectures.

**On the failure patterns that recur**

- The **dead branch**: correct logic gated behind a never-true condition.
- The **lying manifest**: 447 episodes on disk reading as 128.
- The **lying bar**: a gate encoding a belief the ledger had already refuted.
- The **bar that reads 0/N** while a 4 GB model loads behind it.
- The **tautological positive**: 134/192 mined pairs were a video matched with
  its own re-encode.

`native/EXPERIMENTS.md` exists so none of these are paid for twice. Grep it
before proposing anything.

---

## Where it goes next

### 1. The one open gap: P@support 0.40 → 0.70

UC1 (a robot's memory layer beside a VLA — top-k and latency) **ships today**.
UC2 (mine an archive for every past instance of a failure, to retrain on)
**does not**: precision falls off past the top ~20, and finding 35 of 35 lands
near 0.40.

The read side is closed — diffusion, verification re-ranking, PRF, spatial
channels and fusion whitening are all measured dead, and head training is null
under two very different objectives. **There is no third door.** Two remain:

**(a) Features.** A V-JEPA layer sweep — layer 6 is the shipped gate layer and
the ledger holds no record of it ever being swept — and V-JEPA 2.1 when it
becomes loadable; the ViT-B drop-in is presently unusable, with
`encoder.layernorm` missing (silent random init) and `predictor.proj` at 1664
against an expected 768.

**(b) Data.** 13 video-hours across two domains is the measured binding
constraint. That is what the stream pipeline is for.

### 2. `STREAM_PIPELINE.md` — designed, on hold

> Raw video is a **transient**. It exists only long enough to become a trace,
> then it is deleted.

Four daemons — fetcher, ingester, compactor, trainer — plus an evaluator that
gates on the sealed corpus every 6 hours. They coordinate through marker files
in a work directory: no queue server, no RPC, and **state is re-derived from the
filesystem, never from a manifest**. The disk never holds the corpus; it holds a
sliding window of it, and the model is the only thing that accumulates. Steps 1–2
(fetch → ingest → delete) are a working cycle on their own and worth running for
a week before the rest exists.

### 3. The text layer — designed, not built

Query-by-example cannot be the only modality forever. Three layers, in order,
and the first two require no training:

- **L1, state match**: SigLIP text tower against the stored per-step `sig`.
- **L2, transition match**: the query is phrased *before → after*, and spans are
  scored by `cos(Δsig_t, embed(after) − embed(before))`. **Direction comes from
  the trace, not from the text** — the direct descendant of the transition
  anchor and the motion-vector channel, both of which beat text at exactly this.
- **L3, verb adapter** (deferred): only if L1+L2 verb recall is short, trained on
  noun-stripped instructions from **≥2 sources**, never one, or the vocabulary
  becomes corpus-defined.

### 4. Sub-span results

Results today are whole recordings, not the matching sub-span inside them. The
matcher already computes the alignment path; surfacing it is the smallest
remaining product gap.

### 5. Longer horizon

- **A Rust core** for the read path, on the owner's gate. The store format is
  deliberately language-neutral so this is a rewrite of the engine, not of the
  data.
- **ElideDB as a cloud service**, not a local database — the C++ v1 engine ended
  with the Parquet pivot, and the deployment work in Act VII (S3 corpus,
  CockroachDB metadata, one interpreter in the container) is the first half of
  that path.

---

## Appendix — the system as it stands

```
video ──► V-JEPA 2 (frozen, fp16) ──► what changed, moment to moment  (1024-d)
      └─► SigLIP 2  (frozen, fp16) ──► what it looks like              (768-d)
                    └──► one trace per recording, ~0.8 GB / video-hour

query ──► same two encoders ──► trace
      └─► whitened pooled prefilter over the store (milliseconds)
          └─► anchored symmetric2 DTW, per-reference band, on the top 100
              └─► ranked timestamps
```

| | |
|---|---|
| trained parameters | **0** |
| write | 14.4 compute-min per video-hour (4.17× real time), one pass, resumable |
| read | 554 ms p50 / 1.9 s p99 on 3,556 recordings |
| storage | ~0.8 GB per video-hour |
| stores | 5, never mixed — every statistic is computed inside the store |

**Reading order for the internals:** [`README.md`](../README.md) (product) →
[`native/SYSTEM.md`](../native/SYSTEM.md) (architecture and numbers) →
[`native/EXPERIMENTS.md`](../native/EXPERIMENTS.md) (every measured dead end) →
[`BENCHMARKS.md`](../BENCHMARKS.md) (the full raw ledger, 2,677 lines).
