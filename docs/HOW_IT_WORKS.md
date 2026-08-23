# How moment retrieval works

This explains what happens between "here is a clip" and "here are the
timestamps", and why each step is the way it is. Nothing here is required to
use the system; it is here so that the behaviour you see is explainable rather
than magic. The measured basis for every design choice is in
[native/EXPERIMENTS.md](../native/EXPERIMENTS.md).

---

## The problem, stated precisely

Given a store of recordings and a query clip, return the recordings in which
*something like the query happened*, best first.

"Something like" is defined by what changes over time, not only by what the
frames look like. Two clips of the same drawer, one being opened and one being
closed, contain nearly identical frames in the opposite order. A system that
compares frames, or averages frames, cannot tell them apart. This one has to.

---

## Step 1: a recording becomes a trace

Every recording is converted once, at ingest, into a **trace**: a sequence of
vectors, one every two seconds, describing what was happening in that window.

Two frozen, off-the-shelf encoders each contribute a channel:

| channel | encoder | what it carries |
|---|---|---|
| **change** | V-JEPA 2 (ViT-L, `facebook/vjepa2-vitl-fpc64-256`) | how the scene is changing from moment to moment. V-JEPA 2 is a video model trained to predict its own future; the signal used here is its observed change over a window, pooled by where the model's own attention concentrates |
| **appearance** | SigLIP 2 (`google/siglip2-base-patch16-224`) | what things look like: objects, surfaces, layout |

The geometry is fixed and does not depend on the clip:

- video is resampled to **8 frames per second**
- the encoder window is **32 frames, so 4 seconds**
- windows advance by **2 seconds**, so they tile the recording exactly
- frames are centre-cropped to **256 px** for the change channel and
  **224 px** for the appearance channel

A 20-second recording therefore becomes a trace of about 10 steps. Time in
the trace is *stream time*, absolute seconds into the recording, which is
what lets windows tile and lets a step be mapped straight back to a
timestamp.

Both encoders run in half precision. Ingest is one pass: decode, encode both
channels, write. Roughly two thirds of the cost is the V-JEPA 2 encoder.

**Nothing is trained here.** No weights in the store belong to ElideDB. The
measured reason is in the last section.

---

## Step 2: normalisation, the step that matters most

Before matching, each recording's trace is standardised **against itself**:
every dimension is shifted to zero mean and unit variance using that
recording's own statistics, separately for the two channels, and the two are
then concatenated (1024 + 768 = 1792 dimensions) and unit-normalised per step.

Why per recording rather than per store or not at all:

- A recording carries a large constant component (the kitchen, the camera,
  the lighting) that is identical across all its steps. Per-recording
  standardisation removes it, leaving *what varied over time*, which is the
  part that describes events.
- Because it is fitted on the recording alone, it cannot encode anything
  about other recordings or other customers. It is label-free and
  store-isolated by construction.

This is the single largest measured gain in the read path. Representations
that were whitened instead, or not normalised, scored measurably lower, and
normalisation alone accounted for what earlier looked like gains from
learned components.

---

## Step 3: the store, and the prefilter

A **store** is one deployment's memory. All its traces are loaded once and
held as a padded bank; a query addresses one store and never mixes stores.

Comparing the query trace against every trace with full time-alignment is
exact but costs time proportional to (query length × recording length) per
recording. On a store of a few thousand recordings with long episodes that
is seconds per query. Two standard cuts bring it to about half a second
without changing the answer in any measured way:

**Prefilter.** Each recording is summarised by the plain mean of its raw
channels (not the standardised trace: that has zero mean by construction and
would carry nothing). Those pooled vectors are whitened per store, because
raw pooled features sit in a narrow cone where cosine barely discriminates.
One matrix-vector product against the whole store selects the top 100
candidates.

**Band.** Time alignment is restricted to a diagonal band of 25% of each
recording's length. Alignments further from the diagonal than that would
mean the same event happening at wildly different speeds, which the system
treats as a different event anyway.

Measured on the sealed corpus: the exact scan scores P@10 0.765; the
prefiltered, banded default scores 0.752 in 554 ms instead of 20 s. Every
speed change ships with that fidelity column next to it, because two real
defects were caught by exactly that comparison.

---

## Step 4: elastic time alignment

The 100 candidates are scored by **dynamic time warping** between the query
trace and each candidate trace. Each trace is first re-indexed by *arc
length*, the cumulative amount of change, so that stretches where nothing
happens do not dominate the alignment and stretches where a lot happens get
their due.

The alignment is anchored at both ends and symmetric; cost is the mean of
(1 − cosine similarity) along the best path, normalised by path length. The
result is therefore elastic: the same action performed somewhat faster or
slower still aligns. It is deliberately **not** fully tempo-invariant. A
7-second pour and a 20-second pour are different events, and duration is
treated as content rather than as noise to be normalised away. This was a
measured decision: training or matching for tempo invariance made results
worse.

The score shown to the user is one minus that cost, so it reads as a
similarity in [0, 1].

---

## Step 5: ranking and the robot's special case

Candidates are sorted by alignment score and the top k are returned with the
recording id, its source path, and its time span.

When the querier is a robot asking about its own history, the query is a
trace that already exists in the store. Encoding is skipped entirely; only
the search runs. The recording's own rollout is excluded from its results,
so a recording never matches itself or another camera's view of the same
moment.

---

## Why nothing is trained

Every trained component this project built performed better on the data it
was trained on and worse on data it had not seen. That was measured four
times across four architectures: a discriminative head, a world model over
frozen latents, a learned ranker, and two self-supervised objectives as
different from one another as self-supervision allows.

The decisive control was run on a sealed corpus of which 91% of the tasks
had never been seen:

| representation | precision at support |
|---|---|
| frozen change + appearance, standardised, no learned component | 0.400 |
| the same plus a trained head (two different objectives) | 0.404, 0.406 |

The head contributed +0.006 against a confidence interval of ±0.04. It was
inert; the gains previously attributed to it were normalisation.

So the shipped system trains nothing, and that is why its precision on
unseen tasks (0.404) is not below its precision on seen ones (0.375): there
is no home corpus to be biased toward. How adaptation returns without giving
this up, per store and gated on a sealed benchmark, is in
[ROADMAP.md](ROADMAP.md).

---

## What the numbers mean

- **P@k**: of the top k results, the fraction that are the same kind of
  event as the query. Measured on a held-out corpus with random-guess
  baseline 2.9%.
- **Precision at support**: when an event has *n* true instances in the
  store, the fraction of the top *n* results that are correct. This is the
  exhaustive-recall number, and it is where the system is weakest (about
  0.40). The top of the ranking is strong; the tail is not yet.
- **Chance**: support divided by pool size, always quoted beside a precision,
  because a precision over 447 recordings and one over 2,250 are not the
  same distance above chance.

---

## The whole path in one picture

```
                 ingest (once per recording)
video ──► decode @ 8 fps ──► 4 s windows, 2 s hop
            ├─► V-JEPA 2 (frozen, fp16) ──► change channel     (1024-d / step)
            └─► SigLIP 2  (frozen, fp16) ──► appearance channel ( 768-d / step)
                                              └──► trace on disk

                 query (per request)
clip ──► same encoders ──► trace
       └─► standardise per recording ──► 1792-d unit steps
           ├─► whitened pooled prefilter over the store ──► top 100   (ms)
           └─► arc-length DTW, 25% band, anchored ──► scores          (~0.5 s)
               └─► top k ──► (recording, source path, span, similarity)
```
