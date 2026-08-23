# Roadmap

Where ElideDB is today, where it is going, and how customer data compounds
along the way. This document is deliberately honest about limitations; the
numbers behind every claim are in [BENCHMARKS.md](../BENCHMARKS.md) and the
full experimental record is in [native/EXPERIMENTS.md](../native/EXPERIMENTS.md).

---

## Where the system is today

**What works now**

- Query any store with a video clip and get ranked results back in about
  half a second, each with the span inside the recording where the match
  lies.
- Works on day one, on video the system has never seen. There is no setup
  model, no labeling step, and no tuning period. Measured precision is as
  good on unfamiliar tasks as on familiar ones.
- A robot querying its own memory pays no encoding cost at all; its live
  trace is already in the store.
- Ingest is one pass, resumable, and runs at roughly four times real time.

**Current limitations, stated plainly**

- Precision falls off past roughly the top 20 results. The system is strong
  at "show me the closest matches" and not yet strong at "show me every
  instance." Exhaustive recall is the gap that matters most for the
  fleet-mining use case, and it is the top of this roadmap.
- Queries are by example. The v1 text search is deprecated; its
  replacement, a vocabulary layer on top of the memory, is being optimised
  for top-tier performance before it is released.
- Search cost grows linearly with store size.
- Clips shorter than four seconds cannot be encoded.

---

## Why we ship an untrained system today, and why we will not stay there

Every model we trained on a corpus performed better on that corpus and worse
everywhere else. We measured this repeatedly, across four different
architectures, before accepting what it meant: for a memory product that has
to work the day it is installed, generalization beats specialization, so the
version that ships today trains nothing.

But the same measurements contain the growth path. Corpus-aware components
genuinely are better, on the corpus they know. The mistake is not adaptation
itself; the mistake is shipping one adapted model as if it were universal.
The trajectory below captures the upside of adaptation without repeating
that mistake, and it is powered at every stage by customer data.

---

## The growth trajectory

### Phase 1, shipping now: every store adapts to itself, without training

Each store already fits its own normalisation statistics from its own data
at ingest. No gradient descent, no drift, nothing to version. This is why a
store gets sharper as it grows: the statistics that shape ranking are
computed from the customer's own corpus, automatically, and are always safe
because they cannot overfit in the way a trained model can.

### Phase 2, next: per-store adaptation, with guardrails

Small learned ranking components fitted on one store and serving only that
store. In our measurements, corpus-fitted components gained forty percent
in-domain; the failure was only ever in pretending those gains transfer.
Scoping them per store removes the failure by construction. Three guardrails
make this a product feature rather than a liability:

1. A store's adapted ranker serves that store and no other. Isolation is
   already architectural: stores never mix, by construction.
2. An adapted ranker ships only if it beats the untrained baseline on that
   store's own held-out data.
3. The untrained path remains as the permanent fallback, so an adaptation
   can never make a deployment worse than day one.

The customer-visible version of this: *your memory gets better the longer
you run it, and only yours benefits from your data.*

### Phase 3, the flywheel: fleet learning

The decisive finding of our training experiments was that the model was
never the bottleneck; the breadth of the data was. Thirteen hours across two
domains was too narrow for any self-supervised method to improve on the
frozen baseline. A fleet of deployments is exactly the breadth that was
missing.

Customer data feeds growth in three distinct ways, each with its own
consent boundary:

- **Traces, not video.** What a deployment could contribute is the compact
  derived representation, never the footage. Raw video does not leave the
  customer.
- **Aggregate diversity trains the next representation.** Opt-in traces
  across many deployments, embodiments, and environments form the wide
  corpus that single-customer training could not. Every candidate trained on
  it is judged on a sealed benchmark no model ever trains on, and nothing
  ships that scores below the untrained baseline there. That gate is already
  built and already enforced.
- **Query outcomes calibrate confidence.** Which results users accept and
  which they discard is a label-free signal for calibrating ranking
  confidence and abstention, the "know when to say nothing" behaviour that
  buyers value most.

Participation is tiered: local-only by default, per-store adaptation as an
upgrade, fleet contribution as an opt-in with commercial terms that reflect
the value contributed.

### Phase 4, the moat: the distilled encoder

Once fleet-scale training beats the frozen baseline on the sealed benchmark,
the result is distilled into a small, fast encoder. This codebase has
already demonstrated the mechanics at over one hundred times speedup on the
write path. The strategic point is simple: that encoder can only be trained
by whoever has the fleet. The data network effect becomes a model nobody
else can reproduce, and the cost of ingest drops with it.

---

## Near-term delivery items

| item | status |
|---|---|
| Exhaustive recall for fleet mining (the top-20 falloff) | the active work; the read side is fully explored, so the levers are features and data breadth |
| Vocabulary layer: words as an entry point, built on the motion representation | being optimised for top-tier performance; replaces the deprecated v1 text search |
| Continuous ingest pipeline (video in, trace kept, video deleted) | designed, not yet running |
| Per-store adaptation (Phase 2) | next engineering phase |
| Wire the memory layer onto the v1 database engine (Parquet store, elision reads) | engine functional, integration parked while the model layer matures |
| Faster core and a hosted service | the store format is language-neutral by design, so the engine can be rewritten under it without touching data |

---

## What never changes

- Raw video stays with the customer.
- Stores never mix, and nothing computed in one store can influence another.
- No labels are ever required, at ingest or at query time.
- Every shipped change must hold or improve the sealed-benchmark score, so
  generalization is a permanent contract, not a launch-day property.
