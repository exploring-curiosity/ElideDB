---
title: Precedent
emoji: 🎞️
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: An agent whose memory is CockroachDB, triaging robot video
---

# Precedent

An agent that watches robot video it has never seen, decides what each clip is,
and gets better at deciding because it remembers what it decided before.

Two pages:

**`/`** is the problem. Type *"taking something out of the drawer"* at 3,402
clips and you get drawers **closing**. `cos("open the drawer", "close the
drawer") = 0.9766`, so to a text encoder those are the same sentence. Hand the
system one clip instead and precision goes from 0.285 to 0.840 against a 2%
chance line. Every result is graded on screen from the query you just ran.

**`/agent`** is the memory. A cold agent escalates 94% of its queue to a human.
With memory on, that falls to 38% while it stays 96.2% correct on what it filed
alone. Turn memory off and it never improves: 100% escalation, forever. Then
overturn one filing and watch the correction reach 34 others through the
precedent graph, in one transaction.

## Where the state lives

```
CockroachDB   inbox, filings, filing_precedents, verdicts
              plus the vectors: stage 1 is one SQL query that
              never scores 98.6% of the corpus
S3            3,556 traces and 3,402 clips, private.
              The ranker fetches the 48 traces a query shortlisted.
              The browser fetches video by presigned URL, never
              through this app.
this Space    a text tower and no corpus at all, about 1.4 GB
```

The encoder that produced all of it is V-JEPA 2 plus SigLIP 2, and it is
offline. Nothing in the request path does a forward pass over video.

Source and the measured numbers: see the repository.
