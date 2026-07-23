# ElideDB — Pilot Guide

ElideDB is a local-first video+sensor database for robotics teams: load your
recordings, every frame is embedded **at write time**, and natural-language
context queries ("closing the drawer", "pick up a green toy and put it in
the bin") answer in tens of milliseconds from the index alone. Heavier
model-based verification runs in the background and is cached, so **the
database gets more precise the more you query it**. Everything runs on one
Apple-Silicon machine; no cloud, no per-minute API, your data never leaves
the box.

## Quickstart

```bash
# environment (Apple Silicon, macOS; ffmpeg via homebrew)
python -m venv env && env/bin/pip install -e .
brew install ffmpeg

# smoke-test the install against the bundled stores
env/bin/python scripts/pilot_smoke.py

# browse + search in the desktop UI
env/bin/python -m elidedb.desk      # opens http://localhost:8787
```

From Python:

```python
from elidedb import Store
db = Store.open("lake/bridge4h")
hits, stats = db.search_context("closing the drawer", k=5)
# each hit: stream, t0, t1 (ns), score, verified flag
res = db.window(hits[0]["t0"], hits[0]["t1"])   # decoded frames + sensors
```

Loading your own data: video files + a timestamp per frame (or fps).
See `scripts/bridge_full.py` for the full ingest pattern (source-pipe
embedding parallel to transcode, one atomic commit per table).

## Measured capabilities (as of 2026-07-23, BridgeData2 corpus)

| capability | 4 h store | 100 h store (1.8M frames) |
|---|---|---|
| ingest + full embed | 79 s (178× real time) | ~21 min embed track; every frame |
| query latency (warm, index-only) | 25–55 ms | 65–84 ms (flat-scan wall; pruning is the known next step) |
| direction queries (open vs close) | rank-1 correct, 0/20 opposite | 9/20 verb-strict, 2/20 opposite |
| query battery (8 mixed queries, top-3) | 16/24 | 13/24 |
| motion index build | 2.5 s | 20 s (50,414 recordings) |
| background verification | 2B screen (AUC 0.86) → 7B judge (AUC 0.91), cached | same |

The 100 h numbers are COLD and index-only with two of four ranking
channels absent (no captions, no ctx events on that store yet) and no
accumulated verdicts — they are the floor, not the ceiling. Episode-label
grading also undercounts: "pull out the drawer" ranked #1 for "opening
the drawer" scores as a miss because the label lacks the word "open".

## Known limitations — read before judging results

1. **Attribute binding is verify-tier, not index-tier.** "Pick up the
   *green* toy" — the index finds pick-up-and-place-in-drawer clips fast,
   but whether the *green* object is the one acted on is decided by the
   background verifier. First query: expect plausible-but-imperfect
   ordering; re-ask after ~30 s for the verified ranking.
2. **The verifier is right ~86–91% of the time**, not always. A "verified"
   tick is a strong signal, not ground truth.
3. **Numbers above are from one robot corpus.** Cross-corpus validation
   (street driving, lab capture) is in progress; expect quality to vary on
   your data until the verdict cache adapts to your query patterns.
4. **Single node, single writer, Apple Silicon.** Tested to ~90 h /
   ~1.6 M frames per store. Not a fleet-scale system yet.
5. **Ingest formats today**: video (anything ffmpeg reads) + per-frame
   timestamps; sensor tables as CSV/Parquet. ROS bag / MCAP not yet —
   tell us if that blocks you and it moves up the list.
6. **Captions** (the lexical search channel) require an offline VLM pass
   (`store.index_context()`), ~1–2 h per 4 h of video. Search works
   without them; recall for uncommon nouns improves with them.

## What we want from the pilot

- Queries that returned wrong results — the query text alone is enough;
  the store logs candidates and verdicts, so failures are reproducible.
- Your data's shape: cameras, rates, formats, typical clip length.
- Which of the limitations above you actually hit.

Every query you run makes the store better (verdicts accumulate and are
training data for the next model iteration) — please query liberally.
