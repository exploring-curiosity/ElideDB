# ElideDB

**Video memory. Ask "when did something like this happen?" and get timestamps back.**

No labels. No captions. No fine-tuning. No per-dataset configuration. Point it
at video, then query it with more video.

```bash
elidedb add     kitchen  /videos/robot_runs      # ingest
elidedb query   kitchen  /clips/spill.mp4        # ask
```
```
 #   match             span  source
------------------------------------------------
 1   41.2%    0.0-  23.2s  run_0114/frames.mp4
 2   38.7%    0.0-  13.0s  run_0088/frames.mp4
 3   31.1%    0.0-  14.4s  run_0203/frames.mp4

3 hits in 554 ms
```

---

## What it's for

**A memory layer for a robot / VLA.** The robot asks "have I been in a
situation like this before?" and gets its own past experiences back, ranked,
in about half a second. Nothing has to be labelled first, and it works on
scenes and tasks the system has never seen.

**Mining your own archive.** Find every past instance of a behaviour to review
or retrain on, without having tagged any of it.

## Performance

Measured on a held-out corpus where **91% of the tasks were never seen during
setup** (1,152 recordings, random-guess baseline 2.9%):

| you ask for | you get right |
|---|---|
| top 1 | **95.0%** |
| top 5 | **88.6%** |
| top 10 | **79.8%** |
| top 20 | 56.7% |

| operation | cost |
|---|---|
| query (3,556-recording store) | **554 ms** median, 1.9 s p99 |
| query, robot's own live trace | search only — nothing to encode |
| ingest | ~14 min of compute per hour of video (4× real time), resumable |
| storage | ~0.8 GB per hour of video |

Precision holds *equally well on unfamiliar material* — on the held-out corpus
it scores marginally higher on tasks it has never seen (0.404) than on
familiar ones (0.375). There is no model trained on your data, so there is
nothing to drift, retrain, or version when you point it somewhere new.

**Where it is weak, stated plainly:** precision falls off past the top ~20.
If you need to retrieve *every* instance of something (say 35 of 35), it
currently finds about 40%. It is strong at "show me the closest matches",
not yet at "show me all of them".

## Install

Requires Python 3.11+, `ffmpeg`, and ~8 GB RAM. Apple Silicon (MPS) or CPU;
CUDA untested.

```bash
git clone <this repo> && cd StreetDex
pip install torch transformers numpy tqdm
brew install ffmpeg            # or apt install ffmpeg
cd native
```

Models download automatically on first use (V-JEPA 2 ViT-L, SigLIP 2 base;
~1.5 GB total, cached).

## Use it

### Command line

```bash
python -m relmo.cli stores                       # what stores exist
python -m relmo.cli add   kitchen /videos        # ingest a folder
python -m relmo.cli stats kitchen                # size, hours, median length
python -m relmo.cli query kitchen clip.mp4 --top 10
python -m relmo.cli query kitchen long.mp4 --start 120 --end 145
python -m relmo.cli like  kitchen <recording-id> # query with something stored
python -m relmo.cli query kitchen clip.mp4 --json  # machine-readable
```

Add `--exact` to any query for the exhaustive scan: ~40× slower, about 1.3
percentage points more accurate. The default fast path is recommended.

### Python

```python
from relmo.api import Memory

mem = Memory.open("kitchen")
mem.add("/videos/robot_runs")              # ingest, resumable

for hit in mem.query("/clips/spill.mp4", top_k=10):
    print(f"{hit.score:.0%}  {hit.label}  {hit.start:.1f}-{hit.end:.1f}s")

# the robot's own case: the trace is already in memory, so no re-encoding
hits = mem.query_recording("run_0114", top_k=5)

mem.stats()          # {'store': 'kitchen', 'recordings': 812, 'hours': 6.4, ...}
Memory.list()        # {'kitchen': 812, 'warehouse': 2401}
```

`Hit` carries `id`, `score` (similarity, 0–1), `video`, `start`, `end`, and
`label` (a human-readable source name).

## Stores

A **store** is one deployment's memory — one robot, one site, one customer.

Stores never mix. Every statistic the search uses is computed inside the
store, so two deployments cannot leak into each other's results. Query one
store at a time; searching across stores means looping over them deliberately.

## How it works

```
video ──► V-JEPA 2 (frozen) ──► what changed, moment to moment
      └─► SigLIP 2  (frozen) ──► what it looks like
                    └──► one trace per recording, ~0.8 GB / video-hour

query ──► same two encoders ──► trace
      └─► pooled prefilter (whole store, milliseconds)
          └─► elastic time-alignment on the top 100 (DTW)
              └─► ranked timestamps
```

Both encoders are **frozen, off-the-shelf, and never fine-tuned**. Matching is
elastic in time, so the same action performed faster or slower still matches —
but not so elastic that a 7-second event matches a 20-second one, which are
treated as genuinely different.

## Limits

- Clips shorter than **4 seconds** cannot be encoded (the model's window) and
  are skipped at ingest with a count.
- Results are whole recordings, not the matching sub-span inside them.
- Text queries ("show me spills") are **not supported** — query by example
  only.
- Single machine. No server, no auth, no replication.
- Search cost grows linearly with store size; ~550 ms at 3.5k recordings.

## Repo layout

| path | what |
|---|---|
| `native/relmo/api.py` | **the public API** (`Memory`, `Hit`) |
| `native/relmo/cli.py` | **the command line** |
| `native/relmo/vjrec8.py` | write path — one encoder pass |
| `native/relmo/vjstore.py` | read path — stores, prefilter, alignment |
| `native/SYSTEM.md` | architecture and measured numbers |
| `native/EXPERIMENTS.md` | every approach tried and what it scored |

Anything in `native/relmo/` not listed above is internal.
