<div align="center">

# ElideDB

**Video memory. Ask "when did something like this happen?" and get timestamps back.**

No labels. No captions. No fine-tuning. No per-dataset configuration.
Point it at video, then query it with more video.

[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-green)](#install)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20%7C%20CPU-lightgrey)](#install)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-query%20by%20example-yellow)](https://huggingface.co/spaces/SudharshanR/elidedb-qbe)

[Live demos](#try-it-without-installing-anything) ·
[Performance](#performance) ·
[Install](#install) ·
[Roadmap](docs/ROADMAP.md) ·
[The story](docs/STORY.md) ·
[Licensing](#license)

</div>

---

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

## Try it without installing anything

| demo | what it shows | cold start |
|---|---|---|
| **[Query by example](https://huggingface.co/spaces/SudharshanR/elidedb-qbe)** | pick a few clips, get their kind back. Loads **no model at all**; everything it ranks was computed at ingest | seconds |
| **[Text search](https://huggingface.co/spaces/SudharshanR/elidedb-demo)** | describe what you want in words. Five text towers, so the first boot is slow | ~15 min |

Both run on the public [demo store](https://huggingface.co/datasets/SudharshanR/elidedb-demo-store)
(1,122 recordings, ~1.3 GB), the same store you get locally with the commands
under [Run the demos yourself](#run-the-demos-yourself).

## What it's for

**A memory layer for a robot or VLA.** The robot asks "have I been in a
situation like this before?" and gets its own past experiences back, ranked,
in about half a second. Nothing has to be labelled first, and it works on
scenes and tasks the system has never seen. A robot querying its own live
trace pays no encoding cost at all.

**Mining your own archive.** Find past instances of a behaviour to review
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
| query, robot's own live trace | search only; nothing to encode |
| ingest | ~14 min of compute per hour of video (4x real time), resumable |
| storage | ~0.8 GB per hour of video |

Precision holds equally well on unfamiliar material: on the held-out corpus
it scores marginally *higher* on tasks it has never seen (0.404) than on
familiar ones (0.375). Nothing in the shipped system is trained on your
data, so there is nothing to drift, retrain, or version when you point it
somewhere new.

### Current limitations, stated plainly

- Precision falls off past roughly the top 20 results. Strong at "show me
  the closest matches", not yet at "show me every instance". Closing this
  is the top of the [roadmap](docs/ROADMAP.md).
- Results are whole recordings, not the exact matching moment inside them.
- Query by example only; text search is designed but not yet built.
- Clips shorter than 4 seconds cannot be encoded and are skipped at ingest
  with a count.
- Single machine. No server, no auth, no replication. Search cost grows
  linearly with store size (~550 ms at 3.5k recordings).

Where each of these is going, and how the system improves as it runs, is in
**[docs/ROADMAP.md](docs/ROADMAP.md)**.

## Install

Requires Python 3.11+, `ffmpeg`, and ~8 GB RAM. Apple Silicon (MPS) or CPU;
CUDA untested.

```bash
git clone https://github.com/exploring-curiosity/ElideDB.git
cd ElideDB
pip install torch transformers numpy tqdm
brew install ffmpeg            # or apt install ffmpeg
cd native
```

Models download automatically on first use (V-JEPA 2 ViT-L, SigLIP 2 base;
~1.5 GB total, cached). Both are frozen, off-the-shelf, permissively
licensed, and never fine-tuned; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

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

Add `--exact` to any query for the exhaustive scan: ~40x slower, about 1.3
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

`Hit` carries `id`, `score` (similarity, 0-1), `video`, `start`, `end`, and
`label` (a human-readable source name).

## Stores

A **store** is one deployment's memory: one robot, one site, one customer.

Stores never mix. Every statistic the search uses is computed inside the
store, so two deployments cannot leak into each other's results. Query one
store at a time; searching across stores means looping over them
deliberately. This isolation is architectural, and it is also the foundation
of the [growth trajectory](docs/ROADMAP.md#the-growth-trajectory): a store
adapts to its own data and only its own.

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

Both encoders are **frozen, off-the-shelf, and never fine-tuned**. Matching
is elastic in time, so the same action performed faster or slower still
matches, but not so elastic that a 7-second event matches a 20-second one;
those are treated as genuinely different.

**There are no ElideDB weights.** Nothing in the shipped system is trained
on your data, which is why the numbers above hold on unfamiliar material.
Why an untrained system ships today, and how adaptation arrives without
giving that property up, is the subject of [docs/ROADMAP.md](docs/ROADMAP.md).

The two encoders download from Hugging Face on first use (~1.5 GB, cached
under `~/.cache/huggingface`):

| checkpoint | role | license |
|---|---|---|
| [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | what changed, moment to moment | MIT |
| [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) | what it looks like | Apache-2.0 |

To pre-fetch them (air-gapped machines, or to get the download out of the way):

```bash
huggingface-cli download facebook/vjepa2-vitl-fpc64-256
huggingface-cli download google/siglip2-base-patch16-224
```

`native/wm_*.pt` are artifacts of a retired world-model experiment. They are
not on any read path and you do not need them.

## Run the demos yourself

Both Hugging Face Spaces above are this repo's `deploy/` directory pointed at
a downloaded store. To run the same thing locally:

```bash
pip install -r deploy/requirements-qbe.txt
huggingface-cli download SudharshanR/elidedb-demo-store --repo-type dataset --local-dir lake
python deploy/qbe_serve.py --port 7860        # http://localhost:7860
```

The query-by-example demo loads no model, so it warms in about a second and
answers in ~250 ms. The text demo is `deploy/serve.py` with
`deploy/requirements.txt`; it downloads five text towers first, so give it
time. `deploy/README.md` covers deployment and the staging scripts.

## Documentation

| for | read |
|---|---|
| **Evaluators and customers** | this page, then [docs/ROADMAP.md](docs/ROADMAP.md) for limitations and where the system is going |
| **Investors** | [docs/STORY.md](docs/STORY.md), the story of how the product got here and why the design is what it is |
| **Engineers** | [native/SYSTEM.md](native/SYSTEM.md), the architecture and its measured numbers |
| **Reviewers of the method** | [native/EXPERIMENTS.md](native/EXPERIMENTS.md), every approach tried and what it scored, including the failures |
| **Auditors of the claims** | [BENCHMARKS.md](BENCHMARKS.md), the raw benchmark ledger, and [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md), the full engineering history |
| **Licensing and compliance** | [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

## Repo layout

| path | what |
|---|---|
| `native/relmo/api.py` | **the public API** (`Memory`, `Hit`) |
| `native/relmo/cli.py` | **the command line** |
| `native/relmo/vjrec8.py` | write path, one encoder pass |
| `native/relmo/vjstore.py` | read path: stores, prefilter, alignment |
| `python/elidedb/` | the store layer the web demos serve (Parquet tables, channels, text search) |
| `deploy/` | the two web demos and the scripts that stage them to Hugging Face |
| `docs/FORMAT.md` | on-disk format, byte layout |
| `docs/ARCHITECTURE.md` | how the pieces fit |

Anything in `native/relmo/` not listed above is internal. `deprecated/`,
`bench/`, `eval/` and `scripts/` are research history kept for provenance.

## Contributing

Issues and pull requests are welcome for evaluation, testing, and research
use. Two things make a change easy to accept:

1. **Numbers, not intuitions.** A claim about retrieval quality comes with
   the run that produced it. `BENCHMARKS.md` has the format;
   `native/EXPERIMENTS.md` records approaches that were tried and *failed*,
   which is just as useful.
2. **Say why in the code.** Comments here explain the design decision behind
   a line, not what the line does.

Please do not add: a network layer, auth, or per-dataset configuration.
Nothing may be hardwired about a particular corpus.

## License

ElideDB is **source-available** under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

**You can**: read the code, run it, evaluate it, test it, use it for
research, teaching, and personal projects, and open issues and pull
requests.

**You cannot**: use it for commercial purposes. For commercial licensing,
contact **sr7431@nyu.edu**.

Earlier snapshots of this repository were published under MIT; the current
release and everything after it are PolyForm Noncommercial. The product
pipeline's own dependencies (V-JEPA 2, SigLIP 2, PyTorch, Transformers) are
all permissively licensed; the noncommercial term is ElideDB's choice, not
an inherited restriction. Full dependency and dataset terms:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
