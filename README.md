<div align="center">

# ElideDB

**Video memory. Ask "when did something like this happen?" and get timestamps back.**

No labels. No captions. No fine-tuning. No per-dataset configuration.
Point it at video, then query it with more video.

[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-green)](docs/USAGE.md#1-install)
[![Device](https://img.shields.io/badge/device-CUDA%20%7C%20Apple%20Silicon%20%7C%20CPU-lightgrey)](docs/USAGE.md#device-selection)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-query%20by%20example-yellow)](https://huggingface.co/spaces/SudharshanR/elidedb-qbe)

[Live demos](#see-it-in-two-clicks) ·
[What it does](#what-it-does) ·
[Numbers](#the-numbers) ·
[Vision](#where-this-is-going) ·
[Get started](docs/USAGE.md) ·
[Documentation](#documentation) ·
[License](#license)

</div>

---

> **Early release, moving fast.** ElideDB is a new implementation under
> active development. If something breaks on your data,
> [open an issue](https://github.com/exploring-curiosity/ElideDB/issues);
> star the repository to follow progress.

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

Everything about installing, ingesting your own footage, querying, and
evaluating on your data is in **[docs/USAGE.md](docs/USAGE.md)**.

## See it in two clicks

| demo | what it shows | cold start |
|---|---|---|
| **[Query by example](https://huggingface.co/spaces/SudharshanR/elidedb-qbe)** | pick a few clips, get their kind back. Loads **no model at all**; everything it ranks was computed at ingest | seconds |
| **[Text search](https://huggingface.co/spaces/SudharshanR/elidedb-demo)** | describe what you want in words. Five text towers, so the first boot is slow | ~15 min |

Both run on the public [demo store](https://huggingface.co/datasets/SudharshanR/elidedb-demo-store)
(1,122 robot manipulation recordings).

## The problem

Every robot, vehicle, and factory camera is recording right now, and almost
none of it will ever be watched again. Not because it is worthless: because
there is no way to ask it anything. Finding the six times a robot dropped a
plate means either paying people to watch the footage or paying people to
label it first and hoping the labels anticipated the question. Both cost
more than the answer is worth, so the footage sits there, and the fleet
keeps making the same mistake.

The property people actually care about, *what happened and in what order*,
is precisely what text and still-image tools are worst at. To a model
trained on captioned photographs, opening a drawer and closing it are nearly
the same thing, because they are nearly the same picture. The difference is
in the order the frames arrive.

## What it does

ElideDB is a memory for video. Ingest footage once; from then on, show it a
clip and it returns every other time something like that happened, ranked,
with timestamps, in about half a second.

**A memory layer for a robot or VLA.** The robot asks "have I been in a
situation like this before?" and gets its own past experiences back. Its
live trace is already in the store, so the question costs nothing to ask.
Nothing has to be labelled first, and it works on scenes and tasks the
system has never seen.

**Mining an archive.** Find past instances of a behaviour across a fleet's
history, to review, or to turn into training data for the next policy,
without having tagged any of it.

### What makes it different

- **Nothing is trained on your data.** Two frozen, off-the-shelf encoders
  and a time-alignment step. There are no ElideDB weights. That is why it
  works on day one, on any corpus, with nothing to drift, retrain, or
  version.
- **It matches what changed, not just what it looked like.** One encoder
  watches motion and change, the other watches appearance. Matching is
  elastic in time, so the same action done faster or slower still matches,
  without pretending a 7-second event and a 20-second one are the same.
- **Stores never mix.** One store per robot, site, or customer. Every
  statistic the search uses is computed inside the store.
- **Your video is never copied or modified.** The store holds compact
  derived traces and a manifest that points at your files.

How the retrieval actually works, step by step, is in
**[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**.

## The numbers

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
| query, 3,556-recording store | **554 ms** median, 1.9 s p99 |
| query, robot's own live trace | search only; nothing to encode |
| ingest | ~14 min of compute per hour of video, one pass, resumable |
| storage | ~0.8 GB per hour of video |

Precision is as good on unfamiliar material as on familiar: on the held-out
corpus it scores marginally *higher* on tasks it has never seen (0.404) than
on familiar ones (0.375). There is no home corpus to be biased toward.

### Where it is weak, stated plainly

- **Precision falls off past roughly the top 20.** Strong at "show me the
  closest matches", not yet at "show me every instance". If you need all 35
  of 35, it currently finds about 40%. This is the top of the roadmap.
- Results are whole recordings, not the exact matching moment inside them.
- Query by example only; text search is designed but not yet built into
  the memory layer.
- Clips under 4 seconds cannot be encoded.
- Single machine. No server, no auth, no replication.

### What is and is not released

The current release is the **memory layer**: ingest, store, query by
example. The v1 database engine in this repository (`python/elidedb`:
Parquet lake, byte-elision reads, multi-channel text search) is functional
and serves the text demo, but it is **parked** while the model layer
matures, and the memory layer does not run on it yet. Wiring the two
together is on the roadmap.

## Where this is going

The thing that finally worked was refusing to train anything, and that is a
far better business than it is a slide: no onboarding, nothing to retrain,
no per-customer model to host or explain. But it is the floor, not the
destination.

Every model this project trained on a corpus got better on that corpus and
worse everywhere else. The mistake was never adaptation; it was shipping one
adapted model as if it were universal. The trajectory captures the upside
without repeating that:

1. **Every store adapts to itself, without training** (shipping now). A
   store's own statistics shape its ranking, so it gets sharper as it
   grows, and only it benefits from its data.
2. **Per-store learned ranking, with guardrails.** Fitted on one store,
   serving only that store, shipped only if it beats the untrained baseline
   on that store's own held-out data, with the untrained path as the
   permanent fallback.
3. **Fleet learning.** The measured bottleneck was never the model; it was
   the breadth of the data. Opt-in, aggregated traces across deployments
   (never the footage) form the wide corpus single-customer training could
   not, and query outcomes calibrate confidence and abstention.
4. **The distilled encoder.** Once fleet-scale training beats the frozen
   baseline on a sealed benchmark no model ever trains on, it is distilled
   into a small, fast encoder that only whoever has the fleet can train.

Raw video stays with the customer at every phase. Stores never mix. No
labels, ever. And every shipped change must hold the sealed-benchmark score,
so generalization is a permanent contract, not a launch-day property.

The full plan, with near-term delivery items, is in
**[docs/ROADMAP.md](docs/ROADMAP.md)**. How the product arrived here, told
for a reader deciding whether to back it, is **[docs/STORY.md](docs/STORY.md)**.

## Documentation

| document | what it talks about |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | the operating manual: install, ingest your corpus, query, evaluate on your data, manage stores, tune, troubleshoot |
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | how moment retrieval works, step by step, and why nothing is trained |
| [docs/ROADMAP.md](docs/ROADMAP.md) | current limitations, the growth trajectory, and how customer data compounds |
| [docs/STORY.md](docs/STORY.md) | the story of the project: what was tried, what failed, and why the design is what it is |
| [native/SYSTEM.md](native/SYSTEM.md) | the engine internals, module map, and measured numbers |
| [native/EXPERIMENTS.md](native/EXPERIMENTS.md) | every approach tried and what it scored, including the dead ends, kept so nothing is paid for twice |
| [BENCHMARKS.md](BENCHMARKS.md) | the raw benchmark ledger, every run |
| [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md) | the full engineering history, act by act, with retractions next to results |
| [docs/FORMAT.md](docs/FORMAT.md) | the v1 on-disk format, byte layout |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the v1 database pieces fit together |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | every model, library, and dataset used, with verified licenses |
| [LICENSE](LICENSE) | PolyForm Noncommercial 1.0.0 |

## Repo layout

Two generations live here, at different stages:

| path | what |
|---|---|
| `native/relmo/` | **the current release**: the memory layer. `api.py` (`Memory`, `Hit`) and `cli.py` are the supported surface; `vjrec8.py` is the write path, `vjstore.py` the read path. Everything else is internal |
| `python/elidedb/` | the v1 database engine, parked |
| `deploy/` | the two web demos and the scripts that stage them to Hugging Face |
| `deprecated/`, `bench/`, `eval/`, `scripts/` | research history, kept for provenance |

## Contributing

Issues and pull requests are welcome for evaluation, testing, and research
use. Two things make a change easy to accept: a claim about retrieval
quality comes with the run that produced it, and a comment explains the
design decision behind a line rather than what the line does. Nothing may
be hardwired about a particular corpus: no label vocabularies, no class
lists, no dataset-specific priors.

## License

ElideDB is **source-available** under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

**You can**: read the code, run it, evaluate it, test it, use it for
research, teaching, and personal projects, and open issues and pull
requests.

**You cannot**: use it for commercial purposes. For commercial licensing,
contact **sr7431@nyu.edu**.

The product pipeline's own dependencies (V-JEPA 2, SigLIP 2, PyTorch,
Transformers) are all permissively licensed; the noncommercial term is
ElideDB's choice, not an inherited restriction. Full dependency and dataset
terms: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
