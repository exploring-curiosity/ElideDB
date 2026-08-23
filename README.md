<div align="center">

# ElideDB

**Video memory. Ask "when did something like this happen?" and get timestamps back.**

No labels. No captions. No fine-tuning. No per-dataset configuration.
Point it at video, then query it with more video.

[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-green)](docs/USAGE.md#1-install)
[![Device](https://img.shields.io/badge/device-CUDA%20%7C%20Apple%20Silicon%20%7C%20CPU-lightgrey)](docs/USAGE.md#device-selection)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-query%20by%20example-yellow)](https://huggingface.co/spaces/SudharshanR/elidedb-qbe)
[![Website](https://img.shields.io/badge/website-elidedb.com-2cff7e)](https://www.elidedb.com/)

[Website](https://www.elidedb.com/) ·
[Know the project](#know-the-project) ·
[See it](#see-it-in-two-clicks) ·
[What it does](#what-it-does) ·
[The numbers](#the-numbers) ·
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
 1   41.2%   12.5-  24.3s  run_0114/frames.mp4
 2   38.7%    0.0-  11.8s  run_0088/frames.mp4
 3   31.1%    3.2-  15.0s  run_0203/frames.mp4

3 hits in 554 ms
```

## Know the project

Everything about ElideDB, one document per subject. Each stands on its own;
pick what you want to know.

| document | what it talks about |
|---|---|
| [**Using ElideDB**](docs/USAGE.md) | the operating manual: install, ingest your own footage, query it, evaluate it on your data, manage stores, tune, troubleshoot |
| [**How it works**](docs/HOW_IT_WORKS.md) | how moment retrieval works, step by step, and why nothing is trained |
| [**Roadmap**](docs/ROADMAP.md) | current limitations, the growth trajectory, and how customer data compounds |
| [**The story**](docs/STORY.md) | how the project got here: what was tried, what failed, and why the design is what it is |
| [**System internals**](native/SYSTEM.md) | the engine's architecture, module map, and measured numbers |
| [**Experiments**](native/EXPERIMENTS.md) | every approach tried and what it scored, including the dead ends, kept so nothing is paid for twice |
| [**Benchmarks**](BENCHMARKS.md) | the raw benchmark ledger, every run |
| [**Engineering log**](docs/DEVELOPMENT_LOG.md) | the full development history, act by act, with retractions next to results |
| [**Third-party notices**](THIRD_PARTY_NOTICES.md) | every model, library, and dataset used, with verified licenses |
| [**License**](LICENSE) | PolyForm Noncommercial 1.0.0 |
| [`docs/FORMAT.md`](docs/FORMAT.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the v1 database engine: on-disk format and how its pieces fit |
| [**elidedb.com**](https://www.elidedb.com/) | the one-page pitch, for sending to someone who will not open a repository |

## See it in two clicks

| demo | what it shows | cold start |
|---|---|---|
| **[Query by example](https://huggingface.co/spaces/SudharshanR/elidedb-qbe)** | pick a few clips, get their kind back. Loads **no model at all**; everything it ranks was computed at ingest | seconds |
| **[Text search](https://huggingface.co/spaces/SudharshanR/elidedb-demo)** | the v1 text search, now deprecated: describe what you want in words. Five text towers, so the first boot is slow | ~15 min |

Both run on the public [demo store](https://huggingface.co/datasets/SudharshanR/elidedb-demo-store)
(1,122 robot manipulation recordings).

## The problem

Every robot, vehicle, and factory camera is recording right now, and almost
none of it will ever be looked at again. Not because it is worthless:
because there is no way to ask it anything. The property people actually
care about, *what happened and in what order*, is precisely what text and
still-image tools are worst at. To a model trained on captioned
photographs, opening a drawer and closing it are nearly the same thing,
because they are nearly the same picture. The difference is in the order
the frames arrive.

What that costs today, on average:

| | the number | what it means |
|---|---|---|
| **Volume** | a self-driving car generates about **4 TB of data a day** ([Intel](https://www.networkworld.com/article/958319/one-autonomous-car-will-use-4000-gb-of-dataday.html)); a test vehicle with a full sensor suite, **11 to 152 TB a day** ([Tuxera](https://www.tuxera.com/blog/autonomous-and-adas-test-cars-produce-over-11-tb-of-data-per-day/)) | ten vehicles on an eight-hour shift produce 80 hours of footage a day. Watching it once, in real time, is ten people doing nothing else |
| **Watching it** | a U.S. software developer's median pay is **$133,080 a year** ([BLS, May 2024](https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm)), about $64 an hour before benefits and overhead | reviewing footage takes at least as long as the footage. Finding "the six times the robot dropped a plate" in one week of recordings (40 hours) costs at least **$2,500 of engineering time, per question** |
| **Labeling it instead** | annotating one hour of autonomous-driving sensor data takes **about 800 person-hours** ([CloudFactory](https://www.cloudfactory.com/blog/training-data-hurdles-for-autonomous-vehicles)); finely annotating a single street-scene image takes over **1.5 hours** ([Cityscapes, Cordts et al. 2016](https://arxiv.org/abs/1604.01685)); commodity bounding boxes run **$0.02 per object** ([Label Your Data](https://labelyourdata.com/pricing)) | at one labeled frame per second and five objects per frame, one hour of video is 18,000 boxes, about **$360 per hour of footage** at the cheapest tier, and the labels only answer the questions the schema anticipated. Every new question is a new labeling pass |
| **The result** | IBM estimated that **90% of data generated by devices, connected vehicles included, is never analyzed or acted on**, and 60% of it loses value within milliseconds ([IBM, 2015](https://www.prnewswire.com/news-releases/ibm-connects-internet-of-things-to-the-enterprise-300058256.html)); the market for collecting and labeling data is **$6.3B in 2025, heading to $17.1B by 2030** ([Grand View Research](https://www.grandviewresearch.com/press-release/global-data-collection-labeling-market)) | the footage sits there, the fleet keeps making the same mistake, and the bill for making a fraction of it askable grows 28% a year |

Against that: ElideDB ingests an hour of video in about **14 minutes of one
GPU, once**, with no labels, and every question afterwards costs **about half
a second**.

## What it does

ElideDB is a memory for video. Ingest footage once; from then on, show it a
clip and it returns every other time something like that happened, ranked,
with the span inside each recording where the match lies, in about half a
second.

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
  of 35, it currently finds about 40%. This is the top of the
  [roadmap](docs/ROADMAP.md).
- Queries are by example. The v1 text search is deprecated; its
  replacement, a vocabulary layer on top of the memory, is being optimised
  for top-tier performance before it is released.
- Clips under 4 seconds cannot be encoded.

### What is and is not released

The current release is the **memory layer**: ingest, store, query by
example. The v1 database engine in this repository (`python/elidedb`:
Parquet lake, byte-elision reads, the deprecated text search) is functional
and serves the text demo, but it is **parked** while the model layer
matures, and the memory layer does not run on it yet. Wiring the two
together is on the [roadmap](docs/ROADMAP.md), along with where everything
else is going.

## Repo layout

Two generations live here, at different stages:

| path | what |
|---|---|
| `native/relmo/` | **the current release**: the memory layer. `api.py` (`Memory`, `Hit`) and `cli.py` are the supported surface; `vjrec8.py` is the write path, `vjstore.py` the read path. Everything else is internal |
| `python/elidedb/` | the v1 database engine, parked |
| `deploy/` | the two web demos and the scripts that stage them to Hugging Face |
| `deprecated/`, `bench/`, `eval/`, `scripts/` | research history, kept for provenance |

## Support

An early release, moving fast. Issues are triaged within **two business
days**. Bug reports and results on your own data each have a template, and
the [manual](docs/USAGE.md) has a troubleshooting table that covers the
failures we have seen. Details, including what is and is not supported and
how to reach us about commercial use, are in [SUPPORT.md](SUPPORT.md).

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
