# Using ElideDB

The complete operating manual: installing, ingesting your own footage,
querying it, evaluating it on your data, managing stores, tuning, and
troubleshooting. For what the system *is*, read the [README](../README.md);
for *how* retrieval works, read [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

Contents

1. [Install](#1-install)
2. [Ingest your corpus](#2-ingest-your-corpus)
3. [Query](#3-query)
4. [Evaluate on your own data](#4-evaluate-on-your-own-data)
5. [The Python API](#5-the-python-api)
6. [Stores](#6-stores)
7. [Performance and tuning](#7-performance-and-tuning)
8. [Troubleshooting](#8-troubleshooting)
9. [Reference: files on disk](#9-reference-files-on-disk)

---

## 1. Install

### Requirements

| | |
|---|---|
| Python | 3.11 or newer |
| `ffmpeg` and `ffprobe` | on the PATH; used for decoding and for probing frame rate and duration |
| Device | NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU |
| Memory | ~8 GB to query. **16 GB recommended to ingest**: both encoders stay resident (~1.5 GB of fp16 weights) next to full-video decode buffers, and a long recording decodes entirely into RAM before encoding |
| Disk | ~0.8 GB of traces per hour of video, plus ~1.5 GB for the cached encoders |

### Steps

```bash
git clone https://github.com/exploring-curiosity/ElideDB.git
cd ElideDB
pip install torch transformers numpy tqdm
brew install ffmpeg                  # macOS;  Debian/Ubuntu: apt install ffmpeg
cd native                            # every command below runs from here
```

All commands in this guide run from the `native/` directory (the package is
imported as `relmo`). Running from elsewhere produces `No module named
relmo`; see [Troubleshooting](#8-troubleshooting).

### Device selection

The device is chosen automatically in the order CUDA, then MPS, then CPU.
Override it with an environment variable:

```bash
RELMO_DEVICE=cuda python -m relmo.cli add mystore /data/footage
RELMO_DEVICE=cpu  python -m relmo.cli query mystore clip.mp4
```

CPU can run everything, but ingest is far slower there. The throughput
figures in this guide (4x real time) are Apple Silicon; CUDA is supported.

### Models

The two encoders download from Hugging Face on first use and are cached
under `~/.cache/huggingface` (about 1.5 GB). To download ahead of time, or
on a machine that will later be air-gapped:

```bash
huggingface-cli download facebook/vjepa2-vitl-fpc64-256
huggingface-cli download google/siglip2-base-patch16-224
```

Both models are frozen and permissively licensed (MIT and Apache-2.0). They
are never modified; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

### Verify the install

```bash
python -m relmo.cli stores
# no stores yet - create one with:  elidedb add <name> <dir>
```

---

## 2. Ingest your corpus

```bash
python -m relmo.cli add <store> <source> [--fps N]
```

| argument | meaning |
|---|---|
| `store` | a name you choose. Creating a store is just ingesting into a new name |
| `source` | a directory (searched recursively), or a single video file. The Python API also accepts a list of paths |
| `--fps` | override the frame rate for every file instead of probing it. Only needed for containers with a wrong or missing rate |

Example:

```bash
python -m relmo.cli add mystore /data/my_robot_footage
# ingesting 812 videos into store 'mystore' (~203 min, resumable)
#   skipping 14 clips shorter than the 4.0s window
# [progress bar, one tick per video]
# store 'mystore' now holds 798 recordings
```

### What ingest does

For each video: decode at 8 frames per second, run both encoders over
4-second windows advancing every 2 seconds, and write the resulting trace to
disk. It is one pass per video; nothing is revisited.

### What ingest does not do

- **It never copies, moves, re-encodes, or modifies your video.** The store
  holds derived traces and a manifest that records each source file's
  absolute path, frame rate, and length. Keep source paths stable if you
  want results to resolve back to the file (search itself does not depend
  on the file still being there).
- It does not need labels, folder structure, or naming conventions.

### Accepted input

- Formats: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, any resolution, any
  frame rate. Timing comes from the frames a file actually holds over its
  duration, not from the container's rate tag, so files whose tag lies
  (a 10 Hz capture muxed as 25 fps is common in research datasets) are
  timed correctly.
- Minimum length: **4 seconds**, the encoder window. Shorter clips are
  skipped and reported as a count.
- Filenames do not need to be unique. Pipelines that write every clip as
  `frames.mp4` inside a per-run folder are handled: a colliding name takes
  its parent folder as a prefix (`run_0114_frames`), and results display
  the folder.

### Cost

About **14 compute-minutes per hour of video** on Apple Silicon (roughly 4x
real time). Of that, two thirds is the V-JEPA 2 encoder. An estimate is
printed before the run starts, and a progress bar advances once per
completed video.

### Interruptions and growth

Ingest is **resumable**. If it is interrupted, run the same command again;
every video whose trace already exists is skipped.

To **add footage later**, run `add` again pointing at the directory that now
contains both the old and the new files. Old files are skipped, new files
are processed. (Pointing a second run at a directory that contains *only*
the new files works for search, but rebuilds the manifest from that
directory alone, so the older recordings lose their source-path labels.
Point at the superset.)

### Verify the result

Always check what is on disk, not the exit code:

```bash
python -m relmo.cli stats mystore
# {
#  "store": "mystore",
#  "recordings": 798,
#  "hours": 6.4,
#  "median_seconds": 21.4
# }
```

`recordings` counts traces actually present on disk.

---

## 3. Query

Four ways to ask. All of them return the same thing: recordings, best
first, each with a similarity score and the span inside the recording where
the match lies.

### With a clip from outside the store

```bash
python -m relmo.cli query mystore /clips/incident.mp4 --top 10
```

```
 #   match             span  source
------------------------------------------------
 1   41.2%    0.0-  23.2s  run_0114/frames.mp4
 2   38.7%    0.0-  13.0s  run_0088/frames.mp4
 3   31.1%    0.0-  14.4s  run_0203/frames.mp4

3 hits in 554 ms
```

The query clip must be at least 4 seconds long.

### With a slice of a longer video

```bash
python -m relmo.cli query mystore /videos/full_shift.mp4 --start 1240 --end 1262
```

`--start` and `--end` are seconds. Only that slice is decoded and encoded.
The slice must be at least 4 seconds.

### With something already in the store

```bash
python -m relmo.cli like mystore run_0114 --top 10
```

This is the robot's own case: the trace already exists, so no encoding
happens at all and the query costs search time only. The recording itself
is excluded from its results.

### Machine-readable output

```bash
python -m relmo.cli query mystore /clips/incident.mp4 --json
```

```json
[
 {"id": "run_0114_frames", "score": 0.412, "video": "/data/.../run_0114/frames.mp4", "start": 0.0, "end": 23.2},
 ...
]
```

| field | meaning |
|---|---|
| `id` | the recording id inside the store |
| `score` | similarity in [0, 1]; higher is closer. It is one minus the normalised alignment cost (see [HOW_IT_WORKS.md](HOW_IT_WORKS.md)) |
| `video` | absolute path of the source file as recorded at ingest |
| `start`, `end` | the matched span inside the recording, in seconds: it starts where the best-matching window begins and runs for the query's duration, clamped to the recording. A recording no longer than the query is reported whole |

### Options common to `query` and `like`

| flag | effect |
|---|---|
| `--top N` | number of results (default 10) |
| `--exact` | exhaustive scan of the whole store instead of the pruned default. About 40x slower; on the reference corpus it is 1.3 percentage points more accurate (P@10 0.765 vs 0.752). Useful as a reference when evaluating, not recommended as the default |
| `--json` | print the result list as JSON |

### What to expect on timing

Two one-time costs per process: opening a store loads every trace and fits
the prefilter (about a minute for a 3,500-recording store, seconds for a
small one), and the first `query` loads the two encoders (a few seconds).
Every query after that is search only: about half a second on a store of
3,500 recordings, growing linearly with store size. `like` never loads the
encoders. For interactive use keep one process alive (the Python API) rather
than paying the store open on every command-line call.

---

## 4. Evaluate on your own data

You should not trust a retrieval number measured on someone else's corpus.
Three checks, in order of effort.

### 4.1 Self-consistency (no labels, one minute)

Query with recordings already in the store. The top hits should visibly be
the same behaviour as the query:

```bash
python -m relmo.cli like mystore <some-recording-id> --top 10
```

Open the top three sources and look. If they are not the same kind of
event, the system is not working on your data, and the rest of this section
will only quantify that.

### 4.2 Fast-path fidelity (no labels)

The default query prunes the search. Compare it against the exact scan on a
handful of queries; the overlap between the two top-10 lists should be
high:

```bash
for clip in /clips/*.mp4; do
  python -m relmo.cli query mystore "$clip" --json         > "fast_$(basename "$clip").json"
  python -m relmo.cli query mystore "$clip" --json --exact > "exact_$(basename "$clip").json"
done
```

```python
import json, glob
ov = []
for f in glob.glob("fast_*.json"):
    fast  = {h["id"] for h in json.load(open(f))}
    exact = {h["id"] for h in json.load(open(f.replace("fast_", "exact_")))}
    ov.append(len(fast & exact) / max(len(fast), 1))
print(f"fast/exact top-10 overlap: {sum(ov)/len(ov):.2f}")
```

Read this number carefully. On the reference corpus the overlap is only
moderate (about 0.56 of the exact top-10 survives a prefilter of 50, about
0.72 at 200) while *precision* is within 1.3 points of exact. The pruned
path returns different, equally correct neighbours, not worse ones. So a
moderate overlap is normal; an overlap near zero means the prefilter is
missing candidates on your data, and the fix is to raise `prefilter_m`; see
[tuning](#7-performance-and-tuning). If you have labels, [4.3](#43-precision-against-your-own-ground-truth)
run on both paths is the better comparison.

### 4.3 Precision against your own ground truth

If you know which class of event each recording contains, score the top k
directly. Labels are used only to *grade*; the system never sees them.

```python
from relmo.api import Memory

mem = Memory.open("mystore")

labels  = {"run_0114_frames": "spill", "run_0088_frames": "spill", ...}   # id -> class
queries = [("/clips/spill_a.mp4", "spill"), ("/clips/drop_b.mp4", "drop"), ...]

def p_at_k(k):
    scores = []
    for path, want in queries:
        hits = mem.query(path, top_k=k)
        scores.append(sum(labels.get(h.id) == want for h in hits) / len(hits))
    return sum(scores) / len(scores)

for k in (1, 5, 10, 20):
    print(f"P@{k}: {p_at_k(k):.3f}")
```

To measure exhaustive recall, the number that matters if you need *every*
instance: for a class with *n* recordings in the store, ask for `top_k=n`
and compute the same fraction. That is "precision at support". On the
reference corpus it is about 0.40, and it is the system's known weak point;
the top of the ranking is strong, the tail is not yet. Expect the same shape
on your data.

Always quote the random baseline beside a precision: for a class with *n*
instances in a store of *N*, a random ranking scores *n / N*.

---

## 5. The Python API

Everything the command line does, from Python. `relmo.api` is the only
supported surface; other modules are internal and may change.

```python
import sys; sys.path.insert(0, "/path/to/ElideDB/native")   # or run from native/
from relmo.api import Memory, Hit
```

### `Memory`

```python
mem = Memory.open("mystore")
```

Opens (or names) a store. Lazy: nothing loads until the first query.

| method | returns | notes |
|---|---|---|
| `Memory.open(name)` | `Memory` | |
| `Memory.list()` | `{store: n_recordings}` | stores, not internal splits |
| `mem.add(source, fps=None, recursive=True)` | recordings now in the store | `source` is a directory, a file, or a list of paths. Resumable. Raises `ValueError` if no video is found, `RuntimeError` if ingest fails |
| `mem.query(video, start=None, end=None, top_k=10, fast=True)` | `[Hit]` | `start`/`end` in seconds slice the file. `fast=False` is the exact scan. Raises `FileNotFoundError`, or `ValueError` if the clip is under 4 s |
| `mem.query_recording(rec_id, top_k=10, fast=True)` | `[Hit]` | no encoding; the recording is excluded from its own results. Raises `KeyError` if the id is not in the store |
| `mem.stats()` | `dict` | `store`, `recordings`, `hours`, `median_seconds` |
| `mem.size` | `int` | number of recordings |

The encoders load once per process on the first `query`; `query_recording`
never loads them.

### `Hit`

An immutable record of one result.

| field | type | meaning |
|---|---|---|
| `id` | `str` | recording id in the store |
| `score` | `float` | similarity in [0, 1], higher is closer |
| `video` | `str` | source path as recorded at ingest |
| `start`, `end` | `float` | the matched span inside the recording, seconds. Starts where the best-matching window begins, runs for the query's duration, clamped to the recording; whole when the recording is no longer than the query |
| `label` | `str` (property) | a human-readable name: the filename, or `folder/filename` when the filename is generic (`frames.mp4`, `video.mp4`, `clip.mp4`) |

### A complete example

```python
from relmo.api import Memory

mem = Memory.open("mystore")
mem.add("/data/my_robot_footage")

for hit in mem.query("/clips/incident.mp4", top_k=10):
    print(f"{hit.score:.0%}  {hit.label}  {hit.start:.1f}-{hit.end:.1f}s  {hit.video}")

# the robot's own case
for hit in mem.query_recording("run_0114_frames", top_k=5):
    print(hit)

print(mem.stats())
print(Memory.list())
```

---

## 6. Stores

A **store** is one deployment's memory: one robot, one site, one customer.

- Stores never mix. Every statistic the search uses is computed inside the
  store, so two deployments cannot influence each other's results.
- A query addresses exactly one store. Searching several means looping over
  them deliberately and merging the results yourself.
- A store is created by the first `add` into a new name. There is no
  separate create step.

### Listing

```bash
python -m relmo.cli stores
```

### Removing a store

There is deliberately no delete command. Everything a store owns lives
under `data/relmo/` at the repository root:

```
data/relmo/
  datasets/<store>/manifest.json     source paths, fps, length per recording
  vjrec6/<store>_L6/*.npz            change channel traces
  vjrec7/<store>_L6/*.npz            per-token detail (not on the query path)
  vjsig6/<store>/*.npz               appearance channel traces
```

Delete those four locations to delete the store and only the store. Your
source video is untouched. Look before you delete: the `vjrec6`, `vjrec7`
and `vjsig6` directories hold every store side by side.

### Moving a store

Copy the same four locations to the same relative paths on the new machine.
Traces are plain NumPy archives and are device-independent. The manifest
records absolute source paths, so result `video` fields will point at the
old locations until you re-ingest or edit it; search is unaffected.

---

## 7. Performance and tuning

| operation | cost (reference machine, Apple Silicon) |
|---|---|
| ingest | ~14 compute-min per video-hour (4x real time), one pass, resumable |
| opening a store (once per process) | ~1 min for 3,500 recordings; seconds for small stores |
| first `query` in a process | a few seconds (encoder load) |
| query, 3,556-recording store | 554 ms median, 1.9 s p99 |
| query, exact scan, same store | ~20 s |
| `like` (no encoding) | search time only |
| storage | ~0.8 GB per video-hour |

Search cost grows linearly with store size.

### The two knobs behind the default

The default query prefilters the store to the 100 nearest recordings by
pooled similarity, then aligns those 100 in time within a band of 25% of
each recording's length. Both are exposed on the internal `Store` class if
you need to trade speed against fidelity:

```python
from relmo.vjstore import Store
st = Store("mystore")                       # loads the whole store once
hits = st.query(fix, sig, top_k=10, prefilter_m=200, band=0.25)   # wider net
hits = st.query(fix, sig, top_k=10, prefilter_m=len(st.ids), band=0.0)  # exact
```

Measured on the reference store:

| setting | p50 | P@10 |
|---|---|---|
| exact (`prefilter_m=all, band=0`) | 20.2 s | 0.765 |
| **`prefilter_m=100, band=0.25` (default)** | **554 ms** | **0.752** |
| `prefilter_m=50, band=0.25` | 286 ms | 0.728 |
| `prefilter_m=200, band=0` | 1.1 s | 0.765 |

If the fidelity check in [4.2](#42-fast-path-fidelity-no-labels) comes out
low on your data, raise `prefilter_m` first. `Store.query` takes raw channel
traces (`fix`, `sig`), which is what `Memory` produces internally; for most
uses stay on `Memory`.

### Ingest throughput

Ingest is encoder-bound; decoding is about 3% of the time. Running several
ingests in parallel on one GPU does not help (the GPU is already saturated,
measured 0.95 to 1.0x). Run one ingest per device.

---

## 8. Troubleshooting

| symptom | cause and fix |
|---|---|
| `No module named relmo` | you are not in `native/`. `cd native` first, or add it to `sys.path` as in [section 5](#5-the-python-api) |
| `ffmpeg: command not found` / `ffprobe` errors | install ffmpeg and make sure it is on the PATH of the shell running Python |
| `clip is 2.8s; the encoder window is 4.0s` | the query clip or slice is too short. Pass a longer clip or widen `--start`/`--end` |
| `skipping N clips shorter than the 4.0s window` at ingest | informational. Those files are not in the store and cannot be |
| `store 'x' has no records on disk` | the store name is wrong, or ingest wrote nothing. Run `stores`, then re-run `add` and watch for errors |
| the first command takes a minute on a big store | opening the store loads every trace once per process. Keep one process alive (the Python API) for interactive use; the command line pays this on every call |
| first query takes 10 s or more | encoder load, once per process |
| every query is slow, even the second | check the device: the log line `loading encoders onto cpu` means no GPU was found. Set `RELMO_DEVICE` or fix the torch install |
| ingest killed, or the machine swaps | memory. Long recordings decode fully into RAM. Use a 16 GB machine for ingest, or split very long files before ingesting |
| results all show `frames.mp4` | they are labelled by folder in the table view (`run_0114/frames.mp4`); in JSON use the `video` path |
| ingest exited 0 but `stats` shows fewer recordings than expected | some files were skipped as too short, or failed to decode. The ingest log names them. `stats` is the truth; trust it over the exit code |
| Hugging Face download fails | the two models are public and ungated; check network, or pre-download them on another machine and copy `~/.cache/huggingface` |
| NaN scores | fp16 overflow has been seen in other ViT encoders on MPS, not yet in these two. Set `RELMO_DEVICE=cpu` to confirm, then report it as an issue with the device and torch version |

If none of these match, [open an issue](https://github.com/exploring-curiosity/ElideDB/issues)
with the command, the full output, and `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"`.

---

## 9. Reference: files on disk

| path | what |
|---|---|
| `native/relmo/api.py` | the public API, `Memory` and `Hit` |
| `native/relmo/cli.py` | the command line |
| `native/relmo/vjrec8.py` | the write path (one encoder pass); what `add` runs |
| `native/relmo/vjstore.py` | the read path: stores, prefilter, banded alignment |
| `native/relmo/device.py` | device selection and `RELMO_DEVICE` |
| `data/relmo/` | every store's traces and manifests (not in git) |
| `~/.cache/huggingface/` | the two cached encoders |

Anything in `native/relmo/` not listed here is internal.
