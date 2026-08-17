# Brigade

**A kitchen robot whose memory is a video database, and which cannot act without it.**

A person says *"put the bowl away."* In the kitchen the robot is standing in,
that sentence has four physically legal answers — the bowl can go on the stove,
on the plate, on top of the cabinet, or inside the top drawer. Nothing in the
words picks one. Nothing in the scene picks one either.

Only what the robot has **seen** picks one. And the only thing it has is video.

---

## The rule that shaped everything

The memory stores video and nothing else. No belief, no norm, no label, no
caption, no object name, no location, no outcome — and not the human's words
either. `clips` has five columns of content: when it was, where the file is,
RelMo's two vectors, and how many descriptor steps it holds. A column a person
could read as a fact about the kitchen is a bug.

That is not asceticism. It is what makes the claim testable. A system that
writes `bowl → cabinet` into a table and reads it back has demonstrated that a
table works. A system with nowhere to write it has to derive it from pixels
every time, and you can switch the pixels off and watch it stop.

The reasoning layer on top is **not generative**. It emits K weights over frozen
instruction prototypes, and the command it hands the policy is a tensor, not a
sentence — so no vocabulary exists anywhere at serve time.

---

## The architecture

```
       cameras (always on, 10 fps)
              │
              ▼  a sliding 15 s span, cut every 5 s
       ┌──────────────┐        RelMo = V-JEPA 2 + SigLIP 2
       │   RelMo      │───────▶ trace  (48 steps × 1792)     → on disk
       │   sidecar    │───────▶ pooled (512)  ─┐
       └──────────────┘───────▶ spread (768)  ─┤
                                               ▼
                                    ┌────────────────────┐
                                    │  Postgres/CRDB     │  HNSW on both vectors
                                    │  clips             │
                                    └────────────────────┘
   human says one thing                        │
              │                                │ STAGE 1  (SQL, vector index)
              ▼                                ▼
       ┌──────────────┐               top-M candidate spans
       │  reasoning   │◀──────────────────────┘
       │  head        │◀── STAGE 2  RelMo's DTW over the traces (exact)
       └──────────────┘
              │  K weights over frozen prototypes
              ▼
        π0.5  ← installed as the language prefix. No string is tokenized.
              │
              ▼
       the kitchen keeps what changed
```

**Retrieval is RelMo's own two-stage read path, with stage 1 moved into SQL.**
The indexed vector is `concat(qf, qs)/√2` of RelMo's whitened pooled channels,
which makes the database's cosine *exactly* RelMo's prefilter rather than an
approximation of it — verified against RelMo's internal score to **2.3e-08**, by
`selfcheck`, not by assertion. Stage 2 is RelMo's DTW over the full traces of
the shortlist, which is the part a pooled vector cannot do. Putting the coarse
stage in the database is the same reason CockroachDB ships a vector index at
all.

---

## What the video actually supports, measured

Nearest-neighbour behaviour match over the store's own spans: given one span,
is the closest *other* span one of the same behaviour? 10 behaviours, chance
0.100.

The store indexes a **sliding** span, so consecutive rows share two thirds of
their video. An unguarded 1-NN therefore mostly measures whether a clip can find
itself shifted by five seconds — which is trivially true and tells you nothing.
Both columns are printed everywhere in this repo, because the gap between them
is large enough to have shipped a false claim:

111 spans, 10 behaviours, chance 0.100:

| reduction of RelMo's trace | 1-NN | **with overlap barred** |
|---|---|---|
| pooled mean, basis refit on this kitchen | 1.000 | 0.324 |
| pooled mean, no whitening | 0.946 | 0.649 |
| **pooled mean, RoboCasa basis** — RelMo's stage 1 | 1.000 | **0.685** |
| **full trace, RelMo's DTW** — stage 2 | 0.982 | **0.685** |
| per-channel temporal spread, both channels | 0.991 | 0.739 |
| **per-channel temporal spread, SigLIP 2 only** | 0.982 | **0.748** |
| net direction (last − first) | 0.541 | 0.117 |
| all six reductions concatenated | 0.703 | 0.225 |

Four things worth taking from that table:

**The mean says what the kitchen looks like; the spread says what moved.** All
ten behaviours happen in one room in front of one fixed camera, so the mean is
nearly the same for all of them. How much each feature varied over the span is
not. Both are indexed — `embedding` and `motion` — and a query can be asked of
either.

**Fusion loses to selection.** Cosine over concatenated unit blocks is the mean
of the per-block cosines, so five uninformative blocks drown one good one:
0.748 alone, 0.225 with everything attached.

**Direction is near chance here.** A 15 s span contains approach and retreat,
and the net displacement of a descriptor across that is noise. It is left in the
code, unused, because a measured null is worth more than silence.

**Refitting the whitening basis on this kitchen LOSES, and that reverses an
earlier reading of ours.** On a 59-span corpus of 5 s clips the refit looked
like a clear win (0.300 → 0.400) and this repo said so. With the span length
fixed and the overlap barred it is 0.685 borrowed against 0.324 refitted.
Whitening removes the variance a corpus *shares* — right when the corpus is
diverse, wrong when it is not, because 111 overlapping spans of one room share
the very thing that separates the behaviours. A basis needs a corpus wider than
the question asked of it. The store ships the RoboCasa basis; `fit` is still
there, and now carries the measurement that says not to use it.

---

## Is the memory load-bearing? — the reasoning head

The head maps a retrieved span plus the human's request onto weights over frozen
instruction prototypes. To measure which input it is using, **three heads are
trained from scratch on exactly the inputs each is allowed** — not one head with
an input zeroed. Zeroing is a bad control: a head trained on both inputs still
knows the label prior when you hand it a zero vector, and here that reads 0.536
for "words alone" where a words-only *model* reads 0.545 on the subset that
matters and much lower is possible.

| trained on | held out | **ambiguous requests only** |
|---|---|---|
| clips + words | 1.000 | **1.000** |
| words only | 0.710 | **0.545** |
| clips only | 0.957 | **0.955** |
| chance | 0.100 | 0.100 |

Half the training requests are deliberately ambiguous — *"put the bowl away"* is
true of four of the ten behaviours — so the words-only column is capped by the
ambiguity no matter how good that model gets. **The video closes the gap
completely, and the video alone gets 0.955 with nobody speaking at all.**

Held-out spans are the last quarter of each behaviour's block, with a two-span
guard band dropped in between. The guard is exactly the overlap depth
(SPAN/HOP − 1), so no training span shares a single frame with a test span.

---

## The loop, end to end

`bench/serve_demo.py` — one continuous kitchen, cameras streaming, four things
said out loud:

```
heard: 'open a drawer'
  memory returned 5 spans in 81.1 ms (stage 1 + DTW)
  command: 'open the middle drawer of the cabinet'  (top 0.98, margin 0.96, 24.7 ms)
  acted: goal satisfied (11s)
```

Retrieval 67–88 ms, reasoning 24 ms, and the command is a `(1, 200, 2048)` tensor
— no string is ever tokenized.

**With memory off there is no degraded command. There is no command:**

```
heard: 'put the bowl away'
  memory OFF — no clips, so no command and nothing to execute
```

That is what "load-bearing" has to mean to be a claim rather than a boast. The
head's only view of the kitchen is the retrieved spans; remove them and its input
does not exist.

**Same words, different history, different command.** Each request is asked
twice against two halves of the robot's own recorded life — a `since`/`until`
pushdown beside the vector scan, which is the other thing the owner's ruling
allows a video memory to return: *"similar clips or its timestamps"*.

| the human says | against earlier history | against later history | |
|---|---|---|---|
| "put the bowl away" | open the top drawer and put the bowl inside | put the bowl on top of the cabinet | **differs** |
| "open a drawer" | open the top drawer and put the bowl inside | open the middle drawer of the cabinet | **differs** |
| "get the stove ready" | turn on the stove | turn on the stove | same |
| "put the wine bottle away" | put the wine bottle on top of the cabinet | put the wine bottle on top of the cabinet | same |

Two of four flipped. Nothing differed but which seconds of video the memory was
allowed to look at.

### What is not solved: an idle kitchen crowds out its own history

The query is the robot's live view, and when a human speaks the robot is usually
standing still. Over a long session the store fills with spans of a kitchen in
which nothing is happening — 94 of 243 after twenty minutes at the dashboard —
and since those are the closest match to *now*, they crowd the shortlist and the
head ends up reading evidence that shows nothing. In the demo above the store is
mostly task video and the retrieved spans are informative; at scale that stops
being true, and the A/B restricts itself to the collection window rather than
pretending otherwise.

This is a retrieval-policy gap, not a representation one, and the fix is
label-free and already latent in the data: a span's motion energy (the
un-normalised spread the `motion` column normalises away) says whether anything
happened in it, so "when did the kitchen last look like this **and something was
going on**" is one more predicate beside the vector scan. Not built; measured
enough to know it is the next thing.

Two smaller defects the dashboard found, both fixed:

**An unpaced render loop is not a camera.** `idle` drew frames as fast as the GPU
allowed, so "10 fps" became 641 spans in ten idle minutes, the encoder fell ~60 s
behind, and a live query timed out waiting for its own span to be indexed. Worse,
every `t0` was a lie: the stamps claim video seconds and the video was running
many times faster than the room. It is paced to the wall clock now.

**A person does not queue behind the cameras.** The write path is asynchronous on
purpose, and a queue has a tail. When a human speaks, the query *is* the span
just cut, so it is encoded ahead of the queue.

---

## What the database's stage buys

Every indexed span used as a query; recall is agreement with the exact
full DTW scan, so 1.000 would mean the elided bytes provably contained nothing
the answer needed.

| stage-1 shortlist | recall@5 | corpus elided | stage 1 | stage 2 |
|---|---|---|---|---|
| M = 8 | 0.650 | 92.8% | 1.50 ms | 13.5 ms |
| M = 16 | 0.778 | 85.6% | 1.60 ms | 24.4 ms |
| M = 32 | 0.836 | 71.2% | 1.75 ms | 46.2 ms |
| M = 64 | 0.867 | 42.3% | 1.95 ms | 88.5 ms |
| exact full scan | 1.000 | 0% | — | 156.1 ms |

The point is not the speedup. It is that a prefilter which elides most of the
corpus and *changes the answers* is a regression wearing a speedup's clothes —
so fidelity is reported beside latency, in the same table, from the same run.

---

### Two defects that had to be fixed before any of this could be measured

Both were structural, and both made the memory look like a representation
problem when it was a plumbing problem.

**A 5 s span is below RelMo's resolution.** The encoder tiles 4.0 s windows on a
2.0 s hop, so a 5 s clip yields exactly ONE window — eight 0.25 s descriptor
steps covering 2.0 s. Every "trace" was a single glance and the DTW stage had
nothing to align. 15 s gives six windows and 48 steps.

**Anything shorter than the window is silently not stored at all.** RelMo
refuses a clip under 4.0 s, so an earlier 3 s segmentation wrote 75 rows and
indexed **none** of them: present in the store, invisible to retrieval, and the
totals looked fine.

---

## Running it

```bash
brew services start postgresql@18          # pgvector needs pg@17 or pg@18
```

```bash
.venv-libero/bin/python bench/train_reasoner.py --collect --episodes 6
```

```bash
.venv-libero/bin/python brigade/bench/train_reasoner.py --train
```

```bash
myenv/bin/python brigade/bench/relmo_probe.py
```

```bash
.venv-libero/bin/python brigade/bench/two_stage.py
```

```bash
.venv-libero/bin/python brigade/bench/serve_demo.py
```

```bash
.venv-libero/bin/python -m brigade.live      # then open http://localhost:8099
```

Two interpreters, on purpose: π0.5 pins `transformers==4.53.2` and RelMo needs
4.57 for `VJEPA2Model`. Neither can move, so the sidecar is the boundary — the
same one CLAUDE.md already draws for ML sidecars. The sidecar writes files; the
core memory-maps them. No RPC framework, no server.

---

## Layout

| path | what it is |
|---|---|
| `brigade/memory/schema_v2.sql` | the store. Read the comments for what is deliberately absent |
| `brigade/memory/store.py` | sliding spans in, two-stage retrieval out |
| `brigade/memory/traces.py` | how the head reads a trace, and the table above |
| `sidecar/relmo_encode.py` | RelMo in its own interpreter: encode, fit, project, rank |
| `brigade/agent/latent.py` | the reasoning layer — prototypes, head, latent prefix |
| `brigade/agent/serve.py` | one human utterance, start to finish |
| `brigade/live.py` | the same loop with the 3D dashboard attached |
| `bench/` | every number in this file, reproducible |
| `PROBLEM.md` | the spec, in the owner's words, including the prohibitions |
