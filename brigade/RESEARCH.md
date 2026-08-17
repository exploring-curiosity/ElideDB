# Video-only memory → reasoning → action: the research

The ruling this answers (owner, 2026-08-17):

> "Under 'no other info goes in', does the human's own words get stored
> alongside the clip? — **no it cant.** See the one and only purpose of the
> hackathon that can make this survive is the **video database and reasoning
> and action on top of it.** research properly on how to do this. **no
> hardwiring no handrolling.**"

So: the database holds video and its RelMo vectors. Nothing else — no captions,
no labels, no locations, no instructions, no text embeddings. Every semantic
fact is derived at **read time** by something that watches retrieved clips.
This document is the research on how to do that, from two sources: what this
repo already measured, and what is published and released.

---

## 1. What this repo already measured (do not re-learn these)

| finding | number | where |
|---|---|---|
| Query-by-example through the trained ranker works | P@support **0.894** test | `RELMO.md`, `vjrel.py` |
| Raw SigLIP text tower on *direction/verbs* is at chance | 0.224 vs 0.221; "Open" vs "Close the cabinet doors" cosine **0.9974** | `vjtext2z.py` header |
| Learning a text→z map memorises instead of generalising | 54 unique strings / 447 episodes — "memorised its targets" | `RELMO.md:1383` |
| The sanctioned text path is **noun routing** through the `sig` channel | "SigLIP 2's text tower already shares a space with the `sig` channel… needs no new corpus" | `RELMO.md:1405-1408` |
| Off-the-shelf VLM judges collapse to constant outputs | "'held' on 95% of clips regardless of content" — two generations; reproduced here on a third (SmolVLM2: `"the plate."` for every clip) | `RELMO.md`; `bench/watchers.py` |
| General VLMs read fine-grained placement off 256² clips at 2/5 | Qwen3-VL-30B: cabinet ✓, wine rack ✓, stove/plate/in-bowl ✗ | `bench/watcher_gate.py` |
| Every label-free readout plateaus ~0.31–0.51; supervised probes 0.77–0.86 | four representation families | `RELMO.md` §1 |

The pattern in rows 5–7 is one fact seen three ways: **reading semantics off
video is the hard part, and it is where supervision or task-fit goes** —
retrieval is not the bottleneck.

## 2. What is published (checked 2026-08-17)

**MemER** (Stanford, arXiv 2510.20328) is the owner's architecture, published
and validated:

* memory = **raw keyframes, no text stored**;
* a VLM reads retrieved keyframes + recent frames and emits a **language
  subtask**;
* **π0.5 executes the subtask** as the low-level policy;
* \>90% success on real tasks needing minutes of memory.

Two deltas from us: their reader is a **fine-tuned** Qwen2.5-VL-7B (which is
exactly the gap our 2/5 off-the-shelf number shows), and their keyframes are
*selected by a policy at write time* — where Brigade's RelMo retrieves by
content from everything, which is the stronger memory claim and the product.

**ReMEmbR** (NVIDIA, arXiv 2409.13682) is the field's default pattern — and the
anti-pattern here: it runs a captioner at **write time** and stores caption
embeddings. Useful as the contrast that makes the pitch sharp: everyone else
interprets at write; ElideDB/RelMo stores uninterpreted video and pays the
interpretation cost only for the moments a query actually touches. Same thesis
as StreetDex's elision: **the best label is the label never written.**

**Retrieve-Reason-Act** (arXiv 2603.02688) measured where such systems break:
*"the bottleneck lies in visual procedural understanding, not retrieval
quality"* — independently, the same conclusion as row 5–7 above.

**Octo** (arXiv 2405.12213) shows the text-free endgame exists: goal-**image**
conditioning, no language anywhere. But no LIBERO-ready checkpoint is released
(the 78.9/85.7/84.6/51.1 LIBERO numbers are other papers' own fine-tunes), so
it fails the owner's "already tested and released" bar today. Noted as the
eventual way to delete text from the loop entirely.

**Readers built for this shape of problem**, both free, local, released:

* **FastVLM** (Apple, CVPR 2025) — hybrid encoder built for **high-resolution
  input at low latency** (<200 ms TTFT on-device; 85× faster TTFT than
  LLaVA-OneVision-0.5B). `mlx-community/FastVLM-0.5B` and 1.5B exist for
  mlx-vlm. High-res single-frame reading is precisely our reduced task.
* **Moondream** (m87-labs) — small model whose native outputs are **structured
  spatial data**: point/detect/VQA, "(x,y) for every instance of an object you
  describe". Purpose-built for "where is X in this image", not general chat.

## 3. The architecture this dictates

```
WRITE (per episode, online)
  cameras ──► clip (memory camera, 512², separate render pass —
                    the policy keeps its own 256² pipeline untouched)
        └──► RelMo encode → 512-d vector → relmo_recordings (pgvector/C-SPANN)
  NOTHING ELSE. No captions, labels, positions, instructions.

READ (per request)
  human words (runtime only, never stored)
        │
        ├─ noun routing: SigLIP2 TEXT tower → sig channel      [RELMO.md:1405]
        │     "bowl" → clips whose appearance matches — zero-shot,
        │     no new corpus, nouns come from the human's sentence
        │
        ├─ RelMo QbE: robot's own recent trace → similar past spans
        │     (recording, t0, t1) — RelMo's native output
        │
        ▼
  top-k spans ──► READER watches them
        │   reduced task: the span's END FRAME at 512², open-vocab
        │   spatial VQA (+ first frame for what-changed); majority
        │   vote across the k retrieved spans for habit questions
        ▼
  answer / instruction ──► π0.5 acts          (MemER's exact split)
```

Memory OFF = retrieval returns nothing = the reader has nothing to watch. The
ablation stays honest and total.

### The no-hardwire audit — where every word comes from

| words | source | stored? |
|---|---|---|
| nouns/verbs in the request | the human, at runtime | never |
| retrieval keys | RelMo vectors (visual) + SigLIP text tower projection of the human's own nouns | vectors only |
| the reader's answer | open-vocabulary decode over the frame | never |
| the instruction π0.5 gets | reader output (+ the human's own words) | never |
| scoring vocabulary in benches | evaluation harness only — never visible to the system | n/a |

No options lists, no place enumerations, no verb maps, no templates carrying
scene nouns. The two things that must die from the current schema:
`object_beliefs`, `norms`/`norm_evidence`, and the MiniLM `events` embeddings —
all derived facts saved as state. `tasks`/`decisions` remain as **operational
audit** (the queue and the decision log); the recall path never reads them.

## 4. The gate ladder (measure in this order; each gates the next)

| gate | question | method | pass |
|---|---|---|---|
| **G1 clip legibility** | is the information IN the clip at 512²? | re-record the 5 labelled episodes with a 512² memory camera; blind open-vocab end-frame VQA; substring-scored | ≥4/5 for some reader |
| **G2 reader latency** | fast enough to live in the loop? | median wall-clock per read on this Mac | ≤1 s |
| **G3 command form** | can the read become an instruction π0.5 executes? | (a) place-VQA + compose with the human's words vs (b) reader speaks the command; judged by π0.5 episode success on the produced string | either arm ≥ the explicit-string rate |
| **G4 noun routing** | does "bowl" retrieve bowl-clips with no text stored? | SigLIP2 text vector → store's whitening → rank `sig`; P@k against known content | top-k contains the right object's clips |
| **G5 end-to-end** | THE SHIFT on clips-only memory | same seven beats, memory ON/OFF | the gap survives |

Readers to run through G1/G2: **FastVLM-0.5B/1.5B (mlx)**, **Moondream2**
(point + VQA), **Qwen2-VL-2B (mlx)** — the last because it is already in this
repo's own code (`vlmrank.py`). Qwen3-30B is out by fiat; SmolVLM2-500M is out
by measurement (constant predictor).

If G1 fails at 512² with every reader, the honest conclusion is that
off-the-shelf reading is not enough and the reader needs task supervision —
which is MemER's finding too (they fine-tuned), and the fallback is a small
fine-tune of the best G1 reader on RoboCasa's released episodes via
`vjlerobot.py`'s indexing path (26,674 episodes, video + task label, no
simulation needed). Fine-tuning on *another corpus's* public labels stores no
text about *this* kitchen and keeps the no-hardwire rule intact.

## 4a. GATE RESULTS — run 2026-08-17

### G1 — can a fast local reader say what happened? **FAIL**

Six blind clips, re-recorded at **512 px** through a dedicated memory camera,
**no options offered** (the earlier version's list of six places was scene
vocabulary in the prompt and has been removed). Every episode succeeded and
every truth label was verified against the simulator.

| reader | correct | median | distinct answers |
|---|---|---|---|
| Moondream2 (agentview) | **1/6** | 787 ms | 6/6 |
| Moondream2 + point-then-crop | **1/6** | 1564 ms | 5/6 |
| Moondream2 (frontview) | 0/6 | 788 ms | 4/6 |
| Moondream2 (sideview) | 0/6 | 782 ms | 6/6 |
| FastVLM-0.5B | **0/6** | **203 ms** | 3/6 |
| Qwen2-VL-2B-4bit | 0/6 | 922 ms | 4/6 |
| *Qwen3-VL-30B-A3B (banned)* | *2/6* | *796 ms* | — |

Pass bar was ≥4/6. Nothing came close. What was tried and did not help:
**resolution** (256 → 512), **viewpoint** (agentview / frontview / sideview),
**frame budget** (1, 4, 8, uniform vs end-weighted), and **localise-then-crop**
using Moondream's native pointing. The failures are not random — readers get
elevated, distinctive placements (cabinet, wine rack) and answer "table" for
anything resting on the counter plane.

**G2 passes**: FastVLM-0.5B reads in **203 ms**, comfortably inside the robot's
~573 ms think interval. Latency was never the problem. Comprehension is.

### G4 — can a noun route to the right clips? **FAIL, and instructively**

SigLIP 2 text tower → the `sig` channel, the one text path `RELMO.md:1405`
sanctions. 0/3 queries, every one **below chance**:

```
"a black bowl"  ->  g2 bottle .0766 | g9 bottle .0709 | g6 cheese .0615
                    g8 bowl .0547 | g1 bowl .0546 | g4 bowl .0519      0/3
```

The bowl clips ranked **last** for "a black bowl". The diagnosis is structural,
not a tuning failure: **every clip in this kitchen contains every object.** The
bowl is present in the wine-bottle clip and the bottle is present in the bowl
clip, so appearance-based noun matching has nothing to separate. What
distinguishes these clips is *which object moved* — motion and relation, not
presence. That is precisely the V-JEPA channel's job and precisely not the
SigLIP channel's.

So noun routing is the wrong retrieval key **for a fixed single-scene kitchen**.
RelMo's query-by-example is the right one, and it already works here: querying
with a bottle clip returned the wine-rack episodes at 0.6747 / 0.6263, ranked
above the bowl and stove episodes.

### What the gates establish

* **Retrieval is not the bottleneck. Reading is.** Independently the same
  conclusion as *Retrieve-Reason-Act* ("the bottleneck lies in visual procedural
  understanding, not retrieval quality") and as this repo's own ladder
  (label-free readouts 0.31–0.51, supervised probes 0.77–0.86).
* **MemER fine-tuned its reader, and that is not incidental.** Their Qwen2.5-VL
  is trained on the task. Every number above is what "off the shelf" costs.
* **The reader must be trained.** That is the only remaining path to a
  video-only memory that answers questions, and it is the fallback this document
  named before the gates were run.

### The training path, unchanged by the results

`native/relmo/vjlerobot.py` already indexes RoboCasa's released LeRobot
episodes — **26,674 episodes, 80,022 rendered mp4s, 63 task releases, present
locally, no simulation needed**, each carrying video plus a task label. Training
a reader on *that* corpus stores nothing about this kitchen: the supervision is
another dataset's public labels, the same status as any pretrained checkpoint,
and the no-hardwire rule holds. It is a training job measured in hours, not a
prompt change.

## 5. What is deliberately NOT being done

* **No captioning at write time** (ReMEmbR's pattern) — the entire product
  claim is that the database holds uninterpreted video.
* **No stored action replay** (VINN-style retrieve-and-copy) — replaying a past
  motor tape is not acting, and the owner's rule against non-agentic playback
  covers it.
* **No text→z bridge training on this kitchen's strings** — measured to
  memorise (`RELMO.md:1383`), and it would smuggle instructions into weights.
* **No goal-image actuation yet** — Octo would delete text from the loop
  entirely, but no released LIBERO checkpoint exists; it fails "tested and
  released now".

---

*Sources:* [MemER](https://arxiv.org/html/2510.20328) ·
[ReMEmbR](https://nvidia-ai-iot.github.io/remembr/) ·
[Retrieve-Reason-Act](https://arxiv.org/html/2603.02688) ·
[Octo](https://octo-models.github.io/paper.pdf) ·
[FastVLM](https://github.com/apple/ml-fastvlm) ·
[FastVLM MLX checkpoints](https://huggingface.co/mlx-community/FastVLM-0.5B-bf16) ·
[Moondream](https://moondream.ai/models) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
