# Brigade — measured results

Everything below came out of runs on this machine (Apple Silicon, macOS 15.5,
PostgreSQL 18.6 + pgvector 0.8.6, π0.5 on MPS).

> **§1–§8 describe v1, whose memory was a fact store** — beliefs, norms, skill
> stats, text-embedded events. The owner's ruling removed all of it: the memory
> is RelMo, nothing else goes in, and the human's words are not stored either.
> **§9 onward is v2**, the video-only system, and where the two disagree v2 wins.
> The v1 sections are kept because several of their findings survive the rewrite
> unchanged — the persistence cost (§0), the autoreset trap, the robot/world
> state split — and because a result that was superseded is more useful on the
> record than deleted.

---

## 9. v2 — the video-only memory

Reproduce:

```bash
.venv-libero/bin/python brigade/bench/train_reasoner.py --collect --episodes 6
```

```bash
myenv/bin/python brigade/bench/relmo_probe.py
```

```bash
.venv-libero/bin/python brigade/bench/two_stage.py
```

### 9.1 Two defects that looked like a representation problem

The previous session ended with "RelMo's vectors do not separate these ten
behaviours — a representation problem". They were both plumbing.

**A 5 s span is below RelMo's resolution.** The encoder tiles 4.0 s windows on a
2.0 s hop, so a 5 s clip yields exactly ONE window: eight 0.25 s descriptor steps
covering 2.0 s. Every trace was a single glance, DTW had nothing to align, and
the "512-d vector" was one window's pooled descriptor. 15 s gives 6 windows and
48 steps.

**The store shipped stage 1 and called it retrieval.** RelMo's read path
(`vjstore.Store.query`) prefilters by pooled cosine and then ranks by DTW over
the full traces. Brigade indexed the prefilter and stopped, which is an IVF
coarse quantiser with no residual scan.

A third, found while fixing them: RelMo prints to stdout on its first encode,
which corrupted the sidecar's JSON line protocol. It cost exactly one clip per
session — the first encode failed, every later one worked — and read as a flake
rather than a protocol bug.

### 9.2 The measurement discipline that changed every number

The store indexes a sliding 15 s span on a 5 s hop, so consecutive rows share two
thirds of their video. A plain nearest-neighbour score therefore mostly asks
whether a clip can find *itself shifted by five seconds*. Barring every
overlapping span from the candidate set is the difference between:

| | naive 1-NN | overlap barred |
|---|---|---|
| pooled mean, kitchen basis | 1.000 | 0.324 |
| pooled mean, RoboCasa basis | 1.000 | 0.685 |

Both columns are printed in every bench in this repo. Without the guard, the
kitchen-fitted basis reads 1.000 and looks like the obvious choice.

### 9.3 Which reduction of the trace separates the behaviours

111 spans, 10 behaviours, chance 0.100, overlap barred:

| reduction | 1-NN | overlap barred |
|---|---|---|
| pooled mean, basis refit on this kitchen | 1.000 | 0.324 |
| pooled mean, no whitening | 0.946 | 0.649 |
| pooled mean, RoboCasa basis (RelMo stage 1) | 1.000 | 0.685 |
| full trace, RelMo's DTW (stage 2) | 0.982 | 0.685 |
| temporal spread, both channels | 0.991 | 0.739 |
| **temporal spread, SigLIP 2 only** | 0.982 | **0.748** |
| net direction (last − first) | 0.541 | 0.117 |
| all six reductions concatenated | 0.703 | 0.225 |

**Refitting the basis on this kitchen loses, reversing what a shorter corpus
said.** On 59 spans of 5 s clips the refit read 0.300 → 0.400 and this repo
recommended it. Whitening removes the variance a corpus *shares*; 111
overlapping spans of one room share exactly what separates the behaviours, so
fitting on them whitens the signal away. The store ships RoboCasa's basis.

**The spread beats the mean and beats DTW.** The mean says what the kitchen looks
like, which is the same for all ten behaviours. Added as a second indexed column,
`motion`.

**Fusing the reductions destroys the best one** (0.748 → 0.225): cosine over
concatenated unit blocks is the mean of the per-block cosines.

### 9.4 Is the memory load-bearing?

Three heads trained from scratch on exactly the inputs each is allowed — not one
head with an input zeroed, which leaks the label prior (that control read 0.536
for "words alone"; a words-only model reads 0.545 where it matters).

| trained on | held out | ambiguous requests only |
|---|---|---|
| clips + words | 1.000 | **1.000** |
| words only | 0.710 | **0.545** |
| clips only | 0.957 | **0.955** |
| chance | 0.100 | 0.100 |

Half the requests are deliberately ambiguous ("put the bowl away" is true of four
behaviours), so the words-only row is capped by the ambiguity however good that
model gets. Held-out spans are the last quarter of each behaviour's block with a
two-span guard dropped between — exactly the overlap depth, so no training span
shares a frame with a test span.

### 9.5 What the database's stage buys

Recall is agreement with the exact full DTW scan over 111 spans:

| shortlist M | recall@5 | corpus elided | stage 1 | stage 2 |
|---|---|---|---|---|
| 8 | 0.650 | 92.8% | 1.50 ms | 13.5 ms |
| 16 | 0.778 | 85.6% | 1.60 ms | 24.4 ms |
| 32 | 0.836 | 71.2% | 1.75 ms | 46.2 ms |
| 64 | 0.867 | 42.3% | 1.95 ms | 88.5 ms |
| exact | 1.000 | 0% | — | 156.1 ms |

The prefilter is RelMo's own, computed by the vector index rather than by numpy —
`selfcheck` measures the two at **2.3e-08** max absolute difference.

---

## 0. One continuous kitchen, and what it costs

LIBERO resets between episodes by design. That was a real hole in everything
below: the robot's spatial memory described a world that had already been
rewound — accurate about the past, useless about the present, and depended on by
nothing. A belief nothing acts on is bookkeeping.

The kitchen is now persistent. State is carried across every episode and across
goal changes, which rebuild the whole environment:

```
python3 brigade/bench/continuity.py --repeat 2
→ carried 7/7 transitions — the kitchen is persistent
```

Drawers and the stove knob carry too — they are ordinary joints in the same
state vector — so a drawer left open stays open. And side effects persist: in
one measured run the robot knocked the bowl off the cabinet while reaching for
the bottle, and it stayed knocked off.

### The price, measured

| start state | policy success |
|---|---|
| the benchmark's own placements (how π0.5 is published) | **97–100%** |
| carried over from the previous episode | **4/8** |
| carried over *including the robot's own pose* | **0/4** |

π0.5 is an episodic policy. A lived-in kitchen is out of its training
distribution by construction, and it costs roughly half the success rate. That
number is not hidden anywhere in this repo, because it is the honest answer to
"why not just make it continuous".

The third row is a bug worth keeping in mind: `get_sim_state()` is the *whole*
simulation, so carrying it verbatim also carries the arm — each episode starting
wherever the last one stopped, sometimes mid-reach or still gripping. Every
action failed, including pressing the stove button. The fix is also the correct
model of a household robot: **the arm returns to a neutral pose, the kitchen
keeps its state.** Splitting them recovered 0/4 → 4/8.

### Two integrity bugs that persistence introduced

Both would have flattered the result, and both were caught by reading a
suspicious timing rather than a suspicious total:

* **Goals that already held counted as successes.** Three memory-off beats
  "succeeded" in **1 second flat**, having inherited a kitchen the memory-on arm
  had already tidied. An episode whose goal was true before the robot moved is
  now reported separately and excluded from the tally.
* **The control arm ran second, in the world the treatment arm had cleaned.**
  Both arms now start from the same snapshot, and both are scored against the
  same goal per beat — otherwise the memory-off arm never changes goal and every
  beat after its first is trivially already-true.

Persistence is a flag, not a commitment: `--episodic` restores LIBERO's default
if you want the 98% policy and a narrower claim.

---

## 0b. THE SHIFT — seven beats, each leaning on the one before

Each beat is a different kind of remembering, and they compound: beat 3 needs
what beat 1 did, beat 4 needs what beat 3 changed, beat 5 needs the verb from
beat 3.

| # | the human says | what memory has to supply | continuous ON / OFF | episodic ON / OFF |
|---|---|---|---|---|
| 1 | "put the bowl on the stove" | nothing — explicit | ok / **ok** | ok / **ok** |
| 2 | "where is the bowl?" | the belief written in beat 1 | ok / fail | ok / fail |
| 3 | "put it back" | a referent for "it" + the bowl's norm | *fail* / fail | ok / fail |
| 4 | "where is the bowl?" | the belief, revised by beat 3 | ok / fail | ok / fail |
| 5 | "and the bottle too" | the verb from beat 3 + the bottle's norm | *fail* / fail | ok / fail |
| 6 | "now get the stove going" | a paraphrase with no shared words | ok / fail | ok / fail |
| 7 | "feed the cat" | nothing — and it must say so | ok / fail | ok / fail |
| | | | **5/7 / 1/7** | **7/7 / 1/7** |

The two italicised failures are **the policy, not the memory**. Memory resolved
both correctly — `"put it back"` → `"put the bowl on top of the cabinet"`,
`"and the bottle too"` → `"put the wine bottle on the rack"` — and π0.5 then ran
to the step limit in a kitchen that no longer matched anything it was trained
on. That is the same ~50% from the table above, showing up beat by beat.

The gap between the arms is what the experiment is about, and it survives in
both worlds: **5/7 vs 1/7** persistent, **7/7 vs 1/7** episodic.

Beat 1 succeeding in both arms is the control, and it is the honest part of the
table: a **complete** instruction does not need memory, and Brigade does not
pretend otherwise — a fully specified request is passed to the policy untouched.

### The two beats worth reading closely

**Beats 2 and 4 are the same question with different correct answers.**

```
[2] "where is the bowl?"   robot: the bowl is on the flat stove
                           truth: the bowl is on the flat stove      CORRECT  (1.4 ms)
[3] "put it back"          -> put the bowl on top of the cabinet     SUCCESS
[4] "where is the bowl?"   robot: the bowl is on the wooden cabinet
                           truth: the bowl is on the wooden cabinet  CORRECT  (1.3 ms)
```

The belief was written three episodes earlier, survived them, and was *updated*
by the one that moved the bowl again. That is retention plus revision, which is
what separates a memory from a cache. With memory off there is no answer at all:
`"the robot has no record of where anything was put"`.

**Beat 3 contains no noun.**

```
[3] human: "put it back"
    rewritten: "put the bowl back"      ("bowl" — the last thing handled)
    executes:  "put the bowl on top of the cabinet"
```

Two reads composing: the referent for *it* comes from the last episode's
subject, and *back* comes from the bowl's norm. A policy handed the three words
"put it back" has nothing — and measurably does nothing, running to the step
limit and failing. Beat 5 is the mirror image: "and the bottle too" has no verb,
which is carried over from beat 3, and resolves to a **different** place because
the bottle's norm is different.

### How "where is the bowl?" is marked

Against a record the harness keeps itself, of where each episode actually left
things — not the robot's memory, and not the live simulator. The live sim cannot
be the answer key here: LIBERO episodes are independent, so the vector env
resets the kitchen between them, and a look at the sim after beat 1 reports the
bowl back on the table. That mistake marked a correct answer WRONG in the first
run of this act.

---

## 1. The headline: memory is load-bearing

Same kitchen, same policy, same seeds, and the **same BDDL goal predicate**
deciding what counts as done. The only variable is whether the sentence handed
to π0.5 came out of the database or straight from the human.

| request | arm | instruction actually executed | result | mean time |
|---|---|---|---|---|
| "put the bowl away" | memory **on** | `put the bowl on top of the cabinet` | **3/3** | 6.9 s |
| | memory **off** | `put the bowl away` | **0/3** | 20.0 s |
| "put the bottle away" | memory **on** | `put the wine bottle on the rack` | **3/3** | 10.3 s |
| | memory **off** | `put the bottle away` | **0/3** | 22.0 s |
| **overall** | **on** | | **6/6 (100%)** | |
| | **off** | | **0/6 (0%)** | |

The timing is its own evidence. The memory-on arm finishes in 7–10 s; the
memory-off arm runs to the 300-step limit every time and is scored a failure at
the horizon. It is not making a different attempt — it never converges on one.

**The same two words, "put ⟨x⟩ away", resolve differently.** The bowl goes to
the cabinet and the bottle to the rack, out of one mechanism reading one table.
That is what makes it retrieval rather than a constant.

### The audit table says the same thing structurally

`decisions.recalled_event_ids` exists so that "was memory actually used" is a
query rather than a claim:

```
              chose              | n_recalled |  ms
---------------------------------+------------+------
 put the wine bottle on the rack |          5 |  8.4    <- memory on
 put the wine bottle on the rack |          5 |  6.6
 put the bottle away             |          0 |  0.0    <- memory off
 put the bottle away             |          0 |  0.0
```

---

## 2. Where the robot's history came from

Nothing was INSERTed. The robot ran eight episodes, each on an instruction read
from the benchmark, and wrote down both what happened and **what it saw
afterwards**. All eight succeeded, and every observation matches its
instruction:

```
[1-3/8] put the bowl on top of the cabinet   ok   saw  bowl   -> wooden cabinet
[4-5/8] put the wine bottle on the rack      ok   saw  bottle -> wine rack
[6/8]   turn on the stove                    ok   (nothing moved — correct)
[7/8]   put the cream cheese in the bowl     ok   saw  cheese -> akita black bowl
[8/8]   put the bowl on the plate            ok   saw  bowl   -> plate
```

"turn on the stove" moving nothing is not a gap — it is a button press, and a
system that invented a placement for it would be writing fiction.

Where an object ended up is worked out from the geometry, not from the sentence:
a place has to contain the object in plan view and either support it from below
or enclose it. Both halves earned their place by failing:

* **nearest-body** answers "the bottle is at the bowl" whenever four objects
  share a tabletop — technically the closest thing, useless as a location;
* **support-only** loses the wine rack, because a bottle in a rack sits *below*
  the rack's top, so "is it above" says no and the answer falls through to the
  table;
* **most-specific** is needed because everything in the room is over the table.

The norms that fell out of it, as tallies rather than last-writes:

```
bottle  -> rack       conf 1.00   [rack x2]
bowl    -> cabinet    conf 0.75   [cabinet x3, plate x1]
cheese  -> bowl       conf 1.00   [bowl x1]
```

The `plate x1` is the point. A single contrary episode lowers confidence from
1.00 to 0.75 and does **not** move the home — under the original last-write
scheme it flipped the norm outright and "put the bowl away" resolved to the
plate.

---

## 3. Recall latency

pgvector HNSW over 384-d MiniLM embeddings of the robot's own event text:

| | |
|---|---|
| recall, warm | **5–27 ms** (typically ~7 ms) |
| recall, first call | 4 508 ms — the lazy encoder load, now warmed at boot |
| robot's control step | 57 ms |
| gap between policy inferences | ~573 ms |

A recall costs roughly a tenth of one inference cycle. Memory is nowhere near
the bottleneck.

**Abstention works.** `"feed the cat"` → best cosine 0.18, below the 0.30 floor,
so the robot declines rather than inventing something:

```
"feed the cat" -> ABSTAINED
   nothing in memory resembles this (best 0.18 < floor 0.30)
```

---

## 4. Video memory (RelMo) — retrieval with no text at all

Every episode is encoded by RelMo (V-JEPA 2 + SigLIP 2) into 512-d and indexed
in the same database. Querying with a **failed** "put the bottle away" clip:

```
0.6747  put the wine bottle on the rack
0.6263  put the wine bottle on the rack
0.5994  put the bowl away
0.5897  turn on the stove
0.5801  put the bowl away
0.5593  put the wine bottle on the rack
```

The three nearest-by-appearance neighbours include the wine-rack episodes it
resembles, ranked above bowl and stove episodes. No caption, no label, no text
embedding is involved anywhere in that ranking.

### The index is RelMo's stage 1, not an approximation of it

RelMo scores prefilter candidates with `0.5·(pf·qf + ps·qs)`. Brigade stores
`concat(qf, qs)/√2`, for which that is exactly the inner product of two unit
vectors — so the database's cosine operator computes RelMo's own quantity.

```
{"cmd":"selfcheck"} -> {"agrees": true, "max_abs_err": 2.31e-08, "dim": 512}
```

Basis: `rcasa`, fitted over **3 556** recordings of robots working in kitchens.

---

## 5. The execution layer this sits on

From the earlier 400-episode benchmark (`eval_logs/pi05_full`), unchanged:

```
libero_spatial 99.0% | libero_object 100.0% | libero_goal 97.0% | libero_10 96.0%
OVERALL 98.0%   (400 episodes, 10.7 s per episode)
```

Above both published reproductions (LeRobot 97.5%, Physical Intelligence
96.85%). That is what makes the A/B interpretable: the hands are a known
quantity, so a failure is the memory layer's.

Per-goal, for the ten goals in the demo kitchen:

```
0  100%  open the middle drawer of the cabinet     5  100%  push the plate to the front of the stove
1  100%  put the bowl on the stove                 6  100%  put the cream cheese in the bowl
2   70%  put the wine bottle on top of the cabinet 7  100%  turn on the stove
3  100%  open the top drawer and put the bowl in   8  100%  put the bowl on the plate
4  100%  put the bowl on top of the cabinet        9  100%  put the wine bottle on the rack
```

---

## 6. The 3D view

| | |
|---|---|
| visual geoms streamed | 86 (of 239 total; group 1, not group 0) |
| per-frame payload | **3 840 bytes** |
| cost to produce a frame | **0.004 ms**, no GL context |
| one-time geometry | 16.2 MB packed float32 |
| the same geometry as JSON | **134 MB** — why it is binary |

---

## 7. What a measurement changed

Each of these was an assumption that turned out to be wrong, caught before it
reached the demo.

| assumed | measured | consequence |
|---|---|---|
| π0.5 generalises across LIBERO | KITCHEN_SCENE4 `libero_90`: **5/5** on two tasks that overlap training, **0/5** on three that do not | demo moved to `libero_goal`, which is measured at 97% |
| the world persists after an episode | gymnasium runs `AutoresetMode.NEXT_STEP`: bowl at z=1.138 on the cabinet at step 86, z=0.898 on the table at step 87 | the terminal state is captured *during* the rollout; observing after `run()` returns recorded "nothing moved" for every success |
| `geom_rbound` bounds a geom | it is a bounding **sphere** radius — a 1.7 m tabletop got a 0.85 m half-height and the table's box reached z=1.66 | per-axis extents from geom type and mesh vertices; nothing was above the table until this was fixed |
| the live sim is the answer key | episodes are independent, so the kitchen resets between them | "where is the bowl?" is marked against the harness's own record of episode outcomes |
| success follows the instruction given | `is_success = self._env.check_success()` — the *scene's* BDDL predicate | a resolution must name a goal, so memory stores one |
| goal scenes differ | identical object sets, init states differ by ≤0.08 across 79 dims | one kitchen, ten goals, honestly |
| group 0 is the visual mesh | group 0 is the **collision hull** | the kitchen was grey capsules |
| a norm is where it last went | one contrary episode flipped a 3-episode norm | evidence tallied per place, norm is the argmax |
| the subject's head noun identifies the object | "bowl away" → head noun *away* | request matched against learned labels instead |
| bigger model = slower episode | π0.5 (3.6 B) 12.5 s vs SmolVLA (0.45 B) 50.1 s | wall-clock tracks forward passes, not parameters |
