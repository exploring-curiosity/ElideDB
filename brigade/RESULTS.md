# Brigade — measured results

Everything below came out of runs on this machine (Apple Silicon, macOS 15.5,
PostgreSQL 18.6 + pgvector 0.8.6, π0.5 on MPS). Raw output is in
`eval_logs/brigade_act.log`, `eval_logs/brigade_full.log` and the matching
`.json` files.

Reproduce:

```bash
python -m brigade.act --wipe --seed          # THE SHIFT, both arms
python -m brigade.run --wipe --seed --ab "put the bowl away" --trials 3
```

---

## 0. THE SHIFT — seven beats, each leaning on the one before

**7/7 with memory, 1/7 without.** Each beat is a different kind of remembering,
and they compound: beat 3 needs what beat 1 did, beat 4 needs what beat 3
changed, beat 5 needs the verb from beat 3.

| # | the human says | what memory has to supply | mem ON | mem OFF |
|---|---|---|---|---|
| 1 | "put the bowl on the stove" | nothing — explicit | ok | **ok** |
| 2 | "where is the bowl?" | the belief written in beat 1 | ok | fail |
| 3 | "put it back" | a referent for "it" + the bowl's norm | ok | fail |
| 4 | "where is the bowl?" | the belief, now **changed** by beat 3 | ok | fail |
| 5 | "and the bottle too" | the verb from beat 3 + the bottle's norm | ok | fail |
| 6 | "now get the stove going" | a paraphrase with no shared words | ok | fail |
| 7 | "feed the cat" | nothing — and it must say so | ok | fail |
| | | | **7/7** | **1/7** |

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
