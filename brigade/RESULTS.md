# Brigade — measured results

Everything below came out of a run on this machine (Apple Silicon, macOS 15.5,
PostgreSQL 18.6 + pgvector 0.8.6, π0.5 on MPS). Raw output is in
`eval_logs/brigade_full.log` and `eval_logs/brigade_results.json`.

Reproduce:

```bash
python -m brigade.run --wipe --seed \
  --ab "put the bowl away" --ab "put the bottle away" --trials 3
```

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
from the benchmark, and wrote down what happened. All eight succeeded.

```
[1-3/8] put the bowl on top of the cabinet   ok  (12s, 6s, 6s)
[4-5/8] put the wine bottle on the rack      ok  (14s, 10s)
[6/8]   turn on the stove                    ok  (5s)
[7/8]   put the cream cheese in the bowl     ok  (6s)
[8/8]   put the bowl on the plate            ok  (5s)
```

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
| success follows the instruction given | `is_success = self._env.check_success()` — the *scene's* BDDL predicate | a resolution must name a goal, so memory stores one |
| goal scenes differ | identical object sets, init states differ by ≤0.08 across 79 dims | one kitchen, ten goals, honestly |
| group 0 is the visual mesh | group 0 is the **collision hull** | the kitchen was grey capsules |
| a norm is where it last went | one contrary episode flipped a 3-episode norm | evidence tallied per place, norm is the argmax |
| the subject's head noun identifies the object | "bowl away" → head noun *away* | request matched against learned labels instead |
| bigger model = slower episode | π0.5 (3.6 B) 12.5 s vs SmolVLA (0.45 B) 50.1 s | wall-clock tracks forward passes, not parameters |
