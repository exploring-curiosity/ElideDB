# Brigade

**A kitchen robot whose memory is a database, and which cannot work without it.**

A person says *"put the bowl away."* In the kitchen the robot is standing in,
that sentence has four physically legal answers — the bowl can go on the stove,
on the plate, on top of the cabinet, or inside the top drawer. Nothing in the
words picks one. Nothing in the scene picks one either.

Only history picks one: **this household keeps bowls on top of the cabinet**,
and the robot knows that because it did it three times and wrote each one down.

That is the whole system. Memory is not a log the robot writes and never reads,
and not a cache in front of something that would work anyway. It is the only
thing standing between an underspecified sentence and an actuator, and you can
switch it off and watch the robot fail.

---

## The claim, measured

Same kitchen, same policy, same seeds, same goal predicate deciding what counts
as done. The only variable is whether the sentence handed to the policy came out
of the database or straight from the human.

```bash
python -m brigade.act --wipe --seed     # THE SHIFT — 7 beats, both arms
```

**7/7 with memory. 1/7 without.**

| # | the human says | needs from memory | ON | OFF |
|---|---|---|---|---|
| 1 | "put the bowl on the stove" | nothing — explicit | ok | **ok** |
| 2 | "where is the bowl?" | the belief written in beat 1 | ok | fail |
| 3 | "put it back" | a referent for "it" + the bowl's norm | ok | fail |
| 4 | "where is the bowl?" | the belief, **changed** by beat 3 | ok | fail |
| 5 | "and the bottle too" | the verb from beat 3 + the bottle's norm | ok | fail |
| 6 | "now get the stove going" | a paraphrase with no shared words | ok | fail |
| 7 | "feed the cat" | nothing — and it must say so | ok | fail |

Beat 1 succeeding in both arms is the point of including it: a **complete**
instruction does not need memory, and Brigade passes one to the policy
untouched rather than pretending otherwise.

Beats 2 and 4 are the same question with different correct answers — the bowl is
on the stove, then on the cabinet — because the world moved and the robot
noticed. Beat 3 contains **no noun at all**; "it" is filled in from the last
episode and "back" from the norm, two reads composing into one instruction.

See [`RESULTS.md`](RESULTS.md) for the full numbers, including the single-request
A/B (6/6 vs 0/6) and the timings.

---

## What is underneath

| layer | what it is | why it is that |
|---|---|---|
| **execution** | π0.5 (3.6 B) via LeRobot on LIBERO | measured **98.0%** over 400 episodes on this machine, beating both published reproductions (LeRobot 97.5%, Physical Intelligence 96.85%). Failures in the demo are therefore attributable to the memory layer, not the hands. |
| **episodic + procedural memory** | PostgreSQL 18 + pgvector 0.8.6, HNSW, `<=>` | the schema is written in CockroachDB dialect and translated for Postgres. Same tables, same queries, same operator — `BRIGADE_DSN` is the only thing that changes to move it. |
| **spatial memory** | `object_beliefs`, written from what the robot sees after each episode | a *belief* is "the bowl is on the stove" and goes stale; a *norm* is "bowls live on the cabinet" and outlives the object. Conflating them is the classic mistake, so they are separate tables and separate panels. |
| **video memory** | RelMo (V-JEPA 2 + SigLIP 2), 512-d, HNSW | answers "have I ever *seen* anything like this?", which text cannot. |
| **view** | three.js over the simulator's own geometry | not a video stream — see below. |

### The 3D view is the simulator, not a recording of it

The browser holds MuJoCo's own meshes and redraws them from `geom_xpos` and
`geom_xmat` — the arrays `mj_step` writes and nothing else can. Consequences:

* orbiting and zooming are free and never touch the robot;
* what moves on screen moved in the physics, because those arrays have no other
  author;
* it costs **3.8 KB per frame** and **0.004 ms** to produce, with no OpenGL
  context involved — which is what lets the web thread read it while the main
  thread renders the policy's camera input, a hard requirement on macOS where a
  GL context belongs to the thread that made it.

The one-time geometry payload is ~16 MB of packed float32. Serialising the same
meshes as JSON measured **134 MB**, which is why it is binary.

### RelMo's first stage runs *in the database*

RelMo ranks candidates with `0.5·(pf·qf + ps·qs)` over whitened pooled V-JEPA
and SigLIP channels. Brigade stores `concat(qf, qs)/√2`, for which that quantity
is exactly the inner product of two unit vectors. So the pgvector index is not
an approximation of RelMo's retrieval — it **is** its first stage, executed by
the database.

Checked rather than claimed:

```
python brigade/sidecar/relmo_encode.py  <<< '{"cmd":"selfcheck"}'
→ {"agrees": true, "max_abs_err": 2.3e-08, "dim": 512}
```

RelMo runs as a subprocess because it needs `transformers==4.57` for
`VJEPA2Model` while π0.5 is pinned to `4.53.2` plus openpi's SigLIP overlay.
Both pins are real; one interpreter cannot hold both. The sidecar is that
boundary.

---

## Running it

Prerequisites: the LIBERO stack on Apple Silicon
(see [`../docs/LIBERO_ON_APPLE_SILICON.md`](../docs/LIBERO_ON_APPLE_SILICON.md)),
PostgreSQL 18 with pgvector, and `brew services start postgresql@18`.

```bash
createdb -p 5433 brigade
psql -p 5433 -d brigade -c 'CREATE EXTENSION IF NOT EXISTS vector'

# give the robot a history by making it live one, then serve the dashboard
python -m brigade.run --wipe --seed
#   → http://127.0.0.1:8099
```

Then type something underspecified into the dashboard and flip the **MEMORY**
switch.

| flag | effect |
|---|---|
| `--seed` | run the history episodes before serving |
| `--wipe` | forget everything first |
| `--ab REQUEST --trials N` | the controlled memory-on/off measurement |
| `--no-relmo` | skip video memory (saves an ~85 s startup) |
| `--no-serve` | headless |

`BRIGADE_DSN` points the whole system at a different database. Nothing above
`memory/db.py` knows the difference.

---

## Things worth knowing that were found by measuring

Each of these was a wrong assumption that a measurement caught. They are in the
code as comments where they bite.

* **π0.5 does not generalise to `libero_90`.** On KITCHEN_SCENE4 it scores 5/5
  on the two tasks that overlap its training set and **0/5** on three that do
  not. The demo was designed around this rather than into it.
* **LIBERO's success check is the scene's BDDL predicate, not the instruction.**
  `is_success = self._env.check_success()`. An episode therefore has to name a
  goal, or "it worked" means nothing — which is why a resolution carries one.
* **Group 1 is the visual mesh in robosuite; group 0 is the collision hull.**
  The opposite of the usual MuJoCo convention. Drawing group 0 renders a kitchen
  of grey capsules.
* **A norm is a frequency, not a most-recent write.** With one row per label,
  three episodes of the bowl going on the cabinet followed by one on the plate
  left the norm reading "plate, 1 episode". Evidence is now tallied per place
  and the norm is the argmax.
* **"put the bowl away" parses to the subject "bowl away", whose head noun is
  *away*.** The norm lookup found nothing and a correct norm sat unused. The
  request is now matched against the labels the robot has actually learned.
* **Episode wall-clock tracks forward passes, not parameters.** π0.5 (3.6 B)
  runs an episode in 12.5 s; SmolVLA (0.45 B) takes 50.1 s, because π0.5 chunks
  ten actions per inference and that checkpoint infers every step.

---

## Layout

```
brigade/
  agent/resolver.py   the load-bearing module: request -> instruction, from memory
  agent/pilot.py      one episode of real control on an instruction chosen at runtime
  world/scene.py      MuJoCo model -> three.js payload; per-step transforms
  memory/store.py     episodic / spatial / procedural / working / audit
  memory/db.py        pool, retries, CockroachDB<->Postgres dialect
  memory/relmo.py     video memory; RelMo stage 1 in the database
  api/server.py       web thread reading a sim that owns the main thread
  api/static/         the dashboard
  run.py              all of it, one process
sidecar/relmo_encode.py   RelMo in its own interpreter
bench/                    the measurements
```
