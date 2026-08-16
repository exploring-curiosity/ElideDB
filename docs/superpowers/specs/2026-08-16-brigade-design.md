# Brigade — an autonomous kitchen agent whose memory is ElideDB on CockroachDB

**Date:** 2026-08-16
**Target:** CockroachDB × AWS Hackathon (~4000 participants). Build budget: 1 day, $0.
**Directory:** `brigade/` (inside the StreetDex repo, so it can import `native/relmo`).

---

## 1. The pitch

> A kitchen robot that gives itself work. It watches the kitchen, notices what changed,
> decides what needs doing, and does it — and every one of those decisions is a read
> against its memory. Brigade's memory is **ElideDB running on CockroachDB**: video-native
> episodic recall backed by a distributed vector index, alongside the spatial, procedural
> and working memory that make the robot act differently tomorrow than it did today.

Two products ship together, and each makes the other legible:

- **ElideDB (RelMo)** gets what it has never had — a distributed backend. Its stage-1
  retrieval runs on CockroachDB's `VECTOR` index instead of a pile of local `.npz` files.
- **CockroachDB** gets a demonstration that is not a chatbot: an embodied agent whose
  every tick is a transaction, where memory is not context stuffing but the thing that
  selects the next physical action.

**Anti-goal:** crash-resume as a headline. Persistence is table stakes and is demoted to
a footnote. What is being demonstrated is *agency driven by recall*.

---

## 2. Verified ground truth

Everything below was measured on this machine on 2026-08-16, not assumed.

| Fact | Value | How verified |
|---|---|---|
| robosuite / robocasa / mujoco | 1.5.2 / 1.0.1 (365-task) / 3.3.1 | import + version |
| Kitchen envs registered | **396** after importing one atomic module | `REGISTERED_ENVS` |
| Base `Kitchen` env instantiable standalone | yes, 44 fixtures at layout 1 | `robosuite.make(env_name="Kitchen", ...)` |
| Custom object pool via `_get_obj_cfgs()` | yes — `bowl1, bowl2, mug1, kettle1` spawned | `BrigadeKitchen` probe |
| Semantic object labels | `env.get_obj_lang(name)` → `'bowl'`, `'mug'`, `'kettle'` | probe |
| Object pose | `sim.data.body_xpos[env.obj_body_id[name]]` | probe |
| **Teleport an object** (the "drop a bowl" mechanic) | `sim.data.set_joint_qpos(joint, qpos)` moved bowl2 +0.25 m | probe |
| Door state readable / actuable | `cab.is_open(env)`, `open_door`, `close_door`, `get_door_state` → `{'left_door': 0.956, ...}` | probe |
| Control rate | **60–71 Hz** with 1–2 cameras at 256², vs 20 Hz needed → **3× headroom** | 60-step timing |
| Raw physics | 5952 Hz | `sim.step()` |
| Action space | `action_dim=12` — arm `0:6`, gripper `6:7`, base `7:10`, torso `10`, mode `11` | probe |
| Cameras | `robot0_agentview_{center,left,right}`, `frontview`, `robotview`, `eye_in_hand` | probe |
| Reset cost | **~44 s** with an object pool (one-time at boot); 2.1 s without | probe |
| MPS | available, torch 2.7.1 | probe |
| Perception assets | ultralytics 8.4.113 + `FastSAM-s.pt` (23 MB) present | ls |
| Embedding/VLM assets cached | `siglip2-base-patch16-224`, `vjepa2-vitl-fpc64-256`, `Qwen3-VL-30B-4bit` (MLX) | HF cache |
| Text embedder | `sentence_transformers` 5.1.2 present | import |
| Web stack | fastapi 0.139.2, uvicorn 0.35.0 | import |
| DB driver | `psycopg2` 2.9.10 present (`psycopg` v3 absent) | import |
| Docker | 29.4.0 running, **no cockroach image cached** | `docker info` |
| **Missing, must install** | `cockroach` (docker image), `boto3`, `ccloud` CLI, `aws` CLI | `command -v` |

Relevant RoboCasa envs that already exist and match the demo verbatim:
`StackBowlsCabinet`, `RestockBowls`, `SetupBowls`, `NavigateKitchen`, `OpenCabinet`,
`CloseCabinet`, `OpenDrawer`, `TurnOnStove`, `ArrangeTea`, `KettleBoiling`.

RelMo assets already on disk: **3,556 encoded rcasa recordings** (~19.6 video-hours),
the store's own best-measured domain (P@10 0.752, p50 554 ms).

---

## 3. Architecture

```
┌──────────────────── LOCAL (Mac, $0) ─────────────────────┐   ┌──── CockroachDB ────┐
│                                                          │   │                     │
│  world/    BrigadeKitchen (one persistent kitchen)        │   │ events    VECTOR384 │
│            SimRunner @20Hz, own thread, command queue     │◄─►│ relmo_rec VECTOR512 │
│            skills: navigate/pick/place/open/close/turn    │   │ object_beliefs      │
│            spawn: object-pool teleport                    │   │ norms, skill_stats  │
│                                                          │   │ tasks (SKIP LOCKED) │
│  perceive/ FastSAM → crops → SigLIP2 → detections         │   │ decisions (audit)   │
│                                                          │   └─────────┬───────────┘
│  memory/   DAL + retry(40001) + all four memory kinds     │             │ MCP (read-only)
│            relmo_index: stage1 in CRDB, DTW rerank local  │             ▼
│                                                          │      agent analytics
│  agent/    tick: observe→recall→decide→act→learn          │
│            brain: Claude Agent SDK  (deterministic fallback)   ┌──── AWS ($0) ────┐
│                                                          │    │ Lambda consolidator│
│  api/      FastAPI: MJPEG stream, SSE, instructions       │◄──►│ EventBridge sched  │
│            dashboard: live view + memory panel            │    │ S3 clips + banks   │
└──────────────────────────────────────────────────────────┘    └───────────────────┘
```

### Module boundaries

Each is independently testable and owns one thing.

| Module | Does | Depends on |
|---|---|---|
| `brigade/world/kitchen.py` | `BrigadeKitchen(Kitchen)` — persistent scene, fixed object pool, fixture refs | robosuite, robocasa |
| `brigade/world/sim.py` | `SimRunner` — owns the env on its own thread, steps at 20 Hz, accepts commands via queue, publishes frames + state snapshots | kitchen.py |
| `brigade/world/skills.py` | `navigate_to`, `pick`, `place`, `open_fixture`, `close_fixture`, `turn_on` — each returns `SkillResult(ok, seconds, strategy, detail)` | sim.py |
| `brigade/world/spawn.py` | object-pool teleport: `drop(label, fixture)` / `move_secretly(instance, fixture)` | sim.py |
| `brigade/perceive/vision.py` | frame → `[Detection(label, bbox, score, pos3d)]` via FastSAM + SigLIP2 | ultralytics, transformers |
| `brigade/memory/db.py` | pool, `@retry_serializable`, health, schema apply | psycopg2 |
| `brigade/memory/events.py` | append event, MiniLM embed, `recall(text,k)`, `recall_by_kind` | db, sentence_transformers |
| `brigade/memory/beliefs.py` | upsert belief, `where_is(label)`, `mark_stale`, decay | db |
| `brigade/memory/norms.py` | `home_for(label)`, `learn_norm`, `set_norm(instructed)` | db |
| `brigade/memory/tasks.py` | enqueue, `claim()` via `FOR UPDATE SKIP LOCKED`, complete, preempt | db |
| `brigade/memory/relmo_index.py` | push RelMo 512-d vectors to CRDB; `qbe(recording_id, k)` = CRDB KNN → local DTW rerank | db, native/relmo |
| `brigade/agent/loop.py` | the autonomous tick | everything |
| `brigade/agent/brain.py` | LLM tool-calling; deterministic fallback planner | memory |
| `brigade/api/server.py` | MJPEG stream, SSE event feed, POST instruction, POST drop-object | sim, memory |

---

## 4. Memory schema (CockroachDB)

Four memory kinds, one database, joined by ids. Exact DDL is applied by
`brigade/memory/schema.sql` and **verified against a live cluster at build time** — any
syntax the server rejects is fixed against the server, not against documentation.

- **`events`** — episodic. `text` is a self-description the robot writes
  (`"placed bowl1 in cab_2_main_group — success — 12.4s"`), `embedding VECTOR(384)`
  (MiniLM `all-MiniLM-L6-v2`), `payload JSONB`, `outcome`, `clip_key`, `task_id`.
  Vector index for recall; inverted index on payload; `(robot_id, ts DESC)` for scans.
- **`object_beliefs`** — spatial. `(kitchen_id, instance_id)` PK, `label`, `location`
  (fixture name), `pos FLOAT[]`, `confidence`, `last_seen`, `last_verified`, `stale BOOL`.
- **`norms`** + **`skill_stats`** — procedural. Where a label *belongs*
  (`home_location`, `confidence`, `n_episodes`, `source ∈ {learned, instructed}`), and
  per-`(skill, label, strategy)` success counts that drive strategy choice.
- **`tasks`** — working. `state ∈ {pending, claimed, running, done, failed, preempted}`,
  `priority`, `origin ∈ {self, user}`. Claimed with `SELECT … FOR UPDATE SKIP LOCKED`.
- **`decisions`** — audit. What was chosen, why, `recalled_event_ids UUID[]`,
  `recall_latency_ms`. This is what the dashboard renders as "the memories it used".
- **`relmo_recordings`** — ElideDB stage 1. `embedding VECTOR(512)`, `basis_id`,
  `bank_key` (S3 pointer to the `(T,1792)` DTW bank), `lang`, `outcome`.

### The RelMo ↔ CockroachDB seam (the load-bearing technical claim)

RelMo's prefilter score is `0.5 * (pf @ qf + ps @ qs)`. This is *exactly* an inner product
on `concat(pf, ps) / sqrt(2)` — verified numerically to **5.55e-17**. So moving stage 1
into a CockroachDB `VECTOR(512)` column under a vector index is **not an approximation of
RelMo's retrieval; it is RelMo's retrieval**. Stage 2 (banded DTW — order-dependent,
non-metric, not decomposable into inner products) stays in the compute layer, reading
banks from S3 with a local cache.

Two traps, both handled: (a) the PCA basis is store-fitted, so adding recordings would
invalidate every stored vector — the basis is **frozen**, sha256'd, and carried as a
`basis_id` column so a refit becomes a new namespace instead of silent corruption;
(b) `k = min(256, N-1)` must be pinned at 256.

**One sentence for judges:** *RelMo remembers what moments looked like; CockroachDB
remembers what they meant.*

### Where RelMo is used — and where it deliberately is not

Used only where it is measured to be strong, and never as a single point of failure:

- Visual déjà-vu (`query_recording`, zero re-encode) on the robot's own history.
- Failure recall **by example** — the query is the failure *video*, which text cannot do.
- Fleet seeding from the existing 3,556 rcasa recordings ("born with the fleet's memory").
- The dashboard "find similar moments" trick, with scores shown so quality is visible.

Structurally out of the load path: no TextBridge (cut), no sub-span DTW, no rank-1 trust
(strategy is decided by **joining outcomes over top-K**, so 0.75 precision cannot sink a
beat), and the autonomous loop never *blocks* on a QbE result — correctness is carried by
norms and beliefs, which are exact SQL.

---

## 5. The autonomous loop

```
tick (every ~2 s, or on event):
  OBSERVE  render cameras → FastSAM+SigLIP2 → detections
           diff against object_beliefs → changes
  NOTICE   changes → self-generated tasks (origin='self')
  CLAIM    tasks.claim()  -- FOR UPDATE SKIP LOCKED, priority DESC
  RECALL   events.recall(goal)  +  norms.home_for(label)  +  beliefs.where_is(label)
           + relmo.qbe(current_window) when a comparable clip exists
  DECIDE   brain: pick skill + strategy   (deterministic fallback always available)
  ACT      skills.* on SimRunner
  LEARN    write event(outcome) + update beliefs + skill_stats
           + norms.learn_norm() when a placement repeats
```

`origin='self'` is what makes it an agent rather than a command executor: the robot writes
its own tasks from what it notices. User instructions enter the same queue at higher
priority, which is how preemption falls out for free.

### The five acts (each a real execution of the loop)

1. **Cold memory → exploration.** A bowl is dropped. Recall returns nothing above
   threshold. The robot opens cabinets until it finds the existing bowls, infers
   "bowls live in `cab_2`", puts it away, and writes that as a **norm it learned itself**.
2. **Memory replaces exploration.** Second bowl. One recall, straight there. Dashboard
   shows the counter: act 1 = N cabinet-openings and ~90 s; act 2 = 0 and ~20 s.
   *Same code, different memory contents, different behaviour.*
3. **Instruction → standing behaviour.** "Bowls go on the island from now on" updates the
   norm (`source='instructed'`). A later, unprompted bowl goes to the island.
4. **Learning from its own failure.** A grasp fails. On retry, recall over its own event
   log (and RelMo by example) surfaces "side approach failed / top approach succeeded" →
   strategy flips → success → `skill_stats` updated → next object of that kind is right
   first try.
5. **Realtime juggling.** Mid-task, a user instruction preempts; the interrupted task
   returns to the queue and resumes after.

Meanwhile the **Lambda consolidator** (EventBridge schedule) is the memory's sleep cycle:
distils repeated episodes into norms, decays belief confidence, marks stale beliefs. Real
memory work, not decoration.

---

## 6. Hackathon requirements

**CockroachDB tools — need ≥ 2, committing to 3 (4th if time):**

1. **Distributed vector indexing** — two vector families (`VECTOR(384)` event text,
   `VECTOR(512)` RelMo visual) in the operational DB. Core, not optional.
2. **Cloud Managed MCP Server** — configured read-only; the agent queries its own
   analytics through it ("my grasp success rate on bowls by strategy"), and it is how a
   human inspects the robot's mind. Endpoint `https://cockroachlabs.cloud/mcp`.
3. **ccloud CLI** — `scripts/setup_crdb.sh` provisions/inspects the cluster with
   `--format json`, committed and documented.
4. *(stretch)* a **Brigade Agent Skill** contributed in the CockroachDB agent-skills format.

**AWS — need ≥ 1, using 3:** **Lambda** (consolidator, always-free tier),
**EventBridge** (schedule), **S3** (clips + DTW banks, free tier).

**Cost: $0.** The reasoning brain runs on the existing Claude subscription via the Agent
SDK — zero marginal cost, and squarely inside the "works natively with Claude Code"
ecosystem the sponsors are selling.

---

## 7. Error handling, resilience, observability

- **DB**: every write wrapped in `@retry_serializable` (CockroachDB `40001` retry loop with
  jittered backoff). Connection pool with health check. If the DB is unreachable the loop
  **halts and says so on the dashboard** — it does not fake memory from a local cache.
  That is the honest demonstration of "an agent whose memory goes offline stops".
- **Skills**: every skill is time-bounded and returns a structured `SkillResult`; a
  timeout is a `failure` outcome, which is itself a memory write (act 4 depends on this).
- **Perception**: detections carry scores; below-threshold detections do not overwrite a
  belief, they lower its confidence.
- **Sim**: `SimRunner` runs on its own thread with a watchdog; a MuJoCo exception is
  caught, logged as an event, and the runner restarts from current state.
- **Provenance**: `basis_id` + git commit + model ids recorded per run, reusing the
  existing `vjsnap`-style pinning discipline.
- **Observability**: every tick emits an SSE event; `decisions` is a queryable audit trail
  of what was recalled and what was chosen, with latencies.

## 8. Testing

- `tests/test_world.py` — kitchen boots, pool spawns, teleport moves an object, door
  opens/closes, step rate ≥ 30 Hz.
- `tests/test_memory.py` — against a **live local CockroachDB**: schema applies, event
  round-trips, vector KNN returns the planted neighbour, `SKIP LOCKED` gives two workers
  disjoint tasks, serializable retry works under a forced conflict.
- `tests/test_relmo_index.py` — the 5.55e-17 identity is re-asserted as a test: CRDB
  KNN top-M ⊇ local exact top-M on the same vectors.
- `tests/test_loop.py` — the act-1/act-2 contrast is an automated assertion:
  with empty memory the loop explores; with the norm present it does not.
- Skills are tested by outcome (object ends up in the target fixture), not by trajectory.

## 9. Build order (1 day)

**Base first — this is what gets built and tested before anything else:**

1. `world/` — persistent kitchen, object pool, SimRunner, teleport. *(mechanics verified)*
2. `memory/` — schema on a live local CockroachDB, DAL, all four memory kinds + vector KNN.
3. Smoke test proving 1 and 2 together: boot kitchen → observe → write events → recall.

**Then:** skills → perception → agent loop → dashboard → the five acts → RelMo index +
seeding → Lambda/S3/ccloud/MCP → video + README + diagram.

**Cut list, in order:** contributed agent-skill, RelMo fleet seeding, act 5, multi-region.

## 10. What requires the user, not me

- **Downloads** (need a green light): `cockroachdb/cockroach` docker image (~1 GB),
  `boto3`, `ccloud` CLI, `aws` CLI, MiniLM weights (~90 MB).
- **Accounts I will not create**: CockroachDB Cloud free cluster, AWS account. I build and
  test against **local CockroachDB in docker** (real CRDB, real vector index, offline);
  pointing at CRDB Cloud is then one `BRIGADE_DSN` env var, and Lambda deployment is one
  documented command they run.

## 11. Explicit non-goals

No VLA training. No sub-span DTW. No TextBridge. No multi-region cluster (designed for,
not deployed). No re-encoding of the existing rcasa store. No fabricated numbers — every
figure in the README comes from a script in `brigade/bench/`.
