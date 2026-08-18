# PRECEDENT

**An agent that starts with no memory and no vocabulary, and earns both.**

A robot fleet records more video than anyone can watch. Every episode still has
to be dispositioned — filed under what it is — before it can be counted,
compared or acted on. Today that means defining a taxonomy up front and buying a
labelling contract before a single clip is searchable.

This agent is handed an empty database and a queue. For each episode it asks its
own memory for precedent, files the episode if the precedent agrees, escalates it
to a person if it does not — and writes down what it decided either way. Its own
past filings are what let it stop asking.

```
episodes seen        0    140    280    420    560    700
MEMORY ON         0.94   0.73   0.57   0.56   0.41   0.38
MEMORY OFF        1.00   1.00   1.00   1.00   1.00   1.00
```

The OFF arm is the identical agent over the identical feed with **only the write
removed**. It never accumulates a precedent, so every episode reaches a person,
forever. Without its memory the agent does not degrade — it does no work at all.

---

## Why this problem, and why text search cannot solve it

Before building the agent we measured the obvious alternative: just search the
video with words. Over 57 kinds of moment, one query per kind, precision@8:

| | precision@8 | chance | |
|---|---|---|---|
| **type it** — SigLIP 2 text tower → video | 0.285 | 0.018 | 16× |
| **show it** — query by example | **0.840** | 0.021 | **41×** |

**24 of the 57 typed queries land at or below chance.** Six return nothing
correct at all. The reason is one number:

```
cos( "open the drawer" , "close the drawer" )  =  0.9766
```

To a text encoder those are the same sentence. The distinction the query needs
was destroyed at the embedding, before search began — and no prompt fixes it.
The console at `/` shows this live: type "taking something out of the drawer"
and watch drawers *closing* come back.

That is why the agent's memory is indexed by what episodes **looked like**, and
why its vocabulary has to be earned from human answers rather than assumed.

---

## What CockroachDB holds — four tables, four jobs

Memory here is not a vector column with an app around it.

| table | job | why it must be transactional |
|---|---|---|
| `moments` | 3,402 episodes, three vector views, all indexed | the recall |
| `inbox` | the work queue, claimed atomically | two workers must never take one episode |
| `filings` | every decision and how sure the agent was | the agent's context |
| `filing_precedents` | **which filings convinced it** | the graph a correction walks |
| `verdicts` | the only place a human writes | the trigger |

### The cascade — the thing a vector store cannot do

A reviewer overturns one filing. A recursive CTE walks the precedent graph
**backwards** — not "what did this lean on" but "what leaned on this" — and
reopens every filing that inherited the mistake, transitively, in one
serialisable transaction with the verdict and the queue rows.

```
1 overturn → 16 filings cited it directly → 18 reopened transitively
```

Similarity search can tell you what looks alike. Only the graph can tell you
which decisions inherited an error. That is the entire argument for a
transactional database under an agent's memory, and it is why this is not
pgvector with more nodes.

---

## The one knob, swept rather than quoted

Consensus is the fraction of retrieved precedents that must agree before the
agent acts without a human.

| consensus | escalated | filed alone | of those, right |
|---|---|---|---|
| 0.5 | 0% | 858 | 4.1% |
| 0.6 | 36% | 548 | 81.8% |
| **0.8** | **60%** | **346** | **96.2%** |
| 1.0 | 79% | 179 | 100.0% |

Below 0.6 the agent acts on nearly everything and is usually wrong: early in a
cold start the memory holds three filings, two agree by accident, and that
clears a low bar. The default is 0.8 **because of this table**. An operator with
a different cost of being wrong moves it, which is why it is an argument and not
a constant.

*More memory makes it better:* at 240 episodes the same setting read 92.9%; at
860 it reads 96.2%.

---

## Production readiness

**Resilience.** `reclaim()` returns episodes whose worker claimed them and died —
without it `state='working'` is a permanent leak that raises nothing and simply
means an episode is never dispositioned. Poison episodes dead-letter after three
attempts instead of starving the queue behind them.

**Observability.** `/healthz` is readiness, not liveness: 503 when anything has
dead-lettered or the oldest pending episode is ageing, because that is what
broken looks like here. `/metrics` exposes queue depth by state, the age of the
oldest waiting episode, and **autonomy** — the fraction of filings the agent
made unaided, the one number that says whether the memory is working.

**Access control.** Every table is scoped by `fleet`. Tenancy is enforced inside
the recursive cascade, not only on the read that starts it.

**Secrets.** The connection string lives in a gitignored file and reaches the
Lambda as an environment variable. `sslmode=verify-full` with the CA bundled
into the deployment package.

### Two bugs the tests found, both silent

**Tenancy leaked through the cascade.** The recursive walk followed any edge it
found, so correcting a filing in one fleet superseded another fleet's filings —
a correctness bug and a privacy one, reporting success the whole time.

**`SKIP LOCKED` is a trap on CockroachDB.** Under `SERIALIZABLE` a freshly
committed row is briefly invisible to `SELECT ... FOR UPDATE SKIP LOCKED`: the
clause promises never to wait, so rather than block on the uncertainty window it
returns nothing. The row is not lost and a later claim finds it — but a worker
loop that treats an empty claim as "queue empty" **stops early with work still
pending, and nothing raises.** `claim()` is now a single
`UPDATE ... WHERE item_id = (SELECT ...) RETURNING`, leaving contention to the
`SERIALIZABLE` retry loop, which is what CockroachDB expects of a client.
PostgreSQL keeps `SKIP LOCKED` in the subquery where it is well defined.

**8 tests, green on PostgreSQL, on a local CockroachDB node, and on CockroachDB
Cloud.**

---

## Deployment — free by construction

| | | cost |
|---|---|---|
| CockroachDB Basic | the memory, `aws-us-east-2` | $15/mo credit, scales to zero |
| S3 | 1.34 GB of clips, private + SSE | 5 GiB always free |
| Lambda | the agent worker, on a 5-minute tick | 1M requests always free |
| EventBridge | the tick | free |

No EC2, no NAT gateway, no API Gateway, no RDS — on an account created after
2025-07-15 those draw straight down the $200 of credits, and a NAT gateway alone
is ~$33/month for doing nothing.

*In-region matters, measured:* a vector query from a laptop to the us-east-2
cluster takes ~706 ms, nearly all of it network. From a Lambda in us-east-2 it
is a local hop. Retrieval latency here is a deployment property, not a database
one.

```bash
export PRECEDENT_DSN='postgresql://...cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full'
export BUCKET=precedent-clips-yourname
./showreel/deploy/deploy.sh
```

---

## Run it locally

```bash
.venv-libero/bin/python showreel/ingest.py
```

```bash
.venv-libero/bin/python showreel/run_agent.py --kinds 20 --per 50
```

```bash
.venv-libero/bin/python showreel/server.py     # / and /agent
```

```bash
.venv-libero/bin/python -m pytest showreel/test_precedent.py -q
```

---

## What this is honest about

**One domain.** 3,402 RoboCasa kitchen episodes. The retrieval claim is about
text-vs-example on this corpus, not about generalising to arbitrary CCTV.

**The "human" in the benchmark is the corpus's own folder name** — what a
reviewer would type. It is never read by retrieval, never stored on a clip, and
enters memory only as the answer to an escalation.

**The agent abstains a lot.** 60% of the queue still reaches a person at the
shipped setting. That is the honest cost of being right 96% of the time when it
does act, and the curve above lets you buy a different trade.

**The encoder is offline.** V-JEPA 2 + SigLIP 2 run as a batch job over new
video, not in the request path. Lambda reads vectors; it does not compute them.
