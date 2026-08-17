# The problem statement

Reconstructed from the owner's own instructions across the build, in their
words where they said something exactly. This is the spec the work is answerable
to — not a description of what got built.

---

## 1. The brief

Build an entry for the **CockroachDB × AWS agentic memory hackathon** (~4000
participants). It must be a **real-world use case**, and it has to be good
enough to win, not merely to submit.

Hackathon requirements are non-negotiable and must be **core, not bolted on**:

* at least **two CockroachDB capabilities** used meaningfully;
* at least **one AWS service**.

The starting point is the owner's existing work — **ElideDB / RelMo** — and
**RelMo must be used and marketed as a product**. Both CockroachDB and RelMo are
central. Neither is decoration.

Caveat the owner gave about their own system, which constrains how much can be
leaned on it:

> "dont go anything from memory about the elide db its the most disgusting
> broken system you built. works only barely, so dont overrely on that. it
> simply doesnt work good."

## 2. Hard constraints

| constraint | statement |
|---|---|
| budget | **$0.** "I am not paying." AWS free tier only; CockroachDB free tier only. No Bedrock. |
| time | **1 day.** |
| scope | Single system, one big kitchen. Not a survey of options. |
| honesty | "**No faking things.**" |

## 3. What the system must do

> "the robot must look and things in the kitchen, take action autonomously,
> follow instructions if given any and have a memory system. **The main part is
> it has to retrieve events from memory to remember things.** one big kitchen.
> different tasks getting done. and realtime."

Broken out:

1. **Perceive** the kitchen.
2. **Act autonomously** — decide and do, not be driven.
3. **Follow instructions** when given.
4. **Have a memory system**, and — the stated main part — **retrieve events from
   it to remember things**.
5. **One big kitchen**, several different tasks getting done.
6. **Realtime.**

A VLA is not mandatory ("VLA is not a must"), but if one is used the control
must be genuine.

## 4. What is explicitly forbidden

These were given as corrections, each after seeing something that failed them.
They are the sharpest part of the spec.

| # | forbidden | the owner's words |
|---|---|---|
| 1 | Memory as a one-shot lookup | "simply a robot shown, and reset and spawn from memory is not anything at all… **an agent has to do autonomous tasks**" |
| 2 | State-written world | "the doors are manipulated by states rather than the robot physics… **I want an agent operating the robot not the state**" |
| 3 | Fitted fluff | "**I dont want fitted fluff.**" |
| 4 | Hardwired actions | "**I dont want hardwired robot actions. thats not agentic**" |
| 5 | Untested improvisation | "dont waste time on handrolled approach and **untested not researched lazy fixes** that are one purpose" |
| 6 | Self-validation via own APIs | "**dont rely on your poorly written api's for validation.** Use vision and check for autonomous actions." |
| 7 | Shipping anything faked | "**I dont want something thats not fully agentic. its better to not submit than to submit a fake claim.**" |

Number 7 is a stop condition, and it was used once: the whole hand-rolled
kitchen control layer was halted and abandoned under it.

## 5. The pivot

After the halt:

> "forget the kitchen system. **I want any robot simulation with full autonomous
> control. just check whats already tested and released now**"

and, on being shown a two-arm handover demo:

> "two arms handing a cube across — is it just this one scene? **can it do
> something thats required in a real world?**"

So: an existing, released, tested control stack — not one written here — doing
work that resembles real household tasks.

## 6. Requirements added during this build

| ask | statement |
|---|---|
| Connect the pieces | "connect the reIMO and the local postgres" |
| Real UI | "**a web UI that is not simple html. interactive and 3d** where I can see the robot taking actions" |
| Prove the dependence | "the main reliance on memory has to be shown. **without memory it cant work** small example like that" |
| Spatial memory | "**robot remembers where the bowl is put** with memory on and off" |
| A real demo | "**design a complex act that can show the power of the memory**" |
| Live view | The 3D must actually update while the robot acts, not report completion after the fact |
| One world | "**this is not one continuos simulation. the state resets for every action. whats the use then.**" |

The last one is a requirement, not a comment: the simulation has to be
continuous, or the spatial memory describes a world that has already been
rewound.

## 7. How it gets judged

From the owner's own standards, in the order they were enforced:

1. **Is it genuinely agentic?** No scripted primitives, no state writes, no
   hardwired actions. Fail this and nothing else counts.
2. **Is memory load-bearing?** Not "we used a database" — the system must fail
   without it, demonstrably, on the same task.
3. **Are the numbers real?** Measured on this machine, reproducible, with the
   failures reported alongside the successes.
4. **Are RelMo and CockroachDB core?** Not adjacent, not optional.
5. **Does it look like something?** Interactive 3D, live.

## 8. Open against this spec

Stated plainly because the spec demands it.

* **AWS is not yet used.** Requirement 1 of the hackathon brief is unmet — the
  memory layer runs on local PostgreSQL 18 + pgvector, and the CockroachDB
  dialect is written and translated but the cluster has not been created.
  Both are account-creation steps, not engineering ones.
* **Spatial perception is privileged.** Object positions are read from
  `mjData.xpos`, not from the cameras, so the robot sees through closed drawers.
  The belief can be stale but never wrong, and nothing acts on it — it answers
  questions only. Fixing this means occlusion-tested perception.
* **Persistence costs accuracy.** A continuous kitchen puts π0.5 out of its
  training distribution: 97–100% from benchmark placements, ~50% from carried
  ones. Measured, and it is a real trade rather than a bug.
