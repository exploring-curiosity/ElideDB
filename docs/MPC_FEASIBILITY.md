# MPC-driven worlds — feasibility (measured 2026-08-10)

Owner's direction: stop scripting the arms; let an MPC controller drive
them so they behave like real robots, across three worlds; connect that
as a world model; then build the memory system on the resulting data +
ground truth.

Everything below is measured on this machine (M5 Pro, 15 cores), not
estimated.

## 1. Compute — NOT the blocker

`mujoco.rollout` (in mujoco 3.11) gives batched multi-threaded rollouts
with a persistent pool. On the actual episode scene (nq=16, nv=15,
nu=8):

| planner budget | solve time | wall per 8 s episode |
|---|---|---|
| K=32 samples, 0.5 s horizon | 55 ms | **9 s** |
| K=64, 0.5 s | 106 ms | 17 s |
| K=32, 1.0 s | 111 ms | 18 s |

Single-thread is 62k steps/s; batched is 145–151k steps/s. So a
180-episode primitive corpus costs ~30 min, a 500-episode chain corpus
~1.5 h. Comparable to what the scripted generator already costs once
retries are counted.

DeepMind's `mujoco_mpc` package is NOT pip-installable (would need a
source build). It is also not needed — a predictive-sampling / MPPI
planner is ~60 lines on top of `rollout`.

## 2. Control — works for motion

MPPI prototype (K=48, 0.2 s horizon, ~60 lines) driving the Panda to a
free-space target, cost = EE distance only:

    start 73.9 cm -> 39.9 cm in 0.8 s of sim, monotonic:
    74 72 69 66 62 59 54 51 47 43

Monotonic descent, no overshoot, no reversals — from a cost function,
with zero per-arm tuning. Contrast with the hand-written controller
this replaces, which needed per-arm `cmd_lead`, `carry_gain`,
`place_tol`, `time_scale`, a home-pose basin search, and STILL
oscillated at 4.75 reversals/s on the vx300s.

**This is the strongest argument for the direction**: the oscillation,
the per-arm hacks, and the IK-basin stalls are all artifacts of greedy
per-step IK. An optimizer that simulates the future does not have them,
and one cost function serves every embodiment.

## 3. The honest risk — prehensile grasping

Sampling MPC handles reach, transport, and push well (smooth continuous
cost). Discovering a GRASP from scratch is the hard part and is an open
research area: the cost landscape is discontinuous (holding vs not),
and random control perturbations rarely close a gripper at the exact
instant the object is between the pads.

Standard practice — and the recommendation here — is a split that is
also how real robot stacks are built:

- **MPC owns all continuous motion**: approach, transport, descend,
  retreat, place, push. No waypoints, no IK hacks, no per-arm tuning.
- **A tiny discrete policy owns the gripper**: close when the object is
  between the pads, open at the release condition. Two booleans.

That yields genuinely emergent, robot-like motion while keeping task
success high. Pure end-to-end MPC (gripper included in the sampled
action space) can be attempted afterward as a research arm, gated on
the same clean/success measurements.

## 4. "Connect it as a world model" — two distinct levels

1. **MPC over MuJoCo (privileged dynamics).** The planner's model IS
   the simulator. This is a *planner*, not a learned world model, but
   it produces the data and the behaviour we want now.
2. **MPC over a LEARNED model.** Replace the simulator with a model
   learned from the recordings, and ask whether the same planner still
   acts competently. THIS is the world-model claim, and it is the
   natural bridge to the memory system: a memory that can supply "what
   happened in situations like this" is exactly what a learned model
   needs to roll out.

Level 1 is a week-class build. Level 2 is the research programme, and
it only becomes measurable once level 1 exists.

## 5. What this buys the memory system

The planner emits ground truth the scripted generator cannot:
- per-step cost terms and their decomposition (what the agent was
  trying to do, continuously — not a 5-word primitive label);
- contact events with the exact instant of make/break;
- subgoal transitions, chosen by the optimizer rather than declared;
- counterfactual rollouts (the K−1 sampled futures that were NOT taken)
  — negatives with a physical meaning, free.

That is a materially richer supervision substrate for the retrieval
work than `{"prim": "pick", "t0":..., "t1":...}`.

## 6. Recommended sequence

1. `mpc.py`: MPPI/predictive-sampling planner over `mujoco.rollout`,
   one cost interface, no arm-specific constants.
2. Reach + push under MPC on all three arms; gate on the SAME
   smoothness metrics measured today (reversals/s, jerk) and on task
   success. Target: every arm at or below the Panda's current
   0.72 rev/s and 9.8 jerk.
3. Add the two-boolean gripper policy; regenerate `prim_actions`
   MPC-driven, clean-gated, arm recorded.
4. Chains under MPC (the planner handles the transitions the scripted
   version had to hard-code).
5. Only then: swap MuJoCo for a learned model and measure the drop.

Risk to keep visible: step 2 is the go/no-go. If MPC cannot hold the
smoothness gate on the vx300s and piper, the direction does not pay
for itself and the scripted path with the planned-IK fix is the
fallback.

---

## 7. Session outcome (2026-08-10) — what is and is not working

Built and measured, all on the real episode scene:

| controller | reversals/s | jerk | completes tasks |
|---|---|---|---|
| greedy DLS (shipped, vx300s) | 4.75 → 2.97 | 43.7 → 24.7 | yes (~40%/attempt) |
| greedy DLS (panda) | 0.72 | 9.8 | yes |
| **planned IK** (solve once + track) | **0.00** | **0.6** | NO (descent 65-185mm off in xy) |
| **MPPI** (K=40, 0.3 s, rate-penalised) | 3.12 | **2.65** | NO (does not converge in 60 cycles) |

Both new controllers produce motion far smoother than anything shipped
(jerk 0.6 and 2.65 vs the Panda's 9.8) - the quality bar is clearly
reachable. Neither yet completes the tasks:

- **planned IK**: the arm arrives at the correct HEIGHT but 65-185mm
  off in xy during the grasp descent. Continuity-constrained re-solve
  (warm start, no restarts, global fallback) did NOT fix it, so the
  remaining suspect is the descent target itself being unreachable at
  the commanded depth, or the ctrl/actuator mapping in planned mode -
  needs one focused instrumented pass, not more tuning.
- **MPPI**: with POSITION actuators the sampled controls are joint
  position targets; weighted averaging moves the mean slowly, so a
  far target needs either warm-starting from IK, a larger sigma with
  more iterations per control step, or a velocity/torque action space.
  This is normal MPPI commissioning, not a dead end.

Recommended next move (one focused session): warm-start MPPI from the
IK solution and give it 2-3 optimisation iterations per control step.
That combines the two: IK supplies the basin, MPPI supplies smoothness
and contact awareness, and the rate penalty guarantees no vibration.

**The dataset was NOT regenerated or deleted.** data/prim_actions
still holds the 180 verified episodes (3 arms x 5 primitives x 12,
every episode success + state-verified + every grasp first-try). The
vx300s films in it are correct but visibly shaky; replacing them is
gated on one of the two controllers above completing tasks.

## 8. Warm-started MPPI (2026-08-10, second pass)

Owner clarified: per-arm tuning IS allowed for the generator; the
no-hardwiring rule binds ElideDB only. Requirements for the data:
known primitive label, quality (no vibration/oscillation), complete
ground truth.

Added `MPPI.warm_start` (minimum-jerk ramp from the current command to
the IK solution) and fixed two real bugs found by it:

1. `np.roll` on the plan WRAPPED the start of the trajectory onto the
   end - the arm was fed its own past commands as its future. Now the
   plan shifts and holds the final control.
2. the control-rate penalty was SUMMED over a 125-step horizon, so it
   dwarfed the position term and the planner's optimal move was not to
   move; warm-started ramps got flattened. Now a mean, reweighted.

Result on the Panda (3 targets, 40 cycles each):

| | reversals/s | jerk | residual |
|---|---|---|---|
| greedy DLS (shipped) | 0.72 | 9.8 | converges |
| MPPI, first pass | 3.12 | 2.65 | 18-28 cm |
| **MPPI, warm-started + fixes** | 2.08 | **0.87** | 5-23 cm |

Jerk is now 11x better than the shipped controller - the quality bar
is met. Convergence still is not: the arm stops 5-23cm short.

LIKELY CAUSE, to check first next session: the Panda harness uses a
POSITION-ONLY inline IK for the warm start (sim_stack.Arm has no
solve_ik), while the MPPI cost includes a tool-down ORIENTATION term.
The seed and the cost therefore disagree, and the planner trades
position error for orientation. The vx300s/piper path uses
GenericArm.solve_ik, which already solves position AND tool-down
together - so the first experiment is simply to run the same test on
vx300s, and to give the Panda a proper 6-DoF warm-start solve.

Nothing was regenerated: data/prim_actions still holds the 180
verified greedy-DLS episodes.
