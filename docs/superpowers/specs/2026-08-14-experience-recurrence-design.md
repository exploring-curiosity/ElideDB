# Experience as a recurrence over expectation and outcome

2026-08-14. Design agreed with the owner in brainstorming.

## The problem this replaces

Six earlier signals (commitment, anticipation depth, stability, revision, fan
coherence, convergence rate) were rejected: every one collapses a vector to a
scalar, and every scalar measures the MODEL'S competence rather than the
event's identity. Two different events with equal predictability score
identically under all six. A confidence profile cannot separate experiences.

The error channel `pred(t+k) − actual(t+k)` is barred by standing rule: it is a
function of (event, model prior), so the same event yields different
descriptors as the prior shifts, and an index built from it drifts against
itself.

## Thesis

The forecast carries common sense that raw observation cannot. `act(t+1)−act(t)`
is only "what the pixels did"; `pred(t+1)−act(t)` is "what usually follows from
a state like this", compressed from a million hours of video. That is why a
cabinet opening and a microwave opening can land near each other: their typical
continuations are alike even when their pixels are not.

Supporting measurement: pred_change 0.525 [0.508,0.540] vs obs_change 0.490
[0.471,0.508], intervals disjoint — small but in the predicted direction.

The forecast survives the owner's own objection where error does not: error
drifts toward ZERO as the model improves (toward no signal), while the forecast
drifts toward the TRUTH (toward a stable description of the real dynamics).

**Expectation is the identity. Reality confirms the identity was realised.**

An experience is a PATH, not a set of moments, so the descriptor accumulates.

## Formulation

Per step, both relative to `act(t)` so the scene cancels:

    a_t = pred(t+1) − act(t)     expected change   (identity)
    b_t = act(t+1) − act(t)      realised change   (confirmation)

`pred(t)` is deliberately absent: including it would let f form the barred
error term.

Recurrence, fixed f, no learned parameters:

    g_t = max(0, <â_t, b̂_t>)                       confirmation gate
    n_t = â_t − <â_t, m_{t−1}>·m_{t−1}              novelty vs history
    m_t = normalize( λ·m_{t−1} + (1−λ)·g_t·â_t )    motif
    z_t = [ m_t ; n_t ; g_t ]                       emission

Roles follow what was measured: gating is worth +0.12 over no gate while the
gate is at ceiling (a perfect object mask buys +0.021), so reality GATES and
expectation CARRIES.

`n_t` is the only part unavailable to a history-free descriptor. The same â_t
emits differently depending on `m_{t−1}`, so "door halfway, moving" decomposes
one way after a closed door and another after an open one. That is precisely
the open/close confusion that has beaten this project (61–72% of drawer errors
are the same object reversed).

## λ by principle

λ = 1 − 1/K where K is the predictor's horizon (4 steps here), giving λ = 0.75.
The state's memory matches the timescale the model can actually forecast over;
integrating longer accumulates across a span it has no coherent view of.
Chosen from the architecture, NOT swept against the metric — the layer-6 choice
was made by peeking and that is already on record as the one hyperparameter
that saw labels.

## Data flow

`a_t` and `b_t` are already cached as `pred_change` and `obs_change` in the v4
records, gate-pooled and relative to act(t). The recurrence is pure numpy over
cached data — no re-encoding. `z_t` feeds the existing subsequence-DTW matcher
unchanged, so span retrieval and the time-warp property carry over untouched.

Everything is causal: `z_t` needs frames only up to t+1, so it runs online on a
live stream, and the accumulated z sequence IS the training data for the later
online-training phase.

## Arms to evaluate

    pred_change        current baseline, 0.525 (2.37x), chance 0.222
    m only             motif alone — if this matches baseline, recursion added nothing
    n only             novelty alone — isolates the history-conditioned part
    z = [m;n;g]        full emission

## Honest expectation

The audit found the gate at ceiling, the content carrying no recoverable
physics (R2 ~0.000 even under an oracle object mask), and fusion saturated over
nine schemes. `n_t` is the one genuinely new thing; if the number moves, that is
where it comes from. If `m` alone ties the baseline, the recursion is
decoration and we learn that in one run.
