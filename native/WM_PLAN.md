# RelMo-WM v2 — the world model, by implementation

Owner's specification (2026-08-12), taken literally:
1. understand the STRUCTURE of a robot — pincers, joints, and across
   different robot types;
2. given a 3D trajectory, predict the next trajectory IN 3D;
3. "predict the future based on the current state".

Everything below serves those three. Where I disagree with the owner's
phrasing it is stated, not silently reinterpreted.

---

## 0. One correction, stated up front

"Predict future **actions**" is a policy. A world model predicts future
**states** given the current state (and, if available, an action).

But the owner's instinct points at something real and we should build
it: when actions are NOT labelled — which is always, on real video — a
world model can infer a **latent action**: the low-dimensional cause
that explains the transition s_t -> s_{t+1}. That is Genie / LAPO
territory. And it is a gift for this product, because two moments are
"the same kind of thing happening" exactly when their latent action
sequences match. So we predict states, and we *infer* actions, and the
inferred action sequence becomes a retrieval key.

---

## 1. State: what the model actually sees

**3D point tracks, in a canonical scene frame — not pixels, not 2D.**

    X   (T, P, 3)   3D position of P tracked points over T frames
    V   (T, P)      visibility
    (optionally) per-point local geometry: neighbour offsets

Lifting to 3D:
- TRAIN: exact. Every source now gives depth + camera pose (MuJoCo/
  RoboCasa render it; Kubric ships it; ARCTIC has meshes + camera).
- EVAL on real video: there IS no depth. Two options were considered.
  A pretrained monocular depth model was MEASURED in this project to
  carry a rest-on-surface prior that erases the lifted state exactly
  when it matters (true rises read +0.00). So instead the encoder is
  trained to LIFT INTERNALLY: 2D tracks in, 3D in the latent, with the
  sim's depth supervising the lift during training only. The 3D head
  is then available at eval with no depth model in the loop.

Canonicalisation (this is what makes "different robot types" work):
- translate to the scene centroid, rotate to a per-episode frame
  estimated from the dominant static structure;
- divide all lengths by a per-episode scale s (median inter-entity
  distance) and all times by dt.
  => a 1.2 m arm and a 0.3 m arm produce the same numbers. A car at
  15 m/s becomes 0.35 car-lengths/frame; a hand at 0.1 m/s becomes
  0.03 hand-lengths/frame. The 150x gap collapses to ~10x, which a
  network can hold. WITHOUT this, cross-embodiment is hopeless.

---

## 2. Structure: how "pincers and joints" are discovered

This is the heart of the owner's request, and it is NOT a labelled
classifier over robot types — that would never transfer. It is
geometry, learned.

### 2a. Rigid parts by motion coherence
Points on one rigid link share ONE rigid motion. So: for a candidate
group g and frames t, t+k, fit the SE(3) transform that best maps
X_t[g] -> X_{t+k}[g] (Procrustes / Kabsch, closed form, differentiable).
The residual IS the grouping signal — low residual means "these points
move as one body".

Slot attention over points produces the candidate groups by
competition (no threshold, no k chosen by hand); the Kabsch residual
supervises the assignment. Sim gives per-point body ids, so during
TRAINING this is directly supervised; at eval it runs unsupervised.

### 2b. Joints by relative screw motion
For two parts A, B take the relative transform trajectory
T_AB(t) = T_A(t)^-1 T_B(t). Its screw decomposition classifies the
connection with no learning at all:

FRAME MATTERS, and getting it wrong guarantees failure. A hinge axis
is fixed in the PART'S OWN frame, not in world or camera coordinates:
a person rotating a bottle while opening it sweeps the axis through
world space. MEASURED on ARCTIC across 10 sequences - axis std 0.4332
in camera frame, 0.4154 in world frame, 0.0000 in the object frame
(10/10 stable, mean axis [0,0,+/-1] matching ARCTIC's declared
z_axis). So the joint head classifies T_AB expressed in A's frame.

    T_AB constant over time            -> RIGID (same body, merge)
    rotation about a FIXED axis        -> REVOLUTE  (elbow, hinge, lid)
    translation along a FIXED axis     -> PRISMATIC (slider, drawer,
                                          and GRIPPER FINGERS)
    axis wanders / full 6 DoF          -> FREE (not connected)

A **pincer** is then a specific, checkable pattern, not a label:
two parts in near-mirror prismatic (or revolute) motion about a shared
axis, whose closure coincides with a third object's relative motion
going to zero — i.e. the moment the object becomes rigidly attached to
them. That is grasp, defined physically. It transfers to a two-finger
gripper, a parallel jaw, a suction cup that has no fingers at all, and
a human hand, because the definition never mentions fingers.

Sim gives joint type + axis for supervision (training only). The screw
fit is a differentiable head so the learned features must make it easy.

### 2c. The graph
Nodes  = discovered parts: pose, velocity, extent, point count
Edges  = (i) articulation edges from 2b, with type + axis
         (ii) contact/proximity edges, gated on distance
This is the Interaction Network / FIGNet formulation, with articulation
as a first-class edge type. Message passing over a graph generalises
over graph SIZE, which is exactly why a 7-DoF arm and a 6-DoF arm and
a hand can share one model.

---

## 3. Dynamics: the rollout

    z_t      = Encode(X_{t-K..t}, V)        space-time encoder
    parts    = Slots(z_t)                    -> nodes
    edges    = Screw(parts) + Contact(parts)
    z_{t+1}  = f(z_t, parts, edges)          RECURRENT transition
    ...      applied AUTOREGRESSIVELY for H steps
    dPose_i  = Decode(z_{t+h}, part_i)       SE(3) delta per part
    X_hat    = apply dPose to each part's points

Two properties the previous version lacked:
- the transition is applied step by step, so error compounds the way
  physics does and the latent must actually carry state;
- prediction is per-PART SE(3), not per-point free displacement, so a
  rigid link cannot deform. Rigidity is architectural, not learned.

Loss on the rollout is in LATENT space against an EMA target encoder
(JEPA-style: predicting raw coordinates spends capacity on
unpredictable detail), PLUS the explicit 3D trajectory decode so the
owner's requirement 2 is directly optimised and directly measurable.

### Latent action
A small inverse model a_t = g(z_t, z_{t+1}) with an information
bottleneck; the forward model is conditioned on a_t. Trained
self-supervised. At eval a_t is inferred from observation alone. The
sequence (a_t) is the compact "what was done" descriptor.

---

## 4. Losses

| loss | source | at eval? |
|---|---|---|
| rollout in latent vs EMA target | self-supervised | — |
| 3D trajectory error over H steps | self-supervised | measurable |
| arrow of time | self-supervised | — |
| part segmentation | sim body ids | dropped |
| joint type + axis | sim joint spec | dropped |
| contact onset | sim contacts | dropped |
| depth / lift | sim depth | dropped (head kept) |

Sim labels are SCAFFOLDING: they shape the representation during
training and are gone at eval, which is exactly the owner's rule —
sim state may train, nothing labelled may be required to serve.

---

## 5. Gates (a stage does not ship until its gate passes)

Every gate quotes the trivial baseline beside the model number. This
project has already been burned once by a loss curve that looked
healthy while the model was tying "predict stillness" 1.00x.

- G0 DATA   const-velocity R2 on the 3D targets > 0 at h=1. Proves the
            target is learnable before a step is spent.
- G1 LIFT   3D lift error vs sim depth, beaten against
            "predict the median scene depth".
- G2 PARTS  part segmentation vs sim body ids, beaten against
            "everything is one body" AND "every point its own body".
- G3 JOINTS joint-type accuracy + axis angular error, beaten against
            the majority class.
- G4 ROLL   multi-step 3D trajectory R2 > 0 and rising with capacity,
            beaten against const-velocity AND predict-stillness at
            EVERY horizon step, not just the mean.
- G5 XFER   all of the above on the SEALED arm corpus, never trained
            on. Plus RoboCasa-only vs full-portfolio, so corpus
            narrowness is measured, not argued.
- G6 PRODUCT yield/precision on the retrieval benchmark, against the
            standing symbolic baseline 0.510.

---

## 6. Corpus roles (no source does everything)

| source | unique contribution |
|---|---|
| RoboCasa | multi-phase agent-driven manipulation, 377 task classes, articulated fixtures, contact forces. THE core. |
| ARCTIC | non-rigid agent, bimanual, articulated objects, dense contact — and a completely different embodiment |
| physgen_v3 | driven contact with NO manipulator (push/lift/sweep/press) |
| MOVi-E | ballistic, collision cascades, scanned non-convex geometry, moving camera |
| arm corpus | SEALED. eval only. |

---

## 7. Where FDNN goes

The per-part SE(3) decoder — the head that turns a latent into a
trajectory. Its oscillatory bases are a catalogue of exactly that
physics: poly-chirp = acceleration, FINER = rolling/bouncing, Gabor =
contact as a burst. It stays an A/B against an MLP head with the same
budget, decided by G4, and it is NOT put inside the recurrence (this
project measured high omega inside recurrence producing chaotic
gradients).

---

## 8. Order of work

1. RoboCasa ingest -> episodes + exact GT, 0.000px acceptance gate
2. G0 on RoboCasa 3D targets
3. Encoder + lift (G1)
4. Structure heads (G2, G3)
5. Rollout + latent action (G4)
6. Sealed-corpus transfer + corpus ablation (G5)
7. Retrieval readout (G6)
