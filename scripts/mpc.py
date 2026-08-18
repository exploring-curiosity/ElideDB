"""MPPI planner: the arms are driven by OPTIMISATION, not by scripted IK.

Why this replaces the hand-written controller (all measured 2026-08-10):
  - greedy per-step DLS + a lead clamp oscillated at 4.75 reversals/s
    with 43.7 jerk on the vx300s (Panda 0.72 / 9.8) - visible shaking;
  - it needed per-arm cmd_lead / carry_gain / place_tol / time_scale /
    an IK-basin home search, and STILL stalled 65-185mm off;
  - none of that is the arm: multi-restart IK reaches every zone
    tool-down at 0mm. It was the controller.

An optimiser that SIMULATES THE FUTURE has none of those failure
modes, and smoothness is not tuned - it is a term in the cost:
control-rate and joint-velocity penalties make vibration expensive, so
the planner does not produce it. One cost function serves every
embodiment; nothing here is arm-specific.

Cost (per sampled rollout, evaluated on a subsampled horizon):
    position    ‖ee - target‖               the task
    tool-down   axis misalignment            keeps the gripper vertical
    velocity    ‖qvel‖                       arrive settled, not fast
    rate        ‖u_t - u_{t-1}‖              THE anti-vibration term
"""
from __future__ import annotations

import numpy as np
import mujoco
from mujoco import rollout

FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


class MPPI:
    def __init__(self, m, ee_body, arm_dof, arm_act, grip_act=None,
                 K=40, horizon=0.30, sub=8, sigma=0.30, lam=0.06,
                 w_pos=1.0, w_ori=0.25, w_vel=0.02, w_rate=8.0):
        self.m = m
        self.ee = ee_body
        self.dof = list(arm_dof)
        self.act = list(arm_act)
        self.grip = grip_act
        self.K = K
        self.H = int(horizon / m.opt.timestep)
        self.sub = sub
        self.sigma = sigma
        self.lam = lam
        self.w = (w_pos, w_ori, w_vel, w_rate)
        self.ds = [mujoco.MjData(m) for _ in range(K)]
        self.ms = [m] * K
        self.nstate = mujoco.mj_stateSize(m, FULL)
        self._scratch = mujoco.MjData(m)
        self.U = None
        self.rng = np.random.default_rng(0)

    def reset(self, d):
        self.U = np.tile(d.ctrl.copy(), (self.H, 1))

    def _cost(self, states, U, target):
        w_pos, w_ori, w_vel, w_rate = self.w
        sc = self._scratch
        C = np.zeros(len(states))
        idx = list(range(0, self.H, self.sub)) + [self.H - 1]
        for k in range(len(states)):
            c = 0.0
            for h in idx:
                mujoco.mj_setState(self.m, sc, states[k, h], FULL)
                mujoco.mj_kinematics(self.m, sc)
                p = sc.xpos[self.ee]
                c += w_pos * float(np.linalg.norm(p - target))
                axis = sc.xmat[self.ee].reshape(3, 3) @ np.array(
                    [0.0, 0.0, 1.0])
                c += w_ori * float(np.linalg.norm(
                    axis - np.array([0.0, 0.0, -1.0])))
                c += w_vel * float(np.linalg.norm(
                    sc.qvel[self.dof]))
            # smoothness lives in the COST, so the planner never emits
            # the chatter the greedy controller had to be tuned out of
            # MEAN, not sum: summed over a 125-step horizon the rate
            # term dwarfed the position term and the planner's best
            # move was to not move (measured: warm-started ramps got
            # flattened and the arm stalled 7-19cm short)
            du = np.diff(U[k][:, self.act], axis=0)
            c += w_rate * float(np.square(du).mean())
            C[k] = c
        return C

    def warm_start(self, d, q_goal, act_idx):
        """Seed the nominal control with a MINIMUM-JERK ramp from the
        current command to the IK solution. This is the fix for the
        measured MPPI failure: with POSITION actuators a sampled
        control is a joint-position target, so Gaussian noise explores
        only a small neighbourhood and the weighted mean crawls - 60
        cycles left the arm 18-28cm short. IK supplies the BASIN (it
        reaches every zone tool-down at 0mm); MPPI then only has to
        refine locally, which is what sampling is good at. The ramp
        itself is already smooth, so the planner starts from a
        vibration-free trajectory instead of discovering one.
        Per-arm tuning is allowed here - this is the data generator,
        not ElideDB."""
        q0 = np.array([d.ctrl[a] for a in act_idx])
        H = self.H
        t = np.linspace(0.0, 1.0, H)
        # min-jerk scalar profile 10t^3-15t^4+6t^5
        prof = 10 * t**3 - 15 * t**4 + 6 * t**5
        U = np.tile(d.ctrl.copy(), (H, 1))
        for i, a in enumerate(act_idx):
            U[:, a] = q0[i] + (q_goal[i] - q0[i]) * prof
        self.U = U

    def step(self, d, target, grip_cmd=None, apply=8):
        """One planning cycle: sample, weight, apply the first `apply`
        controls. Returns the EE distance after applying."""
        if self.U is None:
            self.reset(d)
        s = np.zeros(self.nstate)
        mujoco.mj_getState(self.m, d, s, FULL)
        noise = np.zeros((self.K, self.H, self.m.nu))
        # perturb ARM actuators only; the gripper is a discrete
        # decision, not something to explore with noise
        noise[:, :, self.act] = self.rng.normal(
            0, self.sigma, (self.K, self.H, len(self.act)))
        Uk = self.U[None] + noise
        if self.grip is not None and grip_cmd is not None:
            Uk[:, :, self.grip] = grip_cmd
        lo, hi = self.m.actuator_ctrlrange[:, 0], \
            self.m.actuator_ctrlrange[:, 1]
        lim = self.m.actuator_ctrllimited.astype(bool)
        Uk[:, :, lim] = np.clip(Uk[:, :, lim], lo[lim], hi[lim])
        S = np.tile(s, (self.K, 1))
        states, _ = rollout.rollout(self.ms, self.ds, S, Uk,
                                    persistent_pool=True)
        C = self._cost(states, Uk, target)
        wgt = np.exp(-(C - C.min()) / self.lam)
        wgt /= max(wgt.sum(), 1e-12)
        self.U = (wgt[:, None, None] * Uk).sum(0)
        for h in range(apply):
            d.ctrl[:] = self.U[h]
            if self.grip is not None and grip_cmd is not None:
                d.ctrl[self.grip] = grip_cmd
            mujoco.mj_step(self.m, d)
        # shift the plan forward and HOLD the final control; np.roll
        # wraps the start of the trajectory onto the end, which fed
        # the arm its own past commands as its future
        self.U = np.concatenate(
            [self.U[apply:], np.tile(self.U[-1], (apply, 1))])
        mujoco.mj_kinematics(self.m, d)
        return float(np.linalg.norm(d.xpos[self.ee] - target))
