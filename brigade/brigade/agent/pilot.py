"""Runs one episode of real robot control, on an instruction chosen at runtime.

`lerobot-eval` can only run a task's *own* instruction: it reads the string out
of the benchmark and hands it to the policy. Brigade needs the opposite — the
scene is fixed and the **instruction is whatever memory resolved** — so the
rollout loop is reproduced here.

It is reproduced rather than reimplemented, deliberately. Every processor comes
from the same `lerobot` factories the benchmark used, in the same order, so the
measured 98.0% over 400 episodes still describes this code path. A hand-rolled
observation pipeline would silently invalidate that number, and then a failure
in the demo could not be attributed to anything.

The instruction is injected at the one place the policy actually reads it:
`add_envs_task` in lerobot pulls `env.envs[0].task_description` on every step,
so setting that attribute is not a hack around the API, it *is* the API.

Threading note (macOS): MuJoCo renders through CGL, whose context belongs to the
thread that made it. Every method here must therefore be called from the same
thread — the process's main thread — while the web server runs elsewhere. The
3D view does not go through here at all; `world/scene.py` reads mjData directly
and needs no context.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..world.scene import SceneExporter, _raw_model

log = logging.getLogger("brigade.pilot")

POLICY = "lerobot/pi05_libero_finetuned"


@dataclass
class EpisodeResult:
    instruction: str
    success: bool
    steps: int
    seconds: float
    suite: str
    task_id: int
    frames: list = field(default_factory=list)   # RGB uint8, for the video memory
    # Where everything was at the END of the episode, captured mid-rollout
    # because the vector env resets on the step after termination.
    final_obs: list = field(default_factory=list)
    start_obs: list = field(default_factory=list)
    # True when the scene's goal already held before the robot moved.
    already_satisfied: bool = False
    # The world as the episode left it, for the next episode to continue from.
    end_state: object = None

    def to_json(self) -> dict:
        return dict(instruction=self.instruction, success=self.success,
                    steps=self.steps, seconds=round(self.seconds, 2),
                    suite=self.suite, task_id=self.task_id)


class Pilot:
    """The policy plus one open scene. Load once, run many episodes."""

    def __init__(self, policy_path: str = POLICY, device: str = "mps",
                 n_action_steps: int = 10):
        self.policy_path = policy_path
        self.device = device
        self.n_action_steps = n_action_steps
        self.policy = None
        self.suite: str | None = None
        self.task_id: int | None = None
        self.env = None
        self.scene: SceneExporter | None = None
        self._proc = None

    # ---- setup --------------------------------------------------------------

    def load(self) -> None:
        """Load pi0.5 (3.6B, ~40 s). Announced, because a silent 40 s looks hung."""
        if self.policy is not None:
            return
        from lerobot.configs.types import FeatureType  # noqa: F401  (import side effects)
        from lerobot.envs.factory import make_env_config
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        log.info("loading %s on %s (3.6B params, ~40s)", self.policy_path, self.device)
        t0 = time.time()
        env_cfg = make_env_config("libero", task="libero_90", task_ids=[0])
        from lerobot.configs.policies import PreTrainedConfig

        cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        cfg.pretrained_path = self.policy_path
        cfg.device = self.device
        cfg.n_action_steps = self.n_action_steps
        # torch.compile's inductor backend has no valid MPS lowering and dies
        # with NoValidChoicesError; eager is the only option on this machine.
        if hasattr(cfg, "compile_model"):
            cfg.compile_model = False

        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        pre, post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=self.policy_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self._proc = (pre, post)
        self._env_cfg_type = env_cfg
        log.info("policy ready in %.0fs", time.time() - t0)

    def open_scene(self, suite: str, task_id: int) -> dict:
        """Build the LIBERO scene. Returns the three.js scene header.

        The scene, not the task: what the robot is asked to do inside it is
        decided per episode by memory.
        """
        from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors

        if self.env is not None and (self.suite, self.task_id) == (suite, task_id):
            return self._scene_header

        self.close()
        env_cfg = make_env_config("libero", task=suite, task_ids=[task_id],
                                  max_parallel_tasks=1)
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        self.env = envs[suite][task_id]
        self.suite, self.task_id = suite, task_id
        self._env_proc = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=self.policy.config)

        # The benchmark's own placements, kept pristine: carry() overwrites
        # _init_states, and the canonical ROBOT pose has to come from somewhere.
        self._benchmark_states = np.asarray(self.env.envs[0]._init_states).copy()

        self.env.reset()
        # A getter, not the sim: robosuite replaces the MjSim on reset, so
        # anything holding the object streams a world that has stopped updating.
        self.scene = SceneExporter(self._sim)
        header, blob = self.scene.scene(task=self.default_instruction)
        self._scene_header, self._scene_blob = header, blob
        log.info("scene %s/%s open: %d visual geoms, %.1f MB meshes",
                 suite, task_id, header["n_geoms"], len(blob) / 1e6)
        return header

    @property
    def default_instruction(self) -> str:
        """The benchmark's own string for this scene — used only as a label."""
        return getattr(self.env.envs[0], "task_description", "") if self.env else ""

    @property
    def scene_payload(self) -> tuple[dict, bytes]:
        return self._scene_header, self._scene_blob

    def _sim(self):
        """The CURRENT MjSim. Always resolved fresh — see SceneExporter."""
        return self.env.envs[0]._env.env.sim

    def _libero(self):
        """LIBERO's own env wrapper, which owns state save/restore."""
        return self.env.envs[0]._env

    # ---- the memory camera --------------------------------------------------

    MEM_CAM = "agentview"
    MEM_RES = 512

    def memory_frame(self, cam: str | None = None) -> np.ndarray:
        """A high-resolution frame for the MEMORY, not for the policy.

        The clips written to memory were previously whatever the policy ate —
        256x256, because that is what pi0.5 wants. That resolution is a property
        of the POLICY, and the memory has no reason to inherit it: measured, a
        reader could pick the cabinet and the wine rack off those clips but not
        the stove, the plate, or inside-the-bowl, and end-weighted sampling made
        no difference. Fine detail simply was not in the pixels.

        This is a separate render pass at 512, costing ~5 ms, and it leaves the
        policy's own observation pipeline completely untouched — verified, the
        256 observations and the GL context both survive it.
        """
        img = self._sim().render(width=self.MEM_RES, height=self.MEM_RES,
                                 camera_name=cam or self.MEM_CAM)
        # MuJoCo renders bottom-up.
        return np.asarray(img, dtype=np.uint8)[::-1]

    # ---- world continuity ---------------------------------------------------

    def save_state(self) -> np.ndarray:
        """The whole MuJoCo state as one flat vector (qpos + qvel)."""
        return np.asarray(self._libero().get_sim_state(), dtype=np.float64).copy()

    # qpos / dof widths by mjtJoint: free, ball, slide, hinge.
    _NQ = (7, 4, 1, 1)
    _NV = (6, 3, 1, 1)
    # robosuite's naming convention for the arm and its gripper, across robots.
    _ROBOT_PREFIX = ("robot0_", "gripper0_")

    def world_only(self, saved, base) -> np.ndarray:
        """`base` with the WORLD's joints taken from `saved`. -> flat state.

        Carrying the raw `get_sim_state()` carries the robot as well, because it
        is the whole simulation: 7 arm joints and 2 gripper joints sit at the
        front of qpos. Each episode then began with the arm wherever the last one
        stopped — extended, or still gripping — and pi0.5 has only ever started
        from the canonical home pose. Measured: with the full state carried,
        every action beat of the demo failed at the step limit, including
        pressing the stove button, which is otherwise trivial.

        Splitting it is also just correct. A household robot returns to a
        neutral pose between jobs; the kitchen is what keeps its state. So the
        arm is reset and everything else persists — and "everything else" here
        includes the three cabinet drawers and the stove knob, so a drawer left
        open stays open and a lit stove stays lit.

        World velocities are zeroed: between two commands the kitchen is at
        rest, and restoring mid-tumble momentum would have objects drift during
        the settling steps that follow every reset.
        """
        from mujoco import mj_id2name, mjtObj

        m = _raw_model(self._sim())
        saved = np.asarray(saved, dtype=np.float64)
        out = np.asarray(base, dtype=np.float64).copy()
        if saved.shape != out.shape:
            raise ValueError(f"state width {saved.shape} != scene's {out.shape}")

        nq = int(m.nq)
        for j in range(int(m.njnt)):
            name = mj_id2name(m, mjtObj.mjOBJ_JOINT, j) or ""
            if name.startswith(self._ROBOT_PREFIX):
                continue
            t = int(m.jnt_type[j])
            qa, na = int(m.jnt_qposadr[j]), self._NQ[t]
            da, nd = int(m.jnt_dofadr[j]), self._NV[t]
            out[1 + qa: 1 + qa + na] = saved[1 + qa: 1 + qa + na]
            out[1 + nq + da: 1 + nq + da + nd] = 0.0
        return out

    def carry(self, state) -> None:
        """Continue the NEXT episode from `state` instead of the benchmark's.

        LIBERO episodes are independent by design: `reset()` restores one of the
        benchmark's 50 stored initial placements, so anything the robot achieved
        is undone before the next request. That makes a spatial memory
        decorative — it can describe what happened, but nothing depends on the
        description, because the world it describes has already been rewound.

        This makes the kitchen persistent by giving LeRobot's reset path our own
        state where it expects the benchmark's. Nothing is reimplemented: reset
        still restores, still settles the scene with no-op actions, still
        rebuilds observations. It just restores where the robot actually left
        things.
        """
        if state is None:
            return
        sub = self.env.envs[0]
        base = self._benchmark_states[0]
        # Layout is derived from the model. All ten goal scenes carry the same
        # objects so it matches, but a mismatch would silently scatter the
        # kitchen rather than fail, so world_only() checks it.
        sub._init_states = self.world_only(state, base).reshape(1, -1)
        sub.init_state_id = 0

    def restore(self, state) -> None:
        """Put the live simulator INTO `state`, right now, without a reset.

        Needed because the vector env autoresets after an episode: left alone,
        the sim ends up holding whatever state was carried IN, so the 3D view
        visibly snaps back to before the action and `look()` reports a world one
        episode out of date. Restoring makes the live sim the present again,
        which is what lets "where is the bowl?" be checked against the actual
        kitchen rather than against a record kept on the side.
        """
        if state is None:
            return
        self._libero().set_init_state(
            self.world_only(state, self._benchmark_states[0]))

    def goal_satisfied(self) -> bool:
        """Is the current scene's goal ALREADY true, before doing anything?

        Only a meaningful question once the world persists, and then an
        important one: a robot asked to put away something already put away
        should say so, not mime the task and collect a free success.
        """
        return bool(self._libero().check_success())

    def idle(self, seconds: float, on_frame=None, fps: float = 10.0) -> int:
        """Let the cameras run with nobody driving the robot.

        The memory records all the time, not only while a task executes — a
        memory that only records during tasks cannot answer a question about
        the time between them, and it makes the memory depend on the agent
        correctly deciding when something interesting is beginning.

        Physics is NOT stepped. Two reasons, and they point the same way. The
        gym wrapper counts every step against the episode limit and autoresets
        when it hits one, which would rewind the kitchen precisely when nothing
        was happening. And stepping the raw MjSim without the controller running
        drops the arm under gravity — the robot would visibly collapse whenever
        it was idle. Between commands the kitchen is genuinely at rest (world
        velocities are zeroed by `world_only`), so a re-render per tick is what
        an idle camera actually sees.
        """
        n = int(seconds * fps)
        for i in range(n):
            self._sim().forward()
            if on_frame is not None:
                on_frame(self.memory_frame(), i)
        return n

    def look(self) -> list:
        """What is where, right now. See world/observe.py."""
        from ..world.observe import observe

        return observe(self._sim()) if self.env is not None else []

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = self.scene = None

    # ---- the episode --------------------------------------------------------

    def run(self, instruction: str, *, on_frame: Callable[[bytes, int], None] | None = None,
            max_steps: int | None = None, keep_frames: bool = True,
            seed: int = 10000, from_state=None) -> EpisodeResult:
        """One episode, driven by `instruction`.

        `on_frame` receives the packed geom transforms every control step; that
        is what the browser draws. It is called synchronously and must be cheap
        — the sim is the thing on a clock, not the viewer.
        """
        from lerobot.envs.utils import add_envs_task, preprocess_observation
        from lerobot.utils.constants import ACTION

        assert self.policy is not None and self.env is not None, "load() and open_scene() first"
        pre, post = self._proc
        env_pre, env_post = self._env_proc
        env = self.env

        # THE injection point: lerobot reads this attribute on every step.
        env.envs[0].task_description = instruction

        self.policy.reset()
        # Continue the persistent world rather than rewinding to the benchmark's
        # stored placement. See carry().
        self.carry(from_state)
        obs, _ = env.reset(seed=[seed])
        already = self.goal_satisfied()
        # AFTER the reset: `before` captured by a caller would otherwise be the
        # previous episode's leftovers, since run() resets internally.
        start_obs = self.look()
        limit = max_steps or int(env.call("_max_episode_steps")[0])

        frames: list = []
        succeeded = False
        t0 = time.time()
        step = 0
        done = np.array([False])
        # The terminal state has to be grabbed DURING the rollout, not after it.
        # Gymnasium's vector env runs AutoresetMode.NEXT_STEP, so the step that
        # reports termination has already put the kitchen back to its start:
        # traced directly, the bowl reads z=1.138 on the cabinet at step 86 and
        # z=0.898 on the table at step 87. Observing after run() returns
        # therefore records "nothing moved" for every successful episode, and
        # every belief written from it is wrong in the same silent way.
        prev_obs: list = []
        terminal_obs: list | None = None
        # The world state to hand to the next episode. Captured on the same
        # schedule as terminal_obs and for the same reason: one step later the
        # vector env has already rewound the kitchen.
        prev_state = None
        terminal_state = None

        while not done.all() and step < limit:
            o = add_envs_task(env, preprocess_observation(obs))
            o = pre(env_pre(o))
            with torch.inference_mode():
                action = post(self.policy.select_action(o))
            act = {ACTION: action}
            a = env_post(act)[ACTION].to("cpu").numpy()

            obs, _reward, terminated, truncated, info = env.step(a)

            if "final_info" in info:
                # LIBERO's success flag can flicker as contacts settle, so the
                # episode counts as a success if it was EVER satisfied — the
                # same rule lerobot-eval applies ("b n -> b", "any").
                succeeded = succeeded or bool(np.any(info["final_info"]["is_success"]))
                if terminal_obs is None:
                    # The state entering this step, i.e. before the reset it may
                    # have just performed.
                    terminal_obs = prev_obs
                    terminal_state = prev_state
            done = terminated | truncated | done
            step += 1

            prev_obs = self.look()
            prev_state = self.save_state()
            if on_frame is not None:
                on_frame(self.scene.frame_bytes(), step)
            if keep_frames and step % 2 == 0:
                # The MEMORY camera, not the policy's. See memory_frame().
                frames.append(self.memory_frame())

        return EpisodeResult(
            instruction=instruction, success=succeeded, steps=step,
            seconds=time.time() - t0, suite=self.suite, task_id=self.task_id,
            frames=frames, final_obs=terminal_obs if terminal_obs else prev_obs,
            start_obs=start_obs, already_satisfied=already,
            end_state=terminal_state if terminal_state is not None else self.save_state(),
        )
