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

from ..world.scene import SceneExporter

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

        self.env.reset()
        sim = self.env.envs[0]._env.env.sim
        self.scene = SceneExporter(sim)
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

    def look(self) -> list:
        """What is where, right now. See world/observe.py."""
        from ..world.observe import observe

        return observe(self.env.envs[0]._env.env.sim) if self.env is not None else []

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
            seed: int = 10000) -> EpisodeResult:
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
        obs, _ = env.reset(seed=[seed])
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
            done = terminated | truncated | done
            step += 1

            prev_obs = self.look()
            if on_frame is not None:
                on_frame(self.scene.frame_bytes(), step)
            if keep_frames and step % 2 == 0:
                frames.append(np.asarray(obs["pixels"]["image"][0], dtype=np.uint8))

        return EpisodeResult(
            instruction=instruction, success=succeeded, steps=step,
            seconds=time.time() - t0, suite=self.suite, task_id=self.task_id,
            frames=frames, final_obs=terminal_obs if terminal_obs else prev_obs,
            start_obs=start_obs,
        )
