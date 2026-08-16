"""Brigade configuration.

One module, no framework. Everything is an env var with a sane default so the
system runs on a laptop with zero setup and points at CockroachDB Cloud by
flipping BRIGADE_DSN.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class WorldConfig:
    """The kitchen the robot lives in.

    layout/style are pinned so the demo is reproducible: the same kitchen every
    boot, which is what makes "it remembers where the bowls are" meaningful. A
    random kitchen per run would make memory worthless by construction.
    """

    layout_id: int = _env_i("BRIGADE_LAYOUT", 1)
    style_id: int = _env_i("BRIGADE_STYLE", 1)
    seed: int = _env_i("BRIGADE_SEED", 0)
    robot: str = "PandaOmron"
    control_freq: int = 20
    camera_h: int = _env_i("BRIGADE_CAM_H", 256)
    camera_w: int = _env_i("BRIGADE_CAM_W", 256)
    # agentview_left is the "over the shoulder" view a human watches;
    # eye_in_hand is what the robot uses to verify a grasp up close.
    cameras: tuple[str, ...] = ("robot0_agentview_left", "robot0_eye_in_hand")

    kitchen_id: str = _env("BRIGADE_KITCHEN_ID", "kitchen-1")
    robot_id: str = _env("BRIGADE_ROBOT_ID", "brigade-01")


@dataclass(frozen=True)
class MemoryConfig:
    """CockroachDB connection + memory policy."""

    # Local single-node CockroachDB in docker by default; one env var to point
    # at a CockroachDB Cloud cluster.
    dsn: str = _env(
        "BRIGADE_DSN",
        "postgresql://root@localhost:26257/brigade?sslmode=disable",
    )
    pool_min: int = _env_i("BRIGADE_POOL_MIN", 1)
    pool_max: int = _env_i("BRIGADE_POOL_MAX", 8)
    retry_max: int = _env_i("BRIGADE_RETRY_MAX", 5)

    text_model: str = _env("BRIGADE_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    text_dim: int = 384
    relmo_dim: int = 512

    # A belief older than this is suspect and must be re-verified by looking
    # before it is trusted for navigation.
    belief_ttl_s: float = _env_f("BRIGADE_BELIEF_TTL_S", 300.0)
    # Below this cosine similarity a recall counts as "I have no memory of this",
    # which is what triggers exploration in act 1.
    recall_floor: float = _env_f("BRIGADE_RECALL_FLOOR", 0.35)


@dataclass(frozen=True)
class AgentConfig:
    tick_s: float = _env_f("BRIGADE_TICK_S", 2.0)
    skill_timeout_s: float = _env_f("BRIGADE_SKILL_TIMEOUT_S", 45.0)
    # How many placements of the same label before we promote it to a norm.
    norm_promote_n: int = _env_i("BRIGADE_NORM_PROMOTE_N", 2)


@dataclass(frozen=True)
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    artifacts: str = _env(
        "BRIGADE_ARTIFACTS",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"),
    )


CFG = Config()
