"""Minimal contracts for single-agent reinforcement-learning environments."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from uav_swarm_control._arrays import Float32Array, immutable_float32_array


@dataclass(frozen=True, slots=True)
class SingleAgentReset:
    """Observation returned when an episode begins."""

    observation: Float32Array

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation",
            immutable_float32_array(self.observation, name="observation", dimensions=1),
        )


@dataclass(frozen=True, slots=True)
class SingleAgentStep:
    """One transition with termination kept separate from truncation."""

    observation: Float32Array
    reward: float
    terminated: bool
    truncated: bool
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation",
            immutable_float32_array(self.observation, name="observation", dimensions=1),
        )
        if not isinstance(self.reward, float):
            raise TypeError("reward must be a float.")
        if not math.isfinite(self.reward) or any(
            not math.isfinite(value) for value in self.metrics.values()
        ):
            raise ValueError("reward and metrics must be finite.")
        if self.terminated and self.truncated:
            raise ValueError("a transition cannot be both terminated and truncated.")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def episode_done(self) -> bool:
        """Whether the environment must be reset before another action."""
        return self.terminated or self.truncated


@runtime_checkable
class SingleAgentEnvironment(Protocol):
    """Structural interface consumed by the PPO rollout collector."""

    @property
    def observation_size(self) -> int: ...

    @property
    def action_size(self) -> int: ...

    def reset(self, *, seed: int | None = None) -> SingleAgentReset: ...

    def step(self, action: Float32Array) -> SingleAgentStep: ...
