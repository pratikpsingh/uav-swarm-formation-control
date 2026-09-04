"""Simulator-independent multi-agent environment contracts."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from uav_swarm_control._arrays import (
    Float32Array,
    immutable_float32_array,
    validated_agent_ids,
)
from uav_swarm_control.agents import AgentId
from uav_swarm_control.observations import CentralizedState, LocalObservations


@dataclass(frozen=True, slots=True)
class NormalizedVelocityActions:
    """Per-agent 3D velocity commands normalized to the closed interval [-1, 1]."""

    agent_ids: Sequence[AgentId]
    values: Float32Array

    def __post_init__(self) -> None:
        values = immutable_float32_array(self.values, name="actions", dimensions=2)
        if values.shape[0] < 1 or values.shape[1] != 3:
            raise ValueError(
                f"actions must have shape (N, 3) with N >= 1; received {values.shape}."
            )
        if np.any(np.abs(values) > 1.0):
            raise ValueError("normalized actions must lie in the closed interval [-1, 1].")
        identifiers = validated_agent_ids(self.agent_ids, expected=values.shape[0])
        object.__setattr__(self, "agent_ids", identifiers)
        object.__setattr__(self, "values", values)

    @property
    def num_agents(self) -> int:
        """Number of agents represented in the action batch."""
        return self.values.shape[0]


def _immutable_metrics(values: Mapping[str, float]) -> Mapping[str, float]:
    metrics: dict[str, float] = {}
    for name, value in values.items():
        if not name or name.strip() != name:
            raise ValueError(
                "metric names must be non-empty and contain no surrounding whitespace."
            )
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"metric {name!r} must be finite.")
        metrics[name] = result
    return MappingProxyType(metrics)


@dataclass(frozen=True, slots=True)
class ResetResult:
    """Information returned at the beginning of an episode."""

    observations: LocalObservations
    centralized_state: CentralizedState


@dataclass(frozen=True, slots=True)
class StepResult:
    """One cooperative multi-agent transition.

    Terminated means the task reached an environment-defined terminal state.
    Truncated means an external limit, usually the episode horizon, stopped it.
    """

    observations: LocalObservations
    centralized_state: CentralizedState
    rewards: Float32Array
    terminated: bool
    truncated: bool
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        rewards = immutable_float32_array(self.rewards, name="rewards", dimensions=1)
        if rewards.shape != (self.observations.num_agents,):
            raise ValueError(
                "rewards must contain one value per agent; "
                f"received {rewards.shape} for {self.observations.num_agents} agents."
            )
        if self.terminated and self.truncated:
            raise ValueError("a transition cannot be both terminated and truncated.")
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "metrics", _immutable_metrics(self.metrics))

    @property
    def episode_done(self) -> bool:
        """Whether either terminal condition ended the episode."""
        return self.terminated or self.truncated


@runtime_checkable
class MultiAgentEnvironment(Protocol):
    """Boundary implemented by every simulator or lightweight environment."""

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        """Stable identifiers in the row order used by all arrays."""
        ...

    def reset(self, *, seed: int) -> ResetResult:
        """Start a deterministic episode from the supplied root seed."""
        ...

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        """Advance the environment by one control step."""
        ...

    def close(self) -> None:
        """Release simulator resources."""
        ...
