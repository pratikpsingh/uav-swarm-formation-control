"""Controller evaluation over complete environment episodes."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from uav_swarm_control._arrays import Float32Array, immutable_float32_array
from uav_swarm_control.controllers.contracts import Controller
from uav_swarm_control.environments.contracts import MultiAgentEnvironment


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Summary of one deterministic controller rollout."""

    steps: int
    returns: Float32Array
    terminated: bool
    truncated: bool
    success: bool
    final_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "returns",
            immutable_float32_array(self.returns, name="episode returns", dimensions=1),
        )
        object.__setattr__(
            self,
            "final_metrics",
            MappingProxyType(dict(self.final_metrics)),
        )


def run_episode(
    environment: MultiAgentEnvironment,
    controller: Controller,
    *,
    seed: int,
    safety_step_limit: int,
) -> EpisodeResult:
    """Run until the environment ends, with a guard against broken adapters."""
    reset_result = environment.reset(seed=seed)
    observations = reset_result.observations
    returns = np.zeros(len(environment.agent_ids), dtype=np.float32)

    for step_number in range(1, safety_step_limit + 1):
        transition = environment.step(controller.act(observations))
        returns += transition.rewards
        observations = transition.observations
        if transition.episode_done:
            return EpisodeResult(
                steps=step_number,
                returns=returns,
                terminated=transition.terminated,
                truncated=transition.truncated,
                success=bool(transition.metrics.get("success", 0.0)),
                final_metrics=transition.metrics,
            )
    raise RuntimeError("environment did not finish within safety_step_limit.")
