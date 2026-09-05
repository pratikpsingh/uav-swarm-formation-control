"""Multi-agent environment interfaces and simulator adapters."""

from uav_swarm_control.environments.continuous_bandit import ContinuousTargetBandit
from uav_swarm_control.environments.contracts import (
    MultiAgentEnvironment,
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.environments.single_agent import (
    SingleAgentEnvironment,
    SingleAgentReset,
    SingleAgentStep,
)

__all__ = [
    "ContinuousTargetBandit",
    "MultiAgentEnvironment",
    "NormalizedVelocityActions",
    "ResetResult",
    "SingleAgentEnvironment",
    "SingleAgentReset",
    "SingleAgentStep",
    "StepResult",
]
