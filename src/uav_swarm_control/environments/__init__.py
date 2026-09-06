"""Multi-agent environment interfaces and simulator adapters."""

from uav_swarm_control.environments.communication import CommunicationFormationEnvironment
from uav_swarm_control.environments.continuous_bandit import ContinuousTargetBandit
from uav_swarm_control.environments.contracts import (
    MultiAgentEnvironment,
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.environments.generalization import GeneralizedFormationEnvironment
from uav_swarm_control.environments.obstacles import ObstacleFormationEnvironment
from uav_swarm_control.environments.single_agent import (
    SingleAgentEnvironment,
    SingleAgentReset,
    SingleAgentStep,
)

__all__ = [
    "CommunicationFormationEnvironment",
    "ContinuousTargetBandit",
    "GeneralizedFormationEnvironment",
    "MultiAgentEnvironment",
    "NormalizedVelocityActions",
    "ObstacleFormationEnvironment",
    "ResetResult",
    "SingleAgentEnvironment",
    "SingleAgentReset",
    "SingleAgentStep",
    "StepResult",
]
