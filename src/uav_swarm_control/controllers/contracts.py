"""Controller boundary shared by scripted and learned policies."""

from typing import Protocol, runtime_checkable

from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.observations import CentralizedState, LocalObservations


@runtime_checkable
class Controller(Protocol):
    """A decentralized controller acting on batched local observations."""

    def act(self, observations: LocalObservations) -> NormalizedVelocityActions:
        """Return one normalized action for every observed agent."""
        ...


@runtime_checkable
class StateAwareController(Protocol):
    """A controller that explicitly requires a shared world-state snapshot."""

    def act_with_state(
        self,
        observations: LocalObservations,
        state: CentralizedState,
    ) -> NormalizedVelocityActions:
        """Return actions using information unavailable to a decentralized learned actor."""
        ...


@runtime_checkable
class EpisodeResettableController(Protocol):
    """Optional lifecycle hook used to prevent state leaking between evaluation episodes."""

    def reset(self) -> None:
        """Clear controller history and episode diagnostics."""
        ...
