"""Controller boundary shared by scripted and learned policies."""

from typing import Protocol, runtime_checkable

from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.observations import LocalObservations


@runtime_checkable
class Controller(Protocol):
    """A decentralized controller acting on batched local observations."""

    def act(self, observations: LocalObservations) -> NormalizedVelocityActions:
        """Return one normalized action for every observed agent."""
        ...
