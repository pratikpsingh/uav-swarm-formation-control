"""Stateful decentralized execution wrapper for the recurrent actor."""

import numpy as np
import torch

from uav_swarm_control.actions import DirectionSpeedActions, direction_speed_to_normalized_velocity
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.models.recurrent_actor_critic import (
    PaperRecurrentActorCritic,
    RecurrentState,
)
from uav_swarm_control.observations import LocalObservations, encode_local_observations


class RecurrentActorController:
    """Maintain per-UAV memory while exposing only local actor observations."""

    def __init__(
        self,
        model: PaperRecurrentActorCritic,
        *,
        direction_epsilon: float = 1e-6,
    ) -> None:
        if direction_epsilon <= 0.0:
            raise ValueError("direction_epsilon must be positive.")
        self.model = model
        self.direction_epsilon = direction_epsilon
        self._state: RecurrentState | None = None

    def reset(self) -> None:
        self._state = None

    def act(self, observations: LocalObservations) -> NormalizedVelocityActions:
        """Return deterministic direction-speed actions converted to velocity."""
        device = next(self.model.parameters()).device
        if self._state is None:
            self._state = self.model.initial_actor_state(
                observations.num_agents,
                device=device,
            )
            keep = torch.zeros(observations.num_agents, dtype=torch.bool, device=device)
        else:
            keep = torch.ones(observations.num_agents, dtype=torch.bool, device=device)
        local = torch.as_tensor(
            np.array(encode_local_observations(observations), copy=True),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            actions, self._state = self.model.deterministic_actions(
                local.unsqueeze(0),
                self._state,
                keep.unsqueeze(0),
            )
        return direction_speed_to_normalized_velocity(
            DirectionSpeedActions(
                observations.agent_ids,
                actions.squeeze(0).cpu().numpy().astype(np.float32, copy=False),
            ),
            direction_epsilon=self.direction_epsilon,
        )


__all__ = ["RecurrentActorController"]
