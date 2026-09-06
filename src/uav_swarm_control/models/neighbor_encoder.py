"""Masked permutation-invariant neighbor encoder for decentralized actors."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from uav_swarm_control.models._networks import build_mlp


@dataclass(frozen=True, slots=True)
class NeighborEncoderSpec:
    """Layout and capacity of the structured section in a flat local observation."""

    ego_features: int
    max_neighbors: int
    neighbor_features: int
    embedding_size: int
    hidden_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.hidden_sizes
            or min(
                self.ego_features,
                self.neighbor_features,
                self.embedding_size,
                *self.hidden_sizes,
            )
            < 1
        ):
            raise ValueError("neighbor encoder dimensions must be positive.")
        if self.max_neighbors < 1:
            raise ValueError("neighbor encoder requires at least one padded neighbor slot.")

    @property
    def structured_width(self) -> int:
        """Width occupied by ego, neighbor rows, and the neighbor mask."""
        return self.ego_features + self.max_neighbors * (self.neighbor_features + 1)


def masked_mean_neighbor_features(
    observations: Tensor,
    spec: NeighborEncoderSpec,
    neighbor_encoder: nn.Module,
) -> Tensor:
    """Return ego, masked-mean neighbor, and trailing features in a fixed layout."""
    if observations.shape[-1] < spec.structured_width:
        raise ValueError("local observation is too narrow for the neighbor encoder layout.")
    ego_end = spec.ego_features
    neighbors_end = ego_end + spec.max_neighbors * spec.neighbor_features
    mask_end = neighbors_end + spec.max_neighbors
    ego = observations[..., :ego_end]
    neighbors = observations[..., ego_end:neighbors_end].reshape(
        *observations.shape[:-1], spec.max_neighbors, spec.neighbor_features
    )
    mask = observations[..., neighbors_end:mask_end].unsqueeze(-1)
    encoded = neighbor_encoder(neighbors) * mask
    pooled = encoded.sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)
    return torch.cat((ego, pooled, observations[..., mask_end:]), dim=-1)


class MaskedMeanNeighborActor(nn.Module):
    """Encode each neighbor with shared weights and mean-pool only valid rows."""

    def __init__(
        self,
        local_observation_size: int,
        action_size: int,
        policy_hidden_sizes: tuple[int, ...],
        spec: NeighborEncoderSpec,
    ) -> None:
        super().__init__()
        if local_observation_size < spec.structured_width:
            raise ValueError("local observation is too narrow for the neighbor encoder layout.")
        self.spec = spec
        self.neighbor_encoder = build_mlp(
            spec.neighbor_features,
            spec.embedding_size,
            spec.hidden_sizes,
            output_gain=1.0,
        )
        tail_size = local_observation_size - spec.structured_width
        self.policy = build_mlp(
            spec.ego_features + spec.embedding_size + tail_size,
            action_size,
            policy_hidden_sizes,
            output_gain=0.01,
        )

    def forward(self, observations: Tensor) -> Tensor:
        """Return outputs invariant to permutations of neighbor slots and their masks."""
        policy_input = masked_mean_neighbor_features(observations, self.spec, self.neighbor_encoder)
        return self.policy(policy_input)
