"""Sequence-correct policy distillation and structured channel pruning."""

import copy
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from uav_swarm_control.configuration.deployment import (
    CompressionCandidateConfig,
    DistillationConfig,
)
from uav_swarm_control.deployment.models import (
    CompactFeedForwardActor,
    CompactGRUActor,
    CompactLSTMActor,
    DeploymentActor,
    DeploymentArchitecture,
)
from uav_swarm_control.models import NeighborEncoderSpec


@dataclass(frozen=True, slots=True)
class TeacherTrajectory:
    """One complete episode; splitting this unit prevents temporal leakage."""

    observations: Tensor
    actions: Tensor
    condition: str
    episode_seed: int

    def __post_init__(self) -> None:
        if (
            self.observations.ndim != 3
            or self.actions.ndim != 3
            or self.observations.shape[:2] != self.actions.shape[:2]
            or self.observations.shape[0] < 1
        ):
            raise ValueError("trajectory tensors must be non-empty [time, agent, feature] arrays.")


@dataclass(frozen=True, slots=True)
class DistillationResult:
    """Trained actor and held-out imitation diagnostics."""

    model: DeploymentActor
    training_mse: float
    validation_mse: float
    training_trajectories: int
    validation_trajectories: int
    structured_zero_channel_fraction: float


def build_student(
    candidate: CompressionCandidateConfig,
    *,
    local_observation_size: int,
    action_size: int,
    teacher_neighbor_spec: NeighborEncoderSpec,
) -> DeploymentActor:
    """Create one student while retaining the teacher's padded observation contract."""
    compact_spec = NeighborEncoderSpec(
        ego_features=teacher_neighbor_spec.ego_features,
        max_neighbors=teacher_neighbor_spec.max_neighbors,
        neighbor_features=teacher_neighbor_spec.neighbor_features,
        embedding_size=candidate.width,
        hidden_sizes=(candidate.width,),
    )
    if candidate.architecture is DeploymentArchitecture.FEED_FORWARD:
        return CompactFeedForwardActor(
            local_observation_size, action_size, candidate.width, candidate.depth, compact_spec
        )
    if candidate.architecture is DeploymentArchitecture.GRU:
        return CompactGRUActor(local_observation_size, action_size, candidate.width, compact_spec)
    return CompactLSTMActor(local_observation_size, action_size, candidate.width, compact_spec)


def _split_trajectories(
    trajectories: tuple[TeacherTrajectory, ...], fraction: float, seed: int
) -> tuple[tuple[TeacherTrajectory, ...], tuple[TeacherTrajectory, ...]]:
    if len(trajectories) < 2:
        raise ValueError("distillation requires at least two teacher trajectories.")
    generator = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for index, trajectory in enumerate(trajectories):
        groups.setdefault(trajectory.condition, []).append(index)
    validation_indices: set[int] = set()
    if all(len(indices) >= 2 for indices in groups.values()):
        for indices in groups.values():
            order = generator.permutation(indices)
            count = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
            validation_indices.update(int(index) for index in order[:count])
    else:
        order = generator.permutation(len(trajectories))
        count = max(1, min(len(trajectories) - 1, round(len(trajectories) * fraction)))
        validation_indices.update(int(index) for index in order[:count])
    training = tuple(
        item for index, item in enumerate(trajectories) if index not in validation_indices
    )
    validation = tuple(
        item for index, item in enumerate(trajectories) if index in validation_indices
    )
    return training, validation


def _sequences(trajectories: tuple[TeacherTrajectory, ...]) -> tuple[list[Tensor], list[Tensor]]:
    observations: list[Tensor] = []
    actions: list[Tensor] = []
    for trajectory in trajectories:
        for agent in range(trajectory.observations.shape[1]):
            observations.append(trajectory.observations[:, agent])
            actions.append(trajectory.actions[:, agent])
    return observations, actions


def _forward_sequences(
    model: DeploymentActor, architecture: DeploymentArchitecture, observations: Tensor
) -> Tensor:
    if architecture is DeploymentArchitecture.FEED_FORWARD:
        return cast(CompactFeedForwardActor, model)(observations)
    batch = observations.shape[0]
    recurrent = cast(CompactGRUActor | CompactLSTMActor, model)
    hidden = torch.zeros(1, batch, recurrent.width, dtype=observations.dtype)
    if architecture is DeploymentArchitecture.GRU:
        output, _ = cast(CompactGRUActor, recurrent)(observations, hidden)
        return output
    output, _, _ = cast(CompactLSTMActor, recurrent)(observations, hidden, torch.zeros_like(hidden))
    return output


def _masked_mse(predicted: Tensor, target: Tensor, lengths: list[int]) -> Tensor:
    steps = torch.arange(predicted.shape[1]).unsqueeze(0)
    mask = steps < torch.tensor(lengths).unsqueeze(1)
    squared = (predicted - target).square().mean(dim=-1)
    return squared[mask].mean()


def _dataset_mse(
    model: DeploymentActor,
    architecture: DeploymentArchitecture,
    trajectories: tuple[TeacherTrajectory, ...],
) -> float:
    observations, actions = _sequences(trajectories)
    padded_observations = pad_sequence(observations, batch_first=True)
    padded_actions = pad_sequence(actions, batch_first=True)
    with torch.no_grad():
        predicted = _forward_sequences(model, architecture, padded_observations)
        return float(
            _masked_mse(predicted, padded_actions, [len(item) for item in observations]).item()
        )


def _apply_structured_masks(
    model: nn.Module, fraction: float
) -> tuple[list[tuple[nn.Linear, Tensor, Tensor]], int, int]:
    masks: list[tuple[nn.Linear, Tensor, Tensor]] = []
    zero_channels = 0
    total_channels = 0
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
    for module in linears[:-1]:
        remove_count = max(1, round(module.out_features * fraction))
        importance = module.weight.detach().square().sum(dim=1)
        removed = torch.argsort(importance)[:remove_count]
        weight_mask = torch.ones_like(module.weight)
        bias_mask = torch.ones_like(module.bias)
        weight_mask[removed] = 0.0
        bias_mask[removed] = 0.0
        module.weight.data.mul_(weight_mask)
        module.bias.data.mul_(bias_mask)
        masks.append((module, weight_mask, bias_mask))
        zero_channels += remove_count
        total_channels += module.out_features
    return masks, zero_channels, total_channels


def _enforce_structured_masks(masks: list[tuple[nn.Linear, Tensor, Tensor]]) -> None:
    for module, weight_mask, bias_mask in masks:
        module.weight.data.mul_(weight_mask)
        module.bias.data.mul_(bias_mask)


def distill_student(
    candidate: CompressionCandidateConfig,
    trajectories: tuple[TeacherTrajectory, ...],
    config: DistillationConfig,
    *,
    seed: int,
    local_observation_size: int,
    action_size: int,
    teacher_neighbor_spec: NeighborEncoderSpec,
) -> DistillationResult:
    """Fit teacher actions, with recurrent state reset at every episode boundary."""
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    training, validation = _split_trajectories(trajectories, config.validation_fraction, seed)
    model = build_student(
        candidate,
        local_observation_size=local_observation_size,
        action_size=action_size,
        teacher_neighbor_spec=teacher_neighbor_spec,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_observations, train_actions = _sequences(training)
    generator = np.random.default_rng(seed)

    def train_epochs(epochs: int) -> None:
        for _ in range(epochs):
            order = [int(index) for index in generator.permutation(len(train_observations))]
            for start in range(0, len(order), config.batch_size):
                indices = order[start : start + config.batch_size]
                observations = [train_observations[index] for index in indices]
                actions = [train_actions[index] for index in indices]
                padded_observations = pad_sequence(observations, batch_first=True)
                padded_actions = pad_sequence(actions, batch_first=True)
                predicted = _forward_sequences(model, candidate.architecture, padded_observations)
                loss = _masked_mse(predicted, padded_actions, [len(item) for item in observations])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # pyright: ignore[reportUnknownMemberType]
                optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                _enforce_structured_masks(active_masks)

    active_masks: list[tuple[nn.Linear, Tensor, Tensor]] = []
    train_epochs(config.epochs)
    masks: list[tuple[nn.Linear, Tensor, Tensor]] = []
    zero_channels = 0
    total_channels = 0
    if candidate.structured_pruning_fraction:
        masks, zero_channels, total_channels = _apply_structured_masks(
            model, candidate.structured_pruning_fraction
        )
        active_masks = masks
        train_epochs(config.fine_tune_epochs)
    return DistillationResult(
        copy.deepcopy(model).eval(),
        _dataset_mse(model, candidate.architecture, training),
        _dataset_mse(model, candidate.architecture, validation),
        len(training),
        len(validation),
        zero_channels / total_channels if total_channels else 0.0,
    )


__all__ = ["DistillationResult", "TeacherTrajectory", "build_student", "distill_student"]
