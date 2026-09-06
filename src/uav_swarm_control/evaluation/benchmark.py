"""Episode-wide measurements shared by scripted and learned swarm controllers."""

import math
from collections.abc import Callable, Mapping, Sequence
from statistics import mean, stdev
from typing import Protocol

import numpy as np
import torch

from uav_swarm_control.controllers.contracts import (
    Controller,
    EpisodeResettableController,
    StateAwareController,
)
from uav_swarm_control.environments.contracts import (
    MultiAgentEnvironment,
    NormalizedVelocityActions,
)
from uav_swarm_control.formations import pairwise_distances
from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.models import SharedActorCentralCritic
from uav_swarm_control.observations import LocalObservations, encode_local_observations
from uav_swarm_control.seeding import RandomStream, derive_indexed_seed


class PositionedEnvironment(MultiAgentEnvironment, Protocol):
    @property
    def positions(self) -> FloatArray: ...


class ActorController:
    """Deterministic execution that never calls the centralized critic."""

    def __init__(self, model: SharedActorCentralCritic) -> None:
        self.model = model

    def act(self, observations: LocalObservations) -> NormalizedVelocityActions:
        device = next(self.model.parameters()).device
        local = torch.as_tensor(
            np.array(encode_local_observations(observations), copy=True),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            values = self.model.deterministic_actions(local).cpu().numpy().astype(np.float32)
        return NormalizedVelocityActions(observations.agent_ids, values)


def evaluate_benchmark(
    factory: Callable[[], PositionedEnvironment],
    controller: Controller | StateAwareController,
    *,
    episodes: int,
    seed: int,
    horizon: int,
    time_step_seconds: float,
    collision_distance_m: float,
    on_episode_complete: Callable[[Mapping[str, float]], None] | None = None,
    episode_metadata: Callable[[PositionedEnvironment], Mapping[str, float]] | None = None,
    episode_metadata_prefix: str = "pose",
) -> list[dict[str, float]]:
    """Measure whole trajectories, including initial separation and terminal states.

    Clearance is center separation minus the configured collision distance, not mesh clearance.
    Collision pair-steps count exposure, not unique collision events. Path length is averaged
    across agents. Control smoothness is RMS normalized action change, including the first action
    relative to zero. No interpolation between simulator samples is implied.
    """
    if episodes < 1 or horizon < 1 or time_step_seconds <= 0 or collision_distance_m <= 0:
        raise ValueError("episodes, horizon, timestep, and collision distance must be positive.")
    if not episode_metadata_prefix or not episode_metadata_prefix.isidentifier():
        raise ValueError("episode_metadata_prefix must be a non-empty identifier.")
    results: list[dict[str, float]] = []
    environment = factory()
    try:
        for episode in range(episodes):
            if isinstance(controller, EpisodeResettableController):
                controller.reset()
            episode_seed = derive_indexed_seed(seed, RandomStream.EVALUATION, episode)
            reset = environment.reset(seed=episode_seed)
            metadata = dict(episode_metadata(environment)) if episode_metadata is not None else {}
            if any(not name or not math.isfinite(float(value)) for name, value in metadata.items()):
                raise ValueError("episode metadata requires non-empty names and finite values.")
            observations = reset.observations
            centralized_state = reset.centralized_state
            previous = environment.positions
            rows, columns = np.triu_indices(len(previous), k=1)
            if len(rows) == 0:
                raise ValueError("benchmark requires at least two agents.")
            initial_distances = pairwise_distances(previous)[rows, columns]
            separation = float(initial_distances.min())
            collision_any = bool(np.any(initial_distances < collision_distance_m))
            collision_pair_steps = 0.0
            obstacle_collision_any = False
            obstacle_collision_pair_steps = 0.0
            obstacle_clearance: float | None = None
            reports_obstacles = False
            path_length = 0.0
            squared_delta = 0.0
            action_frame_clip = 0.0
            reports_action_frame_clip = False
            formation_error = 0.0
            episode_return = 0.0
            communication_steps = 0
            communication_sums: dict[str, float] = {}
            minimum_actual_degree = math.inf
            minimum_algebraic_connectivity = math.inf
            maximum_connected_components = 0.0
            received_payload_bytes = 0.0
            for step in range(1, horizon + 1):
                actions = (
                    controller.act_with_state(observations, centralized_state)
                    if isinstance(controller, StateAwareController)
                    else controller.act(observations)
                )
                transition = environment.step(actions)
                current = environment.positions
                path_length += float(np.linalg.norm(current - previous, axis=1).mean())
                previous = current
                metrics = transition.metrics
                separation = min(separation, metrics["minimum_separation_m"])
                collision_pair_steps += metrics["collision_pairs"]
                collision_any |= metrics["collision_pairs"] > 0
                if "obstacle_collision_pairs" in metrics:
                    pair_count = metrics["obstacle_collision_pairs"]
                    obstacle_collision_pair_steps += pair_count
                    obstacle_collision_any |= pair_count > 0
                    reports_obstacles = True
                if "minimum_obstacle_clearance_m" in metrics:
                    current_clearance = metrics["minimum_obstacle_clearance_m"]
                    obstacle_clearance = (
                        current_clearance
                        if obstacle_clearance is None
                        else min(obstacle_clearance, current_clearance)
                    )
                squared_delta += metrics["control_delta_rms"] ** 2
                if "action_frame_clip_fraction" in metrics:
                    action_frame_clip += metrics["action_frame_clip_fraction"]
                    reports_action_frame_clip = True
                formation_error += metrics["normalized_shape_rmse"]
                episode_return += float(transition.rewards.mean())
                if "communication/actual_degree_mean" in metrics:
                    communication_steps += 1
                    for name in (
                        "actual_degree_mean",
                        "actual_degree_max",
                        "reciprocal_edge_fraction",
                        "connected",
                        "algebraic_connectivity",
                        "rigidity_rank_fraction",
                        "infinitesimally_rigid",
                    ):
                        communication_sums[name] = (
                            communication_sums.get(name, 0.0) + metrics[f"communication/{name}"]
                        )
                    minimum_actual_degree = min(
                        minimum_actual_degree,
                        metrics["communication/actual_degree_min"],
                    )
                    minimum_algebraic_connectivity = min(
                        minimum_algebraic_connectivity,
                        metrics["communication/algebraic_connectivity"],
                    )
                    maximum_connected_components = max(
                        maximum_connected_components,
                        metrics["communication/connected_components"],
                    )
                    received_payload_bytes += metrics["communication/received_payload_bytes"]
                observations = transition.observations
                centralized_state = transition.centralized_state
                if transition.episode_done:
                    record = {
                        "episode_seed": episode_seed,
                        "success": metrics["success"],
                        "collision_free_success": float(
                            metrics["success"] > 0
                            and not collision_any
                            and not obstacle_collision_any
                        ),
                        "collision_any": float(collision_any),
                        "collision_pair_steps": collision_pair_steps,
                        "minimum_clearance_m": separation - collision_distance_m,
                        "mean_agent_path_length_m": path_length,
                        "control_delta_rms": math.sqrt(squared_delta / step),
                        "mean_normalized_shape_rmse": formation_error / step,
                        "final_normalized_shape_rmse": metrics["normalized_shape_rmse"],
                        "final_position_rmse_m": metrics["position_rmse_m"],
                        "steps": float(step),
                        "simulated_seconds": step * time_step_seconds,
                        "capped_time_to_goal_seconds": (
                            step * time_step_seconds
                            if metrics["success"] > 0
                            else horizon * time_step_seconds
                        ),
                        "mean_agent_return": episode_return,
                        "terminated": float(transition.terminated),
                        "truncated": float(transition.truncated),
                    }
                    if reports_action_frame_clip:
                        record["action_frame_clip_fraction"] = action_frame_clip / step
                    if reports_obstacles:
                        record["obstacle_collision_any"] = float(obstacle_collision_any)
                        record["obstacle_collision_pair_steps"] = obstacle_collision_pair_steps
                    if obstacle_clearance is not None:
                        record["minimum_obstacle_clearance_m"] = obstacle_clearance
                    if communication_steps:
                        record.update(
                            {
                                "mean_actual_neighbor_degree": (
                                    communication_sums["actual_degree_mean"] / communication_steps
                                ),
                                "minimum_actual_neighbor_degree": minimum_actual_degree,
                                "mean_maximum_actual_neighbor_degree": (
                                    communication_sums["actual_degree_max"] / communication_steps
                                ),
                                "mean_reciprocal_edge_fraction": (
                                    communication_sums["reciprocal_edge_fraction"]
                                    / communication_steps
                                ),
                                "connected_step_fraction": (
                                    communication_sums["connected"] / communication_steps
                                ),
                                "maximum_connected_components": maximum_connected_components,
                                "mean_algebraic_connectivity": (
                                    communication_sums["algebraic_connectivity"]
                                    / communication_steps
                                ),
                                "minimum_algebraic_connectivity": (minimum_algebraic_connectivity),
                                "mean_rigidity_rank_fraction": (
                                    communication_sums["rigidity_rank_fraction"]
                                    / communication_steps
                                ),
                                "rigid_step_fraction": (
                                    communication_sums["infinitesimally_rigid"]
                                    / communication_steps
                                ),
                                "received_payload_bytes": received_payload_bytes,
                                "mean_received_payload_bytes_per_step": (
                                    received_payload_bytes / communication_steps
                                ),
                                "mean_received_payload_bytes_per_agent_step": (
                                    received_payload_bytes
                                    / (communication_steps * len(environment.agent_ids))
                                ),
                            }
                        )
                    for name, value in metadata.items():
                        key = f"{episode_metadata_prefix}/{name}"
                        if key in record:
                            raise ValueError(f"episode metadata collides with metric {key!r}.")
                        record[key] = float(value)
                    results.append(record)
                    if on_episode_complete is not None:
                        on_episode_complete(record)
                    break
            else:
                raise RuntimeError("benchmark environment exceeded its configured horizon.")
    finally:
        environment.close()
    return results


def average_episodes(episodes: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """One equally weighted mean per trained seed; episode seed is not a metric."""
    if not episodes:
        raise ValueError("cannot aggregate zero episodes.")
    keys = set(episodes[0]) - {"episode_seed"}
    if any(set(item) - {"episode_seed"} != keys for item in episodes):
        raise ValueError("all episodes must use identical metric keys.")
    return {key: mean(item[key] for item in episodes) for key in sorted(keys)}


def summarize_records(
    records: Sequence[Mapping[str, float]], *, count_key: str
) -> dict[str, object]:
    """Summarize homogeneous records while making their sampling-unit count explicit."""
    if len(records) < 2:
        raise ValueError("at least two records are required.")
    if not count_key or not count_key.isidentifier():
        raise ValueError("count_key must be a non-empty identifier.")
    keys = set(records[0])
    if any(set(record) != keys for record in records):
        raise ValueError("all records must report identical metric keys.")
    return {
        count_key: len(records),
        "metrics": {
            key: {
                "mean": mean(row[key] for row in records),
                "sample_std": stdev(row[key] for row in records),
            }
            for key in sorted(keys)
        },
    }


def summarize_seeds(records: Sequence[Mapping[str, float]]) -> dict[str, object]:
    """Report variation across independently trained seed records."""
    return summarize_records(records, count_key="seed_count")
