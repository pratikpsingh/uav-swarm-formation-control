"""Trajectory evaluation for recurrent research environments and perturbations."""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np

from uav_swarm_control.controllers.contracts import Controller
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.evaluation.benchmark import PositionedEnvironment
from uav_swarm_control.evaluation.recovery import measure_recovery
from uav_swarm_control.seeding import RandomStream, derive_indexed_seed

type EnvironmentFactory = Callable[[], PositionedEnvironment]
type ActionTransform = Callable[[int, NormalizedVelocityActions], NormalizedVelocityActions]
type EpisodeMetadata = Callable[[PositionedEnvironment], Mapping[str, float]]


class RecurrentEvaluationController(Controller, Protocol):
    """Controller with episode-reset support for recurrent evaluation."""

    def reset(self) -> None:
        """Clear recurrent state before a new episode."""
        ...


def evaluate_recurrent_environment(
    factory: EnvironmentFactory,
    controller: RecurrentEvaluationController,
    *,
    episodes: int,
    seed: int,
    horizon: int,
    time_step_seconds: float,
    collision_distance_m: float,
    action_transform: ActionTransform | None = None,
    recovery_start_step: int | None = None,
    recovery_threshold: float | None = None,
    recovery_hold_steps: int = 1,
    episode_metadata: EpisodeMetadata | None = None,
    episode_metadata_prefix: str = "context",
) -> list[dict[str, float]]:
    """Record common, final, and optional sustained-recovery measurements."""
    if episodes < 1 or horizon < 1 or time_step_seconds <= 0.0:
        raise ValueError("episodes, horizon, and time step must be positive.")
    records: list[dict[str, float]] = []
    environment = factory()
    try:
        for episode in range(episodes):
            controller.reset()
            episode_seed = derive_indexed_seed(seed, RandomStream.EVALUATION, episode)
            observations = environment.reset(seed=episode_seed).observations
            previous = environment.positions
            path_length = 0.0
            episode_return = 0.0
            collision_any = False
            obstacle_collision_any = False
            minimum_separation = float("inf")
            minimum_obstacle_clearance = float("inf")
            formation_errors: list[float] = []
            recovery_errors: list[float] = []
            for step in range(1, horizon + 1):
                actions = controller.act(observations)
                if action_transform is not None:
                    actions = action_transform(step, actions)
                transition = environment.step(actions)
                current = environment.positions
                path_length += float(np.linalg.norm(current - previous, axis=1).mean())
                previous = current
                metrics = transition.metrics
                episode_return += float(transition.rewards.mean())
                collision_any |= metrics.get("collision_pairs", 0.0) > 0.0
                obstacle_collision_any |= metrics.get("obstacle_collision_pairs", 0.0) > 0.0
                minimum_separation = min(
                    minimum_separation,
                    metrics.get("minimum_separation_m", minimum_separation),
                )
                minimum_obstacle_clearance = min(
                    minimum_obstacle_clearance,
                    metrics.get("minimum_obstacle_clearance_m", minimum_obstacle_clearance),
                )
                shape_error = metrics.get("normalized_shape_rmse", 0.0)
                formation_errors.append(shape_error)
                if recovery_start_step is not None and step >= recovery_start_step:
                    recovery_errors.append(shape_error)
                observations = transition.observations
                if transition.episode_done:
                    break
            else:
                raise RuntimeError("environment exceeded the configured evaluation horizon.")
            obstacle_clearance = (
                -1.0 if not np.isfinite(minimum_obstacle_clearance) else minimum_obstacle_clearance
            )
            record = {
                "episode_seed": float(episode_seed),
                "success": metrics.get("success", 0.0),
                "collision_free_success": float(
                    metrics.get("success", 0.0) > 0.0
                    and not collision_any
                    and not obstacle_collision_any
                ),
                "collision_any": float(collision_any),
                "obstacle_collision_any": float(obstacle_collision_any),
                "minimum_clearance_m": minimum_separation - collision_distance_m,
                "minimum_obstacle_clearance_m": obstacle_clearance,
                "mean_agent_path_length_m": path_length,
                "mean_normalized_shape_rmse": float(np.mean(formation_errors)),
                "final_normalized_shape_rmse": metrics.get("normalized_shape_rmse", 0.0),
                "final_position_rmse_m": metrics.get("position_rmse_m", 0.0),
                "mean_agent_return": episode_return,
                "steps": float(step),
                "simulated_seconds": step * time_step_seconds,
            }
            record.update({f"final/{name}": float(value) for name, value in metrics.items()})
            if episode_metadata is not None:
                context = episode_metadata(environment)
                for name, value in context.items():
                    key = f"{episode_metadata_prefix}/{name}"
                    if key in record:
                        raise ValueError(f"episode metadata collides with metric {key}.")
                    if not np.isfinite(value):
                        raise ValueError(f"episode metadata {name} must be finite.")
                    record[key] = float(value)
            if recovery_errors:
                if recovery_threshold is None:
                    raise ValueError("recovery_threshold is required for recovery evaluation.")
                recovery = measure_recovery(
                    recovery_errors,
                    threshold=recovery_threshold,
                    hold_steps=recovery_hold_steps,
                    time_step_seconds=time_step_seconds,
                )
                record.update(
                    {
                        "recovery/peak_error": recovery.peak_error,
                        "recovery/error_area": recovery.error_area,
                        "recovery/time_seconds": (
                            -1.0 if recovery.recovery_seconds is None else recovery.recovery_seconds
                        ),
                        "recovery/final_error": recovery.final_error,
                    }
                )
            records.append(record)
    finally:
        environment.close()
    return records


def velocity_disturbance(
    *, step: int, affected_agents: tuple[int, ...], offset: tuple[float, float, float]
) -> ActionTransform:
    """Create a one-control-step normalized velocity-command perturbation."""
    if step < 1 or not affected_agents or min(affected_agents) < 0:
        raise ValueError("disturbance step and affected agents must be valid.")
    vector = np.asarray(offset, dtype=np.float32)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("disturbance offset must be one finite 3D vector.")

    def transform(
        current_step: int,
        actions: NormalizedVelocityActions,
    ) -> NormalizedVelocityActions:
        if max(affected_agents) >= actions.num_agents:
            raise ValueError("disturbance selects an unavailable agent.")
        if current_step != step:
            return actions
        values = np.array(actions.values, copy=True)
        values[list(affected_agents)] = np.clip(values[list(affected_agents)] + vector, -1.0, 1.0)
        return NormalizedVelocityActions(actions.agent_ids, values)

    return transform


def numeric_mean(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean homogeneous episode records, excluding the seed identifier."""
    if not records:
        raise ValueError("cannot summarize zero records.")
    keys = set(records[0]) - {"episode_seed"}
    if any(set(record) - {"episode_seed"} != keys for record in records):
        raise ValueError("episode records must have identical metric keys.")
    return {key: float(np.mean([record[key] for record in records])) for key in sorted(keys)}


__all__ = [
    "ActionTransform",
    "EpisodeMetadata",
    "RecurrentEvaluationController",
    "evaluate_recurrent_environment",
    "numeric_mean",
    "velocity_disturbance",
]
