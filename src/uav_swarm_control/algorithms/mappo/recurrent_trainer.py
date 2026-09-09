"""Sequence-correct recurrent MAPPO training for the paper-aligned controller."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from uav_swarm_control.actions import DirectionSpeedActions, direction_speed_to_normalized_velocity
from uav_swarm_control.algorithms.mappo.recurrent_buffer import (
    RecurrentMAPPORolloutBuffer,
    RecurrentSequenceBatch,
)
from uav_swarm_control.algorithms.ppo import resolve_device, seed_torch
from uav_swarm_control.configuration.recurrent_mappo import (
    RecurrentMAPPOConfig,
    RecurrentObservationProfile,
)
from uav_swarm_control.environments import MultiAgentEnvironment
from uav_swarm_control.environments.contracts import StepResult
from uav_swarm_control.models import NeighborEncoderSpec
from uav_swarm_control.models.recurrent_actor_critic import (
    PaperRecurrentActorCritic,
    RecurrentState,
)
from uav_swarm_control.observations import (
    CentralizedState,
    LocalObservations,
    encode_local_observations,
)
from uav_swarm_control.observations.paper import encode_paper_flat_observations
from uav_swarm_control.seeding import RandomStream, derive_indexed_seed

type EnvironmentFactory = Callable[[], MultiAgentEnvironment]
type ObservationEncoder = Callable[[LocalObservations, MultiAgentEnvironment], np.ndarray]


@dataclass(frozen=True, slots=True)
class RecurrentMAPPOUpdateMetrics:
    """Learning, task, and reward-component diagnostics for one update."""

    update: int
    environment_steps: int
    agent_samples: int
    mean_rollout_reward: float
    mean_normalized_shape_rmse: float
    collision_pair_steps_per_step: float
    mean_control_delta_rms: float
    reward_navigation_mean: float
    reward_formation_mean: float
    reward_collision_mean: float
    reward_smoothness_mean: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float


@dataclass(frozen=True, slots=True)
class RecurrentMAPPOTrainingResult:
    """Trained recurrent model and update history."""

    model: PaperRecurrentActorCritic
    history: tuple[RecurrentMAPPOUpdateMetrics, ...]
    environment_steps: int
    agent_samples: int
    device: torch.device


@dataclass(frozen=True, slots=True)
class _RolloutMetrics:
    reward: float
    shape_error: float
    collision_pair_steps: float
    control_delta: float
    navigation_reward: float
    formation_reward: float
    collision_reward: float
    smoothness_reward: float


def observation_encoder_for_profile(profile: RecurrentObservationProfile) -> ObservationEncoder:
    """Resolve a declared decentralized actor encoding."""
    if profile is RecurrentObservationProfile.PAPER_FLAT:
        return encode_paper_flat_observations

    def masked_set(
        observations: LocalObservations,
        environment: MultiAgentEnvironment,
    ) -> np.ndarray:
        del environment
        return encode_local_observations(observations)

    return masked_set


def _tensor(values: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(np.array(values, copy=True), dtype=torch.float32, device=device)


def _stack_local(
    observations: Sequence[LocalObservations],
    environments: Sequence[MultiAgentEnvironment],
    encoder: ObservationEncoder,
    device: torch.device,
) -> Tensor:
    encoded = [
        encoder(observation, environment)
        for observation, environment in zip(observations, environments, strict=True)
    ]
    return _tensor(np.stack(encoded), device)


def _stack_states(states: Sequence[CentralizedState], device: torch.device) -> Tensor:
    return _tensor(np.stack([state.values for state in states]), device)


def _reset_done_state(state: RecurrentState, done: Tensor, *, repeats: int = 1) -> RecurrentState:
    keep = (~done).to(state.hidden.dtype).repeat_interleave(repeats).view(1, -1, 1)
    return RecurrentState(state.hidden * keep, state.cell * keep)


def _environment_steps(
    environments: Sequence[MultiAgentEnvironment],
    actions: Tensor,
    *,
    direction_epsilon: float,
) -> list[StepResult]:
    transitions: list[StepResult] = []
    for index, environment in enumerate(environments):
        direction_speed = DirectionSpeedActions(
            environment.agent_ids,
            actions[index].cpu().numpy().astype(np.float32, copy=False),
        )
        transitions.append(
            environment.step(
                direction_speed_to_normalized_velocity(
                    direction_speed,
                    direction_epsilon=direction_epsilon,
                )
            )
        )
    return transitions


def _metrics(transitions: Sequence[StepResult]) -> np.ndarray:
    values = np.zeros(8, dtype=np.float64)
    for transition in transitions:
        metrics = transition.metrics
        values += np.array(
            [
                float(transition.rewards.mean()),
                metrics.get("normalized_shape_rmse", 0.0),
                metrics.get("collision_pairs", 0.0),
                metrics.get("control_delta_rms", 0.0),
                metrics.get("reward/navigation_mean", 0.0),
                metrics.get("reward/formation_mean", 0.0),
                metrics.get("reward/collision_mean", 0.0),
                metrics.get("reward/smoothness_mean", 0.0),
            ],
            dtype=np.float64,
        )
    return values


def _collect_rollout(
    environments: Sequence[MultiAgentEnvironment],
    model: PaperRecurrentActorCritic,
    config: RecurrentMAPPOConfig,
    device: torch.device,
    local: Tensor,
    states: Tensor,
    actor_state: RecurrentState,
    critic_state: RecurrentState,
    keep_masks: Tensor,
    episode_indices: list[int],
    root_seed: int,
    encoder: ObservationEncoder,
) -> tuple[
    RecurrentSequenceBatch,
    Tensor,
    Tensor,
    RecurrentState,
    RecurrentState,
    Tensor,
    _RolloutMetrics,
]:
    ppo = config.mappo.ppo
    environment_count, agents, local_size = local.shape
    buffer = RecurrentMAPPORolloutBuffer(
        time_steps=ppo.rollout_steps,
        num_environments=environment_count,
        num_agents=agents,
        local_observation_size=local_size,
        centralized_state_size=states.shape[-1],
        action_size=model.action_size,
        recurrent_layers=model.recurrent_layers,
        recurrent_hidden_size=model.recurrent_hidden_size,
        device=device,
    )
    metric_sums = np.zeros(8, dtype=np.float64)
    model.eval()
    for _ in range(ppo.rollout_steps):
        actor_before = actor_state.detached()
        critic_before = critic_state.detached()
        actor_input = local.reshape(environment_count * agents, local_size)
        actor_keep = keep_masks.repeat_interleave(agents)
        with torch.no_grad():
            policy = model.sample_actions(
                actor_input.unsqueeze(0), actor_state, actor_keep.unsqueeze(0)
            )
            values, critic_state = model.values(
                states.unsqueeze(0), critic_state, keep_masks.unsqueeze(0)
            )
        actor_state = policy.state
        bounded = policy.actions.squeeze(0).reshape(environment_count, agents, model.action_size)
        transitions = _environment_steps(
            environments,
            bounded,
            direction_epsilon=config.direction_epsilon,
        )
        next_observations = [transition.observations for transition in transitions]
        next_states = [transition.centralized_state for transition in transitions]
        terminal_states = _stack_states(next_states, device)
        with torch.no_grad():
            next_values, _ = model.values(
                terminal_states.unsqueeze(0),
                critic_state,
                torch.ones_like(keep_masks).unsqueeze(0),
            )
        rewards = _tensor(
            np.asarray([float(transition.rewards.mean()) for transition in transitions]),
            device,
        )
        terminated = torch.tensor(
            [transition.terminated for transition in transitions],
            dtype=torch.bool,
            device=device,
        )
        truncated = torch.tensor(
            [transition.truncated for transition in transitions],
            dtype=torch.bool,
            device=device,
        )
        buffer.add(
            local_observations=local,
            centralized_states=states,
            latent_actions=policy.latent_actions.squeeze(0).reshape(
                environment_count, agents, model.action_size
            ),
            rewards=rewards,
            log_probabilities=policy.log_probabilities.squeeze(0).reshape(
                environment_count, agents
            ),
            values=values.squeeze(0),
            next_values=next_values.squeeze(0),
            terminated=terminated,
            truncated=truncated,
            keep_masks=keep_masks,
            actor_state=actor_before,
            critic_state=critic_before,
        )
        metric_sums += _metrics(transitions)
        done = terminated | truncated
        for index, transition in enumerate(transitions):
            if transition.episode_done:
                episode_indices[index] += 1
                reset = environments[index].reset(
                    seed=derive_indexed_seed(
                        root_seed,
                        RandomStream.INITIAL_STATE,
                        index,
                        episode_indices[index],
                    )
                )
                next_observations[index] = reset.observations
                next_states[index] = reset.centralized_state
        actor_state = _reset_done_state(actor_state, done, repeats=agents)
        critic_state = _reset_done_state(critic_state, done)
        keep_masks = ~done
        local = _stack_local(next_observations, environments, encoder, device)
        states = _stack_states(next_states, device)

    means = metric_sums / (ppo.rollout_steps * environment_count)
    batch = buffer.as_sequences(
        sequence_length=config.sequence_length,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
    )
    return (
        batch,
        local,
        states,
        actor_state,
        critic_state,
        keep_masks,
        _RolloutMetrics(*(float(value) for value in means)),
    )


def _update_model(
    model: PaperRecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RecurrentSequenceBatch,
    config: RecurrentMAPPOConfig,
) -> tuple[float, float, float, float, float]:
    ppo = config.mappo.ppo
    totals = torch.zeros(5, device=batch.local_observations.device)
    minibatches = 0
    model.train()
    for _ in range(ppo.update_epochs):
        order = torch.randperm(batch.sequence_count, device=batch.local_observations.device)
        for start in range(0, batch.sequence_count, config.sequences_per_minibatch):
            selected = batch.select(order[start : start + config.sequences_per_minibatch])
            actor_observations, actor_masks = selected.actor_inputs()
            sequences, time_steps, agents, _ = selected.local_observations.shape
            latent = selected.latent_actions.permute(1, 0, 2, 3).reshape(
                time_steps, sequences * agents, model.action_size
            )
            policy = model.evaluate_actions(
                actor_observations,
                latent,
                selected.actor_initial_state,
                actor_masks,
            )
            new_log = policy.log_probabilities.reshape(time_steps, sequences, agents)
            old_log = selected.old_log_probabilities.permute(1, 0, 2)
            advantages = selected.advantages.permute(1, 0).unsqueeze(-1)
            ratio = torch.exp(new_log - old_log)
            unclipped = ratio * advantages
            clipped = (
                ratio.clamp(1.0 - ppo.clip_coefficient, 1.0 + ppo.clip_coefficient) * advantages
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            critic_inputs, critic_masks = selected.critic_inputs()
            new_values, _ = model.values(critic_inputs, selected.critic_initial_state, critic_masks)
            value_loss = 0.5 * (new_values - selected.returns.permute(1, 0)).square().mean()
            entropy = policy.entropies.mean()
            total = (
                policy_loss + ppo.value_coefficient * value_loss - ppo.entropy_coefficient * entropy
            )
            with torch.no_grad():
                approximate_kl = (old_log - new_log).mean()
                clip_fraction = ((ratio - 1.0).abs() > ppo.clip_coefficient).float().mean()
            optimizer.zero_grad(set_to_none=True)
            total.backward()  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.max_gradient_norm)
            optimizer.step()
            totals += torch.stack(
                (
                    policy_loss.detach(),
                    value_loss.detach(),
                    entropy.detach(),
                    approximate_kl.detach(),
                    clip_fraction.detach(),
                )
            )
            minibatches += 1
    means = totals.cpu() / minibatches
    return (
        float(means[0]),
        float(means[1]),
        float(means[2]),
        float(means[3]),
        float(means[4]),
    )


def train_recurrent_mappo(
    environment_factory: EnvironmentFactory,
    config: RecurrentMAPPOConfig,
    *,
    seed: int,
    neighbor_encoder: NeighborEncoderSpec | None = None,
    observation_encoder: ObservationEncoder | None = None,
    on_update: Callable[[PaperRecurrentActorCritic, RecurrentMAPPOUpdateMetrics], None]
    | None = None,
) -> RecurrentMAPPOTrainingResult:
    """Train a shared recurrent actor with a centralized recurrent team critic."""
    device = resolve_device(config.mappo.ppo.device)
    seed_torch(seed)
    encoder = observation_encoder or observation_encoder_for_profile(config.observation_profile)
    environments: list[MultiAgentEnvironment] = []
    try:
        environments = [environment_factory() for _ in range(config.mappo.num_environments)]
        resets = [
            environment.reset(seed=derive_indexed_seed(seed, RandomStream.INITIAL_STATE, index, 0))
            for index, environment in enumerate(environments)
        ]
        if not environments:
            raise ValueError("recurrent MAPPO requires at least one environment.")
        if any(environment.agent_ids != environments[0].agent_ids for environment in environments):
            raise ValueError("parallel environments must use identical agent IDs and order.")
        local = _stack_local(
            [reset.observations for reset in resets], environments, encoder, device
        )
        states = _stack_states([reset.centralized_state for reset in resets], device)
        environment_count, agents, local_size = local.shape
        model = PaperRecurrentActorCritic(
            local_size,
            states.shape[-1],
            feature_size=config.feature_size,
            recurrent_hidden_size=config.hidden_size,
            recurrent_layers=config.num_layers,
            initial_log_standard_deviation=(config.mappo.ppo.initial_log_standard_deviation),
            neighbor_encoder=neighbor_encoder,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.mappo.ppo.learning_rate, eps=1e-5
        )
        actor_state = model.initial_actor_state(environment_count * agents, device=device)
        critic_state = model.initial_critic_state(environment_count, device=device)
        keep_masks = torch.zeros(environment_count, dtype=torch.bool, device=device)
        episode_indices = [0] * environment_count
        history: list[RecurrentMAPPOUpdateMetrics] = []
        environment_steps_per_update = config.mappo.ppo.rollout_steps * environment_count
        agent_samples_per_update = environment_steps_per_update * agents
        for update in range(1, config.mappo.num_updates + 1):
            (
                batch,
                local,
                states,
                actor_state,
                critic_state,
                keep_masks,
                rollout,
            ) = _collect_rollout(
                environments,
                model,
                config,
                device,
                local,
                states,
                actor_state,
                critic_state,
                keep_masks,
                episode_indices,
                seed,
                encoder,
            )
            policy, value, entropy, approximate_kl, clip_fraction = _update_model(
                model, optimizer, batch, config
            )
            update_metrics = RecurrentMAPPOUpdateMetrics(
                update,
                update * environment_steps_per_update,
                update * agent_samples_per_update,
                rollout.reward,
                rollout.shape_error,
                rollout.collision_pair_steps,
                rollout.control_delta,
                rollout.navigation_reward,
                rollout.formation_reward,
                rollout.collision_reward,
                rollout.smoothness_reward,
                policy,
                value,
                entropy,
                approximate_kl,
                clip_fraction,
            )
            history.append(update_metrics)
            if on_update is not None:
                on_update(model, update_metrics)
    finally:
        for environment in environments:
            environment.close()
    return RecurrentMAPPOTrainingResult(
        model,
        tuple(history),
        config.mappo.ppo.total_steps,
        config.mappo.ppo.total_steps * agents,
        device,
    )


__all__ = [
    "EnvironmentFactory",
    "ObservationEncoder",
    "RecurrentMAPPOTrainingResult",
    "RecurrentMAPPOUpdateMetrics",
    "observation_encoder_for_profile",
    "train_recurrent_mappo",
]
