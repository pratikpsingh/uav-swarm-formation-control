"""Parameter-shared MAPPO training with a centralized critic."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from uav_swarm_control.algorithms.mappo.buffer import FlatMAPPOBatch, MAPPORolloutBuffer
from uav_swarm_control.algorithms.ppo import ppo_loss, resolve_device, seed_torch
from uav_swarm_control.configuration import MAPPOConfig
from uav_swarm_control.environments import MultiAgentEnvironment, NormalizedVelocityActions
from uav_swarm_control.models import SharedActorCentralCritic
from uav_swarm_control.observations import (
    CentralizedState,
    LocalObservations,
    encode_local_observations,
)
from uav_swarm_control.seeding import RandomStream, derive_indexed_seed

type EnvironmentFactory = Callable[[], MultiAgentEnvironment]


@dataclass(frozen=True, slots=True)
class MAPPOUpdateMetrics:
    """Averaged diagnostics for one parallel collect-update cycle."""

    update: int
    environment_steps: int
    agent_samples: int
    mean_rollout_reward: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float


@dataclass(frozen=True, slots=True)
class MAPPOTrainingResult:
    """Trained centralized-training model and compact update history."""

    model: SharedActorCentralCritic
    history: tuple[MAPPOUpdateMetrics, ...]
    environment_steps: int
    agent_samples: int
    device: torch.device


def _tensor(values: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(np.array(values, copy=True), dtype=torch.float32, device=device)


def _stack_local(observations: Sequence[LocalObservations], device: torch.device) -> Tensor:
    return _tensor(np.stack([encode_local_observations(item) for item in observations]), device)


def _stack_states(states: Sequence[CentralizedState], device: torch.device) -> Tensor:
    return _tensor(np.stack([state.values for state in states]), device)


def _validate_parallel_layout(
    environments: Sequence[MultiAgentEnvironment],
    local_observations: Tensor,
    centralized_states: Tensor,
) -> tuple[int, int, int]:
    if not environments:
        raise ValueError("MAPPO requires at least one environment.")
    agent_ids = environments[0].agent_ids
    if any(environment.agent_ids != agent_ids for environment in environments):
        raise ValueError("parallel environments must use identical agent identifiers and order.")
    if local_observations.ndim != 3 or centralized_states.ndim != 2:
        raise ValueError("parallel observations must use [E,N,F] and states [E,S].")
    if local_observations.shape[:2] != (len(environments), len(agent_ids)):
        raise ValueError("parallel local observation axes do not match environments and agents.")
    if centralized_states.shape[0] != len(environments):
        raise ValueError("centralized state environment axis does not match environments.")
    return len(agent_ids), local_observations.shape[-1], centralized_states.shape[-1]


def _collect_rollout(
    environments: Sequence[MultiAgentEnvironment],
    model: SharedActorCentralCritic,
    config: MAPPOConfig,
    device: torch.device,
    local_observations: Tensor,
    centralized_states: Tensor,
    episode_indices: list[int],
    root_seed: int,
) -> tuple[FlatMAPPOBatch, Tensor, Tensor, float]:
    ppo = config.ppo
    buffer = MAPPORolloutBuffer(
        time_steps=ppo.rollout_steps,
        num_environments=config.num_environments,
        num_agents=model.num_agents,
        local_observation_size=model.local_observation_size,
        centralized_state_size=model.centralized_state_size,
        action_size=model.action_size,
        device=device,
    )
    reward_sum = 0.0
    model.eval()
    for _ in range(ppo.rollout_steps):
        with torch.no_grad():
            policy = model.sample_actions(local_observations)
            values = model.values(centralized_states)
        transitions = [
            environment.step(
                NormalizedVelocityActions(
                    environment.agent_ids,
                    policy.actions[index].detach().cpu().numpy().astype(np.float32, copy=False),
                )
            )
            for index, environment in enumerate(environments)
        ]
        next_local_items = [transition.observations for transition in transitions]
        next_state_items = [transition.centralized_state for transition in transitions]
        transition_states = _stack_states(next_state_items, device)
        with torch.no_grad():
            next_values = model.values(transition_states)
        rewards = _tensor(np.stack([transition.rewards for transition in transitions]), device)
        terminated_environment = torch.tensor(
            [transition.terminated for transition in transitions],
            dtype=torch.bool,
            device=device,
        )
        truncated_environment = torch.tensor(
            [transition.truncated for transition in transitions],
            dtype=torch.bool,
            device=device,
        )
        terminated = terminated_environment.unsqueeze(1).expand(-1, model.num_agents)
        truncated = truncated_environment.unsqueeze(1).expand(-1, model.num_agents)
        buffer.add(
            local_observations=local_observations,
            centralized_states=centralized_states,
            latent_actions=policy.latent_actions,
            rewards=rewards,
            log_probabilities=policy.log_probabilities,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
        )
        reward_sum += float(rewards.mean())

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
                next_local_items[index] = reset.observations
                next_state_items[index] = reset.centralized_state
        local_observations = _stack_local(next_local_items, device)
        centralized_states = _stack_states(next_state_items, device)

    batch = buffer.as_batch(gamma=ppo.gamma, gae_lambda=ppo.gae_lambda).flatten()
    return batch, local_observations, centralized_states, reward_sum / ppo.rollout_steps


def _update_model(
    model: SharedActorCentralCritic,
    optimizer: torch.optim.Optimizer,
    batch: FlatMAPPOBatch,
    config: MAPPOConfig,
) -> tuple[float, float, float, float, float]:
    ppo = config.ppo
    totals = torch.zeros(5, device=batch.local_observations.device)
    minibatches = 0
    model.train()
    for _ in range(ppo.update_epochs):
        indices = torch.randperm(batch.size, device=batch.local_observations.device)
        for start in range(0, batch.size, ppo.minibatch_size):
            selected = indices[start : start + ppo.minibatch_size]
            policy = model.evaluate_latent_actions(
                batch.local_observations[selected],
                batch.latent_actions[selected],
            )
            values = model.values_for_agents(
                batch.centralized_states[selected],
                batch.agent_indices[selected],
            )
            loss = ppo_loss(
                new_log_probabilities=policy.log_probabilities,
                old_log_probabilities=batch.old_log_probabilities[selected],
                advantages=batch.advantages[selected],
                new_values=values,
                returns=batch.returns[selected],
                entropies=policy.entropies,
                clip_coefficient=ppo.clip_coefficient,
                value_coefficient=ppo.value_coefficient,
                entropy_coefficient=ppo.entropy_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.max_gradient_norm)
            optimizer.step()
            totals += torch.stack(
                (
                    loss.policy.detach(),
                    loss.value.detach(),
                    loss.entropy.detach(),
                    loss.approximate_kl.detach(),
                    loss.clip_fraction.detach(),
                )
            )
            minibatches += 1
    means = (totals / minibatches).cpu()
    return (
        float(means[0]),
        float(means[1]),
        float(means[2]),
        float(means[3]),
        float(means[4]),
    )


def train_mappo(
    environment_factory: EnvironmentFactory,
    config: MAPPOConfig,
    *,
    seed: int,
) -> MAPPOTrainingResult:
    """Train MAPPO with shared actors and parallel independent environments."""
    device = resolve_device(config.ppo.device)
    seed_torch(seed)
    environments: list[MultiAgentEnvironment] = []
    try:
        for _ in range(config.num_environments):
            environments.append(environment_factory())
        resets = [
            environment.reset(seed=derive_indexed_seed(seed, RandomStream.INITIAL_STATE, index, 0))
            for index, environment in enumerate(environments)
        ]
        local_observations = _stack_local([reset.observations for reset in resets], device)
        centralized_states = _stack_states([reset.centralized_state for reset in resets], device)
        num_agents, local_size, state_size = _validate_parallel_layout(
            environments,
            local_observations,
            centralized_states,
        )
        model = SharedActorCentralCritic(
            local_size,
            state_size,
            action_size=3,
            num_agents=num_agents,
            actor_hidden_sizes=config.ppo.hidden_sizes,
            critic_hidden_sizes=config.critic_hidden_sizes,
            initial_log_standard_deviation=config.ppo.initial_log_standard_deviation,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.ppo.learning_rate,
            eps=1e-5,
        )
        episode_indices = [0] * config.num_environments
        history: list[MAPPOUpdateMetrics] = []
        agent_samples_per_update = config.ppo.rollout_steps * config.num_environments * num_agents
        environment_steps_per_update = config.ppo.rollout_steps * config.num_environments
        for update in range(1, config.num_updates + 1):
            batch, local_observations, centralized_states, mean_reward = _collect_rollout(
                environments,
                model,
                config,
                device,
                local_observations,
                centralized_states,
                episode_indices,
                seed,
            )
            policy, value, entropy, approximate_kl, clip_fraction = _update_model(
                model,
                optimizer,
                batch,
                config,
            )
            history.append(
                MAPPOUpdateMetrics(
                    update=update,
                    environment_steps=update * environment_steps_per_update,
                    agent_samples=update * agent_samples_per_update,
                    mean_rollout_reward=mean_reward,
                    policy_loss=policy,
                    value_loss=value,
                    entropy=entropy,
                    approximate_kl=approximate_kl,
                    clip_fraction=clip_fraction,
                )
            )
    finally:
        for environment in environments:
            environment.close()
    return MAPPOTrainingResult(
        model=model,
        history=tuple(history),
        environment_steps=config.ppo.total_steps,
        agent_samples=config.ppo.total_steps * model.num_agents,
        device=device,
    )
