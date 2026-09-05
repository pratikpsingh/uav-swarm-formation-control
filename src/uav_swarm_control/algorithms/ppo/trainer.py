"""Single-agent PPO rollout collection and optimization."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from uav_swarm_control.configuration import PPOConfig
from uav_swarm_control.environments.single_agent import SingleAgentEnvironment
from uav_swarm_control.models import ActorCritic
from uav_swarm_control.seeding import RandomStream, derive_seed

from .buffer import RolloutBatch, RolloutBuffer
from .objectives import ppo_loss


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    """Averaged diagnostics for one collect-update cycle."""

    update: int
    environment_steps: int
    mean_rollout_reward: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float


@dataclass(frozen=True, slots=True)
class PPOTrainingResult:
    """Trained model and its compact learning history."""

    model: ActorCritic
    history: tuple[PPOUpdateMetrics, ...]
    environment_steps: int
    device: torch.device


def resolve_device(name: str) -> torch.device:
    """Resolve a configured device and fail clearly when CUDA was requested."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch.")
    if name not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device {name!r}.")
    return torch.device(name)


def seed_torch(root_seed: int) -> int:
    """Seed PyTorch from its independent project random stream."""
    policy_seed = derive_seed(root_seed, RandomStream.POLICY)
    torch.manual_seed(policy_seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    return policy_seed


def _tensor(values: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(np.array(values, copy=True), dtype=torch.float32, device=device)


def _collect_rollout(
    environment: SingleAgentEnvironment,
    model: ActorCritic,
    config: PPOConfig,
    device: torch.device,
    observation: np.ndarray,
) -> tuple[RolloutBatch, np.ndarray, float]:
    buffer = RolloutBuffer(
        config.rollout_steps,
        environment.observation_size,
        environment.action_size,
        device,
    )
    reward_sum = 0.0
    current = observation
    model.eval()
    for _ in range(config.rollout_steps):
        observation_tensor = _tensor(current, device)
        with torch.no_grad():
            output = model.sample(observation_tensor.unsqueeze(0))
        action = output.action[0]
        transition = environment.step(action.detach().cpu().numpy().astype(np.float32, copy=False))
        next_observation = transition.observation
        with torch.no_grad():
            next_value = model.value(_tensor(next_observation, device).unsqueeze(0))[0]
        buffer.add(
            observation=observation_tensor,
            latent_action=output.latent_action[0],
            reward=transition.reward,
            log_probability=output.log_probability[0],
            value=output.value[0],
            next_value=next_value,
            terminated=transition.terminated,
            truncated=transition.truncated,
        )
        reward_sum += transition.reward
        current = environment.reset().observation if transition.episode_done else next_observation
    return (
        buffer.as_batch(gamma=config.gamma, gae_lambda=config.gae_lambda),
        current,
        reward_sum / config.rollout_steps,
    )


def _update_model(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: PPOConfig,
) -> tuple[float, float, float, float, float]:
    totals = torch.zeros(5, device=batch.observations.device)
    minibatches = 0
    model.train()
    for _ in range(config.update_epochs):
        indices = torch.randperm(config.rollout_steps, device=batch.observations.device)
        for start in range(0, config.rollout_steps, config.minibatch_size):
            selected = indices[start : start + config.minibatch_size]
            output = model.evaluate_latent_actions(
                batch.observations[selected],
                batch.latent_actions[selected],
            )
            loss = ppo_loss(
                new_log_probabilities=output.log_probability,
                old_log_probabilities=batch.old_log_probabilities[selected],
                advantages=batch.advantages[selected],
                new_values=output.value,
                returns=batch.returns[selected],
                entropies=output.entropy,
                clip_coefficient=config.clip_coefficient,
                value_coefficient=config.value_coefficient,
                entropy_coefficient=config.entropy_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
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


def train_ppo(
    environment: SingleAgentEnvironment,
    config: PPOConfig,
    *,
    seed: int,
) -> PPOTrainingResult:
    """Train PPO from scratch using one reproducible on-policy interaction stream."""
    device = resolve_device(config.device)
    seed_torch(seed)
    model = ActorCritic(
        environment.observation_size,
        environment.action_size,
        config.hidden_sizes,
        config.initial_log_standard_deviation,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    observation = environment.reset(seed=seed).observation
    history: list[PPOUpdateMetrics] = []
    for update in range(1, config.num_updates + 1):
        batch, observation, mean_reward = _collect_rollout(
            environment,
            model,
            config,
            device,
            observation,
        )
        policy, value, entropy, approximate_kl, clip_fraction = _update_model(
            model,
            optimizer,
            batch,
            config,
        )
        history.append(
            PPOUpdateMetrics(
                update=update,
                environment_steps=update * config.rollout_steps,
                mean_rollout_reward=mean_reward,
                policy_loss=policy,
                value_loss=value,
                entropy=entropy,
                approximate_kl=approximate_kl,
                clip_fraction=clip_fraction,
            )
        )
    return PPOTrainingResult(model, tuple(history), config.total_steps, device)
