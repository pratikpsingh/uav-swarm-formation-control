# Single-agent PPO foundation

Stage 4 provides a small, reviewable implementation of Proximal Policy Optimization (PPO). Its
purpose is to validate the learning algorithm before Stage 5 introduces multi-agent batching and a
centralized critic.

## Reference task

The continuous target bandit samples a target `x` from `[-0.8, 0.8]` and exposes `[x]` as the
observation. The policy chooses one bounded action `a` in `[-1, 1]` and receives:

```text
reward = 1 - (a - x)^2
```

The episode then terminates. The known optimal solution is `a = x`, so action mean-squared error is
a direct, reward-independent measure of learning. The task deliberately excludes UAV dynamics,
formation geometry, and long-horizon credit assignment.

## Data flow

```text
observation -> actor -> Gaussian sample -> tanh -> bounded action
      |                                           |
      +-> critic -> value                         +-> environment
                                                        |
transition -> rollout buffer -> GAE targets -> shuffled minibatches -> PPO update
```

The actor and critic are separate multilayer perceptrons. The actor represents a diagonal Gaussian
in an unbounded latent action space. `tanh` maps each sample into the action contract's `[-1, 1]`
range. The rollout stores the pre-`tanh` latent action losslessly. Log probabilities use a
numerically stable change-of-variables correction and can be recomputed during an update, even when
the bounded action saturates near -1 or 1.

## Returns and advantages

The discounted return is the sum of future discounted rewards. The critic estimates that return.
The temporal-difference residual is:

```text
delta_t = reward_t + gamma * V(next_state) - V(state_t)
```

Generalized Advantage Estimation (GAE) combines residuals with `gamma * lambda`. A true terminal
state has no future value. A truncated state, such as a time limit, may bootstrap from the critic
but must not propagate an advantage into the next episode. Those two masks remain separate through
the rollout buffer.

## PPO update

PPO compares the probability of a stored action under the new and rollout policies:

```text
ratio = exp(new_log_probability - old_log_probability)
```

The policy objective takes the smaller of the ordinary advantage-weighted ratio and a ratio clipped
to `[1 - epsilon, 1 + epsilon]`. This limits how much one batch can change the policy. Training also
uses a mean-squared value loss, a Gaussian entropy bonus, global gradient clipping, multiple epochs,
and shuffled minibatches.

The entropy is an exploration diagnostic for the base Gaussian. It is not the exact entropy of the
tanh-transformed distribution.

## Reproducibility and artifacts

Environment, policy, and evaluation randomness use independent named streams derived from one root
seed. The checked-in reference configuration selects CPU so fixed-seed tests are portable. Set
`algorithm.device` to `cuda` or `auto` for larger experiments.

Policy checkpoints contain a format version, model dimensions, weights, and small provenance
metadata. They use atomic replacement and PyTorch's restricted weights-only loader. These Stage 4
files support exact policy reload, not optimizer-state training resumption.

Train and save the reference policy:

```bash
uv run uav-swarm-control train-ppo \
  --config configs/experiment/ppo_continuous_bandit.yaml \
  --checkpoint artifacts/checkpoints/ppo_continuous_bandit.pt
```

Evaluate the saved deterministic policy:

```bash
uv run uav-swarm-control evaluate-ppo \
  --config configs/experiment/ppo_continuous_bandit.yaml \
  --checkpoint artifacts/checkpoints/ppo_continuous_bandit.pt
```

`artifacts/` is ignored by Git. Configurations, code, tests, and reviewed result summaries are the
version-controlled research record.

## Scope boundary

Passing this reference task demonstrates that the PPO data path and mathematics can learn a known
continuous mapping. It does not demonstrate UAV control, delayed credit assignment, or MAPPO. Stage
5 will reuse these components while adding agent and environment batch axes, parameter sharing, and
a centralized training critic.
