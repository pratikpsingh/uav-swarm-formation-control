# Parameter-shared MAPPO foundation

Stage 5 extends the tested PPO mathematics into a multi-agent control loop. It implements
centralized training with decentralized execution (CTDE): a shared actor controls each UAV from
local information, while a separate critic may inspect the complete swarm state only during
training.

## Information boundary

The actor and critic have different public methods and different input types:

```text
LocalObservations -> encode -> shared actor -> one action per UAV
CentralizedState  ----------------> critic -> one value per UAV
```

The actor input for each agent concatenates its ego features, `K` flattened neighbor rows, and the
`K` validity-mask entries. Its width is therefore `E + K*F + K`. For the three-agent reference task,
`E=6`, `K=2`, and `F=6`, so each actor receives 20 values. Mask entries distinguish a genuinely zero
neighbor feature from an absent neighbor.

One actor network is applied independently to every agent row. Sharing parameters means there is
one policy to train and eventually deploy, not one independently sized policy per vehicle. The
centralized critic receives one global state per environment and predicts a separate value for each
stable agent index.

## Tensor axes

The rollout buffer keeps structure explicit:

| Quantity | Shape | Meaning |
| --- | --- | --- |
| local observations | `[T, E, N, L]` | local vector for each time, environment, and agent |
| centralized states | `[T, E, S]` | one privileged state per time and environment |
| latent actions | `[T, E, N, A]` | pre-`tanh` actions used for exact log-probability replay |
| rewards, values, masks | `[T, E, N]` | one scalar trajectory per agent |

`T` is rollout time, `E` is the number of environment instances, and `N` is the number of agents.
GAE operates along `T` while preserving every `E,N` trajectory. Only after advantages and returns
are computed does the buffer flatten to `T*E*N` agent samples for shuffled PPO minibatches. Each
central state is repeated for its `N` corresponding samples, and an agent-index tensor selects the
correct critic output.

`algorithm.total_steps` counts environment transitions across all parallel environments. Agent
samples equal `total_steps * N`. Reporting both avoids making a multi-agent run look artificially
more or less sample-efficient.

## Collection and update

During collection, the shared actor samples bounded velocity actions and the centralized critic
estimates values. True termination disables bootstrapping; time-limit truncation permits a final
value bootstrap but stops GAE from crossing into a reset episode. Each parallel environment and
episode receives a deterministic seed derived from the experiment root seed.

During an update, stored latent actions are evaluated again under the current actor. The existing
PPO clipped objective, value loss, entropy term, shuffled minibatches, and gradient clipping are
reused rather than reimplemented.

Evaluation is deterministic and calls only the actor. It reports success, whether a collision
occurred anywhere in an episode, return, episode length, final position RMSE, and final shape RMSE.
The physical metrics are kept separate from reward so future reward tuning cannot redefine success.

## Reference experiment

Train and save a checkpoint:

```bash
uv run uav-swarm-control train-mappo \
  --config configs/experiment/mappo_triangle_kinematic.yaml \
  --checkpoint artifacts/checkpoints/mappo_triangle_kinematic.pt
```

Evaluate the saved actor:

```bash
uv run uav-swarm-control evaluate-mappo \
  --config configs/experiment/mappo_triangle_kinematic.yaml \
  --checkpoint artifacts/checkpoints/mappo_triangle_kinematic.pt
```

The completion test trains a shortened configuration for 16,384 environment steps across seeds 11,
22, 33, 44, and 55. Each run must solve all ten deterministic evaluation episodes without a
collision and finish below 0.1 m mean position RMSE.

## Scope boundary

This stage proves the CTDE data path on the lightweight kinematic task. It does not yet prove
rigid-body flight, obstacle avoidance, generalization across agent counts, recurrent memory, or
efficient subprocess simulation. Those concerns remain separate stages so failures can be
localized and comparisons remain defensible.
