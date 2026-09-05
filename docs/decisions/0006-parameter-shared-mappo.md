# ADR 0006: Use parameter-shared actors with a centralized critic

- Status: accepted
- Date: 2026-09-05

## Context

The kinematic environment exposes one local observation per UAV and a separate privileged global
state. Extending PPO to multiple agents introduces two risks: silently mixing time, environment,
and agent axes, and allowing information available only during training to leak into the deployed
policy. Training one unrelated policy per UAV would also increase model size and discard the fact
that every UAV has the same control role.

## Decision

- Use one feed-forward actor whose weights are shared by all agents.
- Give the actor only encoded `LocalObservations`; never give it `CentralizedState`.
- Use a separate centralized critic that maps one global state to one value per stable agent row.
- Preserve rollout tensors as `[time, environment, agent, ...]` until minibatch flattening.
- Count environment transitions and agent samples separately.
- Run independent environments with deterministically derived per-environment, per-episode seeds.
- Store pre-`tanh` actions so rollout log probabilities can be recomputed exactly.
- Save both actor and critic for experiment reproducibility, but evaluate using only the actor.
- Keep the first implementation feed-forward and step environments sequentially.

## Consequences

Parameter sharing reduces the number of actor parameters and lets all agents contribute training
samples to the same behavior. The centralized critic can use global information to reduce variance
and account for coupled rewards without making that information a deployment dependency.

The critic is tied to a fixed agent count, and the current flattened neighbor encoder is tied to a
fixed neighbor capacity and ordering. Sequential environment stepping is simple and deterministic
but will not maximize simulator throughput. Recurrence, permutation-invariant neighbor aggregation,
observation normalization, and resumable optimizer checkpoints remain later, evidence-driven work.
