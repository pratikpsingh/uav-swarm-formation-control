# ADR 0005: Validate single-agent PPO on a known continuous task

- Status: accepted
- Date: 2026-09-05

## Context

Moving directly from the scripted kinematic controller to MAPPO would introduce stochastic policy
learning, value estimation, rollout semantics, multi-agent tensor axes, parameter sharing, and a
centralized critic at once. A learning failure would be hard to localize.

## Decision

- Implement PPO in project-owned, typed modules instead of hiding it behind a training framework.
- Validate it first on a one-step continuous target-matching task with a known optimum.
- Use separate actor and critic multilayer perceptrons.
- Model continuous actions with a diagonal Gaussian followed by `tanh` and correct log probabilities
  for that transformation.
- Store old log probabilities, values, next values, termination, and truncation explicitly.
- Keep return estimation, GAE, and the PPO loss as pure, hand-testable functions.
- Use independent seeded streams for environment sampling, policy sampling, and evaluation.
- Save versioned policy-only checkpoints atomically and load them with `weights_only=True`.
- Require successful learning across five fixed seeds before extending the algorithm to MAPPO.

## Consequences

The algorithm is transparent enough to explain and debug from first principles. The reference task
is fast and has a measurable optimum, but it does not test long-horizon behavior or UAV dynamics.
The implementation intentionally lacks parallel environments, observation normalization, learning-
rate schedules, optimizer-resume checkpoints, and experiment tracking; those should be added only
when a later stage has a concrete need and tests for their semantics.
