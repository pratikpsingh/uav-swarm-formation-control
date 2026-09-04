# ADR 0004: First-order kinematic control loop

- Status: accepted
- Date: 2026-09-05

## Context

Training MAPPO directly in a quadrotor simulator would combine task design, observation semantics,
reward behavior, termination, physics, and optimization. A failure would be difficult to localize.
The project needs a fast reference environment whose equations and expected trajectories can be
checked by hand.

## Decision

- Model every UAV as a first-order 3D point mass controlled by instantaneous velocity.
- Keep normalized actions per axis and scale them by the configured physical component limit.
- Use fixed formation-target assignment.
- Expose target displacement and current velocity as ego features.
- Expose relative position and velocity in masked nearest-neighbor slots.
- Use deterministic distance ordering with AgentId as the tie-break.
- Give the future centralized critic positions, velocities, and targets.
- Sample reset noise only from the named initial-state random stream.
- Decompose reward into navigation, formation, collision, smoothness, and terminal terms.
- Keep evaluation metrics independent of reward weights.
- Require consecutive in-tolerance steps for success.
- Use a proportional controller as an executable correctness baseline.

## Consequences

The environment is fast, deterministic, and suitable for PPO/MAPPO implementation tests. It cannot
establish real flight performance because it omits attitude and actuator dynamics. Results from this
environment must therefore be labeled kinematic and must not be compared directly with physical
flight results without a separate native-system track.

The local observation is translation-equivariant because it uses relative target and neighbor
positions. World-frame velocity and axes remain explicit assumptions. Later changes to body-frame
observations, assignment, or action semantics require separate experiments rather than silent
replacement.
