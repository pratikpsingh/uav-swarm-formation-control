# Kinematic swarm environment

## Purpose

The kinematic environment is the first complete control loop in the project. It is deliberately
small and deterministic so observation, action, reward, termination, and evaluation behavior can be
verified before adding rigid-body simulation or reinforcement learning.

It implements the Stage 2 MultiAgentEnvironment contract and has no dependency on Gymnasium,
PyBullet, or a neural-network framework.

## State and dynamics

Each UAV is a first-order point mass with world position p and commanded velocity v. For control
interval dt:

    v(t) = a(t) * maximum_velocity_component
    p(t + 1) = p(t) + dt * v(t)

The normalized action a has three components in [-1, 1]. Velocity changes instantaneously. There is
no acceleration, attitude, motor, battery, drag, or downwash model.

This model tests task logic rather than flight physics. The future PyBullet adapter will retain the
same external action contract while translating desired velocity into lower-level control.

## Target and initial state

Assigned targets come from the Stage 1 formation generator, Euler rotation, and configured world
center. Agent row i is assigned target row i.

At reset, the same rotated formation is placed at task.initial_center_m and independent uniform
position noise is added in each axis. The INITIAL_STATE random stream makes this perturbation exactly
repeatable for a given root seed.

## Observations

The local ego row has six features:

| Indices | Feature |
| --- | --- |
| 0:3 | Assigned target position minus current position, in meters |
| 3:6 | Current world-frame velocity, in meters per second |

Every neighbor row also has six features:

| Indices | Feature |
| --- | --- |
| 0:3 | Neighbor position minus ego position, in meters |
| 3:6 | Neighbor velocity minus ego velocity, in meters per second |

Neighbors outside neighbor_radius_m are excluded when a radius is configured. Remaining neighbors
are ordered by distance, with AgentId breaking numerical ties, then limited to max_neighbors.
Unused slots are zero padded and masked False.

The centralized state concatenates all positions, velocities, and assigned targets, producing 9N
features. It is for a future training critic and is not used by the scripted controller.

## Proportional controller

The deterministic controller uses only the first three ego features:

    desired_velocity = gain * relative_target_position
    normalized_action = clip(desired_velocity / maximum_velocity_component, -1, 1)

Large errors saturate at the configured velocity limit. Near the target, commanded speed decreases
linearly. It is a correctness baseline, not a competitor to MAPPO or DMPC.

## Reward decomposition

The reward is reported as five separate per-agent components:

| Component | Definition |
| --- | --- |
| Navigation | Negative assigned-target distance times navigation_weight |
| Formation | Negative normalized shape MSE times formation_weight, shared by all agents |
| Collision | Negative collision_penalty for every agent in at least one collision pair |
| Smoothness | Negative mean squared normalized-action change times smoothness_weight |
| Termination | success_bonus on the step that completes the task |

The total reward is the elementwise sum. Step metrics contain the mean of every component under the
reward/ namespace. Reward weights are explicit in YAML.

## Reward-independent metrics

The environment reports:

- assigned-position RMSE in meters;
- rigid-aligned shape RMSE in meters;
- dimensionless normalized shape RMSE;
- centroid error in meters;
- minimum pairwise separation in meters;
- unique collision-pair count;
- RMS normalized-action change;
- current tolerance status, success, collision failure, step count, and success streak.

Success requires assigned-position RMSE at or below success_tolerance_m, no collision, and that
condition remaining true for success_hold_steps consecutive transitions. Holding the condition
prevents a controller from receiving success after only passing briefly through the target region.

A collision is a pair whose center distance is strictly below collision_distance_m. When
terminate_on_collision is true, a collision causes termination. Reaching max_episode_steps without
termination causes truncation.

## Run the baseline

From the repository root:

    uv run uav-swarm-control --log-level INFO run-scripted \
      --config configs/experiment/triangle_kinematic.yaml

The committed seed currently completes the triangle task in 28 steps with approximately 0.0243 m
final position RMSE.

## Intentional limitations

- Point masses, not quadrotor rigid bodies.
- Instantaneous velocity tracking.
- No static or dynamic obstacles.
- No world boundary.
- No sensor delay, noise, packet loss, or communication cost.
- Fixed agent-to-target assignment.
- Global cooperative termination rather than individual-agent respawn.

These are controlled omissions. Stage 3 validates the task and learning interface; later stages add
physics and research variables one at a time.
