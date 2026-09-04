# Multi-agent contracts

## Purpose

The contract layer defines the data exchanged by environments, controllers, and learning
algorithms. It has no simulator or reinforcement-learning framework dependency. A kinematic
environment and a future PyBullet adapter must both obey the same boundary.

## Stable agent identity and row order

Each agent uses a non-negative AgentId. Every batched array follows the order declared by its
agent_ids tuple. Identifiers must be unique and remain stable during an episode.

This prevents an array row from silently changing meaning between observations, actions, and
evaluation. The initial implementation uses agent_0, agent_1, and so on.

## Local actor observations

LocalObservations keeps ego and neighbor information structurally separate:

| Field | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| ego | (N, E) | float32 | Local features for each agent |
| neighbors | (N, K, F) | float32 | Up to K neighbor feature rows |
| neighbor_mask | (N, K) | bool | True only where a neighbor slot is valid |

N is the number of agents, E is the ego feature count, K is the configured neighbor capacity, and F
is the neighbor feature count. The Stage 3 kinematic environment uses E = 6 and F = 6: relative
assigned-target position and velocity for ego, and relative position and velocity for neighbors.

Invalid neighbor slots are zero padded. The mask, rather than the numerical padding value, states
whether a slot is valid. Keeping a separate neighbor axis allows a later permutation-invariant
encoder and controlled neighbor-count experiments.

## Centralized critic state

CentralizedState is a finite one-dimensional float32 array. It is privileged training information
for centralized training with decentralized execution. It is deliberately a different type from
LocalObservations, reducing the risk that information available only during training is
accidentally passed to a deployable actor.

## Actions

NormalizedVelocityActions contains one [vx, vy, vz] row per agent. Each component lies in [-1, 1].
An environment converts it to physical velocity by multiplying each component by
environment.max_velocity_component_mps.

The limit is explicitly per axis. Therefore the maximum possible Euclidean magnitude is
sqrt(3) times max_velocity_component_mps. A future magnitude-limited action mode must use a
different contract rather than changing this behavior silently.

## Reset and step lifecycle

Every environment implements the MultiAgentEnvironment protocol:

    reset(seed=...) -> ResetResult
    step(actions) -> StepResult
    close() -> None

A step returns:

- the next local observations;
- the next centralized state;
- one reward per agent;
- a task-termination flag;
- a time-limit truncation flag;
- finite, named diagnostic metrics.

terminated means the modeled task reached a terminal condition such as success or collision.
truncated means an external boundary such as the maximum step count stopped the episode. They are
mutually exclusive in this cooperative environment contract. Learning code must preserve the
distinction because value targets normally bootstrap after truncation but not after true
termination.

Rewards are optimization signals. Metrics are reporting quantities with physical or experimental
meaning. A reward change must not redefine the common evaluation metrics.

## Numeric and mutability contract

Model-facing arrays are canonicalized to float32, checked for finite values, copied, and made
read-only. Boolean masks must already have boolean dtype. Defensive copies ensure that a simulator
cannot mutate a previously returned transition through a shared NumPy buffer.

Stage 1 geometry remains float64 because it is reference mathematics. The conversion to float32
happens explicitly at the learning/environment boundary.

## Deterministic seeding

One unsigned 64-bit root seed deterministically derives independent named streams:

- environment randomness;
- initial-state sampling;
- action sampling;
- policy randomness.

The implementation uses NumPy SeedSequence and returns isolated Generator instances. It does not
modify NumPy's global random state. Stream enum values are part of the reproducibility contract and
must never be renumbered.

## Configuration

Experiment YAML is loaded with yaml.safe_load, narrowed from untrusted values, and converted into
immutable dataclasses. Validation rejects:

- missing or unknown keys;
- unsupported schema versions;
- invalid experiment names and seeds;
- non-finite or non-positive physical values;
- incompatible formation names and agent counts;
- a neighbor capacity larger than num_agents - 1.

The first complete example is
[configs/experiment/triangle_kinematic.yaml](../configs/experiment/triangle_kinematic.yaml).
Configurations can be converted back to a plain resolved mapping for future experiment artifacts.

One explicit YAML file is used per experiment for now. Hydra-style composition is intentionally
deferred until repeated algorithm, task, and simulator configurations demonstrate a real need.
