# ADR 0003: Multi-agent contracts and configuration

- Status: accepted
- Date: 2026-09-05

## Context

The reference repositories mix simulator state, actor input, centralized information, rewards,
termination flags, random state, and command-line configuration. That makes it difficult to replace
the simulator, compare controllers fairly, or determine what information a deployed policy uses.

Future work requires a lightweight kinematic environment, MAPPO, PyBullet, DMPC, dynamic obstacles,
variable neighbor counts, and deployment measurements. These components need one stable exchange
boundary.

## Decision

- Identify agents with validated non-negative AgentId values and fixed array row order.
- Represent actor-visible data as structured batched ego and masked-neighbor arrays.
- Represent centralized critic state with a separate training-only type.
- Use read-only float32 arrays at environment and learning boundaries.
- Use normalized per-axis 3D velocity actions.
- Return separate termination and truncation flags and reject an ambiguous simultaneous end.
- Keep rewards separate from named evaluation metrics.
- Use one explicit root seed and stable, isolated random-stream namespaces.
- Define immutable typed configuration objects and reject unknown YAML keys.
- Use safe PyYAML loading and one self-contained experiment file initially.
- Defer a configuration composition framework until repeated configurations justify it.

## Consequences

All environment adapters and controllers must perform explicit conversion at this boundary. This is
slightly more code than returning unconstrained dictionaries, but shape, dtype, information-access,
and lifecycle errors are caught before training.

Padded neighbors permit fixed tensor sizes while preserving actual graph degree through a mask.
Separating local and centralized inputs supports MAPPO without leaking privileged critic inputs to
the actor.

Self-contained YAML duplicates some values across experiments. That cost is accepted until the
project has enough repeated configurations to choose composition rules from evidence.
