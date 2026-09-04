# ADR 0002: Separate formation shape, pose, and metrics

- Status: accepted
- Date: 2026-09-04

## Context

The inherited implementation contains duplicate template generators, changes the meaning of spacing
between shapes, mixes safety clipping into geometry, and reports total squared error as mean error.
Those choices prevent controlled comparisons across swarm sizes and simulators.

## Decision

- Represent templates as centered float64 arrays with shape `(N, 3)`.
- Define spacing as minimum pairwise separation.
- Keep template creation independent of world placement.
- Apply world pose with positive scale, proper SO(3) rotation, and translation.
- Use fixed row correspondence in the initial baseline.
- Expose position and rigid-aligned shape errors separately.
- Name SSE, MSE, and RMSE explicitly.
- Normalize shape MSE by squared target diameter when a dimensionless metric is needed.
- Reject ambiguous incomplete square grids, cubes, and pyramids.
- Keep simulator constraints such as altitude clipping outside the geometry package.

## Consequences

- Swarm sizes can be compared without silently changing spacing semantics.
- The same geometry can be used by analytical, PyBullet, and Isaac Sim environments.
- Full 3D orientation is available before simulator integration.
- Fixed assignment remains simple and matches the initial Paper 04 baseline.
- Interchangeable assignment will require a separate, explicit algorithm.
- Some shapes accept fewer agent counts than the inherited best-effort generators.
