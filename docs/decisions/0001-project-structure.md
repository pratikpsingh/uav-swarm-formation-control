# ADR 0001: Use a packaged `src` layout

- Status: accepted
- Date: 2026-09-04

## Context

This repository will grow from deterministic geometry into simulator adapters, multi-agent training,
and deployment tools. The old repositories mix scripts, vendored environments, and reusable logic,
which makes imports and ownership difficult to understand.

## Decision

Use one installable package under `src/uav_swarm_control`. Keep configuration, learning notes, plans,
tests, and thin scripts outside the package. Manage Python and dependencies with `uv`, and commit the
universal `uv.lock` file.

## Consequences

- Tests import the installed package rather than accidentally importing files from the repository.
- Reusable behavior has a clear home and can be packaged for lab machines.
- Simulator dependencies can later be isolated in optional dependency groups.
- A build backend is required; Stage 0 uses `uv_build` because the package is pure Python.
