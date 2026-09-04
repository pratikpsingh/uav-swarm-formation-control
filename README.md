# UAV Swarm Control

A reproducible research platform for studying multi-UAV formation control with classical and
multi-agent reinforcement-learning methods.

## Project status

The project is being built in small, reviewable stages. Stage 0 establishes the Python package,
quality checks, logging, documentation, and experiment-configuration boundaries. It does not yet
contain a simulator or reinforcement-learning algorithm.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

Run the project smoke-test command:

```bash
uv run uav-swarm-control --log-level INFO
```

Run all local quality checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest --cov
```

To apply automatic formatting and safe lint fixes while developing:

```bash
uv run ruff format .
uv run ruff check --fix .
```

## Repository layout

```text
configs/                 Version-controlled experiment inputs
deployment/              Future model export and device benchmarking
docs/                    Architecture and engineering reference documentation
reports/                 Reviewed, reproducible result summaries
scripts/                 Thin executable helpers; reusable logic belongs in src/
src/uav_swarm_control/   Installable Python package
tests/                   Automated tests mirroring the source package
```

Generated checkpoints, tracking logs, and large experiment artifacts are intentionally ignored by
Git. Each future experiment will save a small, version-controlled summary containing its resolved
configuration, seed, code revision, environment details, and evaluation metrics.

## Development principles

- Reproduce a corrected baseline before proposing improvements.
- Keep simulator-specific code behind explicit interfaces.
- Separate training-only information from observations available to a deployed agent.
- Test deterministic mathematics before training stochastic policies.
- Compare methods with reward-independent metrics in a common environment.
- Make one conceptually complete commit per stage.
