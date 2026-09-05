# UAV Swarm Control

A reproducible research platform for studying multi-UAV formation control with classical and
multi-agent reinforcement-learning methods.

## Project status

The project is being built in small, reviewable stages. The current package provides validated
formation geometry, multi-agent contracts, strict experiment configuration, a deterministic 3D
point-mass environment, decomposed rewards and metrics, a proportional-controller baseline, and a
tested single-agent PPO foundation. It now also provides parameter-shared MAPPO with centralized
training and decentralized execution. It does not yet contain a rigid-body simulator.

See [the formation geometry contract](docs/formations.md) and
[the multi-agent contract](docs/multi-agent-contracts.md) for the foundational APIs. The
[kinematic environment](docs/kinematic-environment.md) describes the first complete control loop.
The [PPO foundation](docs/ppo.md) explains the learning algorithm and its deliberately simple
reference task. The [MAPPO foundation](docs/mappo.md) explains parameter sharing, centralized
training, and the explicit time/environment/agent batch axes.

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

Run the deterministic scripted baseline:

```bash
uv run uav-swarm-control --log-level INFO run-scripted \
  --config configs/experiment/triangle_kinematic.yaml
```

Train the single-agent PPO reference policy (the checkpoint is written to ignored artifacts):

```bash
uv run uav-swarm-control train-ppo \
  --config configs/experiment/ppo_continuous_bandit.yaml
```

Evaluate that checkpoint:

```bash
uv run uav-swarm-control evaluate-ppo \
  --config configs/experiment/ppo_continuous_bandit.yaml \
  --checkpoint artifacts/checkpoints/ppo_continuous_bandit.pt
```

Train and evaluate the three-agent MAPPO reference policy:

```bash
uv run uav-swarm-control train-mappo \
  --config configs/experiment/mappo_triangle_kinematic.yaml

uv run uav-swarm-control evaluate-mappo \
  --config configs/experiment/mappo_triangle_kinematic.yaml \
  --checkpoint artifacts/checkpoints/mappo_triangle_kinematic.pt
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
