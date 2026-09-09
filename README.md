# UAV Swarm Control

`uav-swarm-control` is a reproducible research platform for decentralized multi-UAV formation
control. It provides validated experiment configuration, Crazyflie simulation in PyBullet,
multi-agent reinforcement learning, a classical DMPC baseline, held-out evaluation, communication
analysis, obstacle scenarios, policy compression, and immutable research artifacts.

The research direction is a paper-aligned FC-LSTM-FC MAPPO controller that constructs and maintains
2D/3D formations, uses changing local neighborhoods, navigates through waypoints, avoids static and
dynamic obstacles, changes formation when necessary, and can be compressed for embedded execution.

## Status

The repository contains both the feed-forward MAPPO reference and an integrated paper-aligned
FC-LSTM-FC MAPPO research path. The recurrent path is software-tested but has not yet produced the
full-budget, multi-seed evidence required for scientific claims. Obstacle-triggered morphing is an
implemented deterministic high-level baseline with learned continuous tracking, not yet a validated
flight result.

Available components include:

- deterministic formation geometry, assignment, and error metrics;
- kinematic and Crazyflie/PyBullet environments;
- feed-forward PPO and parameter-shared MAPPO with a centralized critic;
- sequence-correct recurrent MAPPO with per-agent memory and direction-plus-speed actions;
- deterministic held-out evaluation over independent training seeds;
- a clean-room DMPC adaptation;
- pose-generalization, oracle-obstacle, and communication-topology experiments;
- pooled equal-size 3D formations, ground construction, waypoints, recovery, and morphing studies;
- `N={4,8,16,32}` neighbor-scaling configurations and graph/rigidity/payload metrics;
- recurrent actor distillation, compact FF/GRU/LSTM candidates, export, and host benchmarks;
- configuration, provenance, source snapshots, checkpoints, and result aggregation.

## Lab Linux setup

The project requires Python 3.12. On an Ubuntu or Debian lab machine, install the system tools,
clone the committed research revision, install `uv`, and reproduce the locked environment:

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential libgl1 libglib2.0-0 jq tmux

git clone git@github.com:pratikpsingh/uav-swarm-control.git
cd uav-swarm-control

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv sync --locked --extra simulation --extra dmpc --extra deployment
```

The SSH clone requires a GitHub key on the lab machine. Use the corresponding HTTPS URL if SSH is
not configured. The deployment extra is needed for the configured INT8 compression candidate. An
NVIDIA driver is needed for GPU training, but a separately installed Python or CUDA environment is
not needed because `uv` manages the project environment from `uv.lock`.

Verify the environment and, on an NVIDIA host, confirm that PyTorch sees the GPU:

```bash
uv run python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
uv run uav-swarm-control --log-level INFO
```

## Verify the repository

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest --cov
uv build
```

## Research experiment quick start

Commit the exact source and configurations before launching a research campaign; every artifact
records the Git revision, dirty-worktree flag, dependency versions, and source hashes. Set a unique
output root once per frozen campaign:

```bash
git status --short
UAV_PROJECT_ROOT="$PWD"
UAV_RUN_TAG="research-v1"
UAV_ARTIFACT_ROOT="$UAV_PROJECT_ROOT/artifacts/$UAV_RUN_TAG"
mkdir -p "$UAV_ARTIFACT_ROOT"
```

`git status --short` must print nothing. Run the corrected Paper 04 feed-forward baseline with the
configured research budgets and five independent seeds:

```bash
uv run uav-swarm-control run-baseline \
  --config configs/experiment/baseline/triangle-3-uav.yaml \
  --config configs/experiment/baseline/square-4-uav.yaml \
  --config configs/experiment/baseline/pentagon-5-uav.yaml \
  --output "$UAV_ARTIFACT_ROOT/baselines" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

The complete ordered command set for DMPC comparison, 3D generalization, obstacles, neighbor
scaling, all seven recurrent treatments, policy compression, and cross-paper reporting is in
[Experiment commands](docs/experiments.md). Those commands intentionally omit `--smoke`: smoke mode
is only a plumbing check and its artifacts are not research evidence.

If you are new to the project, read that runbook before launching training. The recommended order is
setup and verification, baseline and DMPC, 3D/obstacle/neighbor studies, recurrent treatments,
compression of accepted teachers, and finally report generation. Independent studies may run in
parallel on separate GPUs, but two processes must never write to the same experiment directory and
compression timing must be measured on an otherwise idle machine.

## Documentation

- [Architecture](docs/architecture.md)
- [Research design](docs/research-design.md)
- [Experiment commands](docs/experiments.md)
- [Configuration](docs/configuration.md)
- [Results and reproducibility](docs/results.md)
- [Manuscript red-line responses](docs/manuscript-red-line-responses.md)

Generated models and results belong under ignored `artifacts/`. Private learning notes and planning
material belong under ignored `learning/` and `plan/`.

## License

A project license has not yet been added. Choose and add it before public distribution. Third-party
dependencies and supplied reference repositories retain their own licenses.
