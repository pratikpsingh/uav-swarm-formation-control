# UAV Swarm Control

A reproducible research platform for studying multi-UAV formation control with classical and
multi-agent reinforcement-learning methods.

## Project status

The project is being built in small, reviewable stages. The current package provides validated
formation geometry, multi-agent contracts, strict experiment configuration, a deterministic 3D
point-mass environment, decomposed rewards and metrics, a proportional-controller baseline, and a
tested single-agent PPO foundation. It also provides parameter-shared MAPPO with centralized
training and decentralized execution. A pinned `gym-pybullet-drones` adapter now executes the same
formation task with Crazyflie rigid-body physics and an explicit velocity-to-PID action path.
The feed-forward MAPPO baseline inspired by Paper 04 provides multi-seed physical MAPPO training, checkpoint
evaluation, trajectory metrics, and reproducible result records for three, four, and five UAVs.
It is a feed-forward baseline; full-budget scientific validation remains pending lab experiments.
A clean-room classical DMPC adaptation now uses the same physical tasks, held-out episode seeds and
metrics. Guarded comparison artifacts reject incompatible environment/evaluation protocols and keep
DMPC optimization diagnostics separate from task performance.
The pose-generalization study adds episode-randomized plane, pyramid, cube, and sphere targets with disjoint training and
held-out translation, orientation, and scale ranges. Fixed/minimum-distance assignment and
world/target coordinate frames are explicit experimental factors. The bounded smoke workflow is
complete; full-budget lab runs are required before making a generalization claim.
The dynamic-obstacle study adds seeded oracle static and dynamic spherical obstacles, fixed-width masked actor/critic
inputs, a none-to-static-to-dynamic curriculum, equal-budget controls, and matched multi-seed
evaluation. The full smoke workflow is complete; obstacle-avoidance claims require lab training.
The neighbor and communication study adds a masked permutation-invariant neighbor encoder and a fixed-versus-variable topology
study over planar/spatial formations, requested and actual degree, sensing range, dynamic obstacles,
connectivity, rigidity, payload bytes, and across-policy confidence intervals. Scientific claims
remain pending full-budget lab training.
The policy-compression study adds actor-only distillation, architecture/width comparisons, structured pruning, dynamic
INT8 export, and explicit host/target measurement gates. The cross-paper reporting workflow adds a checksum-backed two-track
report: guarded common-environment controller results remain separate from paper-native QuadSwarm,
OmniDrones/Isaac Sim, custom PyBullet, and native DMPC evidence.

See [the formation geometry contract](docs/formations.md) and
[the multi-agent contract](docs/multi-agent-contracts.md) for the foundational APIs. The
[kinematic environment](docs/kinematic-environment.md) describes the first complete control loop.
The [PPO foundation](docs/ppo.md) explains the learning algorithm and its deliberately simple
reference task. The [MAPPO foundation](docs/mappo.md) explains parameter sharing, centralized
training, and the explicit time/environment/agent batch axes. The
[PyBullet adapter](docs/pybullet.md) documents simulator timing, reset behavior, and provenance.
See the [MAPPO baseline protocol](docs/mappo-baseline.md) for source deviations, budget
definitions, local smoke commands, lab commands, and generated artifacts. The
[classical DMPC protocol](docs/dmpc-baseline.md) documents its model, native-repository deviations,
information access, comparison guard and remaining limitations. The
[3D generalization protocol](docs/3d-generalization.md) defines target-pose sampling, assignment,
coordinate frames, held-out evaluation, and the distinction between software and research gates.
The [dynamic-obstacle protocol](docs/dynamic-obstacles.md) documents oracle information, motion and
collision semantics, curriculum controls, safety metrics, artifacts, and current limitations.
The [neighbor and communication protocol](docs/communication-study.md) defines set-invariant actor
inputs, the factorial design, graph metrics, uncertainty, and the Paper 02 comparison boundary.
The [cross-paper comparison protocol](docs/cross-paper-comparison.md) defines evidence status,
compatibility checks, mandatory context columns, provenance, and the research-readiness gate.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

PyBullet is a research-only optional dependency. A normal development sync includes it because the
development test group exercises the adapter. A runtime-only simulator installation uses:

```bash
uv sync --no-dev --extra simulation
```

The classical DMPC command additionally requires SciPy:

```bash
uv sync --no-dev --extra simulation --extra dmpc
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

Run the scripted controller against pinned Crazyflie physics and save a complete JSON record:

```bash
uv run uav-swarm-control run-pybullet \
  --config configs/experiment/pybullet_triangle.yaml \
  --output artifacts/runs/pybullet-triangle.json
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

Run the classical DMPC adaptation on the same bounded three-, four-, and five-UAV smoke protocol:

```bash
uv run uav-swarm-control run-dmpc \
  --task configs/experiment/baseline/triangle-3-uav.yaml \
  --task configs/experiment/baseline/square-4-uav.yaml \
  --task configs/experiment/baseline/pentagon-5-uav.yaml \
  --controller configs/algorithm/dmpc.yaml \
  --smoke \
  --output artifacts/baselines/dmpc-check \
  --project-root .
```

Run the complete bounded 3D pose-generalization suite:

```bash
uv run uav-swarm-control run-generalization \
  --config configs/experiment/pose-generalization/plane-4-uav.yaml \
  --config configs/experiment/pose-generalization/pyramid-5-uav.yaml \
  --config configs/experiment/pose-generalization/cube-8-uav.yaml \
  --config configs/experiment/pose-generalization/sphere-8-uav.yaml \
  --smoke \
  --output artifacts/generalization/check \
  --project-root .
```

Run the complete bounded dynamic-obstacle study:

```bash
uv run uav-swarm-control run-obstacle-study \
  --config configs/experiment/obstacle-avoidance/plane-4-uav.yaml \
  --smoke \
  --output artifacts/obstacles/check \
  --project-root .
```

Run the complete bounded neighbor and communication workflow:

    uv run uav-swarm-control run-communication-study \
      --config configs/experiment/neighbor-study/plane-4-uav.yaml \
      --config configs/experiment/neighbor-study/pyramid-5-uav.yaml \
      --smoke \
      --output artifacts/communication/check \
      --project-root .

Run the bounded policy-compression workflow after creating a matching neighbor-study smoke teacher:

```bash
uv sync --extra deployment
uv run --extra deployment uav-swarm-control run-deployment-study \
  --task configs/experiment/neighbor-study/plane-4-uav.yaml \
  --deployment configs/deployment/policy-compression.yaml \
  --teacher-checkpoint artifacts/communication/<run>/smoke/<task>/<regimen>/seed-11/model.pt \
  --teacher-result artifacts/communication/<run>/smoke/<task>/<regimen>/seed-11/result.json \
  --smoke \
  --output artifacts/deployment/check \
  --project-root .
```

Smoke output validates plumbing only. A research run requires a full-budget teacher that passes the
predeclared behavioral gates. Portable actor graphs and host measurements do not establish embedded
feasibility; repeat flash, peak-RAM, latency, and energy measurements on the target. See
[the compression protocol](docs/policy-compression.md).


Build a two-track cross-paper report from guarded controller comparisons:

```bash
uv run uav-swarm-control build-cross-paper-report \
  --config configs/comparison/cross-paper.yaml \
  --comparison artifacts/baselines/<run>/<profile>/baseline-triangle-3-uav/controller-comparison.json \
  --comparison artifacts/baselines/<run>/<profile>/baseline-square-4-uav/controller-comparison.json \
  --comparison artifacts/baselines/<run>/<profile>/baseline-pentagon-5-uav/controller-comparison.json \
  --output artifacts/comparison/<report-run> \
  --project-root . \
  --evidence-root ..
```

The report verifies paper/repository checksums and controlled protocol fingerprints. Complete smoke
inputs still produce `research_ready: false`.

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
deployment/              Model-export and target-benchmarking guidance
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
