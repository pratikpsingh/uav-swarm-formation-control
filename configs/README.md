# Experiment configuration

Configuration files make scientific choices inspectable without reading implementation code.

```text
configs/
├── algorithm/     controller-specific optimization settings
├── comparison/    evidence catalogs and comparison contracts
├── deployment/    compression candidates, gates, and measurements
├── environment/   reserved for reusable simulator definitions
├── experiment/    complete runnable experiment definitions
└── task/          reserved for reusable task definitions
```

Current experiments are self-contained YAML files. Immutable Python dataclasses provide the schema,
and PyYAML's safe loader handles serialization. Unknown fields and invalid cross-field combinations
are rejected.

## How configurations become experiments

A YAML file describes *what* to run: formation, simulator, reward, training budget, seeds, evaluation
conditions, and profile. The CLI command selects *which runner* interprets that file and *where* its
artifacts are stored. One YAML file can expand into several regimens, conditions, and five separately
trained policies, so one command does not necessarily mean one model.

Do not edit a YAML file after a frozen campaign starts. Copy it to a new descriptive file, change its
internal `name`, commit the change, and use a new run tag. See the experiment runbook for the safe
execution order, parallelization rules, and expected artifact paths.

## Reference experiments

- `triangle_kinematic.yaml`: deterministic multi-agent kinematic task.
- `ppo_continuous_bandit.yaml`: small continuous-action PPO reference.
- `mappo_triangle_kinematic.yaml`: parameter-shared MAPPO reference.
- `pybullet_hover.yaml`: one-UAV physical adapter check.
- `pybullet_triangle.yaml`: three-UAV scripted physical check.

## Feed-forward physical baseline

```text
experiment/baseline/
├── triangle-3-uav.yaml
├── square-4-uav.yaml
└── pentagon-5-uav.yaml
```

Each file declares physics, task, MAPPO, five training seeds, a separate evaluation root, and a
research profile. The configured budgets are large. Research commands use these budgets unchanged;
the optional CLI smoke profile is only for software verification.

## DMPC

`algorithm/dmpc.yaml` contains the horizon, planning rate, objective weights, physical limits, and
solver tolerances. The `run-dmpc` command combines it with one or more baseline task files so MAPPO
and DMPC use the same task protocol.

## Pose generalization

Files under `experiment/pose-generalization/` declare plane, pyramid, cube, and sphere tasks. They
include disjoint training/evaluation pose boxes and assignment/coordinate-frame variants.

The feed-forward generalization runner trains separate policies. The recurrent study implements a
pooled equal-size plane/pyramid/cube/sphere treatment; it remains an experimental capability until
full-budget results pass the predeclared metrics.

## Obstacle avoidance

`experiment/obstacle-avoidance/plane-4-uav.yaml` defines a four-UAV oracle-sphere study with
no-obstacle, static-obstacle, and curriculum training regimens evaluated on paired clear/static/
dynamic scenarios.

## Neighbor topology

Files under `experiment/neighbor-study/` define four-UAV plane and five-UAV pyramid tasks. They cross
requested neighbor count, finite/unlimited range, and clear/mixed-dynamic conditions while recording
actual graph metrics and a 24-byte relative-state payload model.

Files under `experiment/neighbor-scaling/` hold the sphere formation family and physical settings
fixed while crossing `N={4,8,16,32}` with declared `k` grids. They are the controlled feed-forward
suite for estimating the relationship among swarm size, visible neighbors, graph structure,
communication payload, formation error, success, and collision rate.

## Recurrent research study

`experiment/recurrent-study/sphere-8-uav.yaml` defines the shared 8-UAV FC-LSTM-FC MAPPO protocol.
The runner exposes `base`, `pooled-formations`, `mission`, `obstacles`, `neighbors`, `recovery`, and
`morphing` treatments without changing the frozen training seeds or held-out evaluation root.

## Compression

`deployment/policy-compression.yaml` defines teacher acceptance gates, feed-forward and recurrent
student candidates, pruning/quantization treatments, distillation seeds, dataset split, and host
benchmark settings.

## Cross-system evidence

`comparison/cross-paper.yaml` records checksum-backed native evidence and the contract for controlled
MAPPO/DMPC result tables. Native-system numbers remain separate from common-environment comparisons.

## Safe modification

Copy the nearest experiment to a descriptively named file, change its internal `name`, modify one
scientific factor, retain a control, smoke-check it, and use a new artifact output. Files ending in
`.local.yaml` are ignored and must not be the sole record of a published run.

See [configuration documentation](../docs/configuration.md) and
[experiment commands](../docs/experiments.md).
