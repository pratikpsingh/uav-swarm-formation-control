# Experiment configuration

Configuration files will describe experiments without hiding scientific choices in Python scripts.
They will be grouped by responsibility:

```text
configs/
├── algorithm/     Optimizer, network, rollout, and update settings
├── comparison/    Native evidence catalogs and controlled report contracts
├── deployment/    Compression candidates, gates, and measurement protocols
├── environment/   Simulator, dynamics, timing, and world settings
├── experiment/    Reproducible compositions of the other configuration groups
└── task/          Formation, goal, obstacle, observation, and reward settings
```

Stage 2 uses immutable Python dataclasses as the schema and PyYAML's safe loader for serialization.
Unknown keys and invalid cross-field combinations are rejected. Each experiment is currently one
self-contained YAML file, making every scientific choice visible without following an inheritance
graph.

The first schema contains:

- schema_version, experiment name, and root seed;
- formation kind, count, spacing, center, and Euler orientation;
- control time step, episode horizon, and per-axis velocity limit;
- maximum neighbor slots and an optional sensing radius.
- initial-state sampling, success, and collision rules;
- explicit reward-component weights;
- the scripted proportional-controller gain.
- for physics runs, the drone model, physics mode, physics/control rates, GUI, and recording flags.

See [experiment/triangle_kinematic.yaml](experiment/triangle_kinematic.yaml) for a complete example.
The [PPO reference experiment](experiment/ppo_continuous_bandit.yaml) adds validated rollout,
optimization, network, device, and evaluation settings. The
[MAPPO reference experiment](experiment/mappo_triangle_kinematic.yaml) combines those settings with
the kinematic task, parallel-environment count, and centralized-critic widths. A composition
framework will be considered only after repeated configurations create demonstrated duplication.
Local machine overrides should use the suffix .local.yaml, which Git ignores.

The PyBullet smoke gates are [one-drone hover](experiment/pybullet_hover.yaml) and
[three-drone formation](experiment/pybullet_triangle.yaml). Their environment timestep must equal
the reciprocal of the controller frequency, while the physics frequency must be an integer multiple
of that rate.

The `paper04_3uav.yaml`, `paper04_4uav.yaml`, and `paper04_5uav.yaml` experiments combine the
MAPPO and PyBullet schemas with a strict `protocol` section containing five independent training
seeds, an evaluation seed, and a profile label. Use `run-baseline --smoke` for a bounded local
check; the unmodified research budgets are large. See [the baseline protocol](../docs/paper04-baseline.md).

The classical controller is configured independently in
[algorithm/dmpc_native.yaml](algorithm/dmpc_native.yaml). It exposes every horizon, planning-rate,
cost, physical-limit and solver-tolerance choice and rejects missing or unknown keys. `run-dmpc`
composes that controller configuration with one or more Paper 04 task files at runtime. This keeps
the environment protocol identical while preventing controller settings from being duplicated in
each 3/4/5-UAV task. See [the DMPC protocol](../docs/dmpc-baseline.md).

The four `stage9_*` experiment files define plane, pyramid, cube, and sphere generalization tasks.
Each contains two non-overlapping seven-dimensional target-pose boxes: three translation offsets,
three Euler-angle offsets, and one positive scale. They also declare assignment and coordinate-frame
variants, five independent training seeds, and a separate evaluation root seed. The cube task carries
the full 2x2 assignment/frame ablation; the other formations use the selected `minimum-target`
variant. The source configurations are research budgets. `run-generalization --smoke` creates an
explicit bounded copy without changing the source YAML. See
[the 3D generalization protocol](../docs/3d-generalization.md).

`stage10_dynamic_obstacles_4uav.yaml` freezes the four-UAV controlled obstacle study. It declares
the oracle sphere distribution, sensing and safety bounds, one fixed pose range and representation,
three equal-budget training regimens, all four matched evaluation scenarios, and five training
seeds. `run-obstacle-study --smoke` reduces only runtime-related values. See
[the dynamic-obstacle protocol](../docs/dynamic-obstacles.md).

The two stage11 experiment files freeze the four-UAV plane and five-UAV pyramid communication
studies. Each crosses requested neighbor count, finite/unlimited sensing, and clear/mixed-dynamic
obstacles; defines fixed-two, fixed-all, and variable-topology training distributions; retains five
independent training seeds; and fixes a 24-byte relative-state payload model. Use the
run-communication-study command with its smoke option for bounded plumbing validation. See
[the neighbor protocol](../docs/communication-study.md).

`deployment/stage12_policy_compression.yaml` declares the teacher acceptance gates, three
feed-forward widths, GRU/LSTM comparison, structured-pruning and INT8 treatments, five
distillation seeds, episode-level dataset split, host benchmark protocol, and energy measurement
status. The deployment config is composed with exactly one Stage 11 task and matching teacher
checkpoint/result pair at runtime. See [the compression protocol](../docs/policy-compression.md).

`comparison/stage13_cross_paper.yaml` is a strict, checksum-backed evidence catalog for Papers 01-04
and native DMPC plus the contract for normalized common-environment result tables. Missing simulator,
seed, or uncertainty evidence is represented explicitly instead of inferred. See
[the cross-paper protocol](../docs/cross-paper-comparison.md).
