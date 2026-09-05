# Experiment configuration

Configuration files will describe experiments without hiding scientific choices in Python scripts.
They will be grouped by responsibility:

```text
configs/
├── algorithm/     Optimizer, network, rollout, and update settings
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
