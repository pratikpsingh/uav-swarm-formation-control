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

See [experiment/triangle_kinematic.yaml](experiment/triangle_kinematic.yaml) for a complete example.
A composition framework will be considered only when Stage 4 introduces repeated algorithm
settings. Local machine overrides should use the suffix .local.yaml, which Git ignores.
