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

No configuration loader or schema is selected in Stage 0. We will define the domain objects and
validation requirements first, then choose the smallest tool that satisfies them. Local machine
overrides should use the suffix `.local.yaml`, which Git ignores.
