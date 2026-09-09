# Configuration

Experiment choices are stored in YAML and validated by immutable Python configuration models.
Unknown keys, missing required values, and invalid cross-field combinations are rejected.

## Directory layout

```text
configs/
├── algorithm/     controller and optimizer settings
├── comparison/    cross-system evidence and report contracts
├── deployment/    compression candidates and acceptance gates
└── experiment/    complete runnable experiment definitions
```

## Common sections

```yaml
schema_version: 1
name: descriptive-experiment-name
seed: 11

formation:
  kind: plane
  num_agents: 4
  spacing_m: 0.6
  center_m: [0.0, 0.0, 2.0]
  euler_radians: [0.0, 0.0, 0.0]

environment:
  time_step_seconds: 0.03333333333333333
  max_episode_steps: 360
  max_velocity_component_mps: 0.5

task:
  initial_center_m: [-1.4, 0.0, 1.6]
  initial_position_noise_m: 0.05
  success_tolerance_m: 0.12
  success_hold_steps: 15
  collision_distance_m: 0.2
  terminate_on_collision: true
```

## Learning configuration

```yaml
algorithm:
  total_steps: 10000000
  rollout_steps: 1000
  num_environments: 20
  update_epochs: 4
  minibatch_size: 250
  learning_rate: 0.0005
  gamma: 0.99
  gae_lambda: 0.95
  clip_coefficient: 0.2
  value_coefficient: 0.5
  entropy_coefficient: 0.001
  max_gradient_norm: 0.5
  hidden_sizes: [256, 256]
  critic_hidden_sizes: [256, 256]
  initial_log_standard_deviation: -0.5
  device: auto
```

`total_steps` counts environment transitions. Agent samples additionally scale by the number of UAVs.
`num_environments` creates independent environment instances, but the current implementation steps
them sequentially.

## Recurrent research configuration

`configs/experiment/recurrent-study/sphere-8-uav.yaml` composes the common MAPPO, PyBullet,
pose, obstacle, and communication schemas with:

```yaml
recurrent:
  feature_size: 256
  hidden_size: 256
  num_layers: 1
  sequence_length: 20
  sequences_per_minibatch: 10
  observation_profile: masked-set
  action_profile: direction-speed
research:
  formation_kinds: [plane, pyramid, cube, sphere]
  mission:
    initial_ground_altitude_m: 0.08
    construction_altitude_m: 2.0
    waypoint_spacing_m: 1.0
```

`sequence_length` must divide the rollout length, and the number of rollout sequences must divide
cleanly into sequence minibatches. Physical recurrent studies require `masked-set`; the
`paper-flat` profile remains available for architecture-parity tests. Ground altitude is an explicit
world-frame value and must be below construction altitude.

The `neighbor-scaling` directory contains the matched sphere family for `N=4,8,16,32`. Every file
uses capacity `N-1`, omits invalid `k`, and records the condition-specific requested cap and optional
sensing radius. Requested `k` is not a promise that `k` neighbors are in range.

## Editing safely

1. Copy the closest configuration to a descriptively named file.
2. Change its internal `name`.
3. Change one scientific factor at a time.
4. Preserve an otherwise-identical control.
5. run a smoke command to validate the complete path.
6. inspect the resolved configuration in `manifest.json` or `run.json`.
7. commit tracked source and configuration before full execution.
8. use a new output directory.

Do not tune against final held-out evaluation episodes. If those episodes influence a decision, they
have become validation data and a new untouched test set is required.

## Local overrides

Files ending in `.local.yaml` are ignored by Git. They may be useful for machine-specific exploratory
settings but must not be the only record of a reported experiment.

