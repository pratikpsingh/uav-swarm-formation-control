# Architecture

## Design goals

The package separates domain mathematics, simulation, learning, evaluation, and generated artifacts.
This keeps experiment choices visible and allows algorithms to share one environment and metric
contract.

```text
YAML configuration
    -> strict configuration models
    -> formation/task construction
    -> kinematic or PyBullet environment
    -> controller or learned actor
    -> trajectory evaluation
    -> immutable artifacts and summaries
```

## Package responsibilities

| Package | Responsibility |
| --- | --- |
| `configuration` | YAML loading, immutable models, cross-field validation |
| `formations` | templates, transforms, assignment, and errors |
| `observations` | local actor and centralized critic inputs |
| `rewards` | explicit reward components |
| `environments` | kinematic, PyBullet, pose, obstacle, and communication behavior |
| `models` | actors, critics, and neighbor encoders |
| `algorithms` | PPO/MAPPO rollout, advantage estimation, updates, checkpoints |
| `controllers` | proportional control and clean-room DMPC |
| `communication` | graph and payload metrics |
| `obstacles` | sphere state, motion, and clearance mathematics |
| `evaluation` | deterministic rollouts, aggregation, provenance, study runners |
| `deployment` | distillation, compressed actors, export, host benchmarks |

## Learned controllers

### Feed-forward reference

The original reference uses a parameter-shared feed-forward actor, local per-UAV observations, a
centralized feed-forward critic during training, three normalized world-frame velocity components,
and deterministic actor-only evaluation.

### Paper-aligned recurrent path

The integrated recurrent method is:

```text
actor:  local observation -> FC 256 -> ReLU -> LayerNorm
        -> LSTM 256 -> LayerNorm -> FC 4 -> squashed Gaussian action

critic: global state -> FC 256 -> ReLU -> LayerNorm
        -> LSTM 256 -> LayerNorm -> FC 1 -> state value
```

The four actor outputs represent a 3D direction and speed magnitude. Actor weights are shared, while
hidden and cell state are maintained separately for every `(environment, UAV)` row and reset at
episode boundaries. Recurrent PPO stores the initial actor/critic state of chronological sequence
chunks and performs truncated backpropagation through time; it does not shuffle individual time
steps. The critic remains training-only.

Two observation profiles exist. `paper-flat` reproduces `9 + 6K` inputs before optional obstacle
features; `masked-set` uses explicit validity masks and a permutation-invariant neighbor encoder.
Physical recurrent studies currently require `masked-set`.

For the 8-UAV/3-obstacle configuration, the full padded local record has 79 values. Its exported
actor-only teacher contains 547,044 parameters (2,188,176 logical FP32 bytes) plus 2,048 bytes of
LSTM state per UAV. These figures exclude the critic and runtime buffers; they motivate distillation
to the configured compact FF/GRU/LSTM candidates and do not establish fit on a flight processor.

The environment converts the normalized desired velocity into motor RPM through the upstream
Crazyflie PID and advances pinned PyBullet dynamics.

## Environment composition

Physical experiment behavior is composed from increasingly specialized environments:

```text
PyBullet swarm
    -> formation progress and target-pose generalization
    -> static/dynamic obstacles and communication topology
    -> pooled formation, construction/waypoint, or morphing mission wrapper
```

Obstacle state is currently an exact simulator oracle. No camera, lidar, tracking, occlusion, or
sensor-noise claim is implied.

## Information boundary

During training:

```text
actor <- local observation
critic <- joint swarm state
```

During evaluation and deployment:

```text
actor <- local observation
critic is not used
```

Evaluation code must not expose centralized state to the action controller.

## Reproducibility boundary

Every research output records resolved configuration, method identity, source/runtime provenance,
seed protocol, simulator metadata, checkpoint digest, and raw episode outcomes. Existing output
directories are not silently overwritten.
