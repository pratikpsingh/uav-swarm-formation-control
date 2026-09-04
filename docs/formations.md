# Formation geometry

The formation package is independent of simulators and reinforcement-learning frameworks. It defines
desired geometry and evaluation semantics shared by every future controller.

## Coordinate and representation contract

- Coordinates are right-handed and z-up.
- A point set is a finite float64 NumPy array with shape `(N, 3)`.
- Row `i` corresponds to agent `i`; assignment is fixed unless another API explicitly says otherwise.
- Templates are centered at the origin.
- `spacing` is the minimum Euclidean distance between different template points.
- Geometry functions return new arrays and do not intentionally mutate caller inputs.

Using float64 at this boundary gives stable reference calculations. Simulator and neural-network
adapters may explicitly cast to float32.

## Templates

```python
from uav_swarm_control.formations import FormationKind, create_formation

target_shape = create_formation(
    FormationKind.POLYGON,
    num_agents=8,
    spacing=0.5,
)
```

Supported templates:

| Kind | Valid count | Definition |
| --- | --- | --- |
| `line` | Any `N >= 1` | Equally spaced x-axis points |
| `triangle` | `N = 3` | Equilateral triangle |
| `square` | `N = 4` | Regular square |
| `polygon` | Any `N >= 3` | Regular planar polygon |
| `plane` | Perfect square | Complete square lattice |
| `cube` | Perfect cube | Complete cubic lattice |
| `sphere` | Any `N >= 2` | Golden-angle spherical shell approximation |
| `pyramid` | `N = 5` | Regular square pyramid |

Incomplete grids, cubes, and pyramids are rejected because their geometry is not uniquely implied by
the name. A future task may add explicitly named partial structures with their own definitions.

## World placement

```python
from uav_swarm_control.formations import rotation_matrix_from_euler, transform

rotation = rotation_matrix_from_euler(roll=0.2, pitch=-0.3, yaw=1.0)
world_targets = transform(
    target_shape,
    scale_factor=1.2,
    rotation=rotation,
    offset=(2.0, -1.0, 3.0),
)
```

`transform` applies positive scale, proper SO(3) rotation, and translation in that order. It does
not clip points to a world boundary. Environment placement must select a feasible pose.

Euler rotation uses active, right-handed rotations:

```text
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

For row-wise points, rotation is applied as `points @ R.T`.

## Metrics

Position metrics compare assigned world targets directly:

```text
position_mse = mean_i ||current_i - target_i||^2
```

Shape metrics first use proper Kabsch alignment to remove translation and rotation:

```text
shape_mse = mean_i ||aligned_current_i - target_i||^2
```

Scale and reflection are not fitted. A formation with the wrong spacing is therefore penalized.

The dimensionless normalized metric is:

```text
normalized_shape_mse = shape_mse / target_diameter^2
```

Explicit SSE, MSE, and RMSE functions prevent aggregation and unit confusion:

- SSE has squared-distance units and grows with agent count.
- MSE is squared distance per agent.
- RMSE has distance units.
- Normalized MSE and RMSE are dimensionless.

Metrics use fixed row correspondence. Assignment-invariant metrics will be introduced separately if
an experiment permits agents to exchange formation points.
