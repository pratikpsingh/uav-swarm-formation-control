# PyBullet UAV adapter

The adapter adds rigid-body dynamics without changing the scientific task contract. Controllers
still emit one normalized world-frame velocity vector per UAV. The environment scales it to metres
per second, and one upstream `DSLPIDControl` instance per vehicle converts that target into four
motor RPMs. `CtrlAviary` then advances the rigid-body state.

## Timing

There are two explicit clocks:

- the controller runs at 48 Hz, so one action covers 1/48 second;
- physics runs at 240 Hz, so each action is held for five physics updates.

Configuration rejects fractional ratios and rejects an environment timestep that disagrees with
the controller rate. Metrics record both control steps and physics steps.

## Reproducibility boundary

The adapter accepts only the audited simulator version and exact Git revision recorded in
`uv.lock`. At runtime it verifies installation provenance and records the package version, revision,
PyBullet API version, drone model, physics mode, rates, GUI setting, and recording setting.

Every reset supplies the root seed, reconstructs initial positions from the project seed stream,
resets the simulator, and clears every PID controller's internal state. PyBullet may still vary
across platforms or builds, so reproducibility means pinned software plus recorded settings—not a
claim of bit-identical trajectories on every machine.

## Result artifact

`run-pybullet` always writes JSON containing the resolved experiment configuration, independent
simulator provenance, episode returns, terminal flags, and final metrics. The write is atomic.
Generated files belong under `artifacts/` and remain outside Git.

## Completion gates

```bash
uv run uav-swarm-control run-pybullet \
  --config configs/experiment/pybullet_hover.yaml \
  --output artifacts/runs/hover.json

uv run uav-swarm-control run-pybullet \
  --config configs/experiment/pybullet_triangle.yaml \
  --output artifacts/runs/triangle.json
```

The first gate checks stable hover before horizontal motion. The second checks that the shared task,
neighbor observations, PID action path, physics, metrics, and lifecycle work for a swarm before any
RL policy is trained in this simulator.
