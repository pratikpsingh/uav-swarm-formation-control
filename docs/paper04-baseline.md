# Corrected Paper 04 baseline

The implemented method is **corrected-paper04-feedforward-mappo**. It is a controlled baseline
inspired by Paper 04's MA-PPO ablation, not an exact implementation of its MA-LSTM-PPO method.
A completed software smoke test does not establish convergence or reproduce the paper's tables.

## Source audit and deviations

Source: Zhao et al., *UAV Distribution Formation Control with Improved PPO Algorithm*, supplied
Paper 04 PDF (SSRN 5286784, preprint). The audit inspected Section 3.1, Section 4.2, Table 3
(pp. 21-22), Figure 9, and Section 4.3. Repository evidence refers to the supplied sibling
`my-mappo/onpolicy/` tree, which is not a dependency of this package.

| Choice | Paper / supplied code | Corrected baseline |
| --- | --- | --- |
| Swarms | Regular polygons with 3, 4, 5 UAVs | Same counts and target topology |
| Training budget | 10M, 30M, 60M reported steps | Same numeric budgets, explicitly environment transitions per training seed |
| Actor/critic | Dense-LSTM-dense, width 256 (Fig. 9) | Feed-forward 256,256 actor and critic; no recurrent claim |
| PPO | lr 0.0005, gamma 0.99, GAE 0.95, clip 0.2 | Same values |
| Batch/workers | train_batch_size 20000, 80 Ray workers | 1000 rollout steps x 20 independent environments = 20000 environment transitions per update; sequential stepping |
| Episode limit | Table 3: 242; old script overrides to 240 | 242 control steps |
| Timing | Old wrapper defaults to 30 Hz, 8 seconds | 30 Hz control, 240 Hz physics; 242 steps = 8.0667 seconds |
| Physics | Custom PyBullet Crazyflie; paper discusses drag/ground effects | Pinned Stage 6 Crazyflie CF2X; ground, drag, downwash enabled |
| Action | Eq. 18: direction plus magnitude; PID to motors | Three normalized world-velocity components, 0.5 m/s per component |
| Local observations | Eq. 17 describes position, attitude, velocity, rotor state plus neighbors | Assigned-target displacement, own linear velocity, relative neighbor positions/velocities and masks |
| Critic | Joint/shared observations | Positions, velocities and targets for all UAVs; no attitude/PID memory |
| Formation error | Prose says mean; Eq. 20 displays a sum; old code uses SSE | Per-agent MSE divided by squared target diameter; report its square root separately |
| Formation weight | Eq. 22 includes formation; old wrapper defaults to zero | Active weight 1 |
| Navigation | Eq. 24 and old wrapper: distance progress | Same progress formula, using each UAV's assigned destination |
| Weights | Paper defines w1/w2 without numeric values in the inspected reward section; old wrapper uses 10/2 | Navigation 10, collision 2; explicitly inherited design choices |
| Other shaping | Old wrapper adds arrival, velocity, tilt/smoothness terms | No arrival bonus or smoothness shaping; smoothness still measured |
| Spacing | Old polygon spacing changes meaning with N | 1 m nearest-neighbor target separation for every N |
| Placement | Exact paper coordinates/distribution not recoverable from description; old code uses random centers/Gaussian perturbations | Fixed target center (0,0,1), initial center (-1,0,1), independent uniform +/-0.1 m position perturbations |
| Success | Paper reports reward/error curves; no directly reusable success protocol | RMSE <=0.08 m, no threshold collision, held for 15 steps |
| Collision | Eq. 25 threshold penalty | Any neighbor within 0.2 m penalizes the affected UAV once; episode terminates on inter-UAV collision |
| Neighbors | Paper communication radius R; old code supports neighbor limits | All N-1 neighbors, no radius cutoff |
| Stabilization | Paper mentions observation/layer/value normalization (PopArt), value clipping | Existing tested PPO/GAE, advantage normalization, gradient clipping, tanh actions; those additional mechanisms are not implemented |
| Evaluation | Paper reports FAR/FS/NAR/NS and final-20% summaries | Held-out episodes with geometric/safety metrics; no direct numerical comparison of reward totals |

The paper's stability description and displayed inverse-standard-deviation formula do not agree on
whether larger or smaller is better. We report explicitly defined quantities instead of reproducing
that ambiguity. Shape-only error removes translation and proper 3D rotation; it cannot establish
that vehicles reached their assigned destinations. Our geometry also differs from the paper's
displayed planar transform. Full SO(3) alignment can hide a rigidly tilted planar formation, so
assigned-position RMSE must accompany shape error.

Planar *targets* do not constrain the dynamics to a plane. UAVs can move vertically and tilt. The
current task has no dynamic obstacles, workspace bounds, or ground-contact terminal condition.
Inter-UAV collision metrics are sampled center-distance violations, not swept mesh contacts.
Altitude failures can therefore continue until the horizon; do not interpret a low inter-UAV
collision count as proof of flight safety.

## Reward

For UAV i, with target distance d and target diameter D:

```text
reward_i = -(shape_MSE / D^2)
           + 10 * (d_previous_i - d_current_i)
           - 2 * indicator(any UAV within collision distance)
```

A 0.1 m advance gives +1 navigation reward. Equal rigid translation contributes zero shape cost.
The formation term is divided by N before normalization; doubling the swarm does not automatically
double the per-agent shape penalty. Reward weights and all placement assumptions are explicit YAML.

## Experiment protocol

Each task has five independent training roots: 11, 22, 33, 44, 55. Each policy is evaluated
deterministically after reloading its saved checkpoint. Evaluation derives 20 episode seeds from a
separate root (20260907). All training seeds and the proportional controller use the same evaluation
episodes, enabling paired comparisons. Evaluation episodes are not independent training runs.

The report first averages episodes within a training seed, then reports the mean and sample standard
deviation across training-seed means. The repeated scripted evaluations use the same episodes; they
are a reference controller, not five independent learned policies. No confidence interval or
statistical significance claim is inferred from five seeds alone.

Metrics include success, collision-free success, any inter-UAV collision, collision pair-steps,
minimum clearance, average per-UAV path length, action-change RMS, mean/final normalized shape RMSE,
final position RMSE, episode return, simulated duration, training transitions, agent samples, and
training/evaluation wall time. Collision pair-steps measure duration/exposure, not unique collision
events. Clearance is center separation minus the 0.2 m threshold. Path length includes reset-to-first
step movement. Smoothness includes the first command relative to zero and is dimensionless; compare
it only at matching control rates and velocity scales. Minimum separation includes the reset state.
Training time excludes evaluation, final checkpoint I/O, and initial provenance inspection, but
includes environment initialization and periodic training logging/checkpoints.

## Run locally

From the repository root:

```bash
uv sync --locked
uv run uav-swarm-control run-baseline \
  --config configs/experiment/paper04_3uav.yaml \
  --config configs/experiment/paper04_4uav.yaml \
  --config configs/experiment/paper04_5uav.yaml \
  --smoke --output artifacts/baselines/check
```

Smoke mode explicitly changes each seed to 256 training transitions, 128-step rollouts, one
environment, one update epoch, CPU, two evaluation episodes, and a 48-step horizon. The architecture,
reward, physics, and five training seeds remain the same. Its horizon is too short for the nominal
1 m task to reliably finish. It validates the pipeline; its success rate is not research evidence.
The saved configuration records all overrides and outputs live under a distinct `smoke/` directory.

## Run on the lab machine

After committing the source and installing the lockfile:

```bash
uv run uav-swarm-control run-baseline \
  --config configs/experiment/paper04_3uav.yaml \
  --config configs/experiment/paper04_4uav.yaml \
  --config configs/experiment/paper04_5uav.yaml \
  --output artifacts/baselines/paper04-v1
```

This requests **500 million environment transitions and 2.25 billion agent samples** in total over
15 runs. Start with one task and measure throughput before scheduling the full suite. PyBullet
steps on CPU; a large GPU alone does not accelerate that simulation loop. The 20 environments are
independent but stepped sequentially, not 20 Ray workers. `device: auto` chooses CUDA when available.
`--torch-threads` defaults to 1 and is saved in provenance. Benchmark thread settings on the lab host
before committing a research configuration. Do not assume the paper's wall-clock speed.

For a repeat invocation, `--resume` skips completed seeds only when configuration, code/runtime
provenance, and checkpoint hashes match. Existing directories are never silently overwritten.
An interrupted seed has diagnostic `latest.pt` weights, but the optimizer, rollout, and RNG are not
resumable. Restart interrupted training in a new output root. A code commit or runtime change also
requires a new output root. No claim of exact optimizer resumption is made.

## Artifacts and evaluation

Each task writes:

```text
research/paper04-3uav/
  manifest.json       method label, resolved config, source/runtime fingerprints
  source.zip          exact Python source, pyproject.toml, uv.lock
  seed-11/
    run.json          seeded config and installed simulator provenance
    updates.jsonl     one diagnostic record per PPO update
    latest.pt         diagnostic weights from first and every tenth update
    model.pt          final actor/critic weights
    result.json       held-out episodes, scripted reference, means and counters
  seed-22/ ...        remaining independent training runs
  summary.json       means and sample standard deviations across seeds
```

Use a saved policy again without training:

```bash
uv run uav-swarm-control evaluate-baseline \
  --config configs/experiment/paper04_3uav.yaml \
  --checkpoint artifacts/baselines/paper04-v1/research/paper04-3uav/seed-11/model.pt \
  --output artifacts/baselines/paper04-v1/research/paper04-3uav/seed-11/reevaluation.json
```

For a smoke checkpoint, also supply `--smoke`. Task and simulator compatibility are checked before
evaluation. New evaluation outputs must have a new filename. The actor uses only local observations;
the critic is loaded as part of the checkpoint but never queried by the evaluation controller.

Run `uv run pytest tests/evaluation/test_baseline.py -q` for focused verification. The full repository
gate remains Ruff formatting/lint, Pyright, and `uv run pytest --cov`.

## Research completion

The software workflow and smoke checks are distinct from the scientific completion gate. Only after
the lab runs finish and their reports are reviewed can we judge convergence, stability across seeds,
and whether the corrected baseline is strong enough for obstacle, neighbor-count, or compression
experiments. Exact MA-LSTM-PPO reproduction additionally requires sequence-correct recurrent
training and the documented missing normalization mechanisms.
