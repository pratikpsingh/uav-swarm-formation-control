# Research experiment runbook

These commands are for full configured research runs. Run them from the repository root on a clean,
committed revision. The output directories are immutable: use one new run tag whenever source,
configuration, dependencies, or the intended campaign changes.

## How to read this guide

The guide has three parts. Sections 1 and 2 prepare a reproducible computer and campaign. Sections
3-9 generate scientific results. Section 10 makes smaller deployment candidates from successful
policies, and Section 11 builds the final comparison report.

A few terms occur throughout the guide:

- A **policy** is the neural-network controller used by the UAVs.
- **Training** changes the policy weights using simulated experience.
- **Evaluation** freezes those weights and measures the policy on new episodes.
- A **seed** is a controlled random starting value. Five training seeds mean five independently
  trained policies, not five evaluations of one policy.
- A **condition** is one evaluation setting, such as two visible neighbors or moving obstacles.
- A **treatment** is one controlled version of the method used to answer a particular question.
- An **artifact** is a saved checkpoint, result, manifest, log, or source snapshot.

Commands containing `--smoke` are software checks only. The research commands below intentionally
omit that flag and therefore use the complete budgets stored in the YAML files.

## 1. Prepare the lab machine

**Purpose:** install one consistent copy of Python and every package required by the simulator,
controllers, tests, and compression study. This is machine setup, not a scientific experiment, and
normally needs to be done only once for a given checkout.

The commands install basic Linux tools, clone the repository, install `uv`, install Python 3.12, and
recreate the dependency versions recorded in `uv.lock`. `tmux` keeps a terminal session alive when
an SSH connection closes, while `jq` is used later to inspect result files.

For Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential libgl1 libglib2.0-0 jq tmux

git clone git@github.com:pratikpsingh/uav-swarm-control.git
cd uav-swarm-control

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv sync --locked --extra simulation --extra dmpc --extra deployment
```

The clone command assumes the lab machine has an SSH key authorized for the repository. If it does
not, use the corresponding HTTPS clone URL. On an NVIDIA machine, install a compatible host driver
through the lab's normal administration process; the locked project environment supplies the Python
packages and Linux CUDA runtime dependencies.

Confirm the runtime:

```bash
uv run python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
uv run uav-swarm-control --log-level INFO
```

## 2. Freeze and verify the campaign

**Purpose:** prove exactly which code and settings produced a result. Here, “freeze” means committing
the chosen source and configurations before training; it does not mean freezing the Linux machine.

`git status --short` must print nothing before a research run. An empty result means all intended
changes are committed. The experiment manifests record the Git revision, whether the tree was dirty,
package versions, platform, and hashes of source files. This allows another researcher to check out
the same revision and audit the run.

```bash
git status --short
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest --cov
uv build
```

The checks have different jobs: Ruff checks formatting and common programming errors, Pyright checks
Python types, Pytest runs the automated tests and measures coverage, and `uv build` confirms that the
project can be packaged. Do not start a full campaign if one of these commands fails.

Define these variables in every shell or scheduler job. Change `UAV_RUN_TAG` for a new campaign; do
not point a changed configuration at an existing artifact directory.

```bash
UAV_PROJECT_ROOT="$PWD"
UAV_RUN_TAG="research-v1"
UAV_ARTIFACT_ROOT="$UAV_PROJECT_ROOT/artifacts/$UAV_RUN_TAG"
mkdir -p "$UAV_ARTIFACT_ROOT"
```

These are shell variables, not learning parameters:

| Variable | Meaning | Example |
| --- | --- | --- |
| `UAV_PROJECT_ROOT` | Absolute path to this repository. `$PWD` means the current directory, so define it only after `cd uav-swarm-control`. | `/home/user/uav-swarm-control` |
| `UAV_RUN_TAG` | Human-readable name for one frozen campaign. Keep it unchanged across jobs belonging to that campaign; choose a new value after changing code, configuration, or dependencies. | `research-v1` |
| `UAV_ARTIFACT_ROOT` | Common parent directory for every result from that campaign. Later commands use it to find earlier outputs without repeating long paths. | `/home/user/uav-swarm-control/artifacts/research-v1` |

`mkdir -p` creates only the common parent directory. Each runner creates its own immutable child
folder, such as `baselines/research/baseline-triangle-3-uav`. The variables normally last only for
the current terminal. Define them again after reconnecting through SSH or inside every scheduler job.
They do not need to be exported because the current shell expands them before starting the CLI.

The configurations use `device: auto`. To allocate a particular GPU, prefix a command with, for
example, `CUDA_VISIBLE_DEVICES=0`. Keep separate jobs on separate GPUs. The commands below use one
Torch CPU thread per process for reproducible host-side timing.

The frozen campaign is large. These counts exclude evaluation, online DMPC optimization, and
compression/distillation:

| Study | Trained policies | Configured training transitions |
| --- | ---: | ---: |
| Corrected Paper 04 baseline | 15 | 500M |
| 3D pose generalization | 35 | 350M |
| Obstacle study | 15 | 150M |
| Fixed-size neighbor study | 30 | 300M |
| Multi-size neighbor scaling | 60 | 600M |
| Seven recurrent treatments | 85 | 2.125B |
| **Total** | **240** | **4.025B** |

Schedule studies separately and inspect pilot learning curves, wall time, GPU utilization, and
failures before reserving the whole campaign. Do not report a pilot as one of the frozen five seeds.

## Recommended execution order

The section numbers describe the scientific workflow, but not every experiment has to wait for the
previous one. Use this order when taking over the project:

1. Complete Sections 1 and 2. Do this before any experiment.
2. Run a plumbing-only smoke check or a separate pilot configuration to estimate runtime. A pilot is
   not one of the final five-seed results.
3. Run Section 3 to establish the corrected Paper 04 learning baseline.
4. Run the DMPC part of Section 4. After both Section 3 and DMPC finish, create the three controlled
   comparison JSON files from the second half of Section 4.
5. Run Sections 5, 6, 7, and 8. They are independent scientific questions and may be scheduled in
   parallel when hardware permits.
6. Run Section 9 treatment by treatment. Start with `base`, inspect it, and then schedule
   `pooled-formations`, `mission`, `obstacles`, `neighbors`, `recovery`, and `morphing`.
7. Run Section 10 only after a full variable-topology teacher from Section 7 or the `neighbors`
   treatment from Section 9 passes every predeclared gate.
8. Run Section 11 after the three MAPPO-versus-DMPC comparison files exist and the checksum-matched
   paper/reference evidence has been copied to the lab machine.

The hard dependencies are:

| Work | Can start after | Required by |
| --- | --- | --- |
| Section 3 MAPPO baseline | Section 2 | Section 4 comparisons |
| Section 4 DMPC run | Section 2 | Section 4 comparisons |
| Sections 5-8 | Section 2 and a successful pilot | Their own result analyses |
| Section 9 recurrent treatments | Section 2 and a successful recurrent pilot | Recurrent compression |
| Section 10 feed-forward compression | A passing Section 7 teacher | Deployment analysis |
| Section 10 recurrent compression | A passing Section 9 `neighbors` teacher | Deployment analysis |
| Section 11 report | Section 4 comparisons and verified evidence files | Final reporting |

## Can experiments run simultaneously?

Yes, but only when they write to different experiment directories and have separate compute
capacity. The same `UAV_ARTIFACT_ROOT` may be shared because the study names create different child
folders. Never launch two processes for the same configuration or recurrent treatment with the same
output root.

Use these rules:

- Prefer one training process per GPU. Two large MAPPO processes on one GPU can run out of memory,
  become much slower, and make wall-time measurements misleading.
- On a one-GPU machine, run GPU training sequentially. DMPC is mainly CPU work, but it can still
  compete with PyBullet and data collection for CPU resources.
- On a multi-GPU machine, bind each job with `CUDA_VISIBLE_DEVICES`. Monitor `nvidia-smi`, `htop`, and
  available RAM before adding more jobs.
- Combined commands run their listed configurations sequentially. To parallelize, invoke one
  configuration per process. The current CLI still runs every seed declared in that configuration;
  it does not expose safe per-seed scheduling.
- Recurrent treatments are safe parallel units because each treatment has a different output child
  directory. Do not run the same treatment twice against the same artifact root.
- Do not run compression latency/RSS benchmarks while other heavy jobs are active on that machine.
  System contention would invalidate the deployment timing comparison.
- Controller comparison and report commands are lightweight, but they must wait for their required
  input artifacts.

For example, on a machine with two GPUs, these two treatments may run in separate terminals or
scheduler jobs:

```bash
CUDA_VISIBLE_DEVICES=0 uv run uav-swarm-control run-recurrent-study \
  --config configs/experiment/recurrent-study/sphere-8-uav.yaml \
  --treatment base \
  --output "$UAV_ARTIFACT_ROOT/recurrent" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1

CUDA_VISIBLE_DEVICES=1 uv run uav-swarm-control run-recurrent-study \
  --config configs/experiment/recurrent-study/sphere-8-uav.yaml \
  --treatment pooled-formations \
  --output "$UAV_ARTIFACT_ROOT/recurrent" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

The two jobs share the campaign root but write to `recurrent/research/base/...` and
`recurrent/research/pooled-formations/...`, so their artifacts do not collide.

## 3. Corrected Paper 04 baseline

**Purpose:** establish the simplest trusted learning result before adding memory, obstacles, changing
neighbors, or formation morphing. This is the reference against which more complex methods should
be judged.

The command trains separate feed-forward MAPPO policies for a 3-UAV triangle, 4-UAV square, and
5-UAV pentagon. “Feed-forward” means the actor reacts to its current local observation and has no
LSTM memory. Five independent seeds are trained because reinforcement learning can succeed or fail
differently from one initialization to another.

**Question answered:** how well does the corrected Paper 04-style MAPPO baseline form and move these
swarms under the repository's common evaluation protocol?

```bash
uv run uav-swarm-control run-baseline \
  --config configs/experiment/baseline/triangle-3-uav.yaml \
  --config configs/experiment/baseline/square-4-uav.yaml \
  --config configs/experiment/baseline/pentagon-5-uav.yaml \
  --output "$UAV_ARTIFACT_ROOT/baselines" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

This requests the YAML-defined 10M, 30M, and 60M steps for each of five training seeds: 500M
environment transitions in total. Each policy is evaluated for 20 held-out episodes. The resulting
`summary.json` files are required for the MAPPO side of Section 4. This is a corrected extension and
controlled reference, not a claim of exact reproduction of Paper 04.

## 4. Matched DMPC and controlled comparisons

**Purpose:** compare the learned controller with a classical, optimization-based controller without
changing the task. DMPC plans a short future trajectory online; it has no neural-network training
phase.

DMPC uses the same three tasks, evaluation seeds, horizons, simulator, and common metrics as the
feed-forward MAPPO baseline:

```bash
uv run uav-swarm-control run-dmpc \
  --task configs/experiment/baseline/triangle-3-uav.yaml \
  --task configs/experiment/baseline/square-4-uav.yaml \
  --task configs/experiment/baseline/pentagon-5-uav.yaml \
  --controller configs/algorithm/dmpc.yaml \
  --output "$UAV_ARTIFACT_ROOT/dmpc" \
  --project-root "$UAV_PROJECT_ROOT"
```

**Question answered:** under the same simulated conditions, what are the performance and failure
trade-offs between feed-forward MAPPO and clean-room DMPC?

Build one guarded comparison per task only after both methods finish. The command rejects mismatched
protocol fingerprints, preventing results with different initial conditions, horizons, or physics
from being placed in the same table.

```bash
mkdir -p "$UAV_ARTIFACT_ROOT/controller-comparisons"

uv run uav-swarm-control compare-controllers \
  --mappo-summary "$UAV_ARTIFACT_ROOT/baselines/research/baseline-triangle-3-uav/summary.json" \
  --dmpc-result "$UAV_ARTIFACT_ROOT/dmpc/research/baseline-triangle-3-uav/dmpc-swarm-reference/result.json" \
  --output "$UAV_ARTIFACT_ROOT/controller-comparisons/triangle-3-uav.json"

uv run uav-swarm-control compare-controllers \
  --mappo-summary "$UAV_ARTIFACT_ROOT/baselines/research/baseline-square-4-uav/summary.json" \
  --dmpc-result "$UAV_ARTIFACT_ROOT/dmpc/research/baseline-square-4-uav/dmpc-swarm-reference/result.json" \
  --output "$UAV_ARTIFACT_ROOT/controller-comparisons/square-4-uav.json"

uv run uav-swarm-control compare-controllers \
  --mappo-summary "$UAV_ARTIFACT_ROOT/baselines/research/baseline-pentagon-5-uav/summary.json" \
  --dmpc-result "$UAV_ARTIFACT_ROOT/dmpc/research/baseline-pentagon-5-uav/dmpc-swarm-reference/result.json" \
  --output "$UAV_ARTIFACT_ROOT/controller-comparisons/pentagon-5-uav.json"
```

## 5. Held-out 3D pose generalization

**Purpose:** determine whether a policy learned a reusable 3D formation rule instead of memorizing
one fixed target pose. Training samples one range of translations, rotations, and scales; evaluation
uses a separate, held-out range that was not used for learning.

The plane, pyramid, cube, and sphere tasks test several geometries. The cube additionally compares
fixed versus minimum-distance assignment and world versus target coordinate frames.

**Question answered:** does performance remain acceptable when the target formation is moved,
rotated, or resized outside the training distribution?

```bash
uv run uav-swarm-control run-generalization \
  --config configs/experiment/pose-generalization/plane-4-uav.yaml \
  --config configs/experiment/pose-generalization/pyramid-5-uav.yaml \
  --config configs/experiment/pose-generalization/cube-8-uav.yaml \
  --config configs/experiment/pose-generalization/sphere-8-uav.yaml \
  --output "$UAV_ARTIFACT_ROOT/generalization" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

This is the feed-forward per-formation reference. It trains and evaluates disjoint 3D translation,
rotation, and scale splits.

## 6. Static and dynamic obstacle study

**Purpose:** measure whether obstacle-aware training reduces collisions while preserving navigation
and formation quality. Three matched policies are trained: a clear-environment control, a
static-obstacle control, and a curriculum that progressively includes harder obstacle motion.

Every policy is evaluated on shared clear, static, slow-dynamic, and mixed-dynamic episodes. Shared
episode seeds make differences between training regimens easier to interpret.

**Question answered:** which training strategy transfers best from clear flight to static and moving
obstacles?

This is an oracle study: the policy receives exact simulator obstacle state. It does not yet test
camera/LiDAR detection, sensor noise, or missed observations.

```bash
uv run uav-swarm-control run-obstacle-study \
  --config configs/experiment/obstacle-avoidance/plane-4-uav.yaml \
  --output "$UAV_ARTIFACT_ROOT/obstacles" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

The output reports performance per regimen and the paired differences between regimens.

## 7. Neighbor topology at fixed swarm sizes

**Purpose:** isolate the effect of local communication when the swarm size and formation are fixed.
Here, `k` is the requested number of other UAVs visible to one UAV.

For a 4-UAV plane and 5-UAV pyramid, the runner trains fixed-two, fixed-all, and variable-topology
policies. It then evaluates requested and actually visible neighbors under global/local sensing and
clear/mixed-obstacle conditions.

**Question answered:** for a fixed `N`, how do `k`, sensing range, and link changes affect formation
error, success, collisions, graph connectivity, and communication bytes?

```bash
uv run uav-swarm-control run-communication-study \
  --config configs/experiment/neighbor-study/plane-4-uav.yaml \
  --config configs/experiment/neighbor-study/pyramid-5-uav.yaml \
  --output "$UAV_ARTIFACT_ROOT/communication" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

The output contains condition-level physical and communication metrics plus paired differences
between the three training approaches.

## 8. Swarm-size by neighbor-count relationship

**Purpose:** answer the broader `N`-versus-`k` question. `N` is the total number of UAVs and `k` is
the number of neighbors available to each local actor. This suite varies both while keeping the
sphere formation family and major physical settings fixed.

The results should show whether more neighbors continue improving formation performance as the
swarm grows, and where additional communication begins to give little benefit. Connectivity and
rigidity metrics help explain *why* an error changes instead of reporting only a reward value.

**Question answered:** what neighbor budget is sufficient for different swarm sizes, and what
formation-quality or collision cost is paid when that budget is reduced?

```bash
uv run uav-swarm-control run-communication-study \
  --config configs/experiment/neighbor-scaling/sphere-4-uav.yaml \
  --config configs/experiment/neighbor-scaling/sphere-8-uav.yaml \
  --config configs/experiment/neighbor-scaling/sphere-16-uav.yaml \
  --config configs/experiment/neighbor-scaling/sphere-32-uav.yaml \
  --output "$UAV_ARTIFACT_ROOT/neighbor-scaling" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

This controlled feed-forward suite holds the formation family and physical settings fixed while
crossing `N={4,8,16,32}` with declared `k` grids. Use graph connectivity, algebraic connectivity,
rigidity rank, payload bytes, normalized formation error, success, and collision metrics to analyze
the relationship. Do not pool these results statistically with the 8-UAV recurrent study.

## 9. Paper-aligned recurrent MAPPO treatments

**Purpose:** evaluate the main FC-LSTM-FC MAPPO research path. The LSTM gives every local UAV actor a
memory of recent observations, which may help when neighbors or obstacles appear, disappear, or move.
The policy outputs a 3D direction plus speed rather than directly outputting four unrelated velocity
components.

Each treatment asks one controlled question:

| Treatment | Simple meaning | Main question |
| --- | --- | --- |
| `base` | Recurrent 8-UAV sphere without the added mission factors | Does the recurrent controller learn the basic task? |
| `pooled-formations` | One policy sees plane, pyramid, cube, and sphere | Can one equal-size policy handle several 3D formation shapes? |
| `mission` | Start near the ground, construct the formation, then follow waypoints | Can formation building and navigation happen as one mission? |
| `obstacles` | Train/evaluate with clear, static, slow, and mixed obstacles | Does recurrent memory help obstacle-aware formation flight? |
| `neighbors` | Train fixed and changing `k` policies | How does memory interact with changing local communication? |
| `recovery` | Apply a controlled velocity disturbance to selected UAVs | How quickly and reliably does the formation recover? |
| `morphing` | Change between equal-size formation templates near obstacles | Can the same `N` UAVs choose a more suitable shape and continue? |

The morphing logic is presently a deterministic high-level decision with learned continuous
tracking. It is not yet evidence of an end-to-end learned or physically flown morphing system.

This command runs all seven treatments sequentially. For a scheduler, submit the same command once
per treatment with only that treatment's `--treatment` line and a shared artifact root.

```bash
uv run uav-swarm-control run-recurrent-study \
  --config configs/experiment/recurrent-study/sphere-8-uav.yaml \
  --treatment base \
  --treatment pooled-formations \
  --treatment mission \
  --treatment obstacles \
  --treatment neighbors \
  --treatment recovery \
  --treatment morphing \
  --output "$UAV_ARTIFACT_ROOT/recurrent" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

The configuration uses five seeds, 25M environment steps per trained policy, a 256-unit
FC-LSTM-FC actor/critic, masked local neighbor/obstacle sets, and direction-plus-speed actions. The
full command is therefore much larger than a single 25M-step run; schedule and inspect the treatments
separately when possible.

## 10. Select an accepted teacher and compress it

**Purpose:** reduce actor size and inference cost for small onboard devices without silently
accepting an unsafe teacher or student. A large trained policy acts as the teacher; smaller feed-forward,
GRU, LSTM, pruned, and INT8 candidates learn to imitate it and are then evaluated in closed loop.

This section depends on full results from Section 7 for feed-forward compression and the `neighbors`
treatment in Section 9 for recurrent compression.

Compression must not use an arbitrary or best-looking checkpoint. First inspect every independent
seed from the predeclared `variable-topology` regimen:

```bash
for result in "$UAV_ARTIFACT_ROOT"/communication/research/neighbor-study-plane-4-uav/variable-topology/seed-*/result.json; do
  printf '\n%s\n' "$result"
  jq '{training_seed, k2: (.condition_summaries["k2-global-mixed"] | {collision_free_success, collision_any, mean_normalized_shape_rmse}), kall: (.condition_summaries["kall-global-mixed"] | {collision_free_success, collision_any, mean_normalized_shape_rmse})}' "$result"
done
```

The inspection command prints one seed at a time. `k2` is the difficult mixed-obstacle evaluation
with two requested neighbors; `kall` uses all possible neighbors under the same obstacle class.

A seed is eligible only if both conditions have `collision_free_success >= 0.80`,
`collision_any <= 0.05`, and `mean_normalized_shape_rmse <= 0.15`. Set the accepted seed number and
run the full feed-forward compression study:

```bash
UAV_FF_TEACHER_SEED=11
UAV_FF_TEACHER_DIR="$UAV_ARTIFACT_ROOT/communication/research/neighbor-study-plane-4-uav/variable-topology/seed-$UAV_FF_TEACHER_SEED"

uv run --extra deployment uav-swarm-control run-deployment-study \
  --task configs/experiment/neighbor-study/plane-4-uav.yaml \
  --deployment configs/deployment/policy-compression.yaml \
  --teacher-checkpoint "$UAV_FF_TEACHER_DIR/model.pt" \
  --teacher-result "$UAV_FF_TEACHER_DIR/result.json" \
  --output "$UAV_ARTIFACT_ROOT/deployment" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

Repeat the same gate inspection for the recurrent neighbor treatment:

```bash
for result in "$UAV_ARTIFACT_ROOT"/recurrent/research/neighbors/recurrent-sphere-8-uav/variable-topology/seed-*/result.json; do
  printf '\n%s\n' "$result"
  jq '{training_seed, k2: (.condition_summaries["k2-global-mixed"] | {collision_free_success, collision_any, mean_normalized_shape_rmse}), kall: (.condition_summaries["kall-global-mixed"] | {collision_free_success, collision_any, mean_normalized_shape_rmse})}' "$result"
done
```

Then use one passing seed:

```bash
UAV_RECURRENT_TEACHER_SEED=11
UAV_RECURRENT_TEACHER_DIR="$UAV_ARTIFACT_ROOT/recurrent/research/neighbors/recurrent-sphere-8-uav/variable-topology/seed-$UAV_RECURRENT_TEACHER_SEED"

uv run --extra deployment uav-swarm-control run-recurrent-deployment-study \
  --task configs/experiment/recurrent-study/sphere-8-uav.yaml \
  --deployment configs/deployment/policy-compression.yaml \
  --teacher-checkpoint "$UAV_RECURRENT_TEACHER_DIR/model.pt" \
  --teacher-result "$UAV_RECURRENT_TEACHER_DIR/result.json" \
  --output "$UAV_ARTIFACT_ROOT/deployment" \
  --project-root "$UAV_PROJECT_ROOT" \
  --torch-threads 1
```

Replace the example seed `11`; do not weaken the gate after seeing results. If no seed passes, the
honest result is that compression is blocked pending a stronger preregistered teacher. The exported
unit is the decentralized actor only. Host flash proxies, RSS, and latency are not measurements of
the final flight processor.

## 11. Build the two-track cross-paper report

**Purpose:** create an auditable final report without pretending that results from different papers
used identical tasks. The controlled track contains this repository's MAPPO and DMPC results from
the same simulator. The native track separately summarizes what the original papers and supplied
repositories report in their own settings.

Only the controlled track is suitable for direct numerical method comparison. The native track is
context because action spaces, observations, simulators, budgets, and metrics differ across papers.
SHA-256 checks ensure that the report used the expected PDF and repository evidence files.

The evidence root must contain the checksum-matched `papers/`, `multi-UAV-formation/`, and
`DMPC-to-MARL-Sim/` paths declared in `configs/comparison/cross-paper.yaml`. Copy that evidence tree
to the lab machine unchanged and set its absolute path:

```bash
UAV_EVIDENCE_ROOT="/absolute/path/to/JRF"

uv run uav-swarm-control build-cross-paper-report \
  --config configs/comparison/cross-paper.yaml \
  --comparison "$UAV_ARTIFACT_ROOT/controller-comparisons/triangle-3-uav.json" \
  --comparison "$UAV_ARTIFACT_ROOT/controller-comparisons/square-4-uav.json" \
  --comparison "$UAV_ARTIFACT_ROOT/controller-comparisons/pentagon-5-uav.json" \
  --output "$UAV_ARTIFACT_ROOT/cross-paper" \
  --project-root "$UAV_PROJECT_ROOT" \
  --evidence-root "$UAV_EVIDENCE_ROOT"
```

The report is written to
`$UAV_ARTIFACT_ROOT/cross-paper/cross-paper-comparison/report.md`. Native paper results remain a
separate contextual track; they are not treated as common-environment measurements.

## Resume and failure rules

The training runners support `--resume`. Add it only when the source, configuration, runtime
provenance, and output root are unchanged. It skips completed seeds; it does not restore optimizer
state inside an interrupted seed. If a seed directory is incomplete, preserve it for diagnosis and
start a new run tag. DMPC, compression, and report generation do not have resume mode and require a
new output root after an interrupted or changed run.

## Work that has no research command yet

Direct-action versus waypoint-control ablation, smoothness-penalty ablation, wind and sensor-noise
robustness, continuous collision checking, paper-ready artifact-only plotting, and measurements on
the selected flight processor are not implemented runners. They must be implemented and tested
before commands or claims are added. The deterministic morphing treatment is an implemented
high-level baseline, not yet a validated flight result.
