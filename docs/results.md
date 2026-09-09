# Results and reproducibility

Generated checkpoints and results are stored under ignored `artifacts/`. They are not committed to
Git and must be backed up separately on research storage.

## Learned-policy layout

```text
<output>/<profile>/<experiment>/
├── manifest.json
├── source.zip
├── summary.json
└── seed-<seed>/
    ├── run.json
    ├── updates.jsonl
    ├── latest.pt
    ├── model.pt
    └── result.json
```

- `manifest.json`: method, resolved configuration, protocol, and source/runtime fingerprints.
- `source.zip`: exact project source and dependency metadata used for the run.
- `run.json`: seed-specific configuration and simulator provenance.
- `updates.jsonl`: append-only optimizer/update diagnostics.
- `latest.pt`: periodic diagnostic weights.
- `model.pt`: final actor/critic checkpoint.
- `result.json`: raw held-out episodes and seed-level summary.
- `summary.json`: aggregation across independently trained policies.

Specialized studies add variant, regimen, condition, or candidate directories. Recurrent outputs add a
treatment directory (`base`, `pooled-formations`, `mission`, `obstacles`, `neighbors`, `recovery`, or
`morphing`). Episode records persist `context/*` values such as the sampled formation identity so a
pooled result can be stratified correctly. Recovery records add peak error, error area, sustained
recovery time, and residual error.

Recurrent deployment artifacts record the action profile, actor-only parameter and logical storage
counts, serialized bytes, per-agent recurrent-state bytes, export error, checksum, and host benchmark.
They explicitly exclude the centralized critic. A host-portable graph is not a target firmware image.

## Timing

Training and evaluation wall times are recorded separately. Simulator throughput depends heavily on
CPU execution; GPU acceleration affects neural-network updates but not all PyBullet work. Record host
CPU, GPU, RAM, OS, driver, CUDA, and PyTorch information when comparing runtime.

## Resume behavior

`--resume` skips completed seeds only when configuration, provenance, and checkpoint digests match.
It does not reconstruct the optimizer, rollout, and random-number state of an interrupted seed.
Restart interrupted training in a new output root.

## Interpretation

For learned policies, aggregate evaluation episodes inside each training seed, then report the mean
and sample standard deviation across training-seed means. Evaluation episodes from one checkpoint
are not independent trained policies.

DMPC has no training seeds; its variation across evaluation initial conditions is a different
sampling unit. Do not present both standard deviations as equivalent uncertainty estimates.

Inspect raw failures and trajectories in addition to summary means. For pooled policies, inspect every
formation separately; an aggregate can hide a failed shape. For neighbor studies, analyze requested
`k`, actual degree, connectivity, rigidity, and bytes rather than treating requested `k` as the graph.
Reward, DMPC objective, formation
error, collision, and success answer different questions.

## Publication artifacts

Publication plots and tables should be generated from immutable result directories and include the
input artifact hashes. Source experiment data remain under `artifacts/`; reviewed lightweight
figures can be copied into the manuscript repository with provenance.
