# Stage 12 policy-compression smoke verification

Date: 2026-09-07. Profile: `smoke`. Method:
`stage12-validated-teacher-policy-compression`.

The bounded end-to-end run completed all seven configured treatments and five independent
distillation seeds: 35 student results plus one actor-only teacher result. It regenerated 16 complete
teacher episodes (64 per-agent sequences, 767 environment steps), split only at episode boundaries,
trained every student, performed closed-loop evaluation across all 16 Stage 11 conditions, exported
and reloaded every graph, launched a fresh benchmark process for every artifact, and wrote per-seed,
per-candidate, and root summaries.

The Stage 11 smoke teacher correctly failed the research gate: collision-free success was zero on both
`k2-global-mixed` and `kall-global-mixed`, although collision rate was zero and mean normalized shape
RMSE was 0.0445 and 0.0436. The root artifact therefore says `scientific_valid: false` and the
manifest labels it `plumbing-only-smoke-teacher`. This is the expected safety behavior, not a failed
smoke run. The research command would reject the same teacher.

Representative seed-101 systems records are shown only to verify measurement plumbing:

| Actor | Parameters | Export bytes | Host p50 ms | Host peak RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| teacher | 85,219 | 384,589 | 0.282 | 747,548,672 |
| FF-16 | 1,459 | 48,141 | 0.297 | 787,746,816 |
| FF-32 | 4,451 | 60,109 | 0.269 | 788,664,320 |
| FF-64 | 15,043 | 102,477 | 0.280 | 789,843,968 |
| GRU-32 | 10,595 | 81,997 | 0.300 | 801,853,440 |
| LSTM-32 | 13,667 | 95,629 | 0.385 | 803,864,576 |
| FF-64, 50% masked channels | 15,043 | 102,477 | 0.246 | 814,301,184 |
| FF-32, dynamic INT8 | 4,451 | 179,085 | 1.599 | 816,148,480 |

These are host Python measurements from a nearly untrained teacher and must not be used to rank
architectures. In particular, INT8 is larger and slower here because portable-graph/runtime overhead
dominates this tiny model on this host; target kernels may behave differently. Structured zeros did
not shrink the dense graph, which is reported honestly. Peak RSS includes the Python and PyTorch
runtime and is not a UAV RAM requirement. Energy remains explicitly unavailable until an external
power measurement is provided.

Artifacts use source hash
`fc826509f30f32197668da8e1738bc8431558241f0ac7679285d51b027076349`, committed parent
`54926971ff94b825747fc21e91b65ee5600adc82`, Python 3.12.3, PyTorch 2.14.0+cu130, TorchAO 0.17.0,
NumPy 2.5.2, PyBullet 3.2.7, and one Torch CPU thread. The exact dirty source snapshot is stored with
the ignored run at `artifacts/deployment/stage12-smoke-20260907/smoke/stage12-policy-compression/`.

The scientific gate remains open until a full Stage 11 policy passes the predeclared teacher metrics,
all 35 full-budget distillation/evaluation runs complete, and shortlisted actors are compiled and
measured for flash, peak RAM, latency, and energy on the actual target.
Repository verification passed the locked dependency check, Ruff formatting and lint, Pyright with
zero errors, the complete suite with 256 tests and 78% branch-aware coverage, and package construction
for both source and wheel distributions. The final test run emitted no warnings.
