# Stage 10 obstacle smoke verification

The bounded end-to-end protocol completed after the static placement band was widened from 0.12 m to
0.4 m. The original band produced infeasible two-sphere samples for some seeds. Whole-field restart
now prevents an early valid sphere from trapping sequential placement. A 5,000-root stress check per
scenario, including randomized pose and initial noise, produced zero failures with the final config.

Verified workflow:

- three training regimens: no-obstacle control, static control, staged curriculum;
- five independent training seeds per regimen;
- four matched evaluation scenarios per trained policy;
- checkpoint and per-seed artifact creation;
- obstacle collision/clearance, formation, success, and capped-time metrics;
- cross-seed summaries and three seed-paired regimen comparisons;
- source/runtime provenance and guarded completed-seed resume.

The smoke override used 256 environment steps, one environment, 48-step episodes, two evaluation
episodes per scenario, and CPU execution. It preserved every regimen, scenario, seed, reward, sensor,
and environment factor. The resulting policy performance is intentionally not reported as scientific
evidence: this run verifies plumbing and artifact integrity only. Full 10-million-step training for
each of 15 policies remains the research gate.
