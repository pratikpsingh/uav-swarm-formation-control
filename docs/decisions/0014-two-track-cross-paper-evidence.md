# ADR 0014: Separate controlled results from native-system evidence

## Status

Accepted.

## Context

The four papers and native DMPC code use different simulators, action interfaces, observation access,
training budgets, metrics, replications, and execution architectures. Placing their published numbers
in a single ranking would imply an experimental control that does not exist. At the same time, hiding
the native systems would lose important information about demonstrated capabilities and deployment.

## Decision

Maintain two explicit tracks. Only artifacts sharing a verified common-environment comparison
fingerprint enter the controlled numerical track. Publication and native-repository evidence enters a
separate contextual track with typed reported/measured/not-reported/not-applicable status and source
locators.

Every generated table repeats environment, simulator version, action semantics, observation access,
training budget, seeds, uncertainty, and execution architecture. Source files are checksum-verified.
Smoke inputs cannot produce a research-ready report. Existing output directories are not overwritten.

## Consequences

The thesis can make defensible controlled claims while still explaining how prior systems differ.
Missing paper details remain visible. Adding Paper 01-03 to the controlled track requires new adapters
and matched experiments; copying their native headline numbers is insufficient.

