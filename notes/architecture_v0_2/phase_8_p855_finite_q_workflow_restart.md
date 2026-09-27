# Phase 8 P8.5.5: finite-Q workflow, restart, and CPU closure

Status: `PASS_P8_5_5_FINITE_Q_WORKFLOW_RESTART_CPU_CLOSURE`.

Baseline: `46ddffa`, the recorded P8.5.4 closure. Implementation commit:
`8dadb2e3c73f8a0d9af86f6911179be78cb112e5`.

P8.5.5 composes the five P8.5.3 scalar Robin runtimes selected by the
P8.5.4 quadratic finite-Q anchoring specialization. It adds a CPU-only
reference evolution, physical-Q observation, file-backed workflow, exact
restart, aggregate pre-mutation validation, and installed-wheel coverage.
It remains an internal qualification pilot and does not connect finite
anchoring to the public `Simulation` compiler or a production timestep.

## Reference evolution and scope

`PlaneFiniteQAnchoringRuntime` owns five component runtimes in canonical
`Qxx, Qxy, Qxz, Qyy, Qyz` order. Each component evolves its homogeneous
Robin remainder and synchronized bounded modal coefficients. Physical Q is
reconstructed on demand and returned with the component index last.

The only evolution implemented here is backward-Euler wall-normal elastic
relaxation,

```text
(I - dt K_Q d_z^2) Q^(n+1) = Q^n,
```

with the finite-anchoring Robin law already frozen in P8.5.4. It is a
deterministic workflow/restart oracle, not the complete Beris--Edwards
molecular field, advection/co-rotation update, periodic Fourier execution,
or Stokes coupling. Metadata states this limitation explicitly.

All candidate component values are constructed and validated before any
component is replaced. The five progress clocks are required to agree after
construction, every step, checkpoint capture, and restore.

## Checkpoint identity and atomic restore

The aggregate checkpoint binds:

- the finite-Q lowering plan, source simulation, and anchoring identities;
- the quadratic surface-law identity, Q convention, both target tensors,
  wall strengths, `K_Q`, and outward-normal convention;
- all five raw lower/upper Robin coefficient identities;
- bounded-axis operator kinds, eigenbasis plans, materialized operators, and
  construction-cache identities;
- geometry, full shape and lengths, component order, dtype, CPU device,
  `legacy_production` runtime intent, and `torch_spectral` backend intent;
- the reference evolution law and timestep;
- each homogeneous remainder, bounded modal payload, representation ledger,
  and integrator clock.

The generic scalar runtime now exposes a validation-only checkpoint gate.
The aggregate runtime first validates its own identity and all five scalar
snapshots without mutation. Only after every gate passes does it restore the
five components. Tests demonstrate that coefficient, face convention,
target-Q, Q-convention, surface-law, operator, plan, geometry, dtype,
runtime, backend, and tensor-payload tampering leave the target unchanged.

## File workflow

`PlaneFiniteQAnchoringWorkflow` writes physical `Q_<step>.npy` observations
and pickle-free checkpoint directories. Checkpoints store the evolved
homogeneous remainder and bounded modal representation separately for every
component. Each file record includes path, byte size, SHA-256, shape, and
dtype. Aggregate identity, scalar identities, clocks, and representation
metadata live in strict JSON.

Checkpoint directories are staged and atomically renamed. Existing
checkpoint or non-empty workflow output locations are never overwritten.
Loading rejects duplicate/non-finite JSON, missing or extra component
coverage, path traversal, file-size/checksum mismatch, shape/dtype mismatch,
NaN/Inf, payload mismatch, and unsynchronized representations. `COMPLETE` is
written only after the requested final step succeeds.

Continuous four-step execution and two-step plus file-backed two-step
restart are byte-for-byte identical for physical Q, homogeneous remainders,
and bounded modes. The `W=0` homogeneous-Neumann limit remains executable
without erasing target-Q provenance. The P8.5.2 manufactured and limiting
oracles and the P8.4.4 strong-Dirichlet workflow/restart controls remain
unchanged.

## Installed package and architecture boundary

A wheel built from a clean copied source tree contains both new modules and
imports them from an extracted installed-package location outside the source
checkout. The internal `pssolver.runtime` and `pssolver.workflows` package
surfaces expose the pilot types, while top-level `pssolver`, public
`compile_simulation()`, public `run_simulation()`, production lowering, and
runtime selection remain unchanged. Unsupported production Robin Q still
fails closed.

No production default changed. No GPU or H100 was used. P8.5.5 makes no
paper-benchmark, full-Q-timestep, performance, or production-readiness claim.

## Verification

The new test module passed `19 passed, 1 deselected` before this record was
written. P8.4/P8.5 architecture regression passed `82 passed, 1 deselected`.
The full pre-record CPU suite passed
`2392 passed, 1 deselected, 8 subtests passed`, with zero failures, skips,
or xfails. The deselected item was only the not-yet-created record test.

The architecture archive source list includes these P8.5.5 records. The
user-owned untracked archive PDF was not regenerated or modified.

P8.5.6 may now perform the separately authorized H100 numerical,
performance, live-memory, solve-count, no-fallback, conditioning, restart,
and provenance closure. Until that succeeds, finite-Q anchoring remains an
internal CPU qualification pilot.
