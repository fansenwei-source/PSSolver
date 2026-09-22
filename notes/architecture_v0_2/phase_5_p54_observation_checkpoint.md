# Phase 5 P5.4: observation, diagnostics, and checkpoint adapter

Status: `PASS_P5_4_OBSERVATION_DIAGNOSTICS_CHECKPOINT_ADAPTER`.

Baseline: P5.3 commit
`a4abfbe199cccf9dc4a953074d7f0ad9d22bacba`.

P5.4 adds a private workflow-facing surface around the disconnected compiled
Plane program.  It does not add `compiled_v2` to `PlaneRuntimePath`, expose an
application adapter, or change the default runtime.

## Observation and output views

The adapter freezes zero-copy views of the packed physical storage:

- Q components `[0:5]` in canonical compact order;
- velocity components `[5:8]` as `(ux, uy, uz)`;
- pressure component `[8]`.

Observation capture produces the existing `PlaneObservation` NumPy layout.
Independent compiled and legacy runs produce identical Q, velocity, and
pressure arrays.  Writing those observations through the existing writer
produces the same filenames and byte-identical NPY files.

Observation is rejected unless physical Q is current.  Explicit observation
synchronization recomputes algebraic velocity and pressure from the current Q
state before diagnostics or final output.

## Diagnostics

The adapter exposes the existing projected wall-normal force cache and
pressure-solver iteration/residual diagnostics.  Divergence, relative
divergence, Schur, and wall-normal momentum diagnostics use the existing
`PlaneDiagnostic` schema and agree exactly with the legacy adapter on the CPU
oracle trajectory.

## Checkpoint format v1

P5.4 reuses `PlaneWorkflowCheckpoint` and the existing format-v1 writer and
loader without adding, removing, or renaming top-level fields or tensor files.
Because `PlaneRuntimePath.COMPILED_V2` is intentionally deferred to P5.5, the
disconnected checkpoint temporarily uses `legacy_production` as its format-v1
runtime-path carrier.  It cannot be mistaken for a legacy checkpoint because
`backend_restart` carries and enforces:

- `kind = compiled_v2_plane_stateless`;
- the compiled runtime identity;
- the resolved run-spec SHA-256;
- the transitional path-carrier identity.

P5.5 must replace only the carrier value with the explicit compiled selector;
it must not change checkpoint format version or tensor layout.

## Restore preflight and negative gates

Restore validates the entire checkpoint before copying any tensor:

- format and path carrier;
- run-spec runtime identity SHA-256;
- compiled backend identity;
- every Q spatial/spectral shape, dtype, and finite status;
- spectral-refresh counter consistency inherited from checkpoint v1;
- inactive timestep workspace.

All source tensors are fully validated before mutation.  After that preflight,
they are transferred and copied one at a time so a large GPU restore does not
retain a second complete device-side checkpoint.  Wrong runtime identity,
backend identity, runtime path, shape, dtype, NaN, or Inf is therefore
rejected before state, clock, timestep, or output changes.  Legacy-to-compiled
and compiled-to-legacy restart are both rejected.

## Restart oracle

A seven-step continuous trajectory and a three-plus-four split trajectory are
byte-for-byte identical in complete packed physical and spectral storage.
Completed-step counters, refresh counters, representation ledgers, and
restored-static-field state also agree.

## Deliberately unchanged

P5.4 does not change:

- `PlaneRuntimePath`, parsers, CLI, application factories, or production
  imports;
- either existing runtime or workflow;
- equations, transforms, kernels, boundaries, operation order, output schema,
  or checkpoint format version;
- Phase 6 authorization or production-default status.

## Local qualification

The final local qualification completed with:

- 222 targeted tests passed with no failure, skip, or deselection;
- 1899 full-suite tests and 8 subtests passed with no failure, skip, or
  deselection;
- `git diff --check` passed.

## Next boundary

P5.5 is locally eligible but not implemented.  It may add the explicit
`PlaneRuntimePath.COMPILED_V2` selector, private runtime adapter, parser choice,
and opt-in application connection.  Omitted selection must remain
`legacy_production`; dry-run must allocate no runtime tensors; fallback is
forbidden.

The machine-readable authority is
`phase_5_p54_observation_checkpoint.json`.
