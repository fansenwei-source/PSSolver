# Phase 5 P5.3: disconnected compiled Euler step program

Status: `PASS_P5_3_COMPILED_EULER_STEP_PROGRAM`.

Baseline: P5.2 commit
`f2bf7a3f1a5fa9f1886cdbac71e8ea616b728b28`.

P5.3 implements the private Plane projected semi-implicit Euler program over
the construction-time references qualified by P5.2.  It is executable in
direct tests, but remains disconnected from the runtime selector, application
entry point, CLI, workflow, and production default.

## Fixed execution sequence

The program executes the qualified production order:

1. prepare algebraic velocity and pressure fields;
2. invoke the optional pre-update callback;
3. evaluate the explicit Beris--Edwards Q right-hand side;
4. add `dt * rhs` to the evolved spectrum;
5. divide by the pre-bound semi-implicit denominator;
6. project the dynamic spectra;
7. inverse transform the dynamic fields;
8. perform a scheduled spectral refresh when due;
9. commit progress last.

All tensor and operator references are resolved before the first step.  The
hot path contains no runtime selector, registry, capability, configuration,
JSON, metadata, or string-based field discovery.

## Workspace and tensor lifetime

The compiled step program owns a zero-slot, zero-byte `RuntimeWorkspace`.
This object supplies generation tokens, re-entrancy rejection, and an explicit
failure boundary without introducing tensor storage.  The P5.2 state,
denominator, models, projector, backend, kernel scratch, and caches are reused
by identity.  Construction adds no tensor storage identity.

## Callback, refresh, and restored-static semantics

The callback remains after algebraic preparation and before explicit Q RHS
evaluation.  A restored static field state is consumed once and invalidated;
the following step recomputes it.  Refresh remains after inverse transform and
before progress commit.  Refresh counters and completed-step counters match
the legacy runtime at one, two, and seven steps, including a refresh boundary.

## Failure semantics

P5.3 intentionally preserves the existing production failure boundary:

- the workspace generation is aborted;
- the completed-step and refresh clocks are not committed;
- already performed tensor updates are not rolled back;
- algebraic preparation and callback side effects are not rolled back.

This is progress-commit atomicity, not transactional tensor rollback.  A
synthetic failure after spectral Euler arithmetic produces the same tensor,
representation-ledger, clock, and workspace state as the legacy step program.

## Numerical oracle

Independent legacy and compiled constructions starting from the same float64
initial tensors are byte-for-byte identical in packed physical and spectral
storage after one, two, and seven continuous CPU steps.  The seven-step case
crosses two scheduled refresh boundaries.  Linear operators, progress,
representation ledgers, and restored-static flags also agree.

## Deliberately unchanged

P5.3 does not change:

- `PlaneRuntimePath`, CLI choices, workflows, or the default runtime;
- either existing Plane runtime or its timestep;
- equations, coefficients, boundaries, transforms, kernels, or operation
  order;
- checkpoint, diagnostics, metadata, or output behavior;
- Phase 6 authorization or default-promotion status.

## Local qualification

The final local qualification completed with:

- 215 targeted tests passed with no failure, skip, or deselection;
- 1883 full-suite tests and 8 subtests passed with no failure, skip, or
  deselection;
- `git diff --check` passed.

## Next boundary

P5.4 is locally eligible but not implemented.  It may add a private adapter
for observation synchronization, projected normal-force and pressure
diagnostics, output views, and Plane checkpoint-format-v1 identity.  It must
reject incompatible or cross-runtime restore before a timestep or output and
remain disconnected from the application edge.

The machine-readable authority is
`phase_5_p53_compiled_euler_step.json`.
