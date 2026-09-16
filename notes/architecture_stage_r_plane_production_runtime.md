# Architecture Stage R: package-owned Plane production runtime

Stage R resumes the original Plane production-integration roadmap after the
bounded Stage Q performance investigation.  It does not reopen the rejected
Q.6.2 native-segment candidate and does not promote the separated canary.

## Scope and decision

The validated `legacy_production` backend remains the Plane default and the
rollback reference.  Stage R moves ownership of its numerical construction
from `Plane_beris_edwards_stokes.py` into
`pssolver.runtime.plane_legacy`.  The geometry-specific runtime factory can
now build the default backend directly from an immutable
`PlaneRuntimeBuildRequest`; an application no longer injects a solver-building
closure.

The optional legacy-builder injection remains only for characterization and
rollback tests.  The separated architecture remains an explicit opt-in oracle
and is still imported lazily.

Stage R preserves:

- the Beris--Edwards and complete-nematic-stress Stokes equations;
- free-Q and free-slip mixed FFT/DCT/DST boundary spaces;
- zero-mean versus friction plug-mode semantics;
- semi-implicit Euler ordering and spectral-refresh phase;
- float precision, dealiasing, projected transforms, Hermitian storage, and
  pointwise execution policy;
- output, checkpoint, diagnostics, and metadata semantics;
- the default `legacy_production` runtime selection.

It does not modify the generic solver, Channel, initial-condition convention,
production defaults, or benchmark `develop` branch.

## Responsibility boundary

After Stage R the ownership chain is:

```text
PlaneBerisEdwardsRunSpec
        -> PlaneRuntimeBuildRequest
        -> build_plane_beris_edwards_runtime()
        -> pssolver.runtime.plane_legacy
        -> PlaneBerisEdwardsWorkflow
```

The top-level driver still owns application concerns such as environment
selection, initial-condition files, provenance assembly, and progress output.
Those are intentionally reserved for Stage S entry-point consolidation.

## Verification contract

Stage R requires:

1. package-owned construction without importing `pssolver.experimental`;
2. retained explicit builder injection for isolated characterization tests;
3. no numerical-assembly class or builder definition in the top-level driver;
4. unchanged two-step CPU Q/u/p bytes against the pre-refactor driver;
5. the full CPU regression suite;
6. one bounded H100 production-path smoke before Stage S begins.

Passing CPU tests completes the local ownership migration, but Stage R is not
formally closed until the H100 smoke confirms trajectory and performance
non-regression.  No default promotion is part of Stage R.

The local pre/post extraction comparison used a two-step `8 x 8 x 8`,
float64, eager, CPU run with spectral refresh disabled.  All selected arrays
were byte-identical.  Their shared SHA-256 values were:

- `Q_0.npy`: `ebc9a00d4a4f8ed315adff7ed5d623c2e284816b81fafa9cfc2ceadea334a93a`;
- `Q_1.npy`: `93788008958a50538c665c573c2ea8e568c55b82810eeee5fa624be2948a7721`;
- `Q_2.npy`: `04cca19a8127357516d5bb2a2d147f05da4353d8754da9cfd5d33689f9293212`;
- `u_2.npy`: `f630d781e1297d182d1f8bf240c64763f2c32b6bc754fbb39b1dad28519dd887`;
- `p_2.npy`: `18c31b751a7fc822ca098000d9adb21a8a23500e65c927359c33cd0e184eda0f`.

## Stage S boundary

Stage S may move the remaining application orchestration behind a callable
package API and reduce the historical script to a thin compatibility CLI.  It
must also freeze the limited PSSolver v0.1 support statement and entry-point
contract.  Channel and cross-geometry migration remain deferred until after
the planned pause between Stages S and T.
