# Plane Hermitian half-spectrum default promotion

## Scope

This change promotes `hermitian_half` storage only for the qualified Plane
Beris--Edwards production workflow and its profiling tools.  The generic
`SpectralSolver` default remains `full_complex`, so Channel and other
geometries do not inherit a Plane-specific Hermitian axis assumption.

The explicit Plane rollback is:

```text
--spectral-storage full_complex
```

## H100 qualification evidence

The H100 2x2 experiment compared full versus truncated projected transforms
and full-complex versus Hermitian-half storage at `320 x 320 x 80`.  Corrected
scientific reanalysis Job 10823992 used relative-L2 tolerances for comparisons
between mathematically equivalent transform algorithms.  It classified the
candidate `A_recommended` and eligible for default promotion.

For the production combination (truncated plus Hermitian half-spectrum):

- mean timestep: 49.364779 ms;
- speedup over truncated plus full-complex: 1.732440x;
- speedup over the original full/full-complex configuration: 2.305858x;
- peak allocated memory: 5.720723 GiB;
- peak reserved memory: 8.533203 GiB;
- transform calls per step: 7 forward and 32 inverse.

All cross-algorithm Q, velocity, and pressure comparisons passed
`relative L2 <= 1e-12`; the largest reported relative L2 was approximately
`4.44e-15`.  The implicit historical control and its explicit equivalent were
byte-identical.

## Local promotion checks

The complete CPU suite passed with 616 tests and 8 subtests.  A local RTX 3060
Ti, float64, 20-step Plane smoke test used eager pointwise kernels to isolate
the storage-default change:

- implicit Plane storage and explicit `hermitian_half` produced byte-identical
  Q, velocity, and pressure snapshots;
- implicit half-spectrum versus explicit `full_complex` had relative-L2
  differences of `2.61e-16`, `5.02e-16`, and `6.88e-16`, respectively;
- all three runs wrote valid `COMPLETE` markers.

The local environment could not compile TorchInductor kernels because CUDA
development headers were absent.  This is an environment limitation, not a
storage-path failure; the fully implicit production-default smoke therefore
remains an H100 validation step.
