# PSSolver 0.1.2 release notes

PSSolver 0.1.2 is a bounded transform-execution and dataflow performance
release based on the frozen 0.1.1 release. It does not broaden the scientific
or architectural support boundary documented in `pssolver_v0_1_scope.md`.

## Changes

The release adds a device-bound bounded-axis execution plan below the
tensor-product transform facade. The dense matrix DCT/DST operation remains
the only executor, so this refactor changes ownership and extensibility rather
than transform mathematics.

The production dataflow update removes several repeated materializations
without changing public numerical selectors:

- periodic-gradient multipliers are cached by their complete execution
  identity;
- owned bounded-gradient outputs are written directly outside autograd;
- owned spectral projection buffers are updated in place;
- contiguous stress groups use natural tensor views;
- dynamic field refresh and synchronization use grouped field accessors.

Autograd-sensitive paths retain allocating implementations. Advanced-index
field groups retain explicit stores, and the existing transform algorithms and
rollback selectors remain available.

## Numerical qualification

- The bounded-axis execution abstraction passed CPU and H100 tests without
  changing transform-call counts or peak memory.
- Its 100-step R128 Plane comparison produced byte-for-byte identical Q,
  velocity and pressure arrays.
- The complete dataflow candidate passed 1,131 CPU tests, one allowed optional
  dependency skip, two explicitly relocated CUDA-only tests and eight
  subtests.
- The two CUDA-only tests passed on the assigned H100 before profiling.
- The final 100-step parent/candidate Q, velocity and pressure arrays were
  byte-for-byte identical.

## Performance qualification

Job 10835210 measured the frozen parent and candidate in one H100 allocation
with three balanced trials per grid:

- R128 improved from 7.120957 to 5.750187 ms/step on the mean, a 1.238387x
  speedup;
- R320 improved from 48.960003 to 45.162243 ms/step on the mean, a 1.084091x
  speedup;
- the candidate was faster in all three paired trials at both sizes;
- forward and inverse transforms remained at 7 and 32 calls per step;
- peak reserved memory was unchanged, while peak allocated memory increased by
  less than 0.04%;
- no graph break, compile fallback, transform fallback, OOM, NaN, Inf or CUDA
  error occurred.

The archived 103-entry result manifest passed SHA-256 verification; its own
SHA-256 is
`65a72a34a42a526e8ada36bd436ff1cbbaed0d2213fb88804eed1db6248b7876`.

## Rejected follow-up

A separate static-nematic algebraic-fusion experiment was not promoted. Its
first packed-output implementation regressed locally, while a smaller exact
ablation improved R128 by only about 0.34%. No code from that experiment is in
this release, and no additional H100 job was consumed for it.

## Preserved boundaries

- Plane `legacy_production` remains the supported runtime and rollback oracle.
- The v0.1.0 and v0.1.1 tags and release branches remain immutable.
- The transform basis, normalization, mode order, boundary conditions,
  projection, dealiasing and equations are unchanged.
- Dense bounded-axis execution remains the only implementation; this release
  does not claim a fast or pruned DCT/DST.
- This release does not authorize arbitrary PDE/geometry claims, Channel
  production parity, inertial Shendruk dynamics, paper-identical
  initialization, optimal-control production integration or separated-runtime
  promotion.

## Release gates

The complete release source passed 1,134 CPU tests and eight subtests. The
source distribution and wheel passed content and metadata validation. A clean
isolated wheel installation passed version and import-source checks, both
command-line entry points, a bounded dry run and a one-step CPU smoke with
finite Q, velocity and pressure arrays and a valid completion marker.

The annotated `v0.1.2` tag freezes the result. Subsequent architecture or
feature work must proceed on a new branch and version.
