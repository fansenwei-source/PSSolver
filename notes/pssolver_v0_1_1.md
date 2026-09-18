# PSSolver 0.1.1 release notes

PSSolver 0.1.1 is the first bounded performance update built on the frozen
0.1.0 release. It does not broaden the scientific or architectural support
boundary documented in `pssolver_v0_1_scope.md`.

## Changes

Two independently qualified periodic-transform optimizations are now generic
defaults:

- contiguous transform groups use a zero-copy slice/view;
- compatible transforms with at least two periodic axes use a single
  multidimensional FFT call.

The previous `advanced` field indexing and `axiswise` periodic-transform
execution remain available as explicit rollback modes. Non-contiguous groups
and transforms with fewer than two periodic axes retain guarded fallbacks.

## Numerical qualification

- The implicit defaults and explicit optimized selectors were byte-for-byte
  identical in the H100 default-path smoke.
- The historical rollback remained within `1e-12` for float64 and `5e-6` for
  float32.
- Two 100-step R320 Plane parent/candidate pairs produced byte-for-byte
  identical Q, velocity and pressure arrays.
- The Channel regression smoke produced byte-for-byte identical spatial and
  spectral states. Its single periodic axis correctly retained the axiswise
  transform, while the contiguous Q group used the view path.

## Performance qualification

- The generic H100 rollback/default speedup was approximately 1.565x at
  float64 512x512 and 1.621x at float32 1024x1024.
- Peak allocated GPU memory fell to about 77.45% of the rollback path in both
  default-path cases; peak reserved memory was unchanged.
- The R320 Plane geometry result was performance-neutral: the candidate/parent
  elapsed-time ratio was approximately 1.0124 and passed the 5% non-regression
  gate.

The archived H100 evidence is tied to jobs 10833968, 10834013 and 10834996;
their manifest hashes are recorded in `CHANGELOG.md`.

## Preserved boundaries

- Plane `legacy_production` remains the supported runtime and rollback oracle.
- Channel remains outside the supported production contract.
- The v0.1.0 tag and release branch remain immutable.
- This release does not authorize Stage T, arbitrary PDE/geometry claims,
  inertial Shendruk dynamics, paper-identical initialization, optimal-control
  production integration or separated-runtime promotion.

## Release gates

The 0.1.1 source and wheel passed the complete CPU suite, metadata and archive
checks, clean isolated installation, both CLI entry points, a bounded dry run
and an installed-package CPU smoke test. The annotated `v0.1.1` tag freezes the
resulting source; subsequent solver or feature work must proceed on a new
branch and version.
