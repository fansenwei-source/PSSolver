# PSSolver 0.1.1rc1 release-candidate notes

PSSolver 0.1.1rc1 is a bounded performance release candidate based on the
frozen 0.1.0 release. It does not broaden the scientific or architectural
support boundary documented in `pssolver_v0_1_scope.md`.

## Candidate change

The candidate promotes two generic periodic-transform optimizations that were
introduced and qualified independently:

- contiguous transform groups use a zero-copy slice/view;
- compatible transforms with at least two periodic axes use a single
  multidimensional FFT call.

The previous `advanced` field indexing and `axiswise` periodic transform
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

## Preserved boundaries

- Plane `legacy_production` remains the supported runtime and rollback oracle.
- Channel remains outside the supported production contract.
- The v0.1.0 tag and release branch remain immutable.
- This candidate does not authorize Stage T, arbitrary PDE/geometry claims,
  inertial Shendruk dynamics, paper-identical initialization, optimal-control
  production integration or separated-runtime promotion.

## Release process

The `0.1.1rc1` source and wheel must pass the complete CPU suite, metadata
checks, clean wheel installation, console-entry-point help and a bounded
installed-package dry run. A final `v0.1.1` tag requires a separate explicit
decision after those release-candidate gates pass.
