# PSSolver 0.2.0rc2 release candidate

PSSolver 0.2.0rc2 is the stabilization candidate following `v0.2.0rc1`.  It
contains the bounded audit fixes recorded in
`notes/PSSolver_v0_2_0rc2_bug_audit_fix_record_zh.md`; it does not expand the
v0.2 public capability matrix or promote a compiled runtime to a production
default.

## Qualified identities

- rc1 baseline: `1c5237194c5e4f696bb5d3d01d2caa2c4bdd27a1`;
- rc2 runtime candidate: `1c8b3330139dec7cbb2ea138b261fcaadf5e871f`;
- versioned qualification tip:
  `7abda0966741961220c3b1ba3c4152307485ba06`.

The commits after the runtime candidate contain only qualification tools,
tests, and plans.  They do not modify the installed `pssolver` package.

## Stabilization content

The candidate repairs Periodic Hermitian/Nyquist state handling, Channel
dealiasing and pressure contracts, checkpoint integrity and failure
classification, observation synchronization, TF32 enforcement, public option
validation, Plane lifting/restart behavior, and metadata/provenance accuracy.
Public numerical and output declarations are stricter where rc1 defaults could
otherwise conceal an application-specific choice.

The public contract remains functional API protocol 1.0, two model
declarations, three geometries, eight boundary policies, four qualified
model--geometry combinations, six runtime paths, and functional batch size
one.  PSSolver-Control and `nematics3d` remain independent projects.

## H100 audit qualification

The final cumulative qualification completed on NVIDIA H100 PCIe Job 10857114
with `ExitCode=0:0` and classification
`PASS_V0_2_0RC2_AUDIT_H100_NON_REGRESSION`.

The archived v8 result contains 197 verified files.  Its checksum-manifest
SHA-256 is
`50da02431bbedf9936fd7397dc308cd1cca028fad9bc4bd67fdf32c1405d9561`.
The Periodic profile matrix and numerical regression gates passed.  The final
Periodic and Channel continuous/restart smokes were finite and byte-identical
within each version, with no fallback and with candidate/baseline peak-memory
ratios below the frozen 1.05 limit.

The short Channel G5 smoke measured an approximately 4.77 percent larger mean
step time for the candidate.  This was outside the preregistered G5 rejection
criteria and is recorded as a release observation rather than hidden or
reclassified as a failure.

## Release boundary

The R1--R9 redundancy and architecture cleanup remains post-rc2 work.  This
candidate does not authorize new geometry/model families, larger functional
batches, optimizer campaigns, scientific control conclusions, public general
Robin execution, or nonhomogeneous Neumann lifting.

Local source, archive, wheel, and isolated-install verification is required
before the release branch is frozen.  A final installed-wheel H100 release
smoke is required before creating the annotated `v0.2.0rc2` tag or GitHub
prerelease.

The final local release source suite passed 2,733 tests and eight subtests
with no failure or skip; six CUDA-only tests were deliberately deselected for
execution in the final H100 installed-wheel job.

The installed-wheel CPU smoke and `pip check` also passed in a clean release
environment.  The workstation's unrelated optional `nematics3d` installation
declares a conflicting NumPy pin, so it was deliberately excluded from that
environment; neither `nematics3d` nor its environment was modified.
