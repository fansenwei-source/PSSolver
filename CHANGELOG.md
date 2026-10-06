# Changelog

## 0.2.0rc3 — unreleased

Correctness and compatibility follow-up to `v0.2.0rc2`.

### Fixed

- Make the Channel functional pressure transpose use the same unresolved
  x-Nyquist subspace as the production pressure operator on even grids.
- Reject fixed-cap Channel pressure solves that terminate by operator
  breakdown or amplify the initial relative residual.  Generic pressure
  objectives now retain a consistent VJP in both tolerance and fixed-work
  modes.

### Breaking API migration inherited from rc2

`v0.2.0rc2` intentionally removed ambiguous defaults from two typed public
declarations.  This was a **breaking source-level API change**, not a change
to the selected production runtime:

- `SpectralNumerics(...)` requires `spectral_storage`;
- `Output(...)` requires `save_start_step`, `diagnostics`, and
  `save_hydrodynamics`.

Call sites that intentionally want the former rc1 values must now write them
explicitly:

```python
numerics = SpectralNumerics(
    dtype="float64",
    dealias_rule="cubic_half",
    spectral_storage="full_complex",
)
output = Output(
    directory="data/run",
    steps=20_000,
    save_interval=1_000,
    diagnostic_interval=100,
    save_start_step=0,
    diagnostics=True,
    save_hydrodynamics=True,
)
```

`full_complex` is only the former constructor default; it is not a universal
recommendation.  Periodic half-spectrum applications should continue to
declare `hermitian_half` and their `hermitian_axis`.  Independent consumers,
including PSSolver-Control, must migrate their declarations explicitly rather
than relying on Provider defaults.

## 0.2.0rc2 — 2026-10-06

Second release candidate for the composable v0.2 architecture.  It is a
stabilization release over `v0.2.0rc1`; the public capability matrix,
functional protocol version, production defaults, and explicit v0.2 scope are
unchanged.

### Fixed

- Repair long-horizon Periodic spectral-state drift by enforcing Hermitian
  consistency, including the self-conjugate and unresolved Nyquist modes.
- Preserve Channel incompressibility after dealiasing and align complete-
  stress functional declarations, pressure-solver precedence, and production
  observation synchronization with their qualified runtimes.
- Harden public compilation against unsupported grid placement, malformed
  runtime flags, invalid device/pressure options, unsupported snapshot starts,
  and inconsistent capability rejection paths.
- Validate checkpoint schema, payload hashes, shapes, dtypes, finite values,
  progress, runtime identity, and target immutability before restoration.
- Make periodic and Channel checkpoint identities, failure classifications,
  file permissions, and device-independent lifting identities stable across
  copy, serialization, restart, and installed-wheel execution.
- Correct Plane float32 construction, static lifting projection and metadata,
  restart artifacts, initial wall-compatible remainders, and weak-impedance
  Robin root conditioning.
- Apply the declared TF32 policy consistently and canonicalize disabled
  spectral refresh across every qualified public application.

### Public API stabilization

- **Breaking:** require callers to state `SpectralNumerics.spectral_storage`
  and the `Output.save_start_step`, `Output.diagnostics`, and
  `Output.save_hydrodynamics` policies whose previous implicit defaults could
  disagree across applications.  The rc3 migration section above gives the
  explicit replacement for callers that intended the former rc1 values.
- Adapt the legacy Channel workflow to the geometry-neutral public `Output`
  declaration without changing its default runtime.
- Persist Periodic and Channel diagnostics alongside metadata and results.
- Retain functional protocol `1.0`, batch-one scope, the four qualified
  model--geometry combinations, and the legacy command/runtime compatibility
  entry points.  Typed constructor calls that omitted the newly explicit
  fields are not source compatible and must be migrated.

### Qualification evidence

- Runtime candidate:
  `1c8b3330139dec7cbb2ea138b261fcaadf5e871f`.
- Versioned qualification tip:
  `7abda0966741961220c3b1ba3c4152307485ba06`.
- H100 Job 10857114 completed with `ExitCode=0:0` and classification
  `PASS_V0_2_0RC2_AUDIT_H100_NON_REGRESSION`.
- The cumulative v8 archive passed 197/197 checksums; its manifest SHA-256 is
  `50da02431bbedf9936fd7397dc308cd1cca028fad9bc4bd67fdf32c1405d9561`.
- Periodic and Channel 100-step continuous and 50+50 restart paths were finite
  and byte-identical within each version.  Candidate/baseline peak reserved
  memory ratios were 1.0000 for Periodic and approximately 1.04675 for
  Channel, within the frozen 1.05 limit.

### Scope and release status

- This release does not promote `compiled_v2` or `compiled_channel_v2` and
  does not change any production default.
- The Channel qualification smoke showed an approximately 4.77 percent higher
  mean step time for the candidate in that short run; it was not a failure of
  the preregistered G5 contract and is retained as a release observation.
- The broader R1--R9 architecture and redundancy cleanup remains separate
  from this bug-fix candidate.
- `nematics3d` and PSSolver-Control remain independent and unmodified by the
  Provider release.  Consequently PSSolver-Control call sites using the old
  constructor shorthand require a separate consumer-owned migration.
- The final rc2 source distribution, wheel, isolated install, and H100 release
  smoke must pass before the annotated `v0.2.0rc2` tag is created.

## 0.2.0rc1 — 2026-10-01

First release candidate for the composable v0.2 architecture.  The v0.1 tags,
release branches, and historical qualification records remain unchanged.

### Added

- A package-root `Simulation` declaration with typed model, geometry,
  boundary, numerical, time, initial-condition, execution, and output
  ownership.
- Public `compile_simulation`, `run_simulation`, result protocols, and an
  immutable capability catalog that rejects unsupported combinations before
  tensor allocation.
- Qualified complete-stress Beris--Edwards applications for periodic boxes,
  Plane slabs, and rectangular Channels, plus the retained legacy active-force
  Channel application.
- Generic static prescribed-Dirichlet declarations, Plane Q lifting, and
  strong homeotropic/planar Q convenience policies.  The finite-Q Robin work
  remains an internal qualified pilot rather than a public runtime.
- Versioned `pssolver.functional.api` protocol 1.0 for independent batch-one
  Periodic and Channel consumers, including replay, observations, diagnostics,
  checkpoint bridges, and differentiable execution.

### Changed

- Runtime construction is separated into tensor-free declarations, lowering,
  plans, state, workspace, integrators, and application adapters.
- Model declarations, geometry declarations, and field-level boundary
  policies are orthogonal at the public composition boundary.
- Plane and Channel compiled runtimes remain opt-in; the production defaults
  remain `legacy_production` and `legacy_channel`.

### Fixed

- Periodic half-spectrum states are projected onto the self-conjugate-plane
  Hermitian subspace at every step.  The repair prevents hidden anti-Hermitian
  modes from growing during long-horizon functional execution while retaining
  the physical-Q and short-trajectory numerical contracts.

### Qualification evidence

- Architecture baseline and Phase 9 finalization commit:
  `e43979aa8797ed8c49876612c6ce50b1a1fc4510`.
- Qualified Provider source:
  `0838ecd1cd314a5e8879ce6d1a1ced921d0cf69f`.
- Qualified independent Consumer source:
  `28857a59610df355469dca55372e0d0cd1311f3c`.
- P9.8.5 cumulative closure:
  `PASS_P9_8_5_STABLE_FUNCTIONAL_API_CUMULATIVE_CLOSURE_WITH_CANONICAL_ZERO_START_CHANNEL_ORACLE`.
- Phase 9 final closure:
  `PASS_PHASE_9_STABLE_FUNCTIONAL_API_FINAL_CLOSURE`.

### Scope

- The qualified public capability matrix contains two model declarations,
  three geometries, eight boundary policies, four executable model--geometry
  combinations, and six runtime paths.
- Functional qualification is limited to batch-one Periodic and rectangular-
  Channel consumers.  Plane control, larger batches, production-scale
  optimizers, optimizer campaigns, and scientific control results remain out
  of scope.
- Nonhomogeneous Neumann lifting, dynamic/trainable boundary data, and general
  public Robin execution remain future work.
- PSSolver-Control and `nematics3d` are independent projects and are not
  bundled or modified by this release.

### Release status

- This is a release candidate, not the final 0.2.0 release.
- The source distribution, wheel, isolated installation, and final release
  smoke must pass before the annotated `v0.2.0rc1` tag is created.
- No production default is promoted by the version change.

## 0.1.2 — 2026-09-18

Bounded-axis execution and transform-dataflow performance update. The existing
v0.1.0 and v0.1.1 tags remain unchanged.

### Changed

- DCT/DST axis execution is separated from tensor-product planning through a
  device-bound bounded-axis execution plan. The qualified dense matrix method
  remains the sole executor and preserves the transform algorithm.
- Repeated periodic-gradient multipliers are cached by basis, axis, device and
  dtype.
- Owned bounded-gradient and spectral-projection buffers use direct writes or
  in-place updates outside autograd while retaining differentiable fallback
  paths.
- Contiguous stress batches use natural tensor views instead of repeated
  advanced-index materialization.
- Dynamic field synchronization uses the established grouped field-access
  contract rather than direct advanced indexing.

### Qualification evidence

- Candidate source commit before RC metadata:
  `f121428d7a5e66e4b65c6e6f3777fdf1c8f8b23a`.
- Bounded-axis abstraction Job 10835044: `PASS`; R128 and R320 timings and
  memory were neutral, transform calls were unchanged, and the 100-step Q,
  velocity and pressure arrays were byte-for-byte identical. Its 111-entry
  manifest has SHA-256
  `6ff57b61e27985f842340fa6f3edc7cc5e9cf3944f39a49caad777a994a07efe`.
- Bounded-transform dataflow Job 10835210: `A_recommended`; all CPU, CUDA,
  numerical, memory and performance gates passed. Mean timestep speedups were
  approximately 1.238x at R128 and 1.084x at R320, with byte-identical
  100-step Q, velocity and pressure arrays. Its 103-entry manifest has
  SHA-256
  `65a72a34a42a526e8ada36bd436ff1cbbaed0d2213fb88804eed1db6248b7876`.

### Scope

- The v0.1 scientific and architectural support boundary is unchanged.
- Plane `legacy_production` remains the supported production runtime.
- The transform basis, normalization, mode order, boundary semantics,
  projection, dealiasing, equations and production selectors are unchanged.
- Dense bounded-axis execution remains the production implementation; this
  release does not claim a pruned DCT/DST algorithm.
- Arbitrary geometries, inertial Shendruk dynamics, optimal control and the
  separated/canary runtime remain outside the supported release boundary.

### Release gates

- The complete CPU suite passed with 1,134 tests and eight subtests.
- The source distribution and wheel passed content and metadata validation.
- A clean isolated wheel installation passed package-version and import-source
  checks, both CLI entry points, a bounded dry run and an installed-package
  one-step CPU smoke with finite Q, velocity and pressure output.
- No additional HPCC job was required beyond the two archived H100
  qualification jobs above.

## 0.1.1 — 2026-09-17

First bounded v0.1 performance update. The existing v0.1.0 tag remains
unchanged.

### Changed

- Contiguous transform groups now use the qualified zero-copy slice/view path
  by default.
- Compatible transforms with at least two periodic axes now use one
  multidimensional `torch.fft.fftn`/`torch.fft.ifftn` call by default.
- The historical `advanced` transform-group indexing and `axiswise` periodic
  transform modes remain explicit rollback selectors.

### Qualification evidence

- Candidate source commit before RC metadata:
  `525ba2326de3604e75752364d561af6417063ddc`.
- H100 A/B/C/D job 10833968: `A_recommended`; numerical, performance and
  memory gates passed. The archived 51-entry manifest has SHA-256
  `15ffa359a15eb7a8fd907053b7981dbf494b0f6f31aea95b4a7dfe7f116c5a3a`.
- H100 implicit-default job 10834013: `PASS`; the implicit and explicit
  qualified paths were byte-for-byte identical. Float64 512x512 and float32
  1024x1024 rollback/default speedups were approximately 1.565x and 1.621x.
  The archived 43-entry manifest has SHA-256
  `f9a6f14b217a3f49d8abf065756f0082b124dd74b143f1a8499f1ded2fb76c7e`.
- Plane/Channel geometry continuation job 10834996: `B_neutral`; two R320
  Plane pairs and the Channel state were byte-for-byte identical. Plane
  elapsed time remained within the non-regression envelope. The archived
  84-entry manifest has SHA-256
  `8cf0f0d35f8035290973ac405c7946385824072c6ddff821a4c2e59b2b89ea37`.

### Scope

- The v0.1 scientific and architectural support boundary is unchanged.
- Plane `legacy_production` remains the supported production runtime.
- Channel qualification is regression evidence, not production promotion.
- Arbitrary geometries, inertial Shendruk dynamics, optimal control and the
  separated/canary runtime remain outside the supported release boundary.

### Release gates

- The complete CPU suite passed with 1044 tests and 8 subtests.
- The source distribution and wheel passed content and metadata validation.
- A clean isolated wheel installation passed both CLI entry points, default
  policy checks, a bounded dry run and an installed-package CPU smoke test.

## 0.1.0 — 2026-09-16

Initial bounded PSSolver release.

### Supported

- Plane/slab Beris--Edwards active nematics with complete one-constant nematic
  stress and quasistatic incompressible Stokes--Brinkman flow.
- Mixed Fourier/DCT/DST free-slip/Neumann geometry.
- Immutable PlaneBerisEdwardsRunSpec configuration authority.
- Callable pssolver.applications.run_plane_beris_edwards application API.
- Installed pssolver-plane-beris-edwards command and historical compatibility
  script.
- Float64 production path, provenance metadata, observations, diagnostics,
  checkpoint/restart and atomic completion markers.
- Accepted Plane transform, storage, pointwise and stress-divergence
  optimizations with explicit metadata and rollback paths.

### Qualification evidence

- Source commit: 38cd9335b13328922956a46bb0e9dcbc992a5de0.
- Full HPCC CPU gate: 1017 passed, one allowed optional-dependency skip, eight
  subtests passed.
- H100 job 10832784: COMPLETED, exit code 0:0.
- Three R320 parent/candidate pairs: Q/u/p byte-for-byte identical.
- Same-backend checkpoint/restart: final Q/u/p byte-for-byte identical.
- Aggregate candidate/parent elapsed-time ratio: 1.000991904034.
- Archived HPCC checksum manifest: 58/58 entries passed; manifest SHA-256
  b04d570fbe6968c372703f7ec9a1fa4f043147ca6a80698ea29c21a6cbf9a7a5.
- Clean wheel installation: package version, compatibility module, console
  entry point, help output, dry-run and implementation provenance passed.

### Explicit exclusions

- Arbitrary PDE and arbitrary geometry production support.
- Production Channel parity.
- Inertial or paper-identical Shendruk reproduction.
- Paper-identical initialization.
- Optimal-control production integration.
- Production promotion of experimental separated/canary runtimes.
