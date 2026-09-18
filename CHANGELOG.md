# Changelog

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
