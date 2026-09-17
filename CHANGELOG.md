# Changelog

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
