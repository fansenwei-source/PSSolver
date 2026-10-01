# PSSolver 0.2.0rc1 release candidate

PSSolver 0.2.0rc1 is the first packaged candidate for the Phase 0--9 v0.2
architecture.  It is cut from architecture finalization commit
`e43979aa8797ed8c49876612c6ce50b1a1fc4510` after the independent Provider and
Consumer cumulative closure.

## Qualified source identities

- Provider numerical and functional source:
  `0838ecd1cd314a5e8879ce6d1a1ced921d0cf69f`;
- independent PSSolver-Control source:
  `28857a59610df355469dca55372e0d0cd1311f3c`;
- Provider Phase 9 finalization:
  `e43979aa8797ed8c49876612c6ce50b1a1fc4510`.

The release-preparation commit changes package metadata, documentation, and
release assertions.  It does not change equations, transforms, runtime
selection, checkpoint schemas, numerical tolerances, or production defaults.

## Release surface

The candidate publishes:

- typed `Simulation`, compilation, execution, result, and capability APIs;
- two public active-nematic model declarations and three geometry
  declarations;
- four qualified model--geometry combinations and six runtime paths;
- typed homogeneous and prescribed boundary declarations;
- stable `pssolver.functional.api` protocol 1.0 for batch-one Periodic and
  rectangular-Channel consumers;
- long-horizon Periodic Hermitian-state repair;
- versioned checkpoint, identity, diagnostics, and restart contracts.

The precise supported and excluded behavior is frozen in
`notes/pssolver_v0_2_scope.md`.

## Evidence carried into the candidate

Phase 8 closed the installed-wheel CPU/H100 capability matrix.  Phase 9 closed
the Periodic and Channel functional consumers, checkpoint compatibility,
gradient, finite-difference, Taylor, memory, diagnostics, negative gates, and
long-horizon Periodic Hermitian behavior.  The authoritative records are:

- `notes/architecture_v0_2/phase_8_final_closure.json`;
- `notes/architecture_v0_2/phase_9_p985_h100_closure.json`;
- `notes/architecture_v0_2/phase_9_final_closure.json`.

## Candidate policy

This candidate is intended for installation and bug discovery before the
final 0.2.0 tag.  Compatible bug fixes produce another release candidate.
Large new capabilities, new geometry/model families, larger functional
batches, and optimizer campaigns remain post-release work and do not enter the
0.2.0 stabilization branch.

The annotated `v0.2.0rc1` tag and release artifacts are created only after the
local source, archive, wheel, and isolated-install gates plus the final
installed-wheel release smoke have passed.

## Local release preparation result

The RC source passed 23 focused release/functional/final-closure tests.  The
complete CPU source suite passed 2,646 tests with no failure or skip; the six
optional Nematics3D adapter tests were executed separately in the existing
Nematics3D environment and all passed without modifying that environment.

The source archive and wheel contain the required v0.2 scope and closure
records.  A fresh environment that does not expose Nematics3D installed the
wheel with `--no-deps`, passed `pip check`, imported PSSolver from
`site-packages`, reported package version `0.2.0rc1` and functional protocol
version `1.0`, exposed the four qualified combinations, completed the Plane
CLI dry-run without creating an output directory, and produced finite Q,
velocity, and pressure arrays in a one-step CPU smoke.

The final installed-wheel H100 release smoke remains pending.  Until it passes,
the release tag and GitHub prerelease remain uncreated.
