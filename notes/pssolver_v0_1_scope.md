# PSSolver v0.1 support scope

PSSolver v0.1 is a deliberately bounded, research-grade spectral solver
release.  Its supported production application is the three-dimensional Plane
(periodic-periodic-slab) Beris--Edwards active-nematic model coupled to a
quasistatic incompressible Stokes--Brinkman solve.

## Supported production path

- `PlaneBerisEdwardsRunSpec` is the immutable configuration authority.
- `pssolver.applications.run_plane_beris_edwards` is the supported callable
  application API.
- `pssolver-plane-beris-edwards` is the installed command-line entry point.
- `Plane_beris_edwards_stokes.py` is a compatibility CLI with no import-time
  parsing or simulation side effects.
- The mixed Fourier/DCT/DST free-slip/Neumann Plane geometry, the validated
  float64 path, restart checkpoints, diagnostics, provenance metadata and
  atomic completion marker are supported.
- The accepted Plane optimizations remain enabled through their existing
  explicit metadata and rollback controls.
- `legacy_production` remains the v0.1 production runtime.  Experimental
  separated runtimes remain opt-in and are not production-promoted.

## Scientific meaning

The production application solves the complete one-constant Beris--Edwards
nematic stress coupled to a zero-Reynolds-number Stokes--Brinkman equation.
The free-slip velocity conditions are kinematic.  The default tangential zero
mode is a zero-mean modeling convention, not a pressure gauge or a zero-total-
traction condition.

## Explicitly outside v0.1

- arbitrary PDE composition or arbitrary geometries;
- production Channel parity with the Plane application;
- inertial momentum dynamics and a strict Shendruk LB/finite-difference
  reproduction;
- paper-identical initial conditions;
- optimal-control integration;
- production support for experimental separated/canary schedulers.

These exclusions are capability boundaries, not claims that the corresponding
research directions are invalid.  They prevent the tested Plane application
from being presented as a more general solver than it currently is.

## Release qualification

The Stage S source candidate completed all local and HPCC gates:

- complete CPU suites passed;
- three paired R320 H100 trajectories preserved byte-identical Q/u/p;
- same-backend restart preserved a byte-identical final state;
- the H100 performance non-regression gate passed;
- a clean 0.1.0 wheel install passed package, compatibility-module, console,
  help, dry-run and implementation-provenance checks.

Architecture Stage T remains intentionally paused and requires a separate
decision. The 0.1.0 release does not imply support for any capability listed
under the exclusions above.
