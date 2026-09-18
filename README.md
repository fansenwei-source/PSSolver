# PSSolver

PSSolver 0.1.2 is a bounded research-grade tensor-product spectral solver.
Its scientific support boundary is unchanged from 0.1.0.
Its supported production application is a three-dimensional Plane
(periodic-periodic-slab) Beris--Edwards active-nematic model coupled to a
quasistatic incompressible Stokes--Brinkman solve.

The release deliberately does not claim arbitrary PDE or geometry support.
Channel production parity, inertial Shendruk dynamics and optimal control are
outside the v0.1 support contract. See notes/pssolver_v0_1_scope.md for the
precise scientific and architectural boundary.

## Installation

Install into an environment that provides the desired CPU or CUDA PyTorch
build:

    python -m pip install .

The supported command-line entry point is:

    pssolver-plane-beris-edwards --help

The historical command remains a compatibility entry point:

    python -m Plane_beris_edwards_stokes --help

## Programmatic application

    from pathlib import Path

    from pssolver.applications import run_plane_beris_edwards
    from pssolver.configuration import create_plane_beris_edwards_run_spec

    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=Path("run"),
        device="cpu",
        dtype="float64",
        nx=32,
        ny=32,
        nz=16,
        steps=10,
    )
    result = run_plane_beris_edwards(run_spec)
    print(result.final_step)

legacy_production is the supported v0.1 runtime and rollback oracle.
Experimental separated runtimes remain explicit opt-ins.

## Validation status

The frozen Stage S candidate passed the complete CPU suite, three paired R320
H100 trajectories with byte-identical Q/u/p, same-backend restart with
byte-identical final state, and the performance non-regression gate. Version
0.1.1 additionally qualifies contiguous transform-group views and
multidimensional periodic FFT execution as generic defaults while preserving
explicit rollback selectors and byte-identical Plane/Channel qualification
trajectories. Release evidence is summarized in CHANGELOG.md.

Version 0.1.2 additionally separates bounded-axis execution from
tensor-product planning and reduces qualified bounded-transform data movement.
The dense DCT/DST algorithm and all scientific discretization choices remain
unchanged.

The append-only performance and architecture history is maintained in
`notes/pssolver_version_evolution_ledger_zh.md`. It distinguishes matched H100
comparisons from cross-campaign reference values so future releases can extend
the record without rewriting historical evidence.
