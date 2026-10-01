# PSSolver

PSSolver 0.2.0rc1 is a research-grade, GPU-first tensor-product spectral
solver release candidate.  It introduces a typed public composition layer for
models, geometries, boundary policies, numerical choices, execution policy,
and output scheduling while retaining the qualified legacy application entry
points.

The release candidate does not claim arbitrary PDE or geometry support.
Executable combinations are discovered from the public capability catalog and
unsupported combinations fail before tensor allocation.  See
`notes/pssolver_v0_2_scope.md` for the precise public, numerical, and
qualification boundary.

## Installation

Install into an environment that provides the desired CPU or CUDA PyTorch
build:

    python -m pip install .

The supported command-line entry point for the historical Plane workflow is:

    pssolver-plane-beris-edwards --help

The historical command remains a compatibility entry point:

    python -m Plane_beris_edwards_stokes --help

## Public composition API

The package-root API exposes `Simulation`, `compile_simulation`,
`run_simulation`, typed declaration helpers, and immutable capability
discovery:

```python
from pssolver import capability_catalog

catalog = capability_catalog()
for combination in catalog.qualified_combinations:
    print(
        combination.equation_variant,
        combination.geometry_name,
        combination.runtime_paths,
    )
```

The qualified executable combinations in 0.2.0rc1 are:

| Model | Geometry | Runtime paths |
| --- | --- | --- |
| complete-stress Beris--Edwards | Plane slab | `legacy_production`, `compiled_v2` |
| legacy active-force active nematics | rectangular Channel | `legacy_channel`, `compiled_channel_v2` |
| complete-stress Beris--Edwards | periodic box | `periodic_spectral` |
| complete-stress Beris--Edwards | rectangular Channel | `channel_complete_stress` |

Model declarations, geometry declarations, and field-level boundary policies
are orthogonal.  Runtime dispatch is resolved during compilation, not in the
timestep.  Existing Plane and Channel production defaults remain unchanged;
compiled paths remain explicit opt-ins.

## Stable functional API

`pssolver.functional.api` is the versioned functional interface for independent
consumers.  Protocol version 1.0 qualifies batch-one Periodic and rectangular-
Channel activity runtimes, deterministic replay, checkpoint bridges,
observations, diagnostics, and differentiable execution.  PSSolver-Control is
an independent consumer and is not bundled into this distribution.

## Historical Plane application

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

`legacy_production` remains the Plane default and rollback oracle.

## Validation and limits

The Phase 0--9 architecture program is complete.  Its cumulative CPU, H100,
installed-wheel, restart, checkpoint, gradient, Taylor, memory, negative-gate,
and long-horizon Hermitian evidence is summarized in `CHANGELOG.md` and the
versioned closure records under `notes/architecture_v0_2/`.

The following remain outside this release candidate: arbitrary model--geometry
execution, nonhomogeneous Neumann lifting, dynamic or trainable boundary data,
public finite-Q Robin execution, functional batches larger than one, Plane
optimal control, production-scale optimization campaigns, and scientific
control conclusions.  `nematics3d` remains an optional analysis dependency
and is not modified or bundled.
