# PSSolver

PSSolver 0.2.2 is a research-grade, GPU-first tensor-product spectral solver.
It introduces a typed public composition layer for
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

The qualified executable combinations in 0.2.2 are:

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

Only combinations returned by `capability_catalog().qualified_combinations`
are executable.  A model, geometry, or boundary declaration can exist without
forming a qualified executable combination.

### Exact execution-option contracts

`TorchSpectralExecution` receives `runtime_path`, `device`, and the exact
runtime-specific `options` mapping below.  The constructor adds
`device=<requested device>` and `fallback_allowed=False`; callers must not
repeat those two keys inside `options`.

| Runtime path | Exact keys supplied through `options` |
| --- | --- |
| `legacy_production`, `compiled_v2` | `tf32`, `molecular_field_linear_space`, `stress_divergence_sum_space`, `pointwise_execution`, `disable_q_gradient_reuse` |
| `periodic_spectral` | `tf32`, `molecular_field_linear_space`, `stress_divergence_sum_space`, `pointwise_execution`, `disable_q_gradient_reuse` |
| `channel_complete_stress` | `tf32`, `molecular_field_linear_space`, `stress_divergence_sum_space`, `pointwise_execution`, `disable_q_gradient_reuse` |
| `legacy_channel`, `compiled_channel_v2` | `batch_size`, `dtype` |

Complete-stress Channel and legacy Channel declarations additionally require
their exact `discretization["pressure_solver"]` mapping.  Plane, Periodic, and
complete-stress Channel invocations accept only `dry_run` and
`validation_config_sha256`; legacy Channel accepts no invocation options.
`InvocationSpec` is imported from
`pssolver.configuration.simulation`, rather than from the package root.

### Complete composable Plane example

This small CPU example uses one of the four catalog-qualified combinations.
The output directory must not already contain a run.

<!-- rc5-complete-simulation-example:start -->
```python
import os
from pathlib import Path

from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SimulationRunStatus,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
    run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import CompleteStressBerisEdwards


model = CompleteStressBerisEdwards(
    ldg_a=0.0,
    ldg_b=-0.3,
    ldg_c=0.3,
    ldg_l1=1.0 / 81.0,
    gamma=2.94,
    flow_alignment=0.3,
    activity=0.01,
    beta=-1.0,
    viscosity=2.0 / 3.0,
)
geometry = PlaneSlab(
    shape=(8, 8, 6),
    lengths=(40.0, 40.0, 20.0),
)
boundaries = assign_boundaries(
    model=model,
    geometry=geometry,
    policies={
        "Q": neumann_q(),
        "velocity": free_slip_velocity(),
        "pressure": neumann_pressure_compatibility(),
    },
)

simulation = Simulation(
    model=model,
    geometry=geometry,
    boundaries=boundaries,
    numerics=SpectralNumerics(
        dtype="float64",
        dealias_rule="cubic_half",
        spectral_storage="full_complex",
    ),
    time=TimeStepping(dt=0.005, refresh={"mode": "disabled"}),
    initial_condition=GeneratedInitialCondition(
        "extruded_defect_gas",
        parameters={
            "seed": 24,
            "num_defect_pairs": 1,
            "defect_min_separation": 5.0,
            "defect_core_radius": 1.5,
            "background_angle": 0.0,
            "twist_amplitude": 0.01,
            "twist_modes": [1, 2, 3],
            "initial_s": 1.0 / 3.0,
        },
    ),
    execution=TorchSpectralExecution(
        runtime_path="legacy_production",
        device="cpu",
        options={
            "tf32": "off",
            "molecular_field_linear_space": "spectral",
            "stress_divergence_sum_space": "spectral",
            "pointwise_execution": "eager",
            "disable_q_gradient_reuse": False,
        },
    ),
    output=Output(
        directory=Path(
            os.environ.get(
                "PSSOLVER_EXAMPLE_OUTPUT",
                "data/plane_complete_example",
            )
        ),
        steps=2,
        save_start_step=0,
        save_interval=2,
        diagnostic_interval=1,
        diagnostics=True,
        save_hydrodynamics=True,
    ),
    invocation=InvocationSpec({}),
)

compiled = compile_simulation(simulation)
result = run_simulation(compiled)
assert result.status is SimulationRunStatus.COMPLETE
print(result.final_step, result.output_directory)
```
<!-- rc5-complete-simulation-example:end -->

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
        save_start_step=0,
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
public finite-Q Robin execution, static prescribed-Q lifting, strong
homeotropic/planar-Q lifting, functional batches larger than one, Plane optimal
control, production-scale optimization campaigns, and scientific control
conclusions.  The three static lifting policies remain declarable for schema
compatibility but have `executable=false` in the live catalog and are rejected
before runtime construction with
`unsupported_lifted_wall_molecular_field`.  `nematics3d` remains an optional
analysis dependency and is not modified or bundled.
