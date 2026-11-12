# PSSolver

PSSolver is a research-grade, GPU-first tensor-product spectral solver. The
0.5 development line composes models, geometry, boundaries and execution
settings into one Plan for a shared Engine and Runner.

The release does not claim arbitrary PDE or geometry support.
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

The package-root API exposes `Simulation`, `compile_simulation` and
`run_simulation`. Compilation builds one immutable, tensor-free Plan. The
shared Engine and Runner execute that Plan; callers no longer choose a
runtime implementation with `runtime_path`.

```python
from pssolver import capability_catalog

catalog = capability_catalog()
for combination in catalog.qualified_combinations:
    print(combination.equation_variant, combination.geometry_name)
```

The catalog derives executable model, geometry and boundary combinations from
the assembler. The supported Beris–Edwards geometries are Periodic box, Plane
slab and rectangular Channel. Unsupported combinations report a reason before
tensor allocation. The [migration guide](docs/migration_0_4_to_0_5.md) shows
how to port 0.4 declarations and resume supported v0.4 checkpoints.

### Complete composable Plane example

The output directory must not already contain a run.

<!-- rc5-complete-simulation-example:start -->
```python
import os
from pathlib import Path

from pssolver import (
    GeneratedInitialCondition, Output, PlaneSlab, SemiImplicitEuler,
    ShendrukPlaneBerisEdwards, Simulation, SpectralNumerics,
    TorchSpectralExecution, compile_simulation, free_slip_velocity,
    neumann_pressure_compatibility, neumann_q, run_simulation,
)

simulation = Simulation(
    model=ShendrukPlaneBerisEdwards(activity_number=18.0),
    geometry=PlaneSlab(shape=(8, 8, 6), lengths=(40.0, 40.0, 20.0)),
    boundaries={
        "Q": neumann_q(),
        "velocity": free_slip_velocity(),
        "pressure": neumann_pressure_compatibility(),
    },
    numerics=SpectralNumerics(
        dtype="float64", spectral_storage="full_complex", dealias_rule="cubic_half",
    ),
    time=SemiImplicitEuler(dt=0.005),
    initial_condition=GeneratedInitialCondition("extruded_defect_gas", parameters={
        "seed": 24, "num_defect_pairs": 1, "defect_min_separation": 10.0,
        "defect_core_radius": 1.5, "background_angle": 0.0,
        "twist_amplitude": 0.01, "twist_modes": [1, 2, 3], "initial_s": 1 / 3,
    }),
    execution=TorchSpectralExecution(device="cpu", options={"pointwise_execution": "eager"}),
    output=Output(
        directory=Path(os.environ.get("PSSOLVER_EXAMPLE_OUTPUT", "plane-example")),
        steps=2, save_interval=1, diagnostic_interval=1, save_start_step=0,
        diagnostics=True, save_hydrodynamics=True, checkpoint_interval=1,
    ),
)
compiled = compile_simulation(simulation)
result = run_simulation(compiled)
print(compiled.plan_identity_sha256, result.final_step)
```
<!-- rc5-complete-simulation-example:end -->

## Stable functional API

`pssolver.functional.api` is the versioned functional interface for independent
consumers. Protocol version 1.1 qualifies batch-one Periodic and rectangular-
Channel activity runtimes, deterministic replay, checkpoint bridges,
observations, diagnostics, and differentiable execution. Its public
`runtime.execution_scope()` context owns the declared TF32 policy across a
consumer's forward, replay, and backward calls, including exact restoration
on nested and exceptional exits. A backward pass executed outside that scope
uses the caller's TF32 settings. The additive method changes the functional
API version and runtime provider identity; the API-1.0 current-format
checkpoint reader preserves forward-compatible state restart without
claiming exact API-1.1 identity. PSSolver-Control is an independent consumer
and is not bundled into this distribution.

For 1.1 callers, the v0.4 declaration helpers now live in
`pssolver.functional.legacy_declarations`. Keep calling the versioned
functional entry points from `pssolver.functional.api`.

## Plane command line

The Plane command uses the same public assembly and Runner. It keeps
the numerical flags, but no longer accepts `--runtime-path`.

    pssolver-plane-beris-edwards --activity-number 18 --output-dir run \
        --device cpu --dtype float64 --nx 32 --ny 32 --nz 16 \
        --num-defect-pairs 1 --steps 10 --save-start-step 0

## Validation and limits

The Phase 0--9 architecture program is complete.  Its cumulative CPU, H100,
installed-wheel, restart, checkpoint, gradient, Taylor, memory, negative-gate,
and long-horizon Hermitian evidence is summarized in `CHANGELOG.md` and the
versioned closure records under `notes/architecture_v0_2/`.

The following remain outside this release: arbitrary model--geometry
execution, nonhomogeneous Neumann lifting, dynamic or trainable boundary data,
public finite-Q Robin execution, static prescribed-Q lifting, strong
homeotropic/planar-Q lifting, functional batches larger than one, Plane optimal
control, production-scale optimization campaigns, and scientific control
conclusions.  The three static lifting policies remain declarable for schema
compatibility. Unsupported boundary choices carry a scope or mathematical
rejection in the live catalog and fail before runtime construction.
`nematics3d` remains an optional
analysis dependency and is not modified or bundled.

The internal Robin pilot reports its lower and upper wall-law residuals as an
independent boundary-contract gate. Because the lift and homogeneous basis
satisfy those laws by construction, that gate does not replace the separate
interior Helmholtz apply/solve, evolution, and restart gates. Robin remains
unavailable through the public `Simulation` capability catalog.

Checkpoint publication emits a structured diagnostic identifying the atomic
`rename_noreplace` fast path or the owned-placeholder NFS compatibility path.
That operational observation is not checkpoint or provider identity.
Functional checkpoint imports expose `integrity_status`; consumers that
require sealed provenance must require `sealed_sha256` rather than accepting
the explicitly supported `unsealed_legacy` compatibility path.
