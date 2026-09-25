# Phase 7 P7.7.9: public compiler and runner connection

Status: `P7_7_9_COMPLETE_PHASE_8_NOT_AUTHORIZED`.

Parent baseline: `0ba42120207c1ae7662a0e913ce4c1f6694fa116` on
`next/pssolver-v0.2.0-architecture`.

Classification: `PASS_P7_7_9_PUBLIC_COMPILER_RUNNER_CONNECTION`.

P7.7.9 connects the typed public `Simulation` declaration to the already
qualified complete-stress Plane production application. It adds the supported
root imports:

```python
from pssolver import CompiledSimulation, compile_simulation, run_simulation
```

It does not create a second runtime, replace a numerical kernel, add dispatch
inside the timestep, promote a compiled default, or authorize Phase 8.

The machine-readable authority is
[phase_7_p779_public_runner_connection.json](phase_7_p779_public_runner_connection.json).

## Compile once, run through the existing application

`compile_simulation(simulation)` performs the following work before tensor
allocation:

1. checks that the request is the qualified complete-stress Beris--Edwards
   model on a Plane slab;
2. requires the free-slip/Neumann/pressure-compatibility boundary signature;
3. requires the projected semi-implicit Euler path and a qualified Plane
   runtime (`legacy_production` or `compiled_v2`);
4. translates the physical coefficients to the existing fixed-K Plane
   application parameterization;
5. requires the initial-condition, execution, refresh, and workflow controls
   explicitly rather than filling benchmark defaults;
6. recomposes the application `SimulationSpec` and verifies geometry,
   boundaries, numerics, integrator, initial condition, execution, and
   workflow identities;
7. runs the existing capability lowering and package-construction planning;
8. returns an immutable `CompiledSimulation` containing the source identity,
   application request, lowering plan, construction plan, and normalization
   record.

`run_simulation(...)` accepts either the original declaration or the compiled
product. It imports and calls the existing
`run_plane_beris_edwards(...)` application exactly once. Runtime selection is
therefore completed before construction; there is no fallback and no public
API lookup in the timestep.

## Executable declaration example

The runner intentionally requires the controls that affect reproducibility to
be present in the declaration:

```python
from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
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
    shape=(320, 320, 80),
    lengths=(100.0, 100.0, 20.0),
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
    ),
    time=TimeStepping(
        dt=0.005,
        refresh={"mode": "disabled"},
    ),
    initial_condition=GeneratedInitialCondition(
        "extruded_defect_gas",
        parameters={
            "seed": 24,
            "num_defect_pairs": 6,
            "defect_min_separation": 10.0,
            "defect_core_radius": 1.5,
            "background_angle": 0.0,
            "twist_amplitude": 0.01,
            "twist_modes": [1, 2, 3],
            "initial_s": 1.0 / 3.0,
        },
    ),
    execution=TorchSpectralExecution(
        runtime_path="compiled_v2",
        device="cuda",
        options={
            "tf32": "off",
            "molecular_field_linear_space": "spectral",
            "stress_divergence_sum_space": "spectral",
            "pointwise_execution": "compile",
            "disable_q_gradient_reuse": False,
        },
    ),
    output=Output(
        directory="data/A18",
        steps=20_000,
        save_interval=1_000,
        diagnostic_interval=100,
    ),
)

result = run_simulation(simulation)
```

The declaration is longer than the aspirational sketch because P7.7.9 does
not hide initial-state realization parameters, execution policy, pressure
compatibility, or spectral-refresh behavior. Later typed presets may shorten
the syntax by expanding their defaults into the immutable declaration before
compilation.

## Coefficient mapping

The existing production application is expressed through the Shendruk
activity-number/fixed-K parameterization. The public model instead declares
`activity` and `L1` directly. Compilation derives

```text
q_eq = 3 S_eq / 2
K = 2 q_eq^2 L1
A = H sqrt(activity / K)
```

and requests the existing fixed-K application. It then recomposes the model
and requires the effective activity and L1 to match the public request within
double-precision roundoff. Requested and effective values are retained in
`CompiledSimulation.normalization`; a material discrepancy is fatal.

## Fail-closed boundary

P7.7.9 rejects, before allocation:

- models or geometries without a qualified application connection;
- Channel, snapshot, SBDF2, separated-canary, and new boundary-physics paths;
- missing or extra initial-condition, execution, or workflow controls;
- implicit spectral-refresh behavior;
- positive-friction or non-zero-mean tangential-flow requests;
- runtime fallback;
- any translation that changes a declared geometry, boundary, numerical,
  execution, workflow, or material identity.

These are unsupported combinations, not requests for a nearby default.

## Compatibility and performance

The public runner delegates to the existing Plane application with its
existing `PlaneBerisEdwardsRunSpec`. Runtime construction, fields, transforms,
operators, workflow, metadata, checkpoint format, output arrays, and timestep
code are unchanged. Compilation and dispatch occur once before allocation.

No H100 qualification is required for this connection slice because it adds
no GPU operation, persistent allocation, transform call, kernel, or hot-path
branch. Existing Plane defaults remain unchanged; selecting `compiled_v2` in
the public declaration is explicit rather than a promotion.

## Verification

The P7.7.9 target tests cover successful compilation, coefficient mapping,
immutable evidence, lazy application dispatch, source dependency boundaries,
and fail-closed incomplete or unsupported declarations:

```text
87 passed
```

An actual public dry-run dispatched through the existing Plane application
and returned normally without creating an output directory. The complete CPU
suite passed:

```text
2159 passed, 8 subtests passed
```

No test failed, skipped, or was deselected.

## Next boundary

Phase 8 remains unauthorized. A later ergonomic-preset slice may add typed
initial-condition and execution presets that expand every default into the
declaration. New boundary physics, additional geometries, additional models,
and Channel public execution require independent capability and runtime
qualification.
