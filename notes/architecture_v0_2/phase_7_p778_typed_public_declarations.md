# Phase 7 P7.7.8: typed public convenience declarations

Status: `P7_7_8_COMPLETE_RUNNER_CONNECTION_NOT_AUTHORIZED`.

Parent baseline: `65b8fad3f860bc97fe79b60e4f90ee087a18a527` on
`next/pssolver-v0.2.0-architecture`.

Classification: `PASS_P7_7_8_TYPED_PUBLIC_DECLARATIONS`.

P7.7.8 adds tensor-free, typed convenience constructors around the canonical
declarations frozen by P7.7.7. It does not connect `run_simulation`, lower a
declaration, construct a runtime, allocate a tensor, execute a timestep, or
change a production default.

The machine-readable authority is
[phase_7_p778_typed_public_declarations.json](phase_7_p778_typed_public_declarations.json).

## Supported declaration example

The complete public declaration can now be written without importing internal
configuration classes:

```python
from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
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
    time=TimeStepping(dt=0.005),
    initial_condition=GeneratedInitialCondition(
        "extruded_defect_gas",
        parameters={"seed": 24},
    ),
    execution=TorchSpectralExecution(
        runtime_path="compiled_v2",
        device="cuda",
        options={"tf32": "off"},
    ),
    output=Output(
        directory="data/A18",
        steps=20_000,
        save_interval=1_000,
        diagnostic_interval=100,
    ),
)
```

The declaration is intentionally not executable in this slice.

## Model declaration

`CompleteStressBerisEdwards(...)` returns the canonical
`EquationSystemSpec`, not a second public model hierarchy. It requires all
scientific coefficients that cannot be inferred safely: `L1`, beta, activity,
and viscosity are explicit. It selects the already-qualified zero-mean
tangential Stokes policy with zero friction, but does not attach a geometry,
wall law, transform, runtime, or device.

The constructor lives under `pssolver.models.active_nematics`, so ownership of
the equations remains with the model package.

## Geometry declaration

`PlaneSlab` now accepts either its historical `DomainSpec` positional form or
the public keyword form:

```python
PlaneSlab(shape=(nx, ny, nz), lengths=(lx, ly, lz))
```

Both paths construct the same historical canonical
`pssolver.geometries.tensor_product.PlaneSlab` type. The public constructor
facade preserves `isinstance`, exact registry dispatch, protocol-4 pickle, and
restart-provenance identity. A caller cannot mix a prebuilt domain with shape
or length overrides.

## Boundary declaration

The public helpers are physical and semantic declarations, not DCT/DST
selectors:

- `neumann_q()` declares homogeneous Neumann Q anchoring;
- `free_slip_velocity()` declares no penetration for the wall-normal velocity
  and homogeneous Neumann conditions for tangential velocity;
- `neumann_pressure_compatibility()` declares the algebraic pressure modal
  compatibility space;
- `assign_boundaries(...)` expands the logical-field policies to the canonical
  face-aware `BoundaryAssignment`.

Pressure compatibility is explicit. It is not silently inferred from the
velocity wall law and is not presented as independently prescribed pressure
physics. Every evolved or algebraic logical field must be supplied exactly
once. Periodic versus bounded axes are obtained from the geometry, and the
free-slip normal component follows the bounded-axis index rather than a
hard-coded Plane z label.

This slice adds no nonhomogeneous, Robin, tau, or anchoring law and does not
authorize Phase 8.

## Numerical and run declarations

The following convenience classes subclass their existing canonical types:

- `SpectralNumerics` -> `NumericsConfig`;
- `TimeStepping` -> `TimeIntegrationSpec`;
- `GeneratedInitialCondition` and `SnapshotInitialCondition` ->
  `InitialConditionSpec`;
- `TorchSpectralExecution` -> `ExecutionSpec`;
- `Output` -> `WorkflowSpec`.

Consequently P7.7.7 `Simulation` still receives exactly the canonical objects
it originally required. Its constructor field order and implementation source
remain unchanged.

The ownership corrections relative to the earlier illustrative sketch are
deliberate:

1. `steps` belongs to `Output`, not `TimeStepping`, because total duration is
   run identity rather than time-discretization identity;
2. initial condition and execution remain explicit rather than acquiring
   hidden global defaults;
3. pressure compatibility is named separately from the velocity law;
4. physical model coefficients such as `L1`, beta, and viscosity are not
   guessed from a benchmark name.

`TorchSpectralExecution` records `fallback_allowed=False`. It does not import
or instantiate Torch and it requires an explicit runtime path and device.

## Compatibility and performance boundary

The historical geometry constructor remains supported. Existing Plane and
Channel applications, CLIs, run specifications, runtime factories, workflows,
metadata schemas, output arrays, checkpoints, numerical kernels, transform
counts, and defaults are unchanged.

All P7.7.8 constructors execute once on the CPU while building immutable
metadata. None is reachable from a timestep, and the new modules import no
Torch, NumPy, runtime, transform, operator, linear solver, workflow, or
application implementation. No GPU qualification is required for this slice.

## Verification

The P7.7.1, P7.7.2, P7.7.7, P7.7.8, dependency, and public-root target set
passed:

```text
66 passed
```

The complete local CPU suite passed:

```text
2148 passed, 8 subtests passed
```

No test failed, skipped, or was deselected.

## Next slice

The next slice is not authorized by this result. A separately authorized
public compiler/runner connection may accept this `Simulation`, lower it
through the existing fail-closed capability path, construct one qualified
runtime, and expose `run_simulation`. It must not add runtime fallback, put
dictionary or string dispatch in the timestep, infer missing physics, or
change Plane and Channel production defaults.
