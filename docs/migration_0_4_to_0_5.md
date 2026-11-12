# Migrating from PSSolver 0.4 to 0.5

0.5 uses one Engine and Runner for the qualified Periodic, Plane and Channel
problems. `compile_simulation` assembles a tensor-free Plan; `run_simulation`
accepts either the 0.5 declaration or that compiled object.

| 0.4 | 0.5 |
| --- | --- |
| `assign_boundaries(..., policies=...)` | Pass the same policy dictionary as `boundaries`. |
| `TimeStepping(dt=..., refresh={"mode": "disabled"})` | `SemiImplicitEuler(dt=...)`. For an enabled cadence, use `refresh={"steps": N}` or `refresh={"physical_time": T}`. |
| `discretization={"pressure_solver": ...}` | Put the same pressure options in `SpectralNumerics(pressure_solver=...)`. |
| `InvocationSpec` | Put `dry_run` and `validation_config_sha256` in `Output`. |
| `TorchSpectralExecution(runtime_path=..., device=..., options=...)` | Keep `device` and `options`; remove `runtime_path`. |
| Utility names at the package root | Import from `pssolver.loop_conventions`, `pssolver.run_metadata` or `pssolver.snapshots`. Transform names live in their backend, operator or Stokes modules. |
| v1 solver and integrator | Use `pssolver.engine` and `pssolver.planning.assemble`. |

## Periodic box and rectangular Channel

Use the same input snapshot and the same working directory on both versions.
The first block is shared by both versions. Run the old and new blocks in
separate environments, with different empty output directories. Call
`run_case(False)` for the Periodic example and `run_case(True)` for Channel.

```python
from pathlib import Path

import numpy as np


def inputs(channel):
    shape = (8, 8, 8)
    snapshot = Path("channel-initial" if channel else "periodic-initial")
    snapshot.mkdir(exist_ok=True)
    q = np.zeros((*shape, 5), dtype=np.float64)
    q[..., 0], q[..., 3] = 0.2, -0.1
    np.save(snapshot / "Q_0.npy", q, allow_pickle=False)
    material = dict(
        ldg_a=0.0, ldg_b=-0.3, ldg_c=0.3, ldg_l1=1 / 81, gamma=2.94,
        flow_alignment=0.3, activity=0.01, beta=-1.0, viscosity=2 / 3,
    )
    pressure = dict(
        algorithm="preconditioned_conjugate_gradient", relative_tolerance=1e-10,
        max_iterations=40, fixed_iterations=12, warm_start=True,
    )
    options = dict(
        tf32="off", molecular_field_linear_space="spectral",
        stress_divergence_sum_space="physical" if channel else "spectral",
        pointwise_execution="eager", disable_q_gradient_reuse=False,
    )
    return shape, snapshot, material, pressure, options
```

0.4:

```python
from pssolver import (
    Output, Simulation, SnapshotInitialCondition, SpectralNumerics, TimeStepping,
    TorchSpectralExecution, run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries, free_slip_velocity, neumann_pressure_compatibility,
    neumann_q, no_slip_velocity,
)
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.models.active_nematics import CompleteStressBerisEdwards


def run_case(channel):
    shape, snapshot, material, pressure, options = inputs(channel)
    model = CompleteStressBerisEdwards(**material)
    geometry = (RectangularChannel if channel else PeriodicBox)(
        shape=shape, lengths=(16.0, 12.0, 12.0),
    )
    policies = {
        "Q": neumann_q(),
        "velocity": no_slip_velocity() if channel else free_slip_velocity(),
        "pressure": neumann_pressure_compatibility(),
    }
    return run_simulation(Simulation(
        model=model, geometry=geometry,
        boundaries=assign_boundaries(model=model, geometry=geometry, policies=policies),
        numerics=SpectralNumerics(
            dtype="float64", dealias_rule="cubic_half",
            spectral_storage="full_complex" if channel else "hermitian_half",
            hermitian_axis=None if channel else 1,
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress" if channel else "periodic_spectral",
            device="cpu", options=options,
        ),
        output=Output(
            directory="old-channel" if channel else "old-periodic", steps=2,
            save_interval=1, diagnostic_interval=1, save_start_step=0,
            diagnostics=True, save_hydrodynamics=True, checkpoint_interval=1,
        ),
        discretization={"pressure_solver": pressure} if channel else {},
    ))
```

0.5:

```python
from pssolver import (
    CompleteStressBerisEdwards, Output, PeriodicBox, RectangularChannel,
    SemiImplicitEuler, Simulation, SnapshotInitialCondition, SpectralNumerics,
    TorchSpectralExecution, free_slip_velocity, neumann_pressure_compatibility,
    neumann_q, no_slip_velocity, run_simulation,
)


def run_case(channel):
    shape, snapshot, material, pressure, options = inputs(channel)
    return run_simulation(Simulation(
        model=CompleteStressBerisEdwards(**material),
        geometry=(RectangularChannel if channel else PeriodicBox)(
            shape=shape, lengths=(16.0, 12.0, 12.0),
        ),
        boundaries={
            "Q": neumann_q(),
            "velocity": no_slip_velocity() if channel else free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
        numerics=SpectralNumerics(
            dtype="float64", dealias_rule="cubic_half",
            spectral_storage="full_complex" if channel else "hermitian_half",
            hermitian_axis=None if channel else 1,
            pressure_solver=pressure if channel else None,
        ),
        time=SemiImplicitEuler(dt=0.001),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(device="cpu", options=options),
        output=Output(
            directory="new-channel" if channel else "new-periodic", steps=2,
            save_interval=1, diagnostic_interval=1, save_start_step=0,
            diagnostics=True, save_hydrodynamics=True, checkpoint_interval=1,
        ),
    ))
```

## Plane slab

For an activity-number based Plane run, `ShendrukPlaneBerisEdwards` preserves
the CLI parameterization directly. The CLI option names remain the same except
for `--runtime-path`, which must be removed from 0.5 commands. For example:

```bash
# 0.4
pssolver-plane-beris-edwards --activity-number 18 --nx 8 --ny 8 --nz 6 \
    --num-defect-pairs 1 --device cpu --dtype float64 --pointwise-execution eager \
    --steps 2 --save-start-step 0 --save-interval 1 --diagnostic-interval 1 \
    --diagnostics --save-hydrodynamics --checkpoint-interval 1 \
    --runtime-path legacy_production --output-dir old-plane
```

```bash
# 0.5
pssolver-plane-beris-edwards --activity-number 18 --nx 8 --ny 8 --nz 6 \
    --num-defect-pairs 1 --device cpu --dtype float64 --pointwise-execution eager \
    --steps 2 --save-start-step 0 --save-interval 1 --diagnostic-interval 1 \
    --diagnostics --save-hydrodynamics --checkpoint-interval 1 --output-dir new-plane
```

The 0.5 Python equivalent uses a `PlaneSlab`, a `ShendrukPlaneBerisEdwards`
declaration, Neumann Q, free-slip velocity and the Neumann pressure policy.
For Plane, use a generated `extruded_defect_gas` initial condition. Snapshots
remain the public input for Periodic and Channel.

## Restart and provenance

Use `Output(restart_from=..., directory=..., steps=...)` with a new output
directory. `steps` counts **additional** steps. The reader authenticates the
identity before loading an accepted v0.4 checkpoint; new checkpoints use
`engine_state`. Serialized Python objects from removed module names are not
compatible; the checkpoint format does not rely on pickle.

`metadata.json` keeps the qualified output structure with an incremented
schema version and top-level `plan_identity_sha256` and `declaration_sha256`.
The `configuration` object is the new declaration, not the old compatibility
identity.

## Functional API 1.1

Keep using `pssolver.functional.api`. Import the **0.4** declaration types and
helpers for that API from `pssolver.functional.legacy_declarations`:

```python
from pssolver.functional.api import periodic_activity_functional_request
from pssolver.functional.legacy_declarations import Simulation, TimeStepping
```

These declarations retain their Functional API 1.1 identity and are not
inputs to the 0.5 public runner. The compatibility declarations retire
together with Functional API 1.1, not with this migration.
