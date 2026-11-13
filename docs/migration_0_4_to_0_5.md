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

## v0.4.0 package-root names

| v0.4.0 name | 0.5 use |
| --- | --- |
| `BasisAwareSpectralProjector` | Removed from the root; the lower-level owner is `pssolver.operators.projection.BasisAwareSpectralProjector`. |
| `CONVENTION_NAME` | Import from `pssolver.loop_conventions.CONVENTION_NAME`. |
| `ChannelPressureConvergenceError` | Import `ChannelPressureConvergenceError` from `pssolver`. |
| `CompiledSimulation` | Import `CompiledSimulation` from `pssolver`. |
| `DEALIAS_RULE_FRACTIONS` | Removed from the root; the constant lives at `pssolver.core.numerics.DEALIAS_RULE_FRACTIONS`. |
| `DEFAULT_DEALIAS_RULE` | Removed from the root; the constant lives at `pssolver.core.numerics.DEFAULT_DEALIAS_RULE`. |
| `DEFAULT_PERIODIC_TRANSFORM_EXECUTION` | Removed from the root; the constant lives at `pssolver.backends.tensor_product.DEFAULT_PERIODIC_TRANSFORM_EXECUTION`. |
| `DEFAULT_PROJECTED_TRANSFORM_EXECUTION` | Removed from the root; the constant lives at `pssolver.core.numerics.DEFAULT_PROJECTED_TRANSFORM_EXECUTION`. |
| `DEFAULT_SPECTRAL_STORAGE` | Removed from the root; the constant lives at `pssolver.core.numerics.DEFAULT_SPECTRAL_STORAGE`. |
| `DEFAULT_TRANSFORM_EXECUTION_ORDER` | Removed from the root; the constant lives at `pssolver.core.numerics.DEFAULT_TRANSFORM_EXECUTION_ORDER`. |
| `DEFAULT_TRANSFORM_GROUP_INDEXING` | The v1 root constant is removed; leave `transform_group_indexing` unset for the default. |
| `DeclarationCapability` | Import `DeclarationCapability` from `pssolver`. |
| `Fields` | Use the public declaration and snapshots. The v1 mutable field container is removed. |
| `FreeSlipModalStokesSolver` | Removed from the root; the lower-level owner is `pssolver.linear_solvers.stokes.plane_free_slip.FreeSlipModalStokesSolver`. |
| `GeneratedInitialCondition` | Import `GeneratedInitialCondition` from `pssolver`. |
| `IDEAL_LOOP_MODE_AXES` | Import from `pssolver.loop_conventions.IDEAL_LOOP_MODE_AXES`. |
| `Output` | Import `Output` from `pssolver`. |
| `PDEModel` | Use a model declaration such as `CompleteStressBerisEdwards`; the v1 base class is removed. |
| `PERIODIC_TRANSFORM_EXECUTION_MODES` | Removed from the root; the constant lives at `pssolver.backends.tensor_product.PERIODIC_TRANSFORM_EXECUTION_MODES`. |
| `PROJECTED_TRANSFORM_EXECUTION_MODES` | Removed from the root; the constant lives at `pssolver.core.numerics.PROJECTED_TRANSFORM_EXECUTION_MODES`. |
| `Parameters` | Declare model parameters on the selected model or preset; the v1 container is removed. |
| `PublicCapabilityCatalog` | Import `PublicCapabilityCatalog` from `pssolver`. |
| `PublicCompilationRejectionCode` | Import `PublicCompilationRejectionCode` from `pssolver`. |
| `PublicSimulationCompilationError` | Import `PublicSimulationCompilationError` from `pssolver`. |
| `QualifiedCombination` | Import `QualifiedCombination` from `pssolver`. |
| `REPRESENTATIVE_ORDERED_S_DEFINITION` | Import from `pssolver.snapshots.REPRESENTATIVE_ORDERED_S_DEFINITION`. |
| `SPECTRAL_STORAGE_MODES` | Removed from the root; the constant lives at `pssolver.core.numerics.SPECTRAL_STORAGE_MODES`. |
| `SemiImplicitEulerIntegrator` | Use `SemiImplicitEuler` with `run_simulation`; the v1 integrator is removed. |
| `Simulation` | Import `Simulation` from `pssolver`. |
| `SimulationDiagnosticProtocol` | Import `SimulationDiagnosticProtocol` from `pssolver`. |
| `SimulationExecutionError` | Import `SimulationExecutionError` from `pssolver`. |
| `SimulationInputError` | Import `SimulationInputError` from `pssolver`. |
| `SimulationInputFileNotFoundError` | Import `SimulationInputFileNotFoundError` from `pssolver`. |
| `SimulationInputIOError` | Import `SimulationInputIOError` from `pssolver`. |
| `SimulationObservationProtocol` | Import `SimulationObservationProtocol` from `pssolver`. |
| `SimulationRestartError` | Import `SimulationRestartError` from `pssolver`. |
| `SimulationResult` | Import `SimulationResult` from `pssolver`. |
| `SimulationRunStatus` | Import `SimulationRunStatus` from `pssolver`. |
| `SimulationSnapshot` | Import from `pssolver.snapshots.SimulationSnapshot`. |
| `SimulationStateError` | Import `SimulationStateError` from `pssolver`. |
| `SnapshotInitialCondition` | Import `SnapshotInitialCondition` from `pssolver`. |
| `SpectralNumerics` | Import `SpectralNumerics` from `pssolver`. |
| `SpectralSolver` | Use `compile_simulation` and `run_simulation`; the v1 solver is removed. |
| `TRANSFORM_GROUP_INDEXING_MODES` | The v1 root constant is removed; retain a supported `transform_group_indexing` execution option. |
| `TensorProductTransformBackend` | Removed from the root; the lower-level owner is `pssolver.backends.tensor_product.TensorProductTransformBackend`. |
| `TimeStepping` | Use `SemiImplicitEuler`; map the cadence as described above. |
| `TorchSpectralExecution` | Import `TorchSpectralExecution` from `pssolver`. |
| `__version__` | Import `__version__` from `pssolver`. |
| `apply_snapshot_to_solver` | Use `SnapshotInitialCondition`; the v1 solver mutation helper is removed. |
| `available_boundary_policies` | Import `available_boundary_policies` from `pssolver`. |
| `available_combinations` | Import `available_combinations` from `pssolver`. |
| `available_geometries` | Import `available_geometries` from `pssolver`. |
| `available_models` | Import `available_models` from `pssolver`. |
| `capability_catalog` | Import `capability_catalog` from `pssolver`. |
| `classify_ideal_loop` | Import from `pssolver.loop_conventions.classify_ideal_loop`. |
| `compile_simulation` | Import `compile_simulation` from `pssolver`. |
| `convention_metadata` | Import from `pssolver.loop_conventions.convention_metadata`. |
| `ideal_loop_axis_angles` | Import from `pssolver.loop_conventions.ideal_loop_axis_angles`. |
| `load_snapshot` | Import from `pssolver.snapshots.load_snapshot`. |
| `prepare_new_run_directory` | Import from `pssolver.run_metadata.prepare_new_run_directory`. |
| `projected_common_basis_stress_divergence` | Removed from the root; the lower-level owner is `pssolver.operators.tensor_divergence.projected_common_basis_stress_divergence`. |
| `projected_distortion_stress_divergence` | Removed from the root; the lower-level owner is `pssolver.operators.tensor_divergence.projected_distortion_stress_divergence`. |
| `representative_ordered_S` | Import from `pssolver.snapshots.representative_ordered_S`. |
| `require_distinct_output_directory` | Import from `pssolver.snapshots.require_distinct_output_directory`. |
| `run_simulation` | Import `run_simulation` from `pssolver`. |
| `write_run_metadata` | Import from `pssolver.run_metadata.write_run_metadata`. |

## v0.4.0 Plane CLI options

| v0.4.0 option | 0.5 use |
| --- | --- |
| `--activity-number` | Keep the same option. |
| `--background-angle` | Keep the same option. |
| `--beta` | Keep the same option. |
| `--checkpoint-interval` | Keep the same option. |
| `--coefficient-max` | Keep the same option. |
| `--coefficient-min` | Keep the same option. |
| `--dealias-rule` | Keep the same option. |
| `--defect-core-radius` | Keep the same option. |
| `--defect-min-separation` | Keep the same option. |
| `--device` | Keep the same option. |
| `--diagnostic-interval` | Keep the same option. |
| `--diagnostics` | Keep the same option. |
| `--disable-q-gradient-reuse` | Keep the same option. |
| `--disable-spectral-refresh` | Keep the same option. |
| `--dry-run` | Keep the same option. |
| `--dt` | Keep the same option. |
| `--dtype` | Keep the same option. |
| `--eta` | Keep the same option. |
| `--flow-alignment` | Keep the same option. |
| `--frank-k` | Keep the same option. |
| `--friction-mode-fric` | Keep the same option. |
| `--gamma` | Keep the same option. |
| `--height` | Keep the same option. |
| `--help` | Keep the same option. |
| `--initial-s` | Keep the same option. |
| `--ldg-a` | Keep the same option. |
| `--ldg-b` | Keep the same option. |
| `--ldg-c` | Keep the same option. |
| `--lx` | Keep the same option. |
| `--ly` | Keep the same option. |
| `--molecular-field-linear-space` | Keep the same option. |
| `--num-defect-pairs` | Keep the same option. |
| `--nx` | Keep the same option. |
| `--ny` | Keep the same option. |
| `--nz` | Keep the same option. |
| `--output-dir` | Keep the same option. |
| `--parameterization` | Keep the same option. |
| `--pointwise-execution` | Keep the same option. |
| `--projected-transform-execution` | Keep the same option. |
| `--restart-from` | Keep the same option. |
| `--runtime-path` | Remove it; there is one assembled public path. |
| `--save-hydrodynamics` | Keep the same option. |
| `--save-interval` | Keep the same option. |
| `--save-start-step` | Keep the same option. |
| `--seed` | Keep the same option. |
| `--spectral-refresh-steps` | Keep the same option. |
| `--spectral-refresh-time` | Keep the same option. |
| `--spectral-storage` | Keep the same option. |
| `--steps` | Keep the same option. |
| `--stress-divergence-sum-space` | Keep the same option. |
| `--tf32` | Keep the same option. |
| `--transform-execution-order` | Keep the same option. |
| `--twist-amplitude` | Keep the same option. |
| `--twist-modes` | Keep the same option. |
| `--validation-config-sha256` | Keep the same option. |
| `--zero-mode-policy` | Keep the same option. |
| `-h` | Keep the same option. |

## Retired v0.4.0 root commands

These are historical invocations, not commands supported from the 0.5 tree.
Retrieve their original bytes from the `v0.4.0` tag.

| Historical invocation | 0.5 use |
| --- | --- |
| `python Channel_control.py` | Removed; research and Control workflows are not part of the Provider CLI. |
| `python Channel_control_box.py` | Removed; research and Control workflows are not part of the Provider CLI. |
| `python Channel_control_box_high_temporal_resolution.py` | Removed; research and Control workflows are not part of the Provider CLI. |
| `python Channel_dal.py` | Removed; research and Control workflows are not part of the Provider CLI. |
| `python Channel_optimal_control.py` | Removed; research and Control workflows are not part of the Provider CLI. |
| `python Periodic.py` | Use the Periodic declaration example above. |
| `python Plane.py` | For supported Plane configurations, use the Plane declaration or CLI above; the legacy runner is removed. |
| `python Plane_H15_twist_evolution.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `python Plane_beris_edwards_navier_stokes.py` | For supported Plane configurations, use the Plane declaration or CLI above; the legacy runner is removed. |
| `python Plane_beris_edwards_shadow.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `python Plane_fig4_benchmark.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `python Plane_free_slip.py` | For supported Plane configurations, use the Plane declaration or CLI above; the legacy runner is removed. |
| `python Plane_free_slip_dealiased.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `python Plane_shendruk_stokes.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `python compare_dealias_100k.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `python compare_dealias_final_spectra.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `bash run_dealias_100k_remaining.sh` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `python visualize_nematics3d_snapshot.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `python visualize_nematics3d_snapshot_channel.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
| `python visualize_nematics3d_snapshot_plane.py` | Removed; this was a historical benchmark, audit or visualization driver, not a 0.5 public entry point. |
