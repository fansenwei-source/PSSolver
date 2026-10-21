"""RC6-3 compile-time complete-stress model and snapshot validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pssolver import (
    Output,
    Simulation,
    SimulationInputError,
    SnapshotInitialCondition,
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
    no_slip_velocity,
)
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.models.active_nematics import CompleteStressBerisEdwards


_GRID_SHAPE = (8, 8, 6)


def _simulation(
    tmp_path: Path,
    family: str,
    *,
    snapshot_shape: tuple[int, ...] | None = (*_GRID_SHAPE, 5),
) -> Simulation:
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
    if family == "periodic":
        geometry = PeriodicBox(
            shape=_GRID_SHAPE,
            lengths=(8.0, 8.0, 6.0),
        )
        velocity = free_slip_velocity()
        runtime_path = "periodic_spectral"
        stress_space = "spectral"
        numerics = SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        )
        discretization = {}
    elif family == "channel":
        geometry = RectangularChannel(
            shape=_GRID_SHAPE,
            lengths=(8.0, 8.0, 6.0),
        )
        velocity = no_slip_velocity()
        runtime_path = "channel_complete_stress"
        stress_space = "physical"
        numerics = SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        )
        discretization = {
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        }
    else:  # pragma: no cover - private test helper guard
        raise AssertionError(f"unsupported family: {family}")

    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": velocity,
            "pressure": neumann_pressure_compatibility(),
        },
    )
    snapshot = tmp_path / f"{family}_snapshot"
    snapshot.mkdir()
    if snapshot_shape is not None:
        np.save(
            snapshot / "Q_0.npy",
            np.zeros(snapshot_shape, dtype=np.float64),
            allow_pickle=False,
        )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=numerics,
        time=TimeStepping(dt=1.0e-3, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path=runtime_path,
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": stress_space,
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / f"{family}_output",
            steps=2,
            save_start_step=0,
            save_interval=2,
            diagnostic_interval=1,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization=discretization,
    )


@pytest.mark.parametrize("family", ("periodic", "channel"))
@pytest.mark.parametrize(
    "missing_key",
    ("material", "ldg_l1", "activity_amplitude"),
)
def test_missing_runtime_model_parameters_are_structured_compile_rejections(
    tmp_path,
    family,
    missing_key,
):
    simulation = _simulation(tmp_path, family)
    parameters = dict(simulation.model.parameters)
    parameters.pop(missing_key)
    malformed_model = replace(simulation.model, parameters=parameters)
    malformed = replace(simulation, model=malformed_model)

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(malformed)

    error = caught.value
    assert error.code is (
        PublicCompilationRejectionCode.MALFORMED_APPLICATION_SPEC
    )
    assert error.context == {
        "boundary": "public_compiler_dispatch",
        "exception_type": "KeyError",
        "missing_key": missing_key,
    }
    assert not (tmp_path / f"{family}_output").exists()


@pytest.mark.parametrize("family", ("periodic", "channel"))
@pytest.mark.parametrize(
    "missing_key",
    ("ldg_a", "ldg_b", "ldg_c", "gamma", "flow_alignment", "beta"),
)
def test_missing_material_members_are_structured_compile_rejections(
    tmp_path,
    family,
    missing_key,
):
    simulation = _simulation(tmp_path, family)
    parameters = dict(simulation.model.parameters)
    material = dict(parameters["material"])
    material.pop(missing_key)
    parameters["material"] = material
    malformed = replace(
        simulation,
        model=replace(simulation.model, parameters=parameters),
    )

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(malformed)

    assert caught.value.code is (
        PublicCompilationRejectionCode.MALFORMED_APPLICATION_SPEC
    )
    assert caught.value.context["missing_key"] == missing_key
    assert not (tmp_path / f"{family}_output").exists()


@pytest.mark.parametrize("family", ("periodic", "channel"))
def test_snapshot_shape_mismatch_is_rejected_during_public_compilation(
    tmp_path,
    family,
):
    simulation = _simulation(
        tmp_path,
        family,
        snapshot_shape=(8, 8, 5, 5),
    )

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(simulation)

    error = caught.value
    assert error.code is (
        PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT
    )
    assert "Q snapshot shape does not match the declared geometry" in str(error)
    assert "observed=(8, 8, 5, 5)" in str(error)
    assert not (tmp_path / f"{family}_output").exists()


@pytest.mark.parametrize("family", ("periodic", "channel"))
@pytest.mark.parametrize(
    "snapshot_shape",
    ((*_GRID_SHAPE, 5), (5, *_GRID_SHAPE)),
)
def test_both_canonical_snapshot_layouts_pass_compilation(
    tmp_path,
    family,
    snapshot_shape,
):
    compiled = compile_simulation(
        _simulation(tmp_path, family, snapshot_shape=snapshot_shape)
    )

    assert compiled.application == (
        "periodic_complete_stress_beris_edwards"
        if family == "periodic"
        else "channel_complete_stress_beris_edwards"
    )
    assert not (tmp_path / f"{family}_output").exists()


@pytest.mark.parametrize("family", ("periodic", "channel"))
def test_unreadable_existing_snapshot_header_is_a_structured_rejection(
    tmp_path,
    family,
):
    simulation = _simulation(tmp_path, family, snapshot_shape=None)
    snapshot = Path(simulation.initial_condition.parameters["directory"])
    (snapshot / "Q_0.npy").write_bytes(b"not-a-numpy-array")

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(simulation)

    assert caught.value.code is (
        PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT
    )
    assert "Q snapshot header could not be read" in str(caught.value)
    assert not (tmp_path / f"{family}_output").exists()


@pytest.mark.parametrize("family", ("periodic", "channel"))
def test_unstaged_snapshot_remains_valid_for_allocation_free_planning(
    tmp_path,
    family,
):
    compiled = compile_simulation(
        _simulation(tmp_path, family, snapshot_shape=None)
    )

    assert compiled.application.endswith("complete_stress_beris_edwards")
    assert not (tmp_path / f"{family}_output").exists()


@pytest.mark.parametrize("family", ("periodic", "channel"))
@pytest.mark.parametrize(
    ("parameter_path", "invalid"),
    (
        (("material", "gamma"), "not-a-number"),
        (("material", "gamma"), -1.0),
        (("ldg_l1",), 0.0),
        (("activity_amplitude",), []),
        (("stokes", "parameters", "viscosity"), -1.0),
    ),
)
def test_invalid_runtime_parameter_values_are_compile_rejections(
    tmp_path,
    family,
    parameter_path,
    invalid,
):
    simulation = _simulation(tmp_path, family)
    parameters = dict(simulation.model.parameters)
    cursor = parameters
    for name in parameter_path[:-1]:
        child = dict(cursor[name])
        cursor[name] = child
        cursor = child
    cursor[parameter_path[-1]] = invalid
    malformed = replace(
        simulation,
        model=replace(simulation.model, parameters=parameters),
    )

    with pytest.raises(PublicSimulationCompilationError) as captured:
        compile_simulation(malformed)

    assert captured.value.code is (
        PublicCompilationRejectionCode.MALFORMED_APPLICATION_SPEC
    )
    assert captured.value.context["exception_type"] in {"TypeError", "ValueError"}
    assert not (tmp_path / f"{family}_output").exists()


@pytest.mark.parametrize("family", ("periodic", "channel"))
@pytest.mark.parametrize("kind", ("missing", "dtype", "nonfinite"))
def test_runtime_snapshot_failures_use_public_input_error(
    tmp_path,
    family,
    kind,
):
    simulation = _simulation(tmp_path, family, snapshot_shape=None)
    snapshot = Path(simulation.initial_condition.parameters["directory"])
    path = snapshot / "Q_0.npy"
    if kind == "dtype":
        np.save(path, np.zeros((*_GRID_SHAPE, 5), dtype=np.float32))
    elif kind == "nonfinite":
        values = np.zeros((*_GRID_SHAPE, 5), dtype=np.float64)
        values[0, 0, 0, 0] = np.nan
        np.save(path, values)
    compiled = compile_simulation(simulation)

    with pytest.raises(SimulationInputError) as captured:
        run_simulation(compiled)

    error = captured.value
    assert error.application_family == (
        "periodic_complete_stress_beris_edwards"
        if family == "periodic"
        else "channel_complete_stress_beris_edwards"
    )
    assert error.stage == "initial_condition"
    assert error.step == 0
    assert error.details["reason"] == {
        "missing": "snapshot_unreadable",
        "dtype": "snapshot_dtype_mismatch",
        "nonfinite": "snapshot_nonfinite",
    }[kind]
    assert error.to_metadata()["code"] == "simulation_input_error"
    assert not (tmp_path / f"{family}_output").exists()
