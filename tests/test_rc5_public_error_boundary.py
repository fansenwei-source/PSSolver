"""RC5 structured public-compilation error boundary regressions."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

import pssolver
import pssolver.api
from pssolver import (
    CompleteStressBerisEdwards,
    GeneratedInitialCondition,
    Output,
    PeriodicBox,
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
    SemiImplicitEuler,
    ShendrukPlaneBerisEdwards,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TorchSpectralExecution,
    compile_simulation,
    neumann_q,
)
from pssolver.boundaries import free_slip_velocity, neumann_pressure_compatibility, robin
from pssolver.functional.legacy_declarations import assign_boundaries
from pssolver.functional.legacy_declarations.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.functional.legacy_declarations.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.functional.legacy_declarations.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.functional.legacy_declarations.public_simulation_runner import _json_safe
from pssolver.functional.legacy_declarations.robin import validate_public_robin_impedance_support


def _specification(tmp_path):
    request = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "must_not_materialize",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        nx=8,
        ny=8,
        nz=6,
        steps=2,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        defect_min_separation=2.0,
        defect_core_radius=0.5,
        twist_modes=(1, 2),
    )
    return compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(request)
    )


def _public(tmp_path, specification) -> Simulation:
    return Simulation(
        model=ShendrukPlaneBerisEdwards(activity_number=18.0),
        geometry=specification.geometry,
        boundaries={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
        numerics=SpectralNumerics(dtype="float64", spectral_storage="full_complex"),
        time=SemiImplicitEuler(0.01),
        initial_condition=GeneratedInitialCondition("extruded_defect_gas"),
        execution=TorchSpectralExecution(device="cpu"),
        output=Output(
            directory=tmp_path / "must_not_materialize", steps=2,
            save_interval=1, diagnostic_interval=1, save_start_step=0, diagnostics=False,
            save_hydrodynamics=False,
        ),
    )


def _forbid_tensor_allocation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("public rejection must precede tensor allocation")

    for name in ("empty", "ones", "tensor", "zeros"):
        monkeypatch.setattr(torch, name, forbidden)


def test_public_compilation_error_and_code_are_exported_from_both_surfaces():
    assert pssolver.PublicSimulationCompilationError is (
        PublicSimulationCompilationError
    )
    assert not hasattr(pssolver.api, "PublicSimulationCompilationError")
    assert pssolver.api.errors.PublicSimulationCompilationError is (
        PublicSimulationCompilationError
    )
    assert pssolver.PublicCompilationRejectionCode is (
        PublicCompilationRejectionCode
    )
    assert not hasattr(pssolver.api, "PublicCompilationRejectionCode")
    assert pssolver.api.errors.PublicCompilationRejectionCode is (
        PublicCompilationRejectionCode
    )


@pytest.mark.parametrize(
    ("alpha", "beta"),
    ((1.0, 5.0e-324), (1.0e308, 1.0e-10)),
)
def test_robin_overflow_has_json_safe_structured_context_before_side_effects(
    tmp_path,
    monkeypatch,
    alpha,
    beta,
):
    base = _specification(tmp_path)
    components = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
    q_boundary = robin(
        "Q",
        {
            (component, 2, side): (alpha, beta, 0.0)
            for component in components
            for side in ("lower", "upper")
        },
    )
    boundaries = assign_boundaries(
        model=base.equation_system,
        geometry=base.geometry,
        policies={
            "Q": q_boundary,
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    source = _public(tmp_path, base)
    source = replace(source, boundaries={**source.boundaries, "Q": q_boundary})
    _forbid_tensor_allocation(monkeypatch)

    with pytest.raises(ValueError) as legacy:
        validate_public_robin_impedance_support(replace(base, boundaries=boundaries))
    assert type(legacy.value).__name__ == "RobinImpedanceSupportError"
    context = _json_safe(legacy.value.context)
    assert context["dimensionless_impedance"] == "positive_infinity"
    assert json.loads(json.dumps(context, allow_nan=False)) == context

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(source)

    error = caught.value
    assert error.code is PublicCompilationRejectionCode.ROBIN_OR_ANCHORING
    assert error.kind == "scope"
    assert error.opens_in == "0.6"
    assert error.reason == "Robin and anchoring await general boundary support"
    assert not (tmp_path / "must_not_materialize").exists()


def test_missing_stokes_is_translated_without_allocation_or_output(
    tmp_path,
    monkeypatch,
):
    base = _specification(tmp_path)
    parameters = dict(base.equation_system.parameters)
    parameters.pop("stokes")
    malformed_model = replace(base.equation_system, parameters=parameters)
    _forbid_tensor_allocation(monkeypatch)
    # The typed 0.5 model does not accept the raw v0.4 parameter map.
    with pytest.raises(TypeError, match="unsupported Simulation model declaration"):
        replace(_public(tmp_path, base), model=malformed_model)
    assert not (tmp_path / "must_not_materialize").exists()


def test_public_rejection_codes_distinguish_contract_domains(
    tmp_path,
):
    source = _public(tmp_path, _specification(tmp_path))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    import numpy as np

    np.save(snapshot / "Q_0.npy", np.zeros((8, 8, 6, 5), dtype=np.float64))
    periodic = replace(
        source,
        model=CompleteStressBerisEdwards(
            ldg_a=0.0, ldg_b=-0.3, ldg_c=0.3, ldg_l1=0.01, gamma=2.94,
            flow_alignment=0.3, activity=0.01, beta=-1.0, viscosity=2.0 / 3.0,
        ),
        geometry=PeriodicBox(shape=(8, 8, 6), lengths=(8.0, 8.0, 6.0)),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        time=SemiImplicitEuler(0.01, refresh={"steps": 2}),
    )
    cases = (
        (
            replace(
                source,
                execution=TorchSpectralExecution(options={"runtime_path": "periodic_spectral"}),
            ),
            PublicCompilationRejectionCode.DECLARATION_CONTRACT,
        ),
        (periodic, PublicCompilationRejectionCode.REFRESH_NOT_SUPPORTED),
    )

    for invalid, expected in cases:
        with pytest.raises(PublicSimulationCompilationError) as caught:
            compile_simulation(invalid)
        assert caught.value.code is expected
        assert not (tmp_path / "must_not_materialize").exists()
