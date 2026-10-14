"""RC5 structured public-compilation error boundary regressions."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

import pssolver
import pssolver.api
from pssolver import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
    Simulation,
    TorchSpectralExecution,
    compile_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    robin,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation import (
    ExecutionSpec,
    TimeIntegrationSpec,
)


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


def _public(specification) -> Simulation:
    execution_options = dict(specification.execution.options)
    device = execution_options.pop("device")
    return Simulation(
        model=specification.equation_system,
        geometry=specification.geometry,
        boundaries=specification.boundaries,
        numerics=specification.numerics,
        time=specification.time_integration,
        initial_condition=specification.initial_condition,
        execution=TorchSpectralExecution(
            runtime_path=specification.execution.runtime_path,
            device=device,
            options=execution_options,
        ),
        output=specification.workflow,
        # The compositional public Plane surface intentionally exposes no
        # low-level discretization overrides.
        discretization={},
        invocation=specification.invocation,
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
    assert pssolver.api.PublicSimulationCompilationError is (
        PublicSimulationCompilationError
    )
    assert pssolver.PublicCompilationRejectionCode is (
        PublicCompilationRejectionCode
    )
    assert pssolver.api.PublicCompilationRejectionCode is (
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
    source = _public(replace(base, boundaries=boundaries))
    _forbid_tensor_allocation(monkeypatch)

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(source)

    error = caught.value
    assert error.code is PublicCompilationRejectionCode.BOUNDARY_CONTRACT
    assert error.context["robin_impedance"]["dimensionless_impedance"] == (
        "positive_infinity"
    )
    assert json.loads(json.dumps(error.to_metadata(), allow_nan=False)) == (
        error.to_metadata()
    )
    assert type(error.__cause__).__name__ == "RobinImpedanceSupportError"
    assert not (tmp_path / "must_not_materialize").exists()


def test_missing_stokes_is_translated_without_allocation_or_output(
    tmp_path,
    monkeypatch,
):
    base = _specification(tmp_path)
    parameters = dict(base.equation_system.parameters)
    parameters.pop("stokes")
    malformed_model = replace(base.equation_system, parameters=parameters)
    source = _public(replace(base, equation_system=malformed_model))
    _forbid_tensor_allocation(monkeypatch)

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(source)

    error = caught.value
    assert error.code is (
        PublicCompilationRejectionCode.MALFORMED_APPLICATION_SPEC
    )
    assert error.context == {
        "boundary": "public_compiler_dispatch",
        "exception_type": "KeyError",
        "missing_key": "stokes",
    }
    assert isinstance(error.__cause__, KeyError)
    assert error.to_metadata()["message"] == (
        "the public application declaration is malformed"
    )
    assert not (tmp_path / "must_not_materialize").exists()


def test_public_rejection_codes_distinguish_contract_domains(
    tmp_path,
):
    source = _public(_specification(tmp_path))
    cases = (
        (
                replace(
                    source,
                    execution=ExecutionSpec(
                        backend=source.execution.backend,
                        runtime_path="periodic_spectral",
                        options=source.execution.options,
                    ),
            ),
            PublicCompilationRejectionCode.EXECUTION_CONTRACT,
        ),
        (
                replace(
                    source,
                    time=TimeIntegrationSpec(
                        integrator=source.time.integrator,
                        refresh={"mode": "unsupported"},
                    ),
            ),
            PublicCompilationRejectionCode.TIME_INTEGRATION_CONTRACT,
        ),
    )

    for invalid, expected in cases:
        with pytest.raises(PublicSimulationCompilationError) as caught:
            compile_simulation(invalid)
        assert caught.value.code is expected
        assert caught.value.code is not (
            PublicCompilationRejectionCode.APPLICATION_CONTRACT
        )
        assert not (tmp_path / "must_not_materialize").exists()
