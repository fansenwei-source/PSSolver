"""RC4.3.3 public lifting capability restriction regressions."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import pssolver
from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
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
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)
from pssolver.configuration.simulation_binding import bind_simulation_runtime
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    prescribed_q,
    strong_homeotropic_q,
    strong_planar_q,
)


ANALYSIS_SHA256 = (
    "1157b08b5328996876307fb0333aa28e85932d9d09e9ed0f16f08cc7a3a62ef1"
)
REJECTION_CODE = (
    PublicCompilationRejectionCode.UNSUPPORTED_LIFTED_WALL_MOLECULAR_FIELD
)


def _model() -> CompleteStressBerisEdwards:
    return CompleteStressBerisEdwards(
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


def _prescribed_policy(kind: str):
    normals = {
        (2, "lower"): (0.0, 0.0, -1.0),
        (2, "upper"): (0.0, 0.0, 1.0),
    }
    if kind == "prescribed_q":
        lower = np.diag((0.4, -0.2, -0.2))
        upper = np.diag((-0.2, 0.4, -0.2))
        return prescribed_q(
            {(2, "lower"): lower, (2, "upper"): upper}
        )
    if kind == "strong_homeotropic_q":
        return strong_homeotropic_q(
            scalar_order=0.6,
            face_normals=normals,
        )
    if kind == "strong_planar_q":
        return strong_planar_q(
            scalar_order=0.6,
            face_directors={
                (2, "lower"): (1.0, 0.0, 0.0),
                (2, "upper"): (0.0, 1.0, 0.0),
            },
            face_normals=normals,
        )
    raise AssertionError(f"unknown test policy {kind}")


def _simulation(
    tmp_path: Path,
    *,
    runtime_path: str = "legacy_production",
    molecular_field_linear_space: str = "spectral",
    q_policy=None,
    output_name: str = "run",
) -> Simulation:
    model = _model()
    geometry = PlaneSlab(
        shape=(8, 8, 6),
        lengths=(100.0, 100.0, 20.0),
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q() if q_policy is None else q_policy,
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    return Simulation(
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
                "defect_min_separation": 10.0,
                "defect_core_radius": 1.5,
                "background_angle": 0.0,
                "twist_amplitude": 0.01,
                "twist_modes": [1, 2, 3],
                "initial_s": 1.0 / 3.0,
            },
        ),
        execution=TorchSpectralExecution(
            runtime_path=runtime_path,
            device="cpu",
            options={
                "tf32": "off",
                "fallback_allowed": False,
                "molecular_field_linear_space": (
                    molecular_field_linear_space
                ),
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / output_name,
            steps=1,
            save_start_step=0,
            save_interval=1,
            diagnostic_interval=1,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
    )


def test_catalog_keeps_lifting_declarations_without_execution_claims():
    policies = {
        value.key: value
        for value in pssolver.available_boundary_policies()
    }
    for name in (
        "prescribed_q",
        "strong_homeotropic_q",
        "strong_planar_q",
    ):
        declaration = policies[name]
        assert declaration.constructor.startswith("pssolver.")
        assert declaration.executable is False
        assert declaration.qualified_applications == ()
    assert "plane_complete_stress_beris_edwards" in policies[
        "neumann_q"
    ].qualified_applications


@pytest.mark.parametrize(
    "runtime_path", ("legacy_production", "compiled_v2")
)
@pytest.mark.parametrize(
    "linear_space", ("physical", "spectral")
)
@pytest.mark.parametrize(
    "policy_kind",
    ("prescribed_q", "strong_homeotropic_q", "strong_planar_q"),
)
def test_complete_static_prescribed_q_rejects_with_stable_public_contract(
    tmp_path,
    runtime_path,
    linear_space,
    policy_kind,
):
    simulation = _simulation(
        tmp_path,
        runtime_path=runtime_path,
        molecular_field_linear_space=linear_space,
        q_policy=_prescribed_policy(policy_kind),
        output_name=f"{runtime_path}_{linear_space}_{policy_kind}",
    )

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(simulation)

    error = caught.value
    assert error.code is REJECTION_CODE
    assert error.code.value == "unsupported_lifted_wall_molecular_field"
    assert "nonzero wall molecular-field semantics" in str(error)
    assert dict(error.context) == {
        "equation_variant": "complete_stress_beris_edwards",
        "geometry_name": "plane_slab",
        "field_name": "Q",
        "boundary_semantics": (
            "static_constant_two_sided_prescribed_dirichlet"
        ),
        "requested_runtime_path": runtime_path,
        "requested_molecular_field_linear_space": linear_space,
        "supported_Q_boundary": "homogeneous_neumann",
        "analysis_record_sha256": ANALYSIS_SHA256,
    }
    json.dumps(error.to_metadata(), allow_nan=False, sort_keys=True)
    assert not Path(simulation.output.options["output_dir"]).exists()


def test_lifted_rejection_precedes_run_spec_lowering_and_construction(
    tmp_path,
    monkeypatch,
):
    import pssolver.configuration.public_simulation_runner as runner

    def forbidden(*_args, **_kwargs):
        raise AssertionError("post-rejection construction was reached")

    monkeypatch.setattr(
        runner,
        "create_plane_beris_edwards_run_spec",
        forbidden,
    )
    monkeypatch.setattr(runner, "lower_simulation_spec", forbidden)
    monkeypatch.setattr(
        runner,
        "plan_package_runtime_construction",
        forbidden,
    )
    simulation = _simulation(
        tmp_path,
        q_policy=_prescribed_policy("strong_homeotropic_q"),
    )

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(simulation)

    assert caught.value.code is REJECTION_CODE


def test_partial_prescribed_q_uses_existing_boundary_validation(tmp_path):
    model = _model()
    geometry = PlaneSlab(
        shape=(8, 8, 6),
        lengths=(100.0, 100.0, 20.0),
    )
    partial = prescribed_q(
        {(2, "lower"): np.diag((0.4, -0.2, -0.2))}
    )

    with pytest.raises(
        ValueError,
        match="prescribed Dirichlet values must cover exactly",
    ):
        assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": partial,
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        )


@pytest.mark.parametrize(
    "runtime_path", ("legacy_production", "compiled_v2")
)
@pytest.mark.parametrize(
    "linear_space", ("physical", "spectral")
)
def test_supported_neumann_plane_reports_requested_and_effective_space(
    tmp_path,
    runtime_path,
    linear_space,
):
    compiled = compile_simulation(
        _simulation(
            tmp_path,
            runtime_path=runtime_path,
            molecular_field_linear_space=linear_space,
            output_name=f"neumann_{runtime_path}_{linear_space}",
        )
    )

    assert compiled.lowering_plan.lifting_plan is None
    assert compiled.normalization[
        "requested_molecular_field_linear_space"
    ] == linear_space
    assert compiled.normalization[
        "effective_molecular_field_linear_space"
    ] == linear_space
    assert compiled.normalization[
        "molecular_field_space_fallback_used"
    ] is False


def test_internal_lifting_lowering_and_legacy_binding_remain_available(tmp_path):
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "unused",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        runtime_path="legacy_production",
        nx=8,
        ny=8,
        nz=6,
        lx=100.0,
        ly=100.0,
        height=20.0,
        steps=1,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        defect_min_separation=10.0,
        defect_core_radius=1.5,
        twist_modes=(1, 2, 3),
    )
    base = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )
    lifted_boundaries = assign_boundaries(
        model=base.equation_system,
        geometry=base.geometry,
        policies={
            "Q": _prescribed_policy("strong_homeotropic_q"),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    internal = replace(base, boundaries=lifted_boundaries)

    assert lower_simulation_spec(internal).lifting_plan is not None
    assert bind_simulation_runtime(internal).runtime_path == (
        "legacy_production"
    )
