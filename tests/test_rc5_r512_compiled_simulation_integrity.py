"""R5-12 fail-closed identity for public compiled simulations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType

import pytest

from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
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


def _simulation(directory: Path) -> Simulation:
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
        shape=(16, 16, 8),
        lengths=(100.0, 100.0, 20.0),
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
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
            directory=directory,
            steps=2,
            save_interval=1,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        invocation=InvocationSpec({"dry_run": True}),
    )


def test_compilation_identity_is_stable_and_exposed(tmp_path):
    first = compile_simulation(_simulation(tmp_path / "first"))
    second = compile_simulation(_simulation(tmp_path / "first"))

    assert first.compilation_identity_sha256 == second.compilation_identity_sha256
    assert len(first.compilation_identity_sha256) == 64
    assert first.to_metadata()["compilation_identity_sha256"] == (
        first.compilation_identity_sha256
    )
    first.validate_integrity()


@pytest.mark.parametrize(
    "field",
    (
        "source",
        "application_specification",
        "lowering_plan",
        "construction_plan",
        "application_request",
    ),
)
def test_run_rejects_replaced_compilation_components_before_dispatch(
    tmp_path,
    monkeypatch,
    field,
):
    original = compile_simulation(_simulation(tmp_path / "original"))
    alternate = compile_simulation(_simulation(tmp_path / "alternate"))
    tampered = replace(original, **{field: getattr(alternate, field)})
    invoked = False
    fake = ModuleType("pssolver.applications.plane_beris_edwards")

    def forbidden_dispatch(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("tampered compilation reached application dispatch")

    fake.run_plane_beris_edwards = forbidden_dispatch
    monkeypatch.setitem(
        sys.modules,
        "pssolver.applications.plane_beris_edwards",
        fake,
    )

    with pytest.raises(ValueError, match="compiled simulation integrity mismatch"):
        run_simulation(tampered)

    assert invoked is False
    assert not (tmp_path / "alternate").exists()


@pytest.mark.parametrize(
    "replacement",
    (
        {"application": "unqualified_application"},
        {"normalization": {"runtime_fallback_allowed": True}},
        {"compilation_identity_sha256": "0" * 64},
    ),
)
def test_run_rejects_replaced_identity_fields(tmp_path, replacement):
    compiled = compile_simulation(_simulation(tmp_path / "run"))
    tampered = replace(compiled, **replacement)

    with pytest.raises(ValueError, match="compiled simulation integrity mismatch"):
        run_simulation(tampered)
    with pytest.raises(ValueError, match="compiled simulation integrity mismatch"):
        tampered.to_metadata()

