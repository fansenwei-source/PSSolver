"""P7.7.9 public compiler and runner connection contract."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import pssolver
from pssolver import (
    CompiledSimulation,
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
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_APPLICATION,
    PUBLIC_PLANE_APPLICATION,
    PublicSimulationCompilationError,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.planning.construction import RuntimeConstructionKind


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_7_p779_public_runner_connection.json"
)
def _simulation(**replacements) -> Simulation:
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
    values = {
        "model": model,
        "geometry": geometry,
        "boundaries": assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
        "numerics": SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
        ),
        "time": TimeStepping(
            dt=0.005,
            refresh={"mode": "disabled"},
        ),
        "initial_condition": GeneratedInitialCondition(
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
        "execution": TorchSpectralExecution(
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
        "output": Output(
            directory="data/public_runner_test",
            steps=2,
            save_interval=1,
            diagnostic_interval=1,
        ),
        "invocation": InvocationSpec({"dry_run": True}),
    }
    values.update(replacements)
    return Simulation(**values)


def test_compile_simulation_binds_exact_existing_plane_application():
    source = _simulation()
    compiled = compile_simulation(source)

    assert isinstance(compiled, CompiledSimulation)
    assert compiled.source is source
    assert compiled.application == PUBLIC_PLANE_APPLICATION
    assert compiled.construction_plan.kind is (
        RuntimeConstructionKind.PLANE_COMPILED_V2
    )
    assert compiled.application_request.runtime_path.value == "compiled_v2"
    assert compiled.application_request.device == "cpu"
    assert compiled.application_request.dry_run is True
    assert compiled.application_request.parameterization == "fixed-k"
    assert compiled.application_request.output_dir == Path(
        "data/public_runner_test"
    )
    assert compiled.application_request.numerics.to_metadata() == (
        source.numerics.to_metadata()
    )
    assert compiled.lowering_plan.source_simulation_sha256 == (
        compiled.application_specification.canonical_sha256()
    )
    metadata = compiled.to_metadata()
    assert metadata["source_simulation_sha256"] == source.canonical_sha256()
    assert metadata["fallback_allowed"] is False
    assert metadata["normalization"]["runtime_fallback_allowed"] is False
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_coefficient_translation_preserves_requested_physics_with_roundoff_only():
    compiled = compile_simulation(_simulation())
    normalization = compiled.normalization

    assert normalization["requested_activity"] == 0.01
    assert normalization["effective_activity"] == pytest.approx(
        0.01, rel=2.0e-14, abs=2.0e-15
    )
    assert normalization["requested_ldg_l1"] == 1.0 / 81.0
    assert normalization["effective_ldg_l1"] == pytest.approx(
        1.0 / 81.0, rel=2.0e-14, abs=2.0e-15
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (
            {
                "initial_condition": GeneratedInitialCondition(
                    "extruded_defect_gas", parameters={"seed": 24}
                )
            },
            "must declare every qualified option explicitly",
        ),
        (
            {
                "time": TimeStepping(dt=0.005),
            },
            "time refresh must be exactly one of",
        ),
        (
            {
                "execution": TorchSpectralExecution(
                    runtime_path="separated_canary",
                    device="cpu",
                    options={
                        "tf32": "off",
                        "molecular_field_linear_space": "spectral",
                        "stress_divergence_sum_space": "spectral",
                        "pointwise_execution": "eager",
                        "disable_q_gradient_reuse": False,
                    },
                )
            },
            "supports legacy_production or compiled_v2",
        ),
    ),
)
def test_compile_simulation_fails_closed_without_guessing(replacement, message):
    with pytest.raises(PublicSimulationCompilationError, match=message):
        compile_simulation(_simulation(**replacement))


def test_compile_simulation_rejects_unqualified_negative_activity():
    original = _simulation()
    material = original.model.parameters["material"]
    invalid = CompleteStressBerisEdwards(
        ldg_a=material["ldg_a"],
        ldg_b=material["ldg_b"],
        ldg_c=material["ldg_c"],
        ldg_l1=original.model.parameters["ldg_l1"],
        gamma=material["gamma"],
        flow_alignment=material["flow_alignment"],
        activity=-0.01,
        beta=material["beta"],
        viscosity=original.model.parameters["stokes"]["parameters"][
            "viscosity"
        ],
    )
    boundaries = assign_boundaries(
        model=invalid,
        geometry=original.geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    with pytest.raises(PublicSimulationCompilationError, match="positive activity"):
        compile_simulation(
            _simulation(model=invalid, boundaries=boundaries)
        )


def test_compiled_product_is_immutable():
    compiled = compile_simulation(_simulation())
    with pytest.raises(FrozenInstanceError):
        compiled.application = "other"
    with pytest.raises(TypeError):
        compiled.normalization["runtime_fallback_allowed"] = True


def test_run_simulation_dispatches_once_to_existing_application(monkeypatch):
    compiled = compile_simulation(_simulation())
    observed = {}
    fake = ModuleType("pssolver.applications.plane_beris_edwards")

    def run_plane_beris_edwards(run_spec, *, progress, emit_metadata):
        observed.update(
            run_spec=run_spec,
            progress=progress,
            emit_metadata=emit_metadata,
        )
        return SimpleNamespace(
            start_step=0,
            final_step=2,
            elapsed_seconds=1.25,
            saved_steps=(1, 2),
            checkpoint_steps=(),
            final_observation=SimpleNamespace(
                step=2,
                q=np.zeros((2, 2, 2, 5)),
                velocity=np.zeros((2, 2, 2, 3)),
                pressure=np.zeros((2, 2, 2)),
            ),
            diagnostics=(),
        )

    fake.run_plane_beris_edwards = run_plane_beris_edwards
    monkeypatch.setitem(
        sys.modules,
        "pssolver.applications.plane_beris_edwards",
        fake,
    )
    progress = (0, 1)
    result = run_simulation(
        compiled,
        progress=progress,
        emit_metadata=True,
    )

    assert result.completed is True
    assert result.application == PUBLIC_PLANE_APPLICATION
    assert result.final_step == 2
    assert result.final_observation.step == 2
    assert observed == {
        "run_spec": compiled.application_request,
        "progress": progress,
        "emit_metadata": True,
    }


def test_run_simulation_represents_plane_dry_run_without_false_outputs(
    monkeypatch,
):
    compiled = compile_simulation(_simulation())
    fake = ModuleType("pssolver.applications.plane_beris_edwards")

    def run_plane_beris_edwards(
        run_spec,
        *,
        progress,
        emit_metadata,
    ):
        assert run_spec is compiled.application_request
        assert progress is None
        assert emit_metadata is False
        return None

    fake.run_plane_beris_edwards = run_plane_beris_edwards
    monkeypatch.setitem(
        sys.modules,
        "pssolver.applications.plane_beris_edwards",
        fake,
    )

    result = run_simulation(compiled)

    assert result.status.value == "dry_run"
    assert result.completed is False
    assert result.output_directory is None
    assert result.final_observation is None
    assert result.saved_steps == ()


def test_public_runner_has_no_top_level_application_or_runtime_import():
    forbidden = {
        "pssolver.applications",
        "pssolver.runtime",
        "pssolver.workflows",
    }
    path = ROOT / "pssolver/api/runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    top_level_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not {
        name
        for name in top_level_imports
        if any(name == item or name.startswith(f"{item}.") for item in forbidden)
    }
    assert "pssolver.applications.plane_beris_edwards" in imports


def test_public_exports_include_compile_and_run_connection():
    expected = {
        "CompiledSimulation",
        "compile_simulation",
        "run_simulation",
    }
    assert expected <= set(pssolver.__all__)
    assert pssolver.compile_simulation is compile_simulation
    assert pssolver.run_simulation is run_simulation


def test_machine_record_binds_scope_and_source_identity():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_P7_7_9_TWO_COMBINATION_PUBLIC_COMPILER"
    )
    assert record["p7_7_10_complete"] is False
    assert record["qualified_applications"] == [
        PUBLIC_PLANE_APPLICATION,
        PUBLIC_CHANNEL_APPLICATION,
    ]
    assert record["runtime_fallback_allowed"] is False
    assert record["phase_8_authorized"] is False
    assert record["production_default_changed"] is False
    assert record["source_sha256"] == {
        "pssolver/api/runner.py": (
            "8b72c00725964e6d508a90b0a21bed0ed8a4e58034ea4adc3a214585ef11265a"
        ),
        "pssolver/configuration/public_channel_simulation_compiler.py": (
            "6fb5714b44d4a115837d57167f7a2c8457ff41eaea8ffe4abfe0374ab8741dfc"
        ),
        "pssolver/configuration/public_simulation_runner.py": (
            "6ead3e143f1913e076425cc98776abf5ff3ff626d56092718d466f8e51b1acaa"
        ),
    }
