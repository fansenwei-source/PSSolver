"""P2.5 pre-migration oracles for the legacy Plane runtime consumer."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime import (
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)
from pssolver.runtime import plane_legacy
from pssolver.workflows import (
    capture_plane_checkpoint,
    restore_plane_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RUNTIME_PATH = PROJECT_ROOT / "pssolver" / "runtime" / "plane_legacy.py"

FLAT_RUNTIME_READS = {
    "beta",
    "dealias_rule",
    "diagnostics",
    "disable_q_gradient_reuse",
    "dt",
    "dtype",
    "eta",
    "flow_alignment",
    "friction_mode_fric",
    "height",
    "ldg_a",
    "ldg_b",
    "ldg_c",
    "lx",
    "ly",
    "molecular_field_linear_space",
    "nx",
    "ny",
    "nz",
    "pointwise_execution",
    "projected_transform_execution",
    "shendruk_preset",
    "spectral_refresh_interval_steps",
    "spectral_storage",
    "stress_divergence_sum_space",
    "transform_execution_order",
    "zero_mode_policy",
}
COMPONENT_OWNERS = {
    "geometry": {"height", "lx", "ly", "nx", "ny", "nz"},
    "numerics": {
        "dealias_rule",
        "dtype",
        "projected_transform_execution",
        "spectral_storage",
        "transform_execution_order",
    },
    "physics": {
        "beta",
        "eta",
        "flow_alignment",
        "friction_mode_fric",
        "ldg_a",
        "ldg_b",
        "ldg_c",
        "shendruk_preset",
        "zero_mode_policy",
    },
    "time_stepping": {"dt", "spectral_refresh_interval_steps"},
    "execution": {
        "disable_q_gradient_reuse",
        "molecular_field_linear_space",
        "pointwise_execution",
        "stress_divergence_sum_space",
    },
    "workflow": {"diagnostics"},
}


def _spec(tmp_path: Path, **overrides: object):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / "unused",
        "device": "cpu",
        "dtype": "float64",
        "pointwise_execution": "eager",
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "lx": 8.0,
        "ly": 9.0,
        "height": 20.0,
        "steps": 4,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "spectral_refresh_steps": 3,
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _initial_values() -> dict[str, torch.Tensor]:
    coordinate = torch.arange(8**3, dtype=torch.float64).reshape(8, 8, 8)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


def _request(spec) -> PlaneRuntimeBuildRequest:
    return PlaneRuntimeBuildRequest(
        run_spec=spec,
        production_metadata={
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        },
        initial_values=_initial_values(),
        device="cpu",
    )


def _build(spec):
    return build_plane_beris_edwards_runtime(_request(spec))


def _run_spec_reads() -> set[str]:
    tree = ast.parse(
        LEGACY_RUNTIME_PATH.read_text(encoding="utf-8"),
        filename=str(LEGACY_RUNTIME_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_legacy_plane_runtime"
    )
    return {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "run_spec"
    }


def test_legacy_runtime_consumes_one_component_graph_without_flat_field_reads(
    tmp_path,
    monkeypatch,
):
    assert set().union(*COMPONENT_OWNERS.values()) == FLAT_RUNTIME_READS
    assert _run_spec_reads() == set()

    spec = _spec(tmp_path)
    calls = []
    original = plane_legacy.decompose_plane_beris_edwards_run_spec

    def spy(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(
        plane_legacy,
        "decompose_plane_beris_edwards_run_spec",
        spy,
    )
    plane_legacy.build_legacy_plane_runtime(
        spec,
        device="cpu",
        initial_values=_initial_values(),
    )
    assert calls == [spec]


def test_legacy_runtime_object_graph_preserves_all_resolved_choices(tmp_path):
    spec = _spec(
        tmp_path,
        zero_mode_policy="friction",
        friction_mode_fric=0.125,
        flow_alignment=0.31,
        eta=0.7,
        beta=-0.9,
        molecular_field_linear_space="physical",
        stress_divergence_sum_space="physical",
        dealias_rule="two_thirds",
        projected_transform_execution="full",
        transform_execution_order="legacy",
        spectral_storage="full_complex",
        diagnostics=True,
        disable_q_gradient_reuse=True,
    )
    solver, projector = plane_legacy.build_legacy_plane_runtime(
        spec,
        device="cpu",
        initial_values=_initial_values(),
    )

    assert solver.shape == (8, 8, 8)
    assert solver.L == (8.0, 9.0, 20.0)
    assert solver.dt == spec.dt
    assert solver.dtype is torch.float64
    assert solver.transform_backend.execution_order == "legacy"
    assert solver.transform_backend.spectral_storage == "full_complex"
    # Full-complex storage has no Hermitian truncation axis.  P7.7.11 makes
    # the allocated backend identity agree with the resolved numerics instead
    # of retaining the geometrical half-spectrum axis as unused metadata.
    assert solver.transform_backend.hermitian_axis is None
    assert solver.transform_backend.hermitian_axis is (
        spec.numerics.hermitian_axis
    )
    assert projector.rule == "two_thirds"
    assert projector.transform_execution == "full"
    assert tuple(solver.fields.name_to_idx) == (
        *Q_COMPONENTS,
        "ux",
        "uy",
        "uz",
        "p",
    )
    assert [value[0] for value in solver.model.dyn_fields] == list(Q_COMPONENTS)
    assert [value[0] for value in solver.model.stat_fields] == [
        "ux",
        "uy",
        "uz",
        "p",
    ]

    nonlinear = solver.model.nlmodel
    assert nonlinear.q_boundary_conditions == plane_legacy.Q_BOUNDARIES
    assert nonlinear.flow_alignment == 0.31
    assert nonlinear.q_gradient_cache is None
    assert nonlinear.pointwise_kernels.effective_execution == "eager"

    stokes = solver.model.static_model
    assert stokes.friction == 0.125
    assert stokes.viscosity == 0.7
    assert stokes.zero_mode_policy == "friction"
    assert stokes.beta == -0.9
    assert stokes.ldg_a == spec.ldg_a
    assert stokes.ldg_b == spec.ldg_b
    assert stokes.ldg_c == spec.ldg_c
    assert stokes.ldg_l1 == spec.shendruk_preset.ldg_l1
    assert stokes.flow_alignment == 0.31
    assert stokes.molecular_field_linear_space == "physical"
    assert stokes.stress_divergence_sum_space == "physical"
    assert stokes.cache_force_diagnostics is True
    assert stokes.pressure_diagnostics is True
    assert solver.integrator.spectral_refresh_interval == 3
    assert solver.integrator._static_fields_are_current is False
    assert solver.model.parameters["alpha"].item() == (
        spec.shendruk_preset.zeta
    )


def test_short_cpu_trajectory_and_checkpoint_restart_are_byte_exact(tmp_path):
    spec = _spec(tmp_path)
    continuous = _build(spec)
    segment = _build(spec)
    resumed = _build(spec)

    continuous.advance(4)
    segment.advance(2)
    checkpoint = capture_plane_checkpoint(
        segment,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    assert restore_plane_checkpoint(
        resumed,
        checkpoint,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    ) == 2
    resumed.advance(2)
    continuous.synchronize_for_observation()
    resumed.synchronize_for_observation()

    assert continuous.completed_steps == resumed.completed_steps == 4
    assert continuous.solver.integrator.step_count == (
        resumed.solver.integrator.step_count
    )
    assert continuous.solver.integrator.refresh_count == (
        resumed.solver.integrator.refresh_count
    )
    assert torch.equal(
        continuous.solver.fields.spatial,
        resumed.solver.fields.spatial,
    )
    assert torch.equal(
        continuous.solver.fields.spectral,
        resumed.solver.fields.spectral,
    )
