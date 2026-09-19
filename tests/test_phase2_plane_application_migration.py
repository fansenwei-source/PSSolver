"""P2.5 oracles for the Plane application consumer migration."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from pssolver.applications import plane_beris_edwards as application
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_PATH = (
    PROJECT_ROOT / "pssolver" / "applications" / "plane_beris_edwards.py"
)
FLAT_APPLICATION_READS = {
    "S_initial",
    "activity_number",
    "background_angle",
    "beta",
    "dealias_rule",
    "defect_core_radius",
    "defect_min_separation",
    "device",
    "diagnostic_interval",
    "diagnostics",
    "disable_q_gradient_reuse",
    "dry_run",
    "dt",
    "dtype",
    "eta",
    "flow_alignment",
    "friction_mode_fric",
    "gamma",
    "height",
    "ldg_a",
    "ldg_b",
    "ldg_c",
    "lx",
    "ly",
    "molecular_field_linear_space",
    "num_defect_pairs",
    "nx",
    "ny",
    "nz",
    "output_dir",
    "parameterization",
    "pointwise_execution",
    "projected_transform_execution",
    "restart_from",
    "runtime_path",
    "save_hydrodynamics",
    "save_interval",
    "save_start_step",
    "seed",
    "shendruk_preset",
    "spectral_refresh_effective_time",
    "spectral_refresh_interval_steps",
    "spectral_refresh_mode",
    "spectral_refresh_requested_steps",
    "spectral_refresh_requested_time",
    "spectral_storage",
    "steps",
    "stress_divergence_sum_space",
    "tf32",
    "transform_execution_order",
    "twist_amplitude",
    "twist_modes",
    "validation_config_sha256",
    "zero_mode_policy",
}
FACADE_METHOD_READS = {
    "identity_metadata",
    "runtime_identity_sha256",
    "runtime_selection_metadata",
}


def _spec(tmp_path: Path, **overrides: object):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / "output",
        "device": "cpu",
        "dtype": "float64",
        "pointwise_execution": "eager",
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "steps": 1,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "disable_spectral_refresh": True,
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _run_spec_reads() -> set[str]:
    tree = ast.parse(
        APPLICATION_PATH.read_text(encoding="utf-8"),
        filename=str(APPLICATION_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_plane_beris_edwards"
    )
    return {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"args", "run_spec"}
    }


def _forbid_expensive_application_work(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("application crossed the early-exit gate")

    for name in (
        "create_initial_condition",
        "sample_periodic_neutral_defects_2d",
        "build_plane_beris_edwards_runtime",
    ):
        monkeypatch.setattr(application, name, forbidden)


def test_application_uses_one_component_graph_and_only_facade_identity_methods(
    tmp_path,
    monkeypatch,
):
    assert _run_spec_reads() == FACADE_METHOD_READS
    calls = []
    original = application.decompose_plane_beris_edwards_run_spec

    def spy(value):
        result = original(value)
        calls.append((value, result))
        return result

    monkeypatch.setattr(
        application,
        "decompose_plane_beris_edwards_run_spec",
        spy,
    )
    spec = _spec(tmp_path, dry_run=True)
    assert application.run_plane_beris_edwards(spec) is None
    assert len(calls) == 1
    assert calls[0][0] is spec
    assert FLAT_APPLICATION_READS.isdisjoint(_run_spec_reads())


def test_dry_run_emits_metadata_before_allocation_and_creates_no_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    _forbid_expensive_application_work(monkeypatch)
    spec = _spec(tmp_path, dry_run=True)

    assert application.run_plane_beris_edwards(
        spec,
        emit_metadata=True,
    ) is None
    assert '"script": "Plane_beris_edwards_stokes.py"' in capsys.readouterr().out
    assert not spec.output_dir.exists()


def test_cross_runtime_restart_is_rejected_before_output_or_allocation(
    tmp_path,
    monkeypatch,
):
    _forbid_expensive_application_work(monkeypatch)
    checkpoint = tmp_path / "checkpoint"
    spec = _spec(tmp_path, restart_from=checkpoint)
    monkeypatch.setattr(
        application,
        "read_plane_checkpoint_header",
        lambda path: SimpleNamespace(
            runtime_path=PlaneRuntimePath.SEPARATED_CANARY,
            runtime_identity_sha256="unused",
        ),
    )

    with pytest.raises(
        ValueError,
        match="cross-runtime Plane checkpoint restart is unsupported",
    ):
        application.run_plane_beris_edwards(spec)
    assert not spec.output_dir.exists()
