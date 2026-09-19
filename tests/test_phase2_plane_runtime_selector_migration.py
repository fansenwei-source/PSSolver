"""P2.5 oracles for the Plane runtime-selection consumer migration."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType

import pytest

from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.runtime import (
    LegacyPlaneRuntimeAdapter,
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SELECTOR_PATH = (
    PROJECT_ROOT / "pssolver" / "runtime" / "plane_beris_edwards.py"
)
FLAT_SELECTOR_READS = {"disable_q_gradient_reuse", "runtime_path"}
CANARY_MODULE = "pssolver.experimental.plane_shadow_driver"


class _DummyIntegrator:
    spectral_refresh_interval = None
    step_count = 0
    refresh_count = 0


class _DummySolver:
    fields = {}
    integrator = _DummyIntegrator()

    def run(self, *_args, **_kwargs):
        return None

    def refresh_static_fields(self):
        return None


class _DummyProjector:
    retained_axis_counts = (1, 1, 1)


class _CanaryBuilderReached(RuntimeError):
    pass


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
        "steps": 1,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "disable_spectral_refresh": True,
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _request(spec, *, metadata=None) -> PlaneRuntimeBuildRequest:
    if metadata is None:
        metadata = {
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        }
    return PlaneRuntimeBuildRequest(
        run_spec=spec,
        production_metadata=metadata,
        initial_values={},
        device="cpu",
    )


def _run_spec_reads() -> set[str]:
    tree = ast.parse(
        RUNTIME_SELECTOR_PATH.read_text(encoding="utf-8"),
        filename=str(RUNTIME_SELECTOR_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_plane_beris_edwards_runtime"
    )
    return {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "request"
        and node.value.attr == "run_spec"
    }


def test_runtime_selector_flat_facade_read_set_is_frozen_before_migration():
    assert _run_spec_reads() == FLAT_SELECTOR_READS


def test_request_authority_gate_rejects_mixed_metadata_before_selection(
    tmp_path,
):
    spec = _spec(tmp_path)
    metadata = {
        "configuration": {
            **spec.identity_metadata(),
            "runtime_path": "separated_canary",
        },
        "runtime_selection": spec.runtime_selection_metadata(),
    }
    with pytest.raises(ValueError, match="mixed Plane runtime"):
        _request(spec, metadata=metadata)


def test_request_and_injected_builder_type_gates_keep_their_order(tmp_path):
    with pytest.raises(
        TypeError,
        match="request must be a PlaneRuntimeBuildRequest",
    ):
        build_plane_beris_edwards_runtime(object(), legacy_builder=object())

    with pytest.raises(TypeError, match="legacy_builder must be callable"):
        build_plane_beris_edwards_runtime(
            _request(_spec(tmp_path)),
            legacy_builder=object(),
        )


def test_legacy_selection_uses_only_injected_builder_and_adapter(tmp_path):
    calls = []
    solver = _DummySolver()
    projector = _DummyProjector()

    def builder():
        calls.append("legacy")
        return solver, projector

    adapter = build_plane_beris_edwards_runtime(
        _request(_spec(tmp_path)),
        legacy_builder=builder,
    )
    assert calls == ["legacy"]
    assert isinstance(adapter, LegacyPlaneRuntimeAdapter)
    assert adapter.solver is solver
    assert adapter.projector is projector


def test_canary_import_and_builder_remain_lazy_after_explicit_selection(
    tmp_path,
    monkeypatch,
):
    calls = []
    fake_module = ModuleType(CANARY_MODULE)

    def build(metadata, *, device):
        calls.append((metadata, device))
        raise _CanaryBuilderReached

    fake_module.build_plane_separated_canary_runtime_from_production_metadata = (
        build
    )
    monkeypatch.setitem(sys.modules, CANARY_MODULE, fake_module)

    spec = _spec(tmp_path, runtime_path="separated_canary")
    request = _request(spec)
    with pytest.raises(_CanaryBuilderReached):
        build_plane_beris_edwards_runtime(request)
    assert calls == [(request.production_metadata, "cpu")]
