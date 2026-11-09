from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
import torch

from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.package_construction import (
    normalize_package_construction,
    plan_package_runtime_construction,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation_binding import bind_simulation_runtime
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.planning.construction import RuntimeConstructionKind
from pssolver.runtime import package_construction as package_runtime
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.runtime.plane_beris_edwards import (
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)
from pssolver.workflows.plane_compiled_v2 import build_plane_compiled_v2_runtime


ROOT = Path(__file__).resolve().parents[1]


PLANNING_PATH = ROOT / "pssolver/planning/package_construction.py"


CONFIGURATION_PATH = ROOT / "pssolver/configuration/package_construction.py"


RUNTIME_PATH = ROOT / "pssolver/runtime/package_construction.py"


PLANE_BRIDGE_PATH = ROOT / "pssolver/runtime/plane_application_bridge.py"


CHANNEL_BRIDGE_PATH = ROOT / "pssolver/runtime/channel_application_bridge.py"


RESULT_PATH = ROOT / ("notes/architecture_v0_2/phase_7_p775_package_construction.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plane_spec(tmp_path: Path, runtime_path: str):
    return create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / f"plane_{runtime_path}",
        runtime_path=runtime_path,
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        nx=6,
        ny=6,
        nz=6,
        lx=8.0,
        ly=9.0,
        height=20.0,
        steps=2,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        spectral_refresh_steps=3,
        defect_min_separation=2.0,
        defect_core_radius=0.5,
        twist_modes=(1, 2, 3),
    )


def _plane_simulation(spec):
    return compose_plane_beris_edwards_simulation(decompose_plane_beris_edwards_run_spec(spec))


def _package_build(simulation, request):
    plan = plan_package_runtime_construction(simulation)
    construction = PackageRuntimeConstructionInput(plan, request)
    return build_package_simulation_runtime(construction)


def test_package_plan_declarations_and_resolver_are_tensor_runtime_free():
    forbidden = {
        "numpy",
        "torch",
        "pssolver.backends",
        "pssolver.execution",
        "pssolver.integrators",
        "pssolver.linear_solvers",
        "pssolver.models",
        "pssolver.operators",
        "pssolver.runtime",
        "pssolver.transforms",
        "pssolver.workflows",
    }
    for path in (PLANNING_PATH, CONFIGURATION_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert not {
            name
            for name in imports
            if any(name == blocked or name.startswith(f"{blocked}.") for blocked in forbidden)
        }, path


def test_p775_machine_record_matches_sources_and_scope():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["classification"] == ("PASS_P7_7_5_PACKAGE_CONSTRUCTION_OWNERSHIP_NORMALIZATION")
    assert record["construction"]["application_builder_required"] is False
    assert record["construction"]["hot_path_dispatch_added"] is False
    assert record["production_applications_connected"] is False
    assert record["production_default_changed"] is False
    assert record["phase_8_authorized"] is False
    assert set(record["source_sha256"]) == {
        "pssolver/configuration/package_construction.py",
        "pssolver/planning/package_construction.py",
        "pssolver/runtime/channel_application_bridge.py",
        "pssolver/runtime/package_construction.py",
        "pssolver/runtime/plane_application_bridge.py",
    }
    assert all(len(value) == 64 for value in record["source_sha256"].values())


def _plane_values():
    coordinate = torch.arange(6**3, dtype=torch.float64).reshape(6, 6, 6)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {name: coordinate.mul((index + 1) * 1.0e-3) for index, name in enumerate(Q_COMPONENTS)}


def _plane_request(spec, values=None):
    return PlaneRuntimeBuildRequest(
        run_spec=spec,
        production_metadata={
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        },
        initial_values=_plane_values() if values is None else values,
        device="cpu",
    )


@pytest.mark.parametrize(
    ("family", "runtime_path", "kind", "builder_suffix"),
    (
        (
            "plane",
            "legacy_production",
            RuntimeConstructionKind.PLANE_LEGACY_PRODUCTION,
            "plane_legacy.build_legacy_plane_runtime",
        ),
        (
            "plane",
            "compiled_v2",
            RuntimeConstructionKind.PLANE_COMPILED_V2,
            "plane_application_bridge.build_package_compiled_plane_runtime",
        ),
    ),
)
def test_package_plan_normalizes_all_builder_ownership(
    tmp_path,
    family,
    runtime_path,
    kind,
    builder_suffix,
):
    simulation = _plane_simulation(_plane_spec(tmp_path, runtime_path))
    binding = bind_simulation_runtime(simulation)
    direct = normalize_package_construction(binding)
    composed = plan_package_runtime_construction(simulation)

    assert direct == composed
    assert direct.kind is kind
    assert direct.source_binding_sha256 == binding.canonical_sha256()
    assert direct.source_simulation_sha256 == simulation.canonical_sha256()
    assert direct.implementation_builder.endswith(builder_suffix)
    assert direct.to_metadata()["builder_provision"] == "package"
    assert direct.fallback_allowed is False
    json.dumps(direct.to_metadata(), allow_nan=False, sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        direct.kind = RuntimeConstructionKind.PLANE_LEGACY_PRODUCTION


def test_typed_input_contains_no_builder_callable(tmp_path):
    spec = _plane_spec(tmp_path, "legacy_production")
    plan = plan_package_runtime_construction(_plane_simulation(spec))
    construction = PackageRuntimeConstructionInput(
        plan,
        _plane_request(spec, {}),
    )

    assert tuple(field.name for field in fields(construction)) == (
        "plan",
        "request",
    )
    assert construction.to_metadata()["application_builder_supplied"] is False
    with pytest.raises(FrozenInstanceError):
        construction.request = object()


def test_package_factory_returns_existing_bound_factory_object(
    tmp_path,
    monkeypatch,
):
    spec = _plane_spec(tmp_path, "legacy_production")
    construction = PackageRuntimeConstructionInput(
        plan_package_runtime_construction(_plane_simulation(spec)),
        _plane_request(spec, {}),
    )
    sentinel = object()
    calls = []

    def factory(binding, request, **kwargs):
        calls.append((binding, request, kwargs))
        return sentinel

    monkeypatch.setattr(
        package_runtime,
        "build_bound_simulation_runtime",
        factory,
    )
    result = build_package_simulation_runtime(construction)

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][1] is construction.request
    assert calls[0][2] == {}


def test_plan_mismatch_fails_before_any_builder_runs(tmp_path, monkeypatch):
    legacy = _plane_spec(tmp_path, "legacy_production")
    compiled = _plane_spec(tmp_path, "compiled_v2")
    invalid = PackageRuntimeConstructionInput(
        plan_package_runtime_construction(_plane_simulation(compiled)),
        _plane_request(legacy, {}),
    )
    calls = []
    monkeypatch.setattr(
        package_runtime,
        "build_bound_simulation_runtime",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="does not match"):
        build_package_simulation_runtime(invalid)
    assert calls == []


@pytest.mark.parametrize("runtime_path", ("legacy_production", "compiled_v2"))
def test_plane_package_construction_is_byte_identical_to_existing_path(
    tmp_path,
    runtime_path,
):
    spec = _plane_spec(tmp_path, runtime_path)
    direct_request = _plane_request(spec)
    package_request = _plane_request(spec)
    if runtime_path == "legacy_production":
        direct = build_plane_beris_edwards_runtime(direct_request)
    else:
        direct = build_plane_beris_edwards_runtime(
            direct_request,
            compiled_builder=lambda: build_plane_compiled_v2_runtime(
                spec,
                device="cpu",
                initial_values=direct_request.initial_values,
            ),
        )
    package = _package_build(_plane_simulation(spec), package_request)

    assert type(package) is type(direct)
    direct.advance(2)
    package.advance(2)
    assert torch.equal(package.fields.spatial, direct.fields.spatial)
    assert torch.equal(package.fields.spectral, direct.fields.spectral)
    assert torch.equal(package.fields.L_hat, direct.fields.L_hat)
    assert package.completed_steps == direct.completed_steps == 2
