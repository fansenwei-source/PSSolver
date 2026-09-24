from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver.channel import Q_COMPONENTS, build_active_nematic_channel
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
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
from pssolver.planning.construction import RuntimeConstructionKind
from pssolver.planning.package_construction import (
    PackageRuntimeConstructionPlan,
)
from pssolver.runtime.channel_active_nematics import (
    ChannelRuntimeBuildRequest,
    build_channel_active_nematic_runtime,
)
from pssolver.runtime.channel_application_bridge import (
    build_package_compiled_channel_runtime,
)
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.runtime import package_construction as package_runtime
from pssolver.runtime.plane_beris_edwards import (
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)
from pssolver.workflows.plane_compiled_v2 import (
    build_plane_compiled_v2_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
PLANNING_PATH = ROOT / "pssolver/planning/package_construction.py"
CONFIGURATION_PATH = ROOT / "pssolver/configuration/package_construction.py"
RUNTIME_PATH = ROOT / "pssolver/runtime/package_construction.py"
PLANE_BRIDGE_PATH = ROOT / "pssolver/runtime/plane_application_bridge.py"
CHANNEL_BRIDGE_PATH = ROOT / "pssolver/runtime/channel_application_bridge.py"
RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_7_p775_package_construction.json"
)


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
    return compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(spec)
    )


def _plane_values():
    coordinate = torch.arange(6**3, dtype=torch.float64).reshape(6, 6, 6)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


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


def _channel_spec(runtime_path: str):
    return ChannelActiveNematicRunSpec(
        shape=(8, 6, 5),
        lengths=(8.0, 6.0, 5.0),
        dt=1.0e-3,
        steps=2,
        save_interval=1,
        diagnostic_interval=1,
        activity=0.2,
        pressure_relative_tolerance=1.0e-8,
        pressure_max_iterations=100,
        runtime_path=runtime_path,
    )


def _channel_simulation(spec):
    return compose_channel_active_nematics_simulation(spec.components)


def _channel_values():
    shape = (8, 6, 5)
    coordinate = torch.arange(
        shape[0] * shape[1] * shape[2], dtype=torch.float32
    ).reshape(shape)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


def _channel_request(spec, values):
    return ChannelRuntimeBuildRequest(
        run_spec=spec,
        production_metadata={
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        },
        initial_q=values,
        device="cpu",
    )


def _legacy_channel_builder(spec, values):
    material = spec.components.material
    pressure = spec.components.pressure_solver

    def build():
        solver = build_active_nematic_channel(
            spec.shape,
            spec.lengths,
            spec.dt,
            values,
            device="cpu",
            batchsize=spec.batch_size,
            rho=material.rho,
            elastic_constant=material.elastic_constant,
            beta=material.beta,
            friction=material.friction,
            viscosity=material.viscosity,
            pressure_rel_tol=pressure.relative_tolerance,
            pressure_max_iter=pressure.max_iterations,
            pressure_fixed_iterations=pressure.fixed_iterations,
        )
        solver.parameters["alpha"] = torch.full(
            (spec.batch_size, *spec.shape),
            material.activity,
            dtype=solver.dtype,
            device=solver.device,
        )
        return solver

    return build


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
            if any(
                name == blocked or name.startswith(f"{blocked}.")
                for blocked in forbidden
            )
        }, path


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
        (
            "channel",
            "legacy_channel",
            RuntimeConstructionKind.CHANNEL_LEGACY,
            "channel_application_bridge.build_package_legacy_channel_runtime",
        ),
        (
            "channel",
            "compiled_channel_v2",
            RuntimeConstructionKind.CHANNEL_COMPILED_V2,
            "channel_application_bridge.build_package_compiled_channel_runtime",
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
    simulation = (
        _plane_simulation(_plane_spec(tmp_path, runtime_path))
        if family == "plane"
        else _channel_simulation(_channel_spec(runtime_path))
    )
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
        direct.kind = RuntimeConstructionKind.CHANNEL_LEGACY


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


@pytest.mark.parametrize(
    "runtime_path",
    ("legacy_channel", "compiled_channel_v2"),
)
def test_channel_package_construction_is_byte_identical_to_existing_path(
    runtime_path,
):
    spec = _channel_spec(runtime_path)
    direct_values = _channel_values()
    package_values = {
        name: value.clone() for name, value in direct_values.items()
    }
    direct_request = _channel_request(spec, direct_values)
    package_request = _channel_request(spec, package_values)
    if runtime_path == "legacy_channel":
        direct = build_channel_active_nematic_runtime(
            direct_request,
            legacy_builder=_legacy_channel_builder(spec, direct_values),
        )
    else:
        direct = build_channel_active_nematic_runtime(
            direct_request,
            compiled_builder=lambda: (
                build_package_compiled_channel_runtime(direct_request)
            ),
        )
    package = _package_build(_channel_simulation(spec), package_request)

    assert type(package) is type(direct)
    direct.advance(2)
    package.advance(2)
    assert torch.equal(package.fields.spatial, direct.fields.spatial)
    assert torch.equal(package.fields.spectral, direct.fields.spectral)
    assert package.completed_steps == direct.completed_steps == 2


def test_new_package_factory_remains_opt_in_and_private():
    for path in (
        ROOT / "pssolver/applications/plane_beris_edwards.py",
        ROOT / "pssolver/applications/channel_active_nematics.py",
        ROOT / "pssolver/configuration/__init__.py",
        ROOT / "pssolver/planning/__init__.py",
        ROOT / "pssolver/runtime/__init__.py",
        ROOT / "pssolver/__init__.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "PackageRuntimeConstructionInput" not in source
        assert "build_package_simulation_runtime" not in source
    assert create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=ROOT / "unused_p775_default_output",
    ).runtime_path.value == "legacy_production"
    assert ChannelActiveNematicRunSpec().runtime_path.value == "legacy_channel"


def test_p775_machine_record_matches_sources_and_scope():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["classification"] == (
        "PASS_P7_7_5_PACKAGE_CONSTRUCTION_OWNERSHIP_NORMALIZATION"
    )
    assert record["construction"]["application_builder_required"] is False
    assert record["construction"]["hot_path_dispatch_added"] is False
    assert record["production_applications_connected"] is False
    assert record["production_default_changed"] is False
    assert record["phase_8_authorized"] is False
    assert record["source_sha256"] == {
        "pssolver/configuration/package_construction.py": _sha256(
            CONFIGURATION_PATH
        ),
        "pssolver/planning/package_construction.py": _sha256(PLANNING_PATH),
        "pssolver/runtime/channel_application_bridge.py": _sha256(
            CHANNEL_BRIDGE_PATH
        ),
        "pssolver/runtime/package_construction.py": _sha256(RUNTIME_PATH),
        "pssolver/runtime/plane_application_bridge.py": _sha256(
            PLANE_BRIDGE_PATH
        ),
    }
