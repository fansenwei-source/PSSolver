from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
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
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation_binding import (
    BindingRejectionCode,
    SimulationBindingError,
    bind_simulation_runtime,
)
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.planning.construction import (
    BuilderProvision,
    RuntimeConstructionBinding,
    RuntimeConstructionKind,
)
from pssolver.runtime.channel_active_nematics import (
    ChannelRuntimeBuildRequest,
    build_channel_active_nematic_runtime,
)
from pssolver.runtime.plane_beris_edwards import (
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)
from pssolver.runtime.simulation_construction import (
    build_bound_simulation_runtime,
)
from pssolver.runtime import simulation_construction as connection


ROOT = Path(__file__).resolve().parents[1]
PLANNING_PATH = ROOT / "pssolver/planning/construction.py"
BINDING_PATH = ROOT / "pssolver/configuration/simulation_binding.py"
CONNECTION_PATH = ROOT / "pssolver/runtime/simulation_construction.py"
RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_7_p774_runtime_construction_binding.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plane_spec(tmp_path: Path, runtime_path="legacy_production"):
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


def _channel_spec(runtime_path="legacy_channel"):
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


def _channel_legacy_builder(spec, values):
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


def test_binding_declarations_and_resolver_are_tensor_and_runtime_free():
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
    for path in (PLANNING_PATH, BINDING_PATH):
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
    ("family", "runtime_path", "kind", "provision", "parameter"),
    (
        (
            "plane",
            "legacy_production",
            RuntimeConstructionKind.PLANE_LEGACY_PRODUCTION,
            BuilderProvision.PACKAGE,
            None,
        ),
        (
            "plane",
            "compiled_v2",
            RuntimeConstructionKind.PLANE_COMPILED_V2,
            BuilderProvision.CALLER,
            "compiled_builder",
        ),
        (
            "channel",
            "legacy_channel",
            RuntimeConstructionKind.CHANNEL_LEGACY,
            BuilderProvision.CALLER,
            "legacy_builder",
        ),
        (
            "channel",
            "compiled_channel_v2",
            RuntimeConstructionKind.CHANNEL_COMPILED_V2,
            BuilderProvision.CALLER,
            "compiled_builder",
        ),
    ),
)
def test_binding_is_exact_frozen_and_deterministic(
    tmp_path,
    family,
    runtime_path,
    kind,
    provision,
    parameter,
):
    simulation = (
        _plane_simulation(_plane_spec(tmp_path, runtime_path))
        if family == "plane"
        else _channel_simulation(_channel_spec(runtime_path))
    )
    plan = lower_simulation_spec(simulation)
    first = bind_simulation_runtime(simulation, plan)
    second = bind_simulation_runtime(simulation)

    assert first == second
    assert first.kind is kind
    assert first.builder_provision is provision
    assert first.builder_parameter == parameter
    assert first.source_simulation_sha256 == simulation.canonical_sha256()
    assert first.lowering_plan_sha256 == plan.canonical_sha256()
    assert first.solver_implementation == plan.solver.implementation
    assert first.fallback_allowed is False
    assert first.canonical_sha256() == second.canonical_sha256()
    json.dumps(first.to_metadata(), allow_nan=False, sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        first.runtime_path = "other"


def test_binding_rejects_unqualified_path_and_nonmatching_plan(tmp_path):
    canary = _plane_simulation(_plane_spec(tmp_path, "separated_canary"))
    with pytest.raises(SimulationBindingError) as caught:
        bind_simulation_runtime(canary)
    assert caught.value.rejection.code is (
        BindingRejectionCode.UNSUPPORTED_RUNTIME_PATH
    )
    assert caught.value.rejection.to_metadata()["context"][
        "supported_runtime_paths"
    ] == ["compiled_v2", "legacy_production"]

    simulation = _plane_simulation(_plane_spec(tmp_path))
    plan = lower_simulation_spec(simulation)
    invalid = replace(plan, geometry_name="other_geometry")
    with pytest.raises(SimulationBindingError) as caught:
        bind_simulation_runtime(simulation, invalid)
    assert caught.value.rejection.code is (
        BindingRejectionCode.INVALID_LOWERING_PLAN
    )


def test_connection_returns_the_existing_factory_object_without_wrapping(
    tmp_path,
    monkeypatch,
):
    spec = _plane_spec(tmp_path, "compiled_v2")
    request = _plane_request(spec, {})
    binding = bind_simulation_runtime(_plane_simulation(spec))
    sentinel = object()
    builder = lambda: object()
    calls = []

    def factory(received, **kwargs):
        calls.append((received, kwargs))
        return sentinel

    monkeypatch.setattr(connection, "build_plane_beris_edwards_runtime", factory)
    result = build_bound_simulation_runtime(
        binding,
        request,
        compiled_builder=builder,
    )

    assert result is sentinel
    assert calls == [
        (
            request,
            {"legacy_builder": None, "compiled_builder": builder},
        )
    ]


def test_connection_rejects_identity_or_builder_mismatch_before_factory(
    tmp_path,
    monkeypatch,
):
    spec = _plane_spec(tmp_path)
    request = _plane_request(spec, {})
    binding = bind_simulation_runtime(_plane_simulation(spec))
    calls = []
    monkeypatch.setattr(
        connection,
        "build_plane_beris_edwards_runtime",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    incompatible = replace(binding, runtime_path="compiled_v2")
    with pytest.raises(ValueError, match="does not match"):
        build_bound_simulation_runtime(incompatible, request)
    assert calls == []

    with pytest.raises(ValueError, match="unselected"):
        build_bound_simulation_runtime(
            binding,
            request,
            compiled_builder=lambda: object(),
        )
    assert calls == []


def test_plane_legacy_bound_path_is_byte_identical_to_existing_factory(tmp_path):
    spec = _plane_spec(tmp_path)
    direct = build_plane_beris_edwards_runtime(_plane_request(spec))
    binding = bind_simulation_runtime(_plane_simulation(spec))
    bound = build_bound_simulation_runtime(binding, _plane_request(spec))

    assert type(bound) is type(direct)
    direct.advance(2)
    bound.advance(2)
    assert torch.equal(bound.fields.spatial, direct.fields.spatial)
    assert torch.equal(bound.fields.spectral, direct.fields.spectral)
    assert torch.equal(bound.fields.L_hat, direct.fields.L_hat)
    assert bound.completed_steps == direct.completed_steps == 2


def test_channel_legacy_bound_path_is_byte_identical_to_existing_factory():
    spec = _channel_spec()
    direct_values = _channel_values()
    bound_values = {name: value.clone() for name, value in direct_values.items()}
    direct = build_channel_active_nematic_runtime(
        _channel_request(spec, direct_values),
        legacy_builder=_channel_legacy_builder(spec, direct_values),
    )
    binding = bind_simulation_runtime(_channel_simulation(spec))
    bound = build_bound_simulation_runtime(
        binding,
        _channel_request(spec, bound_values),
        legacy_builder=_channel_legacy_builder(spec, bound_values),
    )

    assert type(bound) is type(direct)
    direct.advance(2)
    bound.advance(2)
    assert torch.equal(bound.fields.spatial, direct.fields.spatial)
    assert torch.equal(bound.fields.spectral, direct.fields.spectral)
    assert bound.completed_steps == direct.completed_steps == 2


def test_connection_is_opt_in_and_production_defaults_remain_unchanged():
    assert create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=ROOT / "unused_p774_default_output",
    ).runtime_path.value == "legacy_production"
    assert ChannelActiveNematicRunSpec().runtime_path.value == "legacy_channel"
    for path in (
        ROOT / "pssolver/applications/plane_beris_edwards.py",
        ROOT / "pssolver/applications/channel_active_nematics.py",
        ROOT / "pssolver/configuration/__init__.py",
        ROOT / "pssolver/planning/__init__.py",
        ROOT / "pssolver/runtime/__init__.py",
        ROOT / "pssolver/__init__.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "build_bound_simulation_runtime" not in source
        assert "bind_simulation_runtime" not in source


def test_p774_machine_record_matches_sources_scope_and_local_evidence():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["classification"] == (
        "PASS_P7_7_4_OPT_IN_RUNTIME_CONSTRUCTION_BINDING"
    )
    assert record["connection"]["opt_in"] is True
    assert record["connection"]["factory_result_returned_directly"] is True
    assert record["connection"]["hot_path_dispatch_added"] is False
    assert record["production_default_changed"] is False
    assert record["phase_8_authorized"] is False
    assert set(record["source_sha256"]) == {
        "pssolver/configuration/simulation_binding.py",
        "pssolver/planning/construction.py",
        "pssolver/runtime/simulation_construction.py",
    }
    assert all(len(value) == 64 for value in record["source_sha256"].values())
