"""Exact old/new selection, rejection order and construction contracts."""

from copy import copy
from dataclasses import replace
from types import MappingProxyType

import pytest

from pssolver.configuration import application_registry as registry
from pssolver.configuration import public_simulation_runner as compiler
from pssolver.configuration import simulation_binding as binding
from pssolver.configuration import simulation_lowering as lowering
from pssolver.configuration.simulation import ExecutionSpec, TimeIntegrationSpec
from pssolver.core.geometry import AxisTopology, GeometrySpec
from pssolver.core.integrators import IntegratorSpec
from pssolver.runtime import package_construction, simulation_construction
from tests.test_v0_4_s18_r1_shadow_registry import _resolve, _simulation
from tests.v0_4_r1_legacy_dispatch import legacy


def _outcome(call):
    try:
        result = call()
    except Exception as exc:
        record = {"type": type(exc).__name__, "message": str(exc)}
        if hasattr(exc, "rejection"):
            record["rejection"] = exc.rejection.to_metadata()
        if hasattr(exc, "context"):
            record["context"] = exc.context
        if hasattr(exc, "code"):
            record["code"] = exc.code
        return "raised", record
    return "returned", result.to_metadata() if hasattr(result, "to_metadata") else result


def _spec(tmp_path, entry):
    spec = _simulation(tmp_path, entry.application)
    return replace(spec, execution=ExecutionSpec(
        backend=spec.execution.backend, runtime_path=entry.runtime_path,
        options=spec.execution.options,
    ))


def _mutate(spec, case):
    if case == "valid":
        return spec
    if case == "type":
        return None
    if case == "model":
        return replace(spec, equation_system=replace(spec.equation_system, variant="unknown"))
    if case in {"runtime", "backend"}:
        return replace(spec, execution=replace(spec.execution, **{
            "runtime_path" if case == "runtime" else "backend": "unknown",
        }))
    if case == "capability":
        return replace(spec, equation_system=replace(spec.equation_system,
            evolution_laws=(replace(spec.equation_system.evolution_laws[0], capability="unknown"),)))
    if case == "time":
        return replace(spec, time_integration=TimeIntegrationSpec(
            integrator=IntegratorSpec.sbdf2(dt=spec.time_integration.integrator.dt)))
    assert case == "topology"
    bad = copy(spec)
    geometry = spec.geometry
    object.__setattr__(bad, "geometry", GeometrySpec(geometry.name, geometry.domain,
        tuple(AxisTopology.BOUNDED if item is AxisTopology.PERIODIC
              else AxisTopology.PERIODIC for item in geometry.axis_topologies)))
    return bad


@pytest.mark.parametrize("entry", registry.RUNTIMES, ids=lambda row: row.runtime_path)
@pytest.mark.parametrize("case", ["valid", "type", "model", "runtime", "backend", "capability", "time", "topology"])
def test_registry_dispatch_exactly_matches_parent_errors_and_identities(tmp_path, entry, case):
    spec = _mutate(_spec(tmp_path, entry), case)
    for module, function in (
        (lowering, "lower_simulation_spec"),
        (binding, "bind_simulation_runtime"),
        (compiler, "compile_public_simulation"),
    ):
        expected = _outcome(lambda: getattr(legacy(module.__name__), function)(spec))
        actual = _outcome(lambda: getattr(module, function)(spec))
        assert actual == expected, (entry.runtime_path, case, function, actual, expected)


@pytest.mark.parametrize("entry", registry.RUNTIMES, ids=lambda row: row.runtime_path)
@pytest.mark.parametrize("case", ["valid", "wrong_request", "missing_builder", "extra_builder", "noncallable", "wrong_binding"])
def test_registry_construction_retains_parent_guard_order(tmp_path, monkeypatch, entry, case):
    source = _spec(tmp_path, entry)
    bound = binding.bind_simulation_runtime(source)
    request = object.__new__(_resolve(entry.application.request_type))
    if case == "wrong_request":
        request = object()
    builder = lambda: "built"
    kwargs = {} if entry.builder_parameter is None else {entry.builder_parameter: builder}
    if case == "missing_builder":
        kwargs = {}
    elif case == "extra_builder":
        kwargs["legacy_builder" if entry.builder_parameter != "legacy_builder" else "compiled_builder"] = builder
    elif case == "noncallable":
        kwargs["legacy_builder"] = 123
    elif case == "wrong_binding":
        bound = replace(bound, source_simulation_sha256="0" * 64)
    outcomes = []
    for module in (legacy(simulation_construction.__name__), simulation_construction):
        monkeypatch.setattr(module, "_request_simulation", lambda _: source)
        calls = []

        def capture(actual_request, **actual_kwargs):
            calls.append((actual_request is request, actual_kwargs))
            return "exact-factory-token"

        monkeypatch.setattr(module, entry.application.runtime_factory.rsplit(".", 1)[1], capture)
        result = _outcome(lambda: module.build_bound_simulation_runtime(bound, request, **kwargs))
        outcomes.append((result, calls))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("entry", registry.RUNTIMES, ids=lambda row: row.runtime_path)
@pytest.mark.parametrize("wrong_request", [False, True])
def test_registry_package_construction_retains_parent_errors(tmp_path, monkeypatch, entry, wrong_request):
    source = _spec(tmp_path, entry)
    plan = package_construction.plan_package_runtime_construction(source)
    request = object() if wrong_request else object.__new__(_resolve(entry.application.request_type))
    construction = object.__new__(package_construction.PackageRuntimeConstructionInput)
    object.__setattr__(construction, "plan", plan)
    object.__setattr__(construction, "request", request)
    outcomes = []
    for module in (legacy(package_construction.__name__), package_construction):
        monkeypatch.setattr(module, "_request_simulation", lambda _: source)
        calls = []

        def capture(actual_binding, actual_request, **kwargs):
            calls.append((actual_binding, actual_request is request, tuple(kwargs)))
            return kwargs[entry.builder_parameter]() if kwargs else "exact-token"

        monkeypatch.setattr(module, "build_bound_simulation_runtime", capture)
        if entry.package_builder:
            monkeypatch.setattr(module, entry.package_builder.rsplit(".", 1)[1], lambda _: "exact-token")
        outcomes.append((_outcome(lambda: module.build_package_simulation_runtime(construction)), calls))
    assert outcomes[0] == outcomes[1]


def test_registry_discovery_matches_parent_byte_order():
    from pssolver.api import capabilities
    import json
    for actual, expected in (
        (compiler.public_compiler_capabilities(), legacy(compiler.__name__).public_compiler_capabilities()),
        (capabilities.capability_catalog().to_metadata(), legacy(capabilities.__name__).capability_catalog().to_metadata()),
    ):
        assert json.dumps(actual) == json.dumps(expected)


@pytest.mark.parametrize("field", ["lowering_kind", "topology", "capabilities"])
def test_registry_parity_probe_detects_tampered_lowering_rows(tmp_path, monkeypatch, field):
    row = registry.APPLICATIONS[0]
    source = _simulation(tmp_path, row)
    changes = {"lowering_kind": "incorrect", "topology": ("bounded",) * 3, "capabilities": ()}
    modified = dict(registry.APPLICATION_BY_PAIR)
    modified[row.key] = replace(row, **{field: changes[field]})
    monkeypatch.setattr(lowering, "APPLICATION_BY_PAIR", MappingProxyType(modified))
    actual = _outcome(lambda: lowering._validate_global_contract(source))
    expected = _outcome(lambda: legacy(lowering.__name__)._validate_global_contract(source))
    assert actual != expected


def test_registry_parity_probe_detects_tampered_binding_target(tmp_path, monkeypatch):
    row = registry.RUNTIMES[0]
    source = _spec(tmp_path, row)
    modified = dict(registry.RUNTIME_BY_KEY)
    modified[row.key] = replace(row, application=replace(row.application, runtime_factory="wrong.factory"))
    monkeypatch.setattr(binding, "RUNTIME_BY_KEY", MappingProxyType(modified))
    assert _outcome(lambda: binding.bind_simulation_runtime(source)) != _outcome(
        lambda: legacy(binding.__name__).bind_simulation_runtime(source))
