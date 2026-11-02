"""Live, repository-only comparisons before any R1 dispatcher is switched."""

import ast
from dataclasses import FrozenInstanceError, replace
from enum import Enum
import importlib
import inspect
from pathlib import Path

import pytest

from pssolver.api.capabilities import available_combinations
from pssolver.configuration import application_registry as shadow
from pssolver.configuration import public_simulation_runner as compiler
from pssolver.configuration import simulation_binding as binding
from pssolver.configuration import simulation_lowering as lowering
from pssolver.configuration.simulation import ExecutionSpec
from pssolver.core.geometry import AxisTopology, GeometrySpec
from pssolver.runtime import package_construction, simulation_construction
from tests.test_phase7_p774_runtime_construction_binding import (
    _channel_simulation, _channel_spec, _plane_simulation, _plane_spec,
)
from tests.test_phase8_p82_periodic_application import _simulation as periodic
from tests.test_phase8_p83_channel_complete_stress_application import (
    _simulation as complete_channel,
)

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def _simulation(tmp_path, entry):
    if entry.lowering_kind == "plane":
        return _plane_simulation(_plane_spec(tmp_path))
    if entry.lowering_kind == "channel_legacy":
        return _channel_simulation(_channel_spec())
    if entry.lowering_kind == "periodic":
        return periodic(tmp_path).specification
    return complete_channel(tmp_path).specification


def _binding_table():
    # Evaluate only the existing literal table (enum members and constants),
    # never execute a numerical kernel or a copy of the binding dispatcher.
    tree = ast.parse(inspect.getsource(binding.bind_simulation_runtime))
    expression, = [
        node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "table" for t in node.targets)
    ]
    assert isinstance(expression, ast.Dict)
    assert not any(isinstance(node, ast.Call) for node in ast.walk(expression))
    return eval(compile(ast.Expression(expression), "<binding-table>", "eval"),
                vars(binding))


def test_shadow_matches_all_compiler_metadata_and_order():
    expected = compiler.public_compiler_capabilities()
    observed = tuple(entry.compiler_metadata() for entry in shadow.APPLICATIONS)
    assert observed == expected
    assert [tuple(row) for row in observed] == [tuple(row) for row in expected]
    assert tuple(shadow.APPLICATION_BY_PAIR) == tuple(compiler._PUBLIC_COMPILER_REGISTRY)
    assert len(shadow.APPLICATIONS) == len(shadow.APPLICATION_BY_PAIR) == 4
    for entry, combination in zip(shadow.APPLICATIONS, available_combinations()):
        assert combination.to_metadata() == {
            **entry.compiler_metadata(), "runtime_paths": list(entry.runtime_paths),
            "qualified": True,
        }


def test_shadow_matches_literal_binding_table_keys_values_and_order():
    table = _binding_table()
    assert len(shadow.RUNTIMES) == len(shadow.RUNTIME_BY_KEY) == len(shadow.RUNTIME_BY_KIND) == 6
    assert tuple(shadow.RUNTIME_BY_KEY) == tuple(table)
    for entry in shadow.RUNTIMES:
        values = tuple(x.value if isinstance(x, Enum) else x for x in table[entry.key])
        assert entry.binding_fields() == values
        assert shadow.RUNTIME_BY_KIND[entry.kind] is entry


@pytest.mark.parametrize("entry", shadow.APPLICATIONS, ids=lambda x: x.lowering_kind)
def test_shadow_matches_lowering_selection_and_compiler_call_identity(tmp_path, monkeypatch, entry):
    source = _simulation(tmp_path, entry)
    assert lowering._validate_global_contract(source) == (entry.lowering_kind, entry.capabilities)
    assert tuple(item.value for item in source.geometry.axis_topologies) == entry.topology
    assert source.equation_system.required_capabilities == entry.capabilities
    target = _resolve(entry.compiler)
    module_name, name = entry.compiler.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert getattr(module, name) is target
    token = object()
    calls = []

    def capture(*args):
        calls.append(args)
        return token

    monkeypatch.setattr(module, name, capture)
    assert compiler.compile_public_simulation(source) is token
    expected_args = (source, compiler.PublicSimulationCompilation) if entry.compiler_accepts_result_type else (source,)
    assert calls == [expected_args]


@pytest.mark.parametrize("entry", shadow.APPLICATIONS, ids=lambda x: x.lowering_kind)
def test_shadow_matches_rejection_order_and_capability_error(tmp_path, entry):
    source = _simulation(tmp_path, entry)
    invalid = replace(source, equation_system=replace(source.equation_system, variant="unqualified"))
    with pytest.raises(lowering.SimulationLoweringError) as caught:
        lowering._validate_global_contract(invalid)
    assert caught.value.rejection.to_metadata() == {
        "code": "unsupported_model_geometry",
        "message": "the model/geometry pair has no qualified implementation",
        "context": {"equation_variant": "unqualified", "geometry_name": entry.geometry_name,
                    "supported_pairs": [list(key) for key in shadow.LOWERING_REJECTION_PAIRS]},
    }
    with pytest.raises(compiler.PublicSimulationCompilationError) as public:
        compiler.compile_public_simulation(invalid)
    assert public.value.context["registered_pairs"] == [list(key) for key in shadow.APPLICATION_BY_PAIR]
    evolution = source.equation_system.evolution_laws[0]
    invalid = replace(source, equation_system=replace(
        source.equation_system,
        evolution_laws=(replace(evolution, capability="unqualified_evolution"),),
    ))
    with pytest.raises(lowering.SimulationLoweringError) as caught:
        lowering._validate_global_contract(invalid)
    assert caught.value.rejection.to_metadata() == {
        "code": "unsupported_capability_set", "message": "the equation capability set or order is not qualified",
        "context": {"expected": list(entry.capabilities),
                    "observed": list(invalid.equation_system.required_capabilities)},
    }


@pytest.mark.parametrize("entry", shadow.RUNTIMES, ids=lambda x: x.runtime_path)
def test_shadow_matches_real_bindings_and_factory_identity(tmp_path, monkeypatch, entry):
    source = _simulation(tmp_path, entry.application)
    source = replace(source, execution=ExecutionSpec(
        backend=source.execution.backend, runtime_path=entry.runtime_path,
        options=source.execution.options,
    ))
    actual = binding.bind_simulation_runtime(source)
    assert (actual.kind.value, actual.request_type, actual.runtime_factory,
            actual.adapter_protocol, actual.builder_provision.value,
            actual.builder_parameter) == entry.binding_fields()
    plan = lowering.lower_simulation_spec(source)
    assert actual.source_simulation_sha256 == source.canonical_sha256()
    assert actual.lowering_plan_sha256 == plan.canonical_sha256()
    assert actual.solver_implementation == plan.solver.implementation
    target = _resolve(entry.application.runtime_factory)
    assert getattr(simulation_construction, target.__name__) is target
    request = object.__new__(_resolve(entry.application.request_type))
    token = object()
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return token

    monkeypatch.setattr(simulation_construction, "_request_simulation", lambda _: source)
    monkeypatch.setattr(simulation_construction, target.__name__, capture)
    builder = lambda: object()
    keywords = {} if entry.builder_parameter is None else {entry.builder_parameter: builder}
    assert simulation_construction.build_bound_simulation_runtime(actual, request, **keywords) is token
    expected_kwargs = {"legacy_builder": None, "compiled_builder": None} if entry.application.lowering_kind in {"plane", "channel_legacy"} else {}
    expected_kwargs.update(keywords)
    assert calls == [((request,), expected_kwargs)]
    if entry.package_builder:
        bridge = _resolve(entry.package_builder)
        assert getattr(package_construction, bridge.__name__) is bridge
    unknown = replace(source, execution=replace(source.execution, runtime_path="unqualified"))
    with pytest.raises(binding.SimulationBindingError) as caught:
        binding.bind_simulation_runtime(unknown)
    assert caught.value.rejection.to_metadata() == {
        "code": "unsupported_runtime_path",
        "message": "the lowered model/geometry pair has no qualified runtime binding",
        "context": {"equation_variant": entry.application.equation_variant,
                    "geometry_name": entry.application.geometry_name, "runtime_path": "unqualified",
                    "supported_runtime_paths": sorted(entry.application.runtime_paths)},
    }


def test_shadow_is_immutable_tensor_free_and_not_public():
    for mapping in (shadow.APPLICATION_BY_PAIR, shadow.RUNTIME_BY_KEY, shadow.RUNTIME_BY_KIND):
        with pytest.raises(TypeError):
            mapping["injected"] = object()
    with pytest.raises(FrozenInstanceError):
        shadow.APPLICATIONS[0].application = "injected"
    with pytest.raises(FrozenInstanceError):
        shadow.RUNTIMES[0].runtime_path = "injected"
    assert shadow.__all__ == []
    tree = ast.parse(inspect.getsource(shadow))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == ["dataclasses", "types"]
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))


@pytest.mark.parametrize("entry", shadow.RUNTIMES, ids=lambda x: x.runtime_path)
def test_shadow_matches_package_builder_selection(tmp_path, monkeypatch, entry):
    source = _simulation(tmp_path, entry.application)
    source = replace(source, execution=ExecutionSpec(
        backend=source.execution.backend, runtime_path=entry.runtime_path,
        options=source.execution.options,
    ))
    plan = package_construction.plan_package_runtime_construction(source)
    request = object.__new__(_resolve(entry.application.request_type))
    construction = package_construction.PackageRuntimeConstructionInput(plan, request)
    monkeypatch.setattr(package_construction, "_request_simulation", lambda _: source)
    token = object()
    calls = []
    bridges = []

    def capture(actual_binding, actual_request, **kwargs):
        calls.append((actual_binding, actual_request, tuple(kwargs)))
        if entry.package_builder is None:
            assert kwargs == {}
            return token
        return kwargs[entry.builder_parameter]()

    def bridge(actual_request):
        bridges.append(actual_request)
        return token

    monkeypatch.setattr(package_construction, "build_bound_simulation_runtime", capture)
    if entry.package_builder is not None:
        monkeypatch.setattr(package_construction, entry.package_builder.rsplit(".", 1)[1], bridge)
    assert package_construction.build_package_simulation_runtime(construction) is token
    assert calls == [(binding.bind_simulation_runtime(source), request,
                      () if entry.package_builder is None else (entry.builder_parameter,))]
    assert bridges == ([] if entry.package_builder is None else [request])


@pytest.mark.parametrize("entry", shadow.APPLICATIONS, ids=lambda x: x.lowering_kind)
def test_shadow_matches_topology_and_capability_order_rejection(tmp_path, entry):
    source = _simulation(tmp_path, entry)
    swapped = tuple(AxisTopology.BOUNDED if item == "periodic"
                    else AxisTopology.PERIODIC for item in entry.topology)
    invalid_geometry = GeometrySpec(source.geometry.name, source.geometry.domain, swapped)
    with pytest.raises(lowering.SimulationLoweringError) as caught:
        lowering._validate_global_contract(replace(source, geometry=invalid_geometry))
    assert caught.value.rejection.to_metadata() == {
        "code": "unsupported_dimension_or_topology",
        "message": "the qualified solver requires its frozen three-dimensional topology",
        "context": {"expected": list(entry.topology),
                    "observed": [item.value for item in swapped]},
    }
    # The declared constitutive order, not a set, determines capability order.
    equation = replace(source.equation_system,
                       constitutive_laws=tuple(reversed(source.equation_system.constitutive_laws)))
    assert equation.required_capabilities != entry.capabilities
    with pytest.raises(lowering.SimulationLoweringError) as caught:
        lowering._validate_global_contract(replace(source, equation_system=equation))
    assert caught.value.rejection.code.value == "unsupported_capability_set"
    assert caught.value.rejection.to_metadata()["context"] == {
        "expected": list(entry.capabilities),
        "observed": list(equation.required_capabilities),
    }


@pytest.mark.parametrize("pair", [
    ("unknown", "plane_slab"),
    ("complete_stress_beris_edwards", "unknown"),
    ("legacy_active_force_active_nematics", "periodic_box"),
    ("legacy_active_force_active_nematics", "plane_slab"),
])
def test_unregistered_shadow_keys_fail_closed_without_fallback(pair):
    assert shadow.APPLICATION_BY_PAIR.get(pair) is None
    with pytest.raises(KeyError):
        shadow.APPLICATION_BY_PAIR[pair]
    for runtime in shadow.RUNTIMES:
        with pytest.raises(KeyError):
            shadow.RUNTIME_BY_KEY[(*pair, runtime.runtime_path)]
