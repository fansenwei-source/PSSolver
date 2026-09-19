"""P2.1S compatibility oracle for algebraic and Stokes declarations.

This suite characterizes the declarations before their mechanical extraction
from :mod:`pssolver.execution` to the tensor-free ``pssolver.systems`` layer.
It deliberately records existing success and failure behavior; it does not
connect the provisional Phase 2 components to a runtime.
"""

from __future__ import annotations

import base64
from dataclasses import FrozenInstanceError, MISSING, fields
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
import pickle
from types import MappingProxyType

import pytest

import pssolver
import pssolver.execution as execution
from pssolver.execution import (
    INCOMPRESSIBLE_STOKES_CAPABILITY,
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)
from pssolver.execution import algebraic as algebraic_module
from pssolver.execution import stokes as stokes_module
import pssolver.systems as systems_package
from pssolver.systems import algebraic as canonical_algebraic_module
from pssolver.systems.algebraic import (
    AlgebraicSystemSpec,
    AlgebraicUpdatePhase,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "systems"


def _load_json(name: str) -> dict[str, object]:
    value = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


API = _load_json("algebraic_stokes_api_v1.json")
CASES = _load_json("algebraic_stokes_cases_v1.json")


def _annotation_text(value: object) -> str | None:
    if value is inspect.Signature.empty:
        return None
    if isinstance(value, str):
        return value
    return inspect.formatannotation(value)


def _callable_name(value: object) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _default_contract(
    value: object,
    *,
    missing_kind: str,
) -> dict[str, object]:
    if value is MISSING or value is inspect.Signature.empty:
        return {"kind": missing_kind}
    if repr(value) == "<factory>":
        return {"kind": "factory"}
    if isinstance(value, Enum):
        return {
            "enum": type(value).__qualname__,
            "kind": "enum",
            "value": value.value,
        }
    return {"kind": "value", "value": value}


def _field_contract(item: object) -> dict[str, object]:
    if item.default_factory is not MISSING:
        default = {
            "callable": _callable_name(item.default_factory),
            "kind": "factory",
        }
    else:
        default = _default_contract(item.default, missing_kind="missing")
    return {
        "compare": item.compare,
        "default": default,
        "init": item.init,
        "name": item.name,
        "repr": item.repr,
        "type": _annotation_text(item.type),
    }


def _signature_contract(value_type: type) -> dict[str, object]:
    signature = inspect.signature(value_type)
    parameters = []
    for parameter in signature.parameters.values():
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters.append(
            {
                "annotation": _annotation_text(parameter.annotation),
                "default": _default_contract(
                    parameter.default,
                    missing_kind="required",
                ),
                "name": parameter.name,
            }
        )
    return {
        "parameters": parameters,
        "return_annotation": _annotation_text(signature.return_annotation),
    }


def _method_contract(value_type: type, name: str) -> dict[str, object]:
    descriptor = value_type.__dict__[name]
    if isinstance(descriptor, classmethod):
        kind = "classmethod"
        function = descriptor.__func__
    else:
        kind = "function"
        function = descriptor
    signature = inspect.signature(function)
    return {
        "kind": kind,
        "name": name,
        "parameters": [
            {
                "annotation": _annotation_text(parameter.annotation),
                "kind": parameter.kind.name,
                "name": parameter.name,
            }
            for parameter in signature.parameters.values()
        ],
        "return_annotation": _annotation_text(signature.return_annotation),
    }


def _canonical_json(value: object) -> str:
    encoding = CASES["canonical_json_encoding"]
    return json.dumps(
        value,
        allow_nan=encoding["allow_nan"],
        separators=tuple(encoding["separators"]),
        sort_keys=encoding["sort_keys"],
    )


def _algebraic_spec() -> AlgebraicSystemSpec:
    return AlgebraicSystemSpec(
        name="flow",
        capability="incompressible_stokes",
        output_components=["ux", "uy", "uz", "p"],
        dependencies=iter(("force_x", "force_y", "force_z")),
        parameters={
            "viscosity": 0.73,
            "nested": {"modes": (1, 2)},
            "friction": 0,
            "enabled": True,
        },
    )


def _stokes_spec(
    *,
    friction: float = 0.0,
    policy: TangentialZeroModePolicy = TangentialZeroModePolicy.ZERO_MEAN,
) -> IncompressibleStokesSystemSpec:
    base = CASES["stokes_base"]
    return IncompressibleStokesSystemSpec(
        name=base["name"],
        force_components=list(base["force_components"]),
        velocity_components=iter(base["velocity_components"]),
        pressure_component=base["pressure_component"],
        viscosity=base["viscosity"],
        friction=friction,
        pressure_gauge=PressureGauge(base["pressure_gauge"]),
        tangential_zero_mode_policy=policy,
    )


@pytest.mark.parametrize(
    ("value_type", "oracle_name", "module_key"),
    (
        (
            AlgebraicSystemSpec,
            "algebraic_system_spec",
            "target_canonical_module",
        ),
        (
            IncompressibleStokesSystemSpec,
            "incompressible_stokes_system_spec",
            "pre_extraction_module",
        ),
    ),
)
def test_dataclass_surfaces_match_characterization_oracle(
    value_type,
    oracle_name,
    module_key,
):
    oracle = API[oracle_name]
    observed_fields = list(fields(value_type))

    assert oracle["pre_extraction_module"] == oracle["legacy_module"]
    assert value_type.__module__ == oracle[module_key]
    assert oracle["legacy_module"].startswith("pssolver.execution.")
    assert oracle["target_canonical_module"].startswith("pssolver.systems.")
    assert value_type.__qualname__ == oracle["qualname"]
    assert [_field_contract(item) for item in observed_fields] == oracle["fields"]
    assert list(value_type.__slots__) == oracle["slots"]
    assert list(value_type.__match_args__) == oracle["match_args"]
    assert _signature_contract(value_type) == {
        "parameters": oracle["signature"],
        "return_annotation": oracle["return_annotation"],
    }
    assert [
        _method_contract(value_type, item["name"])
        for item in oracle["methods"]
    ] == oracle["methods"]

    parameters = value_type.__dataclass_params__
    assert {
        name: getattr(parameters, name)
        for name in oracle["dataclass_parameters"]
    } == oracle["dataclass_parameters"]
    assert all(item.hash is None for item in observed_fields)
    assert all(item.kw_only is False for item in observed_fields)


def test_execution_exports_and_root_absence_match_oracle():
    algebraic_names = {"AlgebraicSystemSpec", "AlgebraicUpdatePhase"}

    for name in API["execution_exports"]:
        assert name in execution.__all__
        leaf = algebraic_module if name in algebraic_names else stokes_module
        assert getattr(execution, name) is getattr(leaf, name)
        if name in algebraic_names:
            assert getattr(execution, name) is getattr(
                canonical_algebraic_module,
                name,
            )
    assert systems_package.__all__ == []
    for name in algebraic_names:
        assert not hasattr(systems_package, name)
    for name in API["root_absent"]:
        assert name not in pssolver.__all__
        assert not hasattr(pssolver, name)
    assert INCOMPRESSIBLE_STOKES_CAPABILITY == (
        API["incompressible_stokes_system_spec"]["capability"]
    )


@pytest.mark.parametrize(
    ("enum_type", "oracle_name", "module_key"),
    (
        (
            AlgebraicUpdatePhase,
            "algebraic_update_phase",
            "target_canonical_module",
        ),
        (PressureGauge, "pressure_gauge", "pre_extraction_module"),
        (
            TangentialZeroModePolicy,
            "tangential_zero_mode_policy",
            "pre_extraction_module",
        ),
    ),
)
def test_enum_surfaces_identity_and_pickle_match_oracle(
    enum_type,
    oracle_name,
    module_key,
):
    oracle = API[oracle_name]

    assert oracle["pre_extraction_module"] == oracle["legacy_module"]
    assert enum_type.__module__ == oracle[module_key]
    assert oracle["legacy_module"].startswith("pssolver.execution.")
    assert oracle["target_canonical_module"].startswith("pssolver.systems.")
    assert enum_type.__qualname__ == oracle["qualname"]
    assert issubclass(enum_type, str)
    assert [(member.name, member.value) for member in enum_type] == [
        tuple(item) for item in oracle["members"]
    ]
    for member in enum_type:
        assert enum_type(member.value) is member
        assert member == member.value
        assert hash(member) == hash(member.value)
        for protocol in (0, 2, 4, pickle.HIGHEST_PROTOCOL):
            assert pickle.loads(pickle.dumps(member, protocol=protocol)) is member
    for protocol in (0, 2, 4, pickle.HIGHEST_PROTOCOL):
        assert pickle.loads(pickle.dumps(enum_type, protocol=protocol)) is enum_type


def test_algebraic_normalization_metadata_and_value_semantics_are_frozen():
    oracle = CASES["algebraic_representative"]
    source_parameters = {
        "viscosity": 0.73,
        "nested": {"modes": (1, 2)},
        "friction": 0,
        "enabled": True,
    }
    value = AlgebraicSystemSpec(
        name="flow",
        capability="incompressible_stokes",
        output_components=["ux", "uy", "uz", "p"],
        dependencies=iter(("force_x", "force_y", "force_z")),
        parameters=source_parameters,
    )
    source_parameters["viscosity"] = 99.0
    source_parameters["nested"] = {"modes": (9,)}

    assert value.output_components == ("ux", "uy", "uz", "p")
    assert value.dependencies == ("force_x", "force_y", "force_z")
    assert isinstance(value.parameters, MappingProxyType)
    assert value._parameters_json == oracle["parameters_json"]
    assert _canonical_json(value.to_metadata()) == oracle["canonical_json"]
    assert hashlib.sha256(
        _canonical_json(value.to_metadata()).encode("utf-8")
    ).hexdigest() == oracle["canonical_sha256"]
    assert hashlib.sha256(repr(value).encode("utf-8")).hexdigest() == (
        oracle["repr_sha256"]
    )
    assert value.update_phase is AlgebraicUpdatePhase.PRE_EXPLICIT_RHS
    assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        value.name = "changed"
    with pytest.raises(TypeError):
        value.parameters["enabled"] = False

    equivalent = AlgebraicSystemSpec(
        name="flow",
        capability="incompressible_stokes",
        output_components=("ux", "uy", "uz", "p"),
        dependencies=("force_x", "force_y", "force_z"),
        parameters={
            "enabled": True,
            "friction": 0,
            "nested": {"modes": [1, 2]},
            "viscosity": 0.73,
        },
        update_phase=AlgebraicUpdatePhase.PRE_EXPLICIT_RHS,
    )
    assert equivalent == value
    with pytest.raises(TypeError, match="unhashable"):
        hash(value)
    metadata = value.to_metadata()
    metadata["parameters"]["enabled"] = False
    assert value.to_metadata()["parameters"]["enabled"] is True


def test_declaration_enum_defaults_preserve_member_identity():
    algebraic_signature = inspect.signature(AlgebraicSystemSpec)
    assert algebraic_signature.parameters["update_phase"].default is (
        AlgebraicUpdatePhase.PRE_EXPLICIT_RHS
    )
    stokes_signature = inspect.signature(IncompressibleStokesSystemSpec)
    assert stokes_signature.parameters["pressure_gauge"].default is (
        PressureGauge.ZERO_MEAN
    )
    assert stokes_signature.parameters[
        "tangential_zero_mode_policy"
    ].default is TangentialZeroModePolicy.ZERO_MEAN


def _invalid_algebraic_kwargs(case_name: str) -> dict[str, object]:
    values: dict[str, object] = {
        "name": "system",
        "capability": "capability",
        "output_components": ("out",),
        "dependencies": ("dep",),
        "parameters": {},
    }
    if case_name == "name_non_string":
        values["name"] = 1
    elif case_name == "name_empty":
        values["name"] = ""
    elif case_name == "name_bad_identifier":
        values["name"] = "bad-name"
    elif case_name == "capability_non_string":
        values["capability"] = None
    elif case_name == "capability_bad_identifier":
        values["capability"] = "bad capability"
    elif case_name == "outputs_string":
        values["output_components"] = "out"
    elif case_name == "dependencies_string":
        values["dependencies"] = "dep"
    elif case_name == "outputs_non_iterable":
        values["output_components"] = 1
    elif case_name == "dependencies_non_iterable":
        values["dependencies"] = 1
    elif case_name == "outputs_empty":
        values["output_components"] = ()
    elif case_name == "dependencies_empty":
        values["dependencies"] = ()
    elif case_name == "outputs_bad_member":
        values["output_components"] = ("out", "bad-name")
    elif case_name == "dependencies_bad_member":
        values["dependencies"] = ("dep", 2)
    elif case_name == "outputs_duplicate":
        values["output_components"] = ("out", "out")
    elif case_name == "dependencies_duplicate":
        values["dependencies"] = ("dep", "dep")
    elif case_name == "output_dependency_overlap":
        values["output_components"] = ("z", "a")
        values["dependencies"] = ("a", "z")
    elif case_name == "phase_string":
        values["update_phase"] = "pre_explicit_rhs"
    elif case_name == "parameters_non_mapping":
        values["parameters"] = []
    elif case_name == "parameter_key_non_string":
        values["parameters"] = {1: "value"}
    elif case_name == "parameter_key_bad_identifier":
        values["parameters"] = {"bad-key": "value"}
    elif case_name == "parameter_nan":
        values["parameters"] = {"value": float("nan")}
    elif case_name == "parameter_infinity":
        values["parameters"] = {"value": float("inf")}
    elif case_name == "parameter_set":
        values["parameters"] = {"value": {1, 2}}
    elif case_name == "parameter_cycle":
        cycle: list[object] = []
        cycle.append(cycle)
        values["parameters"] = {"value": cycle}
    else:
        raise AssertionError(f"unknown algebraic invalid case: {case_name}")
    return values


@pytest.mark.parametrize(
    "case",
    CASES["algebraic_invalid_cases"],
    ids=lambda case: case[0],
)
def test_algebraic_validation_matrix_is_frozen(case):
    name, exception_name, message = case
    exception_type = {"TypeError": TypeError, "ValueError": ValueError}[
        exception_name
    ]
    with pytest.raises(exception_type) as caught:
        AlgebraicSystemSpec(**_invalid_algebraic_kwargs(name))
    assert message in str(caught.value)


def test_algebraic_class_global_loads_but_instances_remain_unpickleable():
    for module_name in (
        "pssolver.systems.algebraic",
        "pssolver.execution.algebraic",
        "pssolver.execution",
    ):
        payload = (
            f"c{module_name}\nAlgebraicSystemSpec\n.".encode("ascii")
        )
        assert pickle.loads(payload) is AlgebraicSystemSpec
    for protocol in (0, 2, 4, pickle.HIGHEST_PROTOCOL):
        assert (
            pickle.loads(pickle.dumps(AlgebraicSystemSpec, protocol=protocol))
            is AlgebraicSystemSpec
        )
    value = _algebraic_spec()
    for protocol in (0, 2, 4, pickle.HIGHEST_PROTOCOL):
        with pytest.raises(
            TypeError,
            match="cannot pickle 'mappingproxy' object",
        ):
            pickle.dumps(value, protocol=protocol)


@pytest.mark.parametrize(
    "case",
    CASES["stokes_valid_cases"],
    ids=lambda case: case["name"],
)
def test_stokes_policy_matrix_metadata_and_round_trip_are_frozen(case):
    policy = TangentialZeroModePolicy(case["policy"])
    value = _stokes_spec(friction=case["friction"], policy=policy)
    generic = value.to_algebraic_system_spec()

    assert value.force_components == ("force_x", "force_y", "force_z")
    assert value.velocity_components == ("ux", "uy", "uz")
    assert type(value.viscosity) is float
    assert type(value.friction) is float
    assert type(generic) is AlgebraicSystemSpec
    assert generic.name == "flow"
    assert generic.capability == INCOMPRESSIBLE_STOKES_CAPABILITY
    assert generic.output_components == ("ux", "uy", "uz", "p")
    assert generic.dependencies == ("force_x", "force_y", "force_z")
    assert generic.update_phase is AlgebraicUpdatePhase.PRE_EXPLICIT_RHS
    assert set(generic.parameters) == {
        "friction",
        "pressure_gauge",
        "tangential_zero_mode_policy",
        "viscosity",
    }
    recovered = IncompressibleStokesSystemSpec.from_algebraic_system_spec(
        generic
    )
    assert type(recovered) is IncompressibleStokesSystemSpec
    assert recovered == value
    canonical = _canonical_json(value.to_metadata())
    assert canonical == case["canonical_json"]
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
        case["canonical_sha256"]
    )
    if "repr_sha256" in case:
        assert hashlib.sha256(repr(value).encode("utf-8")).hexdigest() == (
            case["repr_sha256"]
        )

    equal_value = _stokes_spec(friction=case["friction"], policy=policy)
    assert value == equal_value
    assert hash(value) == hash(equal_value)
    assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        value.viscosity = 2.0
    metadata = value.to_metadata()
    metadata["parameters"]["viscosity"] = 99.0
    assert value.to_metadata()["parameters"]["viscosity"] == 0.73


def _invalid_stokes_kwargs(case_name: str) -> dict[str, object]:
    values: dict[str, object] = {
        "name": "flow",
        "force_components": ("force_x", "force_y", "force_z"),
        "velocity_components": ("ux", "uy", "uz"),
        "pressure_component": "p",
        "viscosity": 0.73,
        "friction": 0.0,
        "pressure_gauge": PressureGauge.ZERO_MEAN,
        "tangential_zero_mode_policy": (
            TangentialZeroModePolicy.ZERO_MEAN
        ),
    }
    if case_name == "name_non_string":
        values["name"] = 1
    elif case_name == "name_bad_identifier":
        values["name"] = "bad-name"
    elif case_name == "forces_string":
        values["force_components"] = "force"
    elif case_name == "forces_non_iterable":
        values["force_components"] = 1
    elif case_name == "forces_wrong_count":
        values["force_components"] = ("fx", "fy")
    elif case_name == "forces_bad_identifier":
        values["force_components"] = ("fx", "fy", "bad-name")
    elif case_name == "forces_duplicate":
        values["force_components"] = ("fx", "fy", "fx")
    elif case_name == "velocities_string":
        values["velocity_components"] = "velocity"
    elif case_name == "velocities_non_iterable":
        values["velocity_components"] = 1
    elif case_name == "velocities_wrong_count":
        values["velocity_components"] = ("ux", "uy")
    elif case_name == "velocities_bad_identifier":
        values["velocity_components"] = ("ux", "uy", "bad-name")
    elif case_name == "velocities_duplicate":
        values["velocity_components"] = ("ux", "uy", "ux")
    elif case_name == "pressure_non_string":
        values["pressure_component"] = 1
    elif case_name == "pressure_bad_identifier":
        values["pressure_component"] = "bad-name"
    elif case_name == "cross_group_overlap":
        values["velocity_components"] = ("force_x", "uy", "uz")
    elif case_name == "force_pressure_overlap":
        values["pressure_component"] = "force_x"
    elif case_name == "velocity_pressure_overlap":
        values["pressure_component"] = "ux"
    elif case_name == "viscosity_bool":
        values["viscosity"] = True
    elif case_name == "viscosity_zero":
        values["viscosity"] = 0.0
    elif case_name == "viscosity_nan":
        values["viscosity"] = float("nan")
    elif case_name == "friction_bool":
        values["friction"] = True
    elif case_name == "friction_negative":
        values["friction"] = -0.1
    elif case_name == "friction_infinite":
        values["friction"] = float("inf")
    elif case_name == "gauge_string":
        values["pressure_gauge"] = "zero_mean"
    elif case_name == "policy_string":
        values["tangential_zero_mode_policy"] = "zero_mean"
    elif case_name == "zero_mean_positive_friction":
        values["friction"] = 0.1
    elif case_name == "friction_policy_zero":
        values["tangential_zero_mode_policy"] = (
            TangentialZeroModePolicy.FRICTION
        )
    else:
        raise AssertionError(f"unknown Stokes invalid case: {case_name}")
    return values


@pytest.mark.parametrize(
    "case",
    CASES["stokes_invalid_cases"],
    ids=lambda case: case[0],
)
def test_stokes_constructor_validation_matrix_is_frozen(case):
    name, exception_name, message = case
    exception_type = {"TypeError": TypeError, "ValueError": ValueError}[
        exception_name
    ]
    with pytest.raises(exception_type) as caught:
        IncompressibleStokesSystemSpec(**_invalid_stokes_kwargs(name))
    assert message in str(caught.value)


def _invalid_generic_stokes(case_name: str) -> object:
    if case_name == "wrong_type":
        return object()
    values: dict[str, object] = {
        "name": "flow",
        "capability": INCOMPRESSIBLE_STOKES_CAPABILITY,
        "output_components": ("ux", "uy", "uz", "p"),
        "dependencies": ("force_x", "force_y", "force_z"),
        "parameters": {
            "friction": 0.0,
            "pressure_gauge": "zero_mean",
            "tangential_zero_mode_policy": "zero_mean",
            "viscosity": 0.73,
        },
    }
    if case_name == "wrong_capability":
        values["capability"] = "other_capability"
    elif case_name == "wrong_dependency_count":
        values["dependencies"] = ("force_x", "force_y")
    elif case_name == "wrong_output_count":
        values["output_components"] = ("ux", "uy", "p")
    elif case_name == "missing_parameter":
        values["parameters"] = {
            key: value
            for key, value in values["parameters"].items()
            if key != "friction"
        }
    elif case_name == "extra_parameter":
        values["parameters"] = {**values["parameters"], "extra": 1}
    elif case_name == "unsupported_gauge":
        values["parameters"] = {
            **values["parameters"],
            "pressure_gauge": "fixed",
        }
    elif case_name == "unsupported_policy":
        values["parameters"] = {
            **values["parameters"],
            "tangential_zero_mode_policy": "unsupported",
        }
    elif case_name == "invalid_viscosity":
        values["parameters"] = {**values["parameters"], "viscosity": 0.0}
    elif case_name == "invalid_zero_mean_pair":
        values["parameters"] = {**values["parameters"], "friction": 0.1}
    elif case_name == "invalid_friction_pair":
        values["parameters"] = {
            **values["parameters"],
            "tangential_zero_mode_policy": "friction",
        }
    else:
        raise AssertionError(f"unknown generic Stokes invalid case: {case_name}")
    return AlgebraicSystemSpec(**values)


@pytest.mark.parametrize(
    "case",
    CASES["stokes_from_algebraic_invalid_cases"],
    ids=lambda case: case[0],
)
def test_stokes_inverse_validation_matrix_is_frozen(case):
    name, exception_name, message = case
    exception_type = {"TypeError": TypeError, "ValueError": ValueError}[
        exception_name
    ]
    with pytest.raises(exception_type) as caught:
        IncompressibleStokesSystemSpec.from_algebraic_system_spec(
            _invalid_generic_stokes(name)
        )
    assert message in str(caught.value)


def test_legacy_algebraic_update_phase_pickles_and_globals_remain_loadable():
    oracle = CASES["legacy_algebraic_update_phase_pickle"]
    class_payload = base64.b64decode(oracle["class_base64"])
    member_payload = base64.b64decode(oracle["member_base64"])

    assert hashlib.sha256(class_payload).hexdigest() == oracle["class_sha256"]
    assert hashlib.sha256(member_payload).hexdigest() == (
        oracle["member_sha256"]
    )
    assert pickle.loads(class_payload) is AlgebraicUpdatePhase
    assert (
        pickle.loads(member_payload)
        is AlgebraicUpdatePhase.PRE_EXPLICIT_RHS
    )

    for module_name in (
        "pssolver.systems.algebraic",
        "pssolver.execution.algebraic",
        "pssolver.execution",
    ):
        payload = (
            f"c{module_name}\nAlgebraicUpdatePhase\n.".encode("ascii")
        )
        enum_type = pickle.loads(payload)
        assert enum_type is AlgebraicUpdatePhase
        assert (
            enum_type("pre_explicit_rhs")
            is AlgebraicUpdatePhase.PRE_EXPLICIT_RHS
        )


def test_legacy_stokes_pickle_and_global_paths_remain_loadable():
    oracle = CASES["legacy_stokes_pickle"]
    payload = base64.b64decode(oracle["base64"])

    assert hashlib.sha256(payload).hexdigest() == oracle["sha256"]
    restored = pickle.loads(payload)
    assert type(restored) is IncompressibleStokesSystemSpec
    assert restored == _stokes_spec()
    assert restored.pressure_gauge is PressureGauge.ZERO_MEAN
    assert (
        restored.tangential_zero_mode_policy
        is TangentialZeroModePolicy.ZERO_MEAN
    )

    leaf_names = (
        "INCOMPRESSIBLE_STOKES_CAPABILITY",
        "IncompressibleStokesSystemSpec",
        "PressureGauge",
        "TangentialZeroModePolicy",
    )
    for module_name in ("pssolver.execution.stokes", "pssolver.execution"):
        for name in leaf_names:
            payload = f"c{module_name}\n{name}\n.".encode("ascii")
            assert pickle.loads(payload) is getattr(execution, name)


@pytest.mark.parametrize(
    "case",
    CASES["stokes_valid_cases"],
    ids=lambda case: case["name"],
)
def test_current_stokes_instances_pickle_round_trip(case):
    value = _stokes_spec(
        friction=case["friction"],
        policy=TangentialZeroModePolicy(case["policy"]),
    )
    for protocol in (0, 2, 4, pickle.HIGHEST_PROTOCOL):
        restored = pickle.loads(pickle.dumps(value, protocol=protocol))
        assert type(restored) is IncompressibleStokesSystemSpec
        assert restored == value
