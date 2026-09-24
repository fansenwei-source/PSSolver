"""Phase 2 compatibility oracles for the flat Plane run specification.

These tests intentionally characterize the v0.1.2 facade before it is
decomposed into smaller configuration objects.  They freeze supported public
shape and persistent schema-v1 identity without exercising the numerical hot
path, which is already covered by the Stage O and Stage S qualification tests.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import FrozenInstanceError, MISSING, fields, replace
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
import pickle

import pytest

import pssolver
import pssolver.configuration as configuration
from pssolver.configuration import (
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneBerisEdwardsRunSpec,
    PlaneFreeSlipBoundaryConditions,
    create_plane_beris_edwards_run_spec,
    parse_plane_beris_edwards_run_spec,
)
from pssolver.configuration import plane_beris_edwards as configuration_module
from pssolver.core import (
    BoundarySet,
    HomogeneousDirichletBC,
    PeriodicBC,
)
from pssolver.workflows import read_plane_checkpoint_header


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "configuration"
API_ORACLE_PATH = FIXTURE_ROOT / "plane_run_spec_api_v1.json"
CASES_ORACLE_PATH = FIXTURE_ROOT / "plane_run_spec_v1_cases.json"
CHECKPOINT_FIXTURE = FIXTURE_ROOT / "plane_checkpoint_v1_header"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


API_ORACLE = _load_json(API_ORACLE_PATH)
CASES_ORACLE = _load_json(CASES_ORACLE_PATH)


def _base_kwargs(**overrides: object) -> dict[str, object]:
    values = dict(CASES_ORACLE["base_factory_kwargs"])
    values.update(overrides)
    return values


def _spec(**overrides: object) -> PlaneBerisEdwardsRunSpec:
    return create_plane_beris_edwards_run_spec(**_base_kwargs(**overrides))


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _callable_name(value: object | None) -> str | None:
    if value is None:
        return None
    return f"{value.__module__}.{value.__qualname__}"


def _annotation_text(value: object) -> str | None:
    if value is inspect.Signature.empty:
        return None
    if isinstance(value, str):
        return value
    return inspect.formatannotation(value)


def _value_contract(
    value: object,
    *,
    missing_kind: str = "required",
) -> dict[str, object]:
    if value is inspect.Signature.empty or value is MISSING:
        return {"kind": missing_kind}
    if value == argparse.SUPPRESS:
        return {"kind": "argparse_suppress"}
    if isinstance(value, PlaneFreeSlipBoundaryConditions):
        return {
            "kind": "plane_free_slip_boundaries",
            "value": value.to_metadata(),
        }
    if isinstance(value, Enum):
        return {
            "kind": "enum",
            "type": _qualified_type(value),
            "value": value.value,
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "value": list(value)}
    return {"kind": "value", "value": value}


def _signature_contract(function: object) -> dict[str, object]:
    signature = inspect.signature(function)
    parameters = []
    for parameter in signature.parameters.values():
        parameters.append(
            {
                "annotation": _annotation_text(parameter.annotation),
                "default": _value_contract(parameter.default),
                "kind": parameter.kind.name,
                "name": parameter.name,
            }
        )
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "parameters": parameters,
        "return_annotation": _annotation_text(signature.return_annotation),
    }


def _dataclass_field_contract(field: object) -> dict[str, object]:
    return {
        "default": _value_contract(field.default, missing_kind="missing"),
        "name": field.name,
        "type": _annotation_text(field.type),
    }


def _argparse_action_name(action: argparse.Action) -> str:
    if isinstance(action, argparse._HelpAction):
        return "help"
    if isinstance(action, argparse._StoreTrueAction):
        return "store_true"
    if isinstance(action, argparse._StoreAction):
        return "store"
    raise AssertionError(f"uncharacterized argparse action: {type(action)!r}")


def _argparse_action_contract(action: argparse.Action) -> dict[str, object]:
    choices = None if action.choices is None else list(action.choices)
    metavar = action.metavar
    if isinstance(metavar, tuple):
        metavar = list(metavar)
    return {
        "action": _argparse_action_name(action),
        "choices": choices,
        "const": _value_contract(action.const),
        "default": _value_contract(action.default),
        "dest": action.dest,
        "metavar": metavar,
        "nargs": action.nargs,
        "option_strings": list(action.option_strings),
        "required": action.required,
        "type": _callable_name(action.type),
    }


def _explicit_member_contract() -> list[dict[str, object]]:
    source_path = Path(configuration_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path.name)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PlaneBerisEdwardsRunSpec"
    )
    observed = []
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        kind = "method"
        if any(
            isinstance(decorator, ast.Name) and decorator.id == "property"
            for decorator in node.decorator_list
        ):
            kind = "property"
        observed.append({"kind": kind, "name": node.name})
    return observed


def _canonical_json(value: object) -> str:
    encoding = CASES_ORACLE["canonical_json_encoding"]
    assert encoding["encoding"] == "utf-8"
    return json.dumps(
        value,
        allow_nan=encoding["allow_nan"],
        separators=tuple(encoding["separators"]),
        sort_keys=encoding["sort_keys"],
    )


def test_run_spec_dataclass_surface_matches_v1_oracle():
    oracle = API_ORACLE["run_spec"]
    observed_fields = list(fields(PlaneBerisEdwardsRunSpec))

    assert PlaneBerisEdwardsRunSpec.__module__ == oracle["module"]
    assert PlaneBerisEdwardsRunSpec.__qualname__ == oracle["qualname"]
    assert [_dataclass_field_contract(item) for item in observed_fields] == (
        oracle["fields"]
    )
    assert list(PlaneBerisEdwardsRunSpec.__slots__) == oracle["slots"]
    assert list(PlaneBerisEdwardsRunSpec.__match_args__) == oracle["match_args"]

    parameters = PlaneBerisEdwardsRunSpec.__dataclass_params__
    observed_parameters = {
        name: getattr(parameters, name)
        for name in oracle["dataclass_parameters"]
    }
    assert observed_parameters == oracle["dataclass_parameters"]

    uniform = oracle["uniform_field_parameters"]
    for item in observed_fields:
        assert item.compare is uniform["compare"]
        assert item.default_factory is MISSING
        assert uniform["default_factory"] == "missing"
        assert item.hash is uniform["hash"]
        assert item.init is uniform["init"]
        assert item.kw_only is uniform["kw_only"]
        assert item.repr is uniform["repr"]

    class_signature = inspect.signature(PlaneBerisEdwardsRunSpec)
    assert list(class_signature.parameters) == oracle["match_args"]
    for parameter, field_oracle in zip(
        class_signature.parameters.values(),
        oracle["fields"],
        strict=True,
    ):
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert _annotation_text(parameter.annotation) == field_oracle["type"]
        expected_default = field_oracle["default"]
        observed_default = _value_contract(
            parameter.default,
            missing_kind="missing",
        )
        assert observed_default == expected_default


def test_run_spec_value_semantics_repr_and_pickle_are_frozen():
    left = _spec()
    right = _spec()

    assert left == right
    assert hash(left) == hash(right)
    assert not hasattr(left, "__dict__")
    with pytest.raises(FrozenInstanceError):
        left.dt = 0.005

    repr_sha256 = hashlib.sha256(repr(left).encode("utf-8")).hexdigest()
    assert repr_sha256 == API_ORACLE["run_spec"]["default_repr_sha256"]

    restored = pickle.loads(pickle.dumps(left))
    assert restored == left
    assert type(restored) is PlaneBerisEdwardsRunSpec
    legacy_global = (
        b"cpssolver.configuration.plane_beris_edwards\n"
        b"PlaneBerisEdwardsRunSpec\n."
    )
    assert pickle.loads(legacy_global) is PlaneBerisEdwardsRunSpec


def test_run_spec_explicit_member_inventory_and_return_types_are_frozen():
    oracle = API_ORACLE["run_spec"]["explicit_members"]
    expected_shape = [
        {"kind": item["kind"], "name": item["name"]} for item in oracle
    ]
    assert _explicit_member_contract() == expected_shape

    spec = _spec()
    for item in oracle:
        name = item["name"]
        if name == "__post_init__":
            continue
        descriptor = PlaneBerisEdwardsRunSpec.__dict__[name]
        if item["kind"] == "property":
            assert isinstance(descriptor, property)
            value = getattr(spec, name)
        else:
            assert inspect.isfunction(descriptor)
            value = getattr(spec, name)()
        assert _qualified_type(value) == item["representative_type"]

    default_case = CASES_ORACLE["cases"][0]
    expected = default_case["expected_metadata"]
    assert spec.S_initial == expected["model"]["initial_s"]
    assert spec.spectral_refresh_mode == "physical_time"
    assert spec.spectral_refresh_requested_time == 0.2
    assert spec.spectral_refresh_requested_steps is None
    assert spec.spectral_refresh_interval_steps == 20
    assert spec.spectral_refresh_effective_time == 0.2
    assert spec.shendruk_preset.to_metadata() == expected["preset"]
    assert spec.domain.to_metadata() == expected["geometry"]["domain"]
    assert spec.geometry.to_metadata() == expected["geometry"]
    assert spec.numerics.to_metadata() == expected["numerics"]
    assert spec.to_metadata() == expected
    assert spec.canonical_sha256() == default_case["canonical_sha256"]
    assert spec.runtime_identity_metadata() == (
        default_case["expected_runtime_identity_metadata"]
    )
    assert spec.runtime_identity_sha256() == (
        default_case["runtime_identity_sha256"]
    )
    assert spec.identity_metadata() == default_case["expected_identity_metadata"]
    assert spec.runtime_selection_metadata() == (
        default_case["expected_runtime_selection_metadata"]
    )


def test_factory_parse_signatures_and_configuration_exports_match_v1_oracle():
    assert _signature_contract(create_plane_beris_edwards_run_spec) == (
        API_ORACLE["factory"]
    )
    assert _signature_contract(parse_plane_beris_edwards_run_spec) == (
        API_ORACLE["parse_function"]
    )
    assert list(configuration.__all__) == API_ORACLE["configuration_exports"]
    for name in configuration.__all__:
        assert getattr(configuration, name) is getattr(configuration_module, name)
        assert name not in pssolver.__all__


def test_parser_semantic_contract_matches_v1_oracle():
    parser = configuration_module._parser()
    oracle = API_ORACLE["parser"]
    help_actions = [action for action in parser._actions if action.dest == "help"]
    domain_actions = [
        action for action in parser._actions if action.dest != "help"
    ]

    assert parser.allow_abbrev is oracle["allow_abbrev"]
    assert len(domain_actions) == oracle["domain_action_count"]
    observed_actions = [
        _argparse_action_contract(action) for action in domain_actions
    ]
    runtime_action = next(
        action for action in observed_actions
        if action["dest"] == "runtime_path"
    )
    oracle_runtime_action = next(
        action for action in oracle["domain_actions"]
        if action["dest"] == "runtime_path"
    )
    assert runtime_action["choices"] == [
        *oracle_runtime_action["choices"],
        "compiled_v2",
    ]
    runtime_action["choices"] = oracle_runtime_action["choices"]
    assert observed_actions == oracle["domain_actions"]
    assert [_argparse_action_contract(action) for action in help_actions] == (
        oracle["help_actions"]
    )
    observed_groups = [
        {
            "destinations": [action.dest for action in group._group_actions],
            "required": group.required,
        }
        for group in parser._mutually_exclusive_groups
    ]
    assert observed_groups == oracle["mutually_exclusive_groups"]
    factory_names = {
        parameter["name"] for parameter in API_ORACLE["factory"]["parameters"]
    }
    assert {action.dest for action in domain_actions} == factory_names


@pytest.mark.parametrize(
    "case",
    CASES_ORACLE["cases"],
    ids=lambda case: case["name"],
)
def test_schema_v1_metadata_and_hash_golden_cases(case):
    spec = _spec(**case["factory_overrides"])
    metadata = spec.to_metadata()
    runtime_metadata = spec.runtime_identity_metadata()

    assert metadata == case["expected_metadata"]
    assert _canonical_json(metadata) == case["expected_canonical_json"]
    assert spec.canonical_sha256() == case["canonical_sha256"]
    assert runtime_metadata == case["expected_runtime_identity_metadata"]
    assert _canonical_json(runtime_metadata) == (
        case["expected_runtime_identity_json"]
    )
    assert spec.runtime_identity_sha256() == case["runtime_identity_sha256"]
    assert spec.identity_metadata() == case["expected_identity_metadata"]
    assert spec.runtime_selection_metadata() == (
        case["expected_runtime_selection_metadata"]
    )


IDENTITY_MUTATIONS = {
    "activity_number": {"activity_number": 19.0},
    "output_dir": {"output_dir": "oracle/identity-output"},
    "height": {"height": 21.0},
    "parameterization": {"parameterization": "fixed-k"},
    "frank_k": {"frank_k": 0.02},
    "coefficient_min": {"coefficient_min": 0.011},
    "coefficient_max": {"coefficient_max": 0.06},
    "lx": {"lx": 101.0},
    "ly": {"ly": 102.0},
    "nx": {"nx": 258},
    "ny": {"ny": 260},
    "nz": {"nz": 80},
    "dt": {"dt": 0.005},
    "steps": {"steps": 12_000},
    "save_start_step": {"save_start_step": 4_000},
    "save_interval": {"save_interval": 750},
    "diagnostic_interval": {"diagnostic_interval": 125},
    "seed": {"seed": 25},
    "num_defect_pairs": {"num_defect_pairs": 7},
    "defect_min_separation": {"defect_min_separation": 11.0},
    "defect_core_radius": {"defect_core_radius": 1.6},
    "background_angle": {"background_angle": 0.1},
    "twist_amplitude": {"twist_amplitude": 0.02},
    "twist_modes": {"twist_modes": (1, 4)},
    "ldg_a": {"ldg_a": -0.01},
    "ldg_b": {"ldg_b": -0.31},
    "ldg_c": {"ldg_c": 0.31},
    "gamma": {"gamma": 3.0},
    "flow_alignment": {"flow_alignment": 0.31},
    "eta": {"eta": 0.7},
    "zero_mode_policy": {"zero_mode_policy": "friction"},
    "friction_mode_fric": {"friction_mode_fric": 0.2},
    "dealias_rule": {"dealias_rule": "two_thirds"},
    "projected_transform_execution": {
        "projected_transform_execution": "full"
    },
    "beta": {"beta": -0.9},
    "initial_s": {"initial_s": 0.25},
    "device": {"device": "cpu"},
    "dtype": {"dtype": "float64"},
    "molecular_field_linear_space": {
        "molecular_field_linear_space": "physical"
    },
    "stress_divergence_sum_space": {
        "stress_divergence_sum_space": "physical"
    },
    "pointwise_execution": {"pointwise_execution": "eager"},
    "transform_execution_order": {"transform_execution_order": "legacy"},
    "spectral_storage": {"spectral_storage": "full_complex"},
    "tf32": {"tf32": "on"},
    "spectral_refresh": {"spectral_refresh_steps": 7},
    "diagnostics": {"diagnostics": True},
    "disable_q_gradient_reuse": {"disable_q_gradient_reuse": True},
    "save_hydrodynamics": {"save_hydrodynamics": True},
    "validation_config_sha256": {"validation_config_sha256": "a" * 64},
    "dry_run": {"dry_run": True},
    "checkpoint_interval": {"checkpoint_interval": 250},
    "restart_from": {"restart_from": "oracle/checkpoint"},
    "runtime_path": {"runtime_path": "separated_canary"},
}


def _alternate_boundaries() -> PlaneFreeSlipBoundaryConditions:
    alternate_q = BoundarySet(
        (PeriodicBC(), PeriodicBC(), HomogeneousDirichletBC())
    )
    return replace(PLANE_FREE_SLIP_BOUNDARIES, q=alternate_q)


def _identity_pair(field_name: str):
    baseline_overrides = {}
    if field_name == "frank_k":
        baseline_overrides["parameterization"] = "fixed-k"
    elif field_name == "transform_execution_order":
        baseline_overrides["spectral_storage"] = "full_complex"

    baseline = _spec(**baseline_overrides)
    if field_name == "boundaries":
        return baseline, replace(baseline, boundaries=_alternate_boundaries())
    changed_overrides = dict(baseline_overrides)
    changed_overrides.update(IDENTITY_MUTATIONS[field_name])
    return baseline, _spec(**changed_overrides)


IDENTITY_CASES = [
    (field_name, partition)
    for partition, field_names in CASES_ORACLE["identity_partition"].items()
    for field_name in field_names
]


def test_identity_partition_covers_every_run_spec_field_once():
    partition_names = [field_name for field_name, _ in IDENTITY_CASES]
    facade_names = [item.name for item in fields(PlaneBerisEdwardsRunSpec)]

    assert len(partition_names) == len(set(partition_names)) == 54
    assert set(partition_names) == set(facade_names)
    assert set(IDENTITY_MUTATIONS) | {"boundaries"} == set(facade_names)


@pytest.mark.parametrize(
    ("field_name", "partition"),
    IDENTITY_CASES,
    ids=lambda value: value,
)
def test_each_field_preserves_the_frozen_identity_partition(
    field_name,
    partition,
):
    baseline, changed = _identity_pair(field_name)
    canonical_changed = (
        baseline.canonical_sha256() != changed.canonical_sha256()
    )
    runtime_changed = (
        baseline.runtime_identity_sha256()
        != changed.runtime_identity_sha256()
    )

    assert canonical_changed is (
        partition in {"canonical_and_runtime", "canonical_only"}
    )
    assert runtime_changed is (
        partition in {"canonical_and_runtime", "runtime_only"}
    )


def test_shendruk_raw_requests_preserve_resolved_identity_behavior():
    paper = _spec()
    paper_changed_raw_k = _spec(frank_k=0.02)
    paper_changed_upper_bound = _spec(coefficient_max=0.06)
    paper_changed_lower_bound = _spec(coefficient_min=0.011)

    assert paper.frank_k != paper_changed_raw_k.frank_k
    assert paper.to_metadata() == paper_changed_raw_k.to_metadata()
    assert paper.canonical_sha256() == paper_changed_raw_k.canonical_sha256()
    assert (
        paper.runtime_identity_sha256()
        == paper_changed_raw_k.runtime_identity_sha256()
    )
    assert paper.to_metadata() == paper_changed_upper_bound.to_metadata()
    assert paper.canonical_sha256() != (
        paper_changed_lower_bound.canonical_sha256()
    )

    fixed = _spec(parameterization="fixed-k")
    fixed_changed_k = _spec(parameterization="fixed-k", frank_k=0.02)
    fixed_changed_bounds = _spec(
        parameterization="fixed-k",
        coefficient_min=0.02,
        coefficient_max=0.06,
    )
    assert fixed.canonical_sha256() != fixed_changed_k.canonical_sha256()
    assert (
        fixed.runtime_identity_sha256()
        != fixed_changed_k.runtime_identity_sha256()
    )
    assert fixed.to_metadata() == fixed_changed_bounds.to_metadata()


def test_dormant_friction_request_remains_bound_into_both_identities():
    default = _spec(zero_mode_policy="zero_mean", friction_mode_fric=0.1)
    changed = _spec(zero_mode_policy="zero_mean", friction_mode_fric=0.2)

    assert default.zero_mode_policy == changed.zero_mode_policy == "zero_mean"
    assert default.to_metadata()["zero_mode"] == {
        "policy": "zero_mean",
        "friction_mode_fric": 0.1,
    }
    assert changed.to_metadata()["zero_mode"]["friction_mode_fric"] == 0.2
    assert default.canonical_sha256() != changed.canonical_sha256()
    assert (
        default.runtime_identity_sha256()
        != changed.runtime_identity_sha256()
    )


@pytest.mark.parametrize(
    ("zero_mode_policy", "expected_effective_friction"),
    (("zero_mean", 0.0), ("friction", 0.2)),
)
def test_dry_run_keeps_requested_and_effective_friction_distinct(
    tmp_path,
    capsys,
    monkeypatch,
    zero_mode_policy,
    expected_effective_friction,
):
    from pssolver.applications import plane_beris_edwards as application

    def _unexpected_runtime_build(*args, **kwargs):
        raise AssertionError("dry-run must not construct a numerical runtime")

    monkeypatch.setattr(
        application,
        "build_package_simulation_runtime",
        _unexpected_runtime_build,
    )
    output_dir = tmp_path / zero_mode_policy
    spec = _spec(
        output_dir=output_dir,
        zero_mode_policy=zero_mode_policy,
        friction_mode_fric=0.2,
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        dry_run=True,
    )

    assert application.run_plane_beris_edwards(spec, emit_metadata=True) is None
    metadata = json.loads(capsys.readouterr().out)

    assert metadata["configuration"] == spec.identity_metadata()
    assert metadata["runtime_selection"] == spec.runtime_selection_metadata()
    assert spec.to_metadata()["zero_mode"]["friction_mode_fric"] == 0.2
    assert metadata["model"]["parameters"]["fric"] == (
        expected_effective_friction
    )
    assert metadata["friction"] == expected_effective_friction
    assert not output_dir.exists()


def test_path_spelling_is_not_resolved_but_is_excluded_from_runtime_identity():
    spelled = _spec(
        output_dir="oracle/a/../run",
        restart_from="oracle/c/../checkpoint",
    )
    normalized = _spec(
        output_dir="oracle/run",
        restart_from="oracle/checkpoint",
    )

    assert spelled.to_metadata()["output_dir"] == "oracle/a/../run"
    assert (
        spelled.to_metadata()["workflow"]["restart_from"]
        == "oracle/c/../checkpoint"
    )
    assert spelled.canonical_sha256() != normalized.canonical_sha256()
    assert (
        spelled.runtime_identity_sha256()
        == normalized.runtime_identity_sha256()
    )


def test_v1_checkpoint_header_binds_the_default_runtime_identity():
    expected = _load_json(CHECKPOINT_FIXTURE / "checkpoint.json")
    header = read_plane_checkpoint_header(CHECKPOINT_FIXTURE)

    assert header.runtime_path.value == expected["runtime_path"]
    assert header.runtime_identity_sha256 == expected["runtime_identity_sha256"]
    assert header.completed_steps == expected["completed_steps"]
    assert _spec().runtime_identity_sha256() == header.runtime_identity_sha256


def test_alternate_boundary_request_changes_identity_but_not_runtime_globals():
    requested = _alternate_boundaries()
    default = _spec()
    changed = replace(default, boundaries=requested)
    qualified_legacy = PLANE_FREE_SLIP_BOUNDARIES.to_legacy()

    assert changed.boundaries == requested
    assert changed.to_metadata()["boundaries"] == requested.to_metadata()
    assert changed.canonical_sha256() != default.canonical_sha256()
    assert (
        changed.runtime_identity_sha256()
        != default.runtime_identity_sha256()
    )

    from pssolver.applications import plane_beris_edwards as application
    from pssolver.runtime import plane_legacy

    assert application.Q_BC == qualified_legacy["q"]
    assert plane_legacy.Q_BOUNDARIES == qualified_legacy["q"]
    assert application.Q_BC != requested.to_legacy()["q"]
    assert plane_legacy.Q_BOUNDARIES != requested.to_legacy()["q"]
    assert "run_spec.boundaries" not in inspect.getsource(
        plane_legacy.build_legacy_plane_runtime
    )


def test_factory_cli_and_direct_constructor_keep_current_validation_timing(
    capsys,
):
    with pytest.raises(ValueError, match="these values must be positive"):
        _spec(dt=-0.01)

    with pytest.raises(SystemExit) as raised:
        parse_plane_beris_edwards_run_spec(
            [
                "--activity-number",
                "18",
                "--output-dir",
                "oracle/invalid",
                "--dt",
                "-0.01",
            ]
        )
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert "these values must be positive: {'dt': -0.01}" in captured.err

    direct = replace(_spec(), dt=-0.01)
    assert direct.dt == -0.01
    assert direct.to_metadata()["workflow"]["dt"] == -0.01
