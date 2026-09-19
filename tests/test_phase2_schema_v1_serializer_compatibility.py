"""Pre-extraction oracles for the Plane schema-v1 serializer authority."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import ast
import hashlib
import json
from pathlib import Path

import pytest

import pssolver
import pssolver.configuration as configuration
from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
    PlaneBerisEdwardsRunSpec,
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration import plane_beris_edwards as facade
from pssolver.configuration import plane_beris_edwards_schema_v1 as schema_v1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "configuration"
    / "plane_run_spec_v1_cases.json"
)
CASES_ORACLE = json.loads(CASES_PATH.read_text(encoding="utf-8"))
FACADE_PATH = (
    PROJECT_ROOT / "pssolver" / "configuration" / "plane_beris_edwards.py"
)
SERIALIZER_PATH = (
    PROJECT_ROOT
    / "pssolver"
    / "configuration"
    / "plane_beris_edwards_schema_v1.py"
)


def _spec(**overrides: object) -> PlaneBerisEdwardsRunSpec:
    values = dict(CASES_ORACLE["base_factory_kwargs"])
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    "case",
    CASES_ORACLE["cases"],
    ids=lambda case: case["name"],
)
def test_all_schema_v1_documents_and_hashes_match_the_frozen_oracle(case):
    spec = _spec(**case["factory_overrides"])

    assert spec.to_metadata() == case["expected_metadata"]
    assert spec.canonical_sha256() == case["canonical_sha256"]
    assert spec.runtime_identity_metadata() == (
        case["expected_runtime_identity_metadata"]
    )
    assert spec.runtime_identity_sha256() == case["runtime_identity_sha256"]
    assert spec.identity_metadata() == case["expected_identity_metadata"]
    assert spec.runtime_selection_metadata() == (
        case["expected_runtime_selection_metadata"]
    )


def test_schema_v1_metadata_calls_return_fresh_mutation_isolated_documents():
    spec = _spec()
    first = spec.to_metadata()
    second = spec.to_metadata()

    assert first == second
    assert first is not second
    assert first["model"] is not second["model"]
    assert first["workflow"] is not second["workflow"]
    assert first["initial_condition"] is not second["initial_condition"]
    first["model"]["ldg_a"] = 123.0
    first["initial_condition"]["twist_modes"].append(99)
    assert spec.to_metadata() == second

    first_runtime = spec.runtime_identity_metadata()
    second_runtime = spec.runtime_identity_metadata()
    assert first_runtime == second_runtime
    assert first_runtime is not second_runtime
    assert first_runtime["model"] is not second_runtime["model"]
    first_runtime["model"]["ldg_a"] = 123.0
    assert spec.runtime_identity_metadata() == second_runtime


def test_canonical_hash_dispatches_through_the_public_metadata_method(
    monkeypatch,
):
    spec = _spec()
    replacement = {"schema_version": 1, "sentinel": ["facade-dispatch"]}
    calls: list[PlaneBerisEdwardsRunSpec] = []

    def metadata(target: PlaneBerisEdwardsRunSpec) -> dict[str, object]:
        calls.append(target)
        return deepcopy(replacement)

    monkeypatch.setattr(PlaneBerisEdwardsRunSpec, "to_metadata", metadata)

    assert spec.canonical_sha256() == _canonical_sha256(replacement)
    assert calls == [spec]


def test_runtime_identity_dispatches_through_public_metadata_but_retains_direct_controls(
    monkeypatch,
):
    spec = _spec()
    replacement = spec.to_metadata()
    replacement["geometry"] = {"sentinel": "geometry"}
    replacement["boundaries"] = {"sentinel": "boundaries"}
    replacement["numerics"] = {"sentinel": "numerics"}
    replacement["model"] = {
        "initial_s": 123.0,
        "sentinel": "model",
    }
    replacement["preset"] = {"sentinel": "preset"}
    replacement["zero_mode"] = {"sentinel": "zero-mode"}
    replacement["workflow"]["spectral_refresh"] = {
        "sentinel": "ignored-workflow-refresh"
    }
    calls: list[PlaneBerisEdwardsRunSpec] = []

    def metadata(target: PlaneBerisEdwardsRunSpec) -> dict[str, object]:
        calls.append(target)
        return deepcopy(replacement)

    monkeypatch.setattr(PlaneBerisEdwardsRunSpec, "to_metadata", metadata)

    observed = spec.runtime_identity_metadata()
    assert calls == [spec]
    assert observed["geometry"] == {"sentinel": "geometry"}
    assert observed["boundaries"] == {"sentinel": "boundaries"}
    assert observed["numerics"] == {"sentinel": "numerics"}
    assert observed["model"] == {"sentinel": "model"}
    assert observed["preset"] == {"sentinel": "preset"}
    assert observed["zero_mode"] == {"sentinel": "zero-mode"}
    assert observed["dt"] == spec.dt
    assert observed["spectral_refresh"] == spec.spectral_refresh.to_metadata()
    assert observed["runtime_controls"] == {
        "device": spec.device,
        "tf32": spec.tf32,
        "disable_q_gradient_reuse": spec.disable_q_gradient_reuse,
    }


def test_runtime_hash_and_identity_document_retain_public_method_dispatch(
    monkeypatch,
):
    spec = _spec()
    runtime_document = {
        "schema_version": 1,
        "sentinel": ["runtime-dispatch"],
    }
    runtime_calls: list[PlaneBerisEdwardsRunSpec] = []

    def runtime_metadata(
        target: PlaneBerisEdwardsRunSpec,
    ) -> dict[str, object]:
        runtime_calls.append(target)
        return deepcopy(runtime_document)

    canonical_calls: list[PlaneBerisEdwardsRunSpec] = []

    def canonical(target: PlaneBerisEdwardsRunSpec) -> str:
        canonical_calls.append(target)
        return "ab" * 32

    monkeypatch.setattr(
        PlaneBerisEdwardsRunSpec,
        "runtime_identity_metadata",
        runtime_metadata,
    )
    monkeypatch.setattr(
        PlaneBerisEdwardsRunSpec,
        "canonical_sha256",
        canonical,
    )

    assert spec.runtime_identity_sha256() == _canonical_sha256(runtime_document)
    assert runtime_calls == [spec]
    assert spec.identity_metadata()["canonical_sha256"] == "ab" * 32
    assert canonical_calls == [spec]


@pytest.mark.parametrize("field", ("beta", "dt"))
def test_schema_v1_hashes_continue_to_reject_non_finite_values(field):
    spec = replace(_spec(), **{field: float("nan")})

    with pytest.raises(ValueError, match="Out of range float values"):
        spec.canonical_sha256()
    with pytest.raises(ValueError, match="Out of range float values"):
        spec.runtime_identity_sha256()


def test_serializer_is_one_stateless_direct_module_only_authority():
    serializer = schema_v1.PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER

    assert isinstance(
        serializer,
        schema_v1.PlaneBerisEdwardsSchemaV1CompatibilitySerializer,
    )
    assert (
        schema_v1.PlaneBerisEdwardsSchemaV1CompatibilitySerializer.__slots__
        == ()
    )
    assert not hasattr(serializer, "__dict__")
    assert facade.PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER is serializer
    assert facade.PLANE_RUN_SPEC_SCHEMA_VERSION == (
        schema_v1.PLANE_RUN_SPEC_SCHEMA_VERSION
    )
    for name in (
        "PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER",
        "PlaneBerisEdwardsSchemaV1CompatibilitySerializer",
    ):
        assert name in schema_v1.__all__
        assert name not in facade.__all__
        assert name not in configuration.__all__
        assert name not in pssolver.__all__
        assert not hasattr(configuration, name)
        assert not hasattr(pssolver, name)


@pytest.mark.parametrize(
    ("facade_method", "serializer_calls"),
    (
        ("to_metadata", ("to_metadata",)),
        ("canonical_sha256", ("canonical_sha256", "to_metadata")),
        (
            "runtime_identity_metadata",
            ("runtime_identity_metadata", "to_metadata"),
        ),
        (
            "runtime_identity_sha256",
            (
                "runtime_identity_sha256",
                "runtime_identity_metadata",
                "to_metadata",
            ),
        ),
        (
            "identity_metadata",
            ("identity_metadata", "canonical_sha256", "to_metadata"),
        ),
        ("runtime_selection_metadata", ("runtime_selection_metadata",)),
    ),
)
def test_each_facade_method_delegates_through_the_single_serializer(
    monkeypatch,
    facade_method,
    serializer_calls,
):
    serializer_type = schema_v1.PlaneBerisEdwardsSchemaV1CompatibilitySerializer
    observed: list[str] = []

    for method_name in (
        "to_metadata",
        "canonical_sha256",
        "runtime_identity_metadata",
        "runtime_identity_sha256",
        "identity_metadata",
        "runtime_selection_metadata",
    ):
        original = getattr(serializer_type, method_name)

        def spy(self, run_spec, __name=method_name, __target=original):
            observed.append(__name)
            return __target(self, run_spec)

        monkeypatch.setattr(serializer_type, method_name, spy)

    getattr(_spec(), facade_method)()
    assert tuple(observed) == serializer_calls


def test_serializer_direct_entry_points_equal_all_facade_documents_and_hashes():
    serializer = schema_v1.PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER
    for case in CASES_ORACLE["cases"]:
        spec = _spec(**case["factory_overrides"])
        assert serializer.to_metadata(spec) == spec.to_metadata()
        assert serializer.canonical_sha256(spec) == spec.canonical_sha256()
        assert serializer.runtime_identity_metadata(spec) == (
            spec.runtime_identity_metadata()
        )
        assert serializer.runtime_identity_sha256(spec) == (
            spec.runtime_identity_sha256()
        )
        assert serializer.identity_metadata(spec) == spec.identity_metadata()
        assert serializer.runtime_selection_metadata(spec) == (
            spec.runtime_selection_metadata()
        )


def test_serializer_module_is_a_one_way_leaf_without_component_or_consumer_edges():
    tree = ast.parse(
        SERIALIZER_PATH.read_text(encoding="utf-8"),
        filename=str(SERIALIZER_PATH),
    )
    source = SERIALIZER_PATH.read_text(encoding="utf-8")

    assert "plane_beris_edwards_components" not in source
    assert "plane_beris_edwards_component_graph" not in source
    assert "pssolver.applications" not in source
    assert "pssolver.runtime" not in source
    assert "pssolver.workflows" not in source
    assert "__import__" not in source
    assert "import_module" not in source
    facade_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "plane_beris_edwards"
    ]
    assert len(facade_imports) == 1
    type_checking_guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    ]
    assert len(type_checking_guards) == 1
    assert facade_imports[0] in type_checking_guards[0].body

    facade_source = FACADE_PATH.read_text(encoding="utf-8")
    assert "import hashlib" not in facade_source
    assert "import json" not in facade_source
    assert "plane_beris_edwards_schema_v1" in facade_source
    for relative in (
        "pssolver/applications/plane_beris_edwards.py",
        "pssolver/runtime/plane_beris_edwards.py",
        "pssolver/runtime/plane_legacy.py",
        "pssolver/workflows/plane_beris_edwards.py",
    ):
        consumer_source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "plane_beris_edwards_schema_v1" not in consumer_source


def test_serializer_and_migrated_component_graph_are_in_source_inventory():
    assert "pssolver/configuration/plane_beris_edwards_schema_v1.py" in (
        PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES
    )
    assert "pssolver/configuration/plane_beris_edwards_components.py" in (
        PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES
    )
    assert (
        "pssolver/configuration/plane_beris_edwards_component_graph.py"
        in PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES
    )
