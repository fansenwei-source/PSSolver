"""Pre-extraction oracles for the Plane schema-v1 serializer authority."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from pssolver.configuration import (
    PlaneBerisEdwardsRunSpec,
    create_plane_beris_edwards_run_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "configuration"
    / "plane_run_spec_v1_cases.json"
)
CASES_ORACLE = json.loads(CASES_PATH.read_text(encoding="utf-8"))


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
