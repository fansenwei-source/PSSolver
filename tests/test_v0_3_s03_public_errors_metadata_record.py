"""Evidence bindings for the v0.3 S-03 local candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pssolver
from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_public_errors_metadata.json"
R0008_IMPORT = ROOT / "notes/PSSolver_v0_3_0_s03_successor_01_r0008_import.json"
REWORK = ROOT / "notes/PSSolver_v0_3_0_s03_successor_02_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
ROADMAP = ROOT / "notes/PSSolver_v0_3_0_roadmap.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _git_sha256(commit: str, relative: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def test_s03_record_binds_the_implementation_commit_and_issue_set() -> None:
    record = _read(RECORD)
    assert record["stage"] == "S-03"
    assert record["classification"] == (
        "PASS_LOCAL_V0_3_S03_PUBLIC_ERROR_AND_METADATA_CANDIDATE"
    )
    identity = record["git_identity"]
    assert identity["development_parent"] == (
        "19dcd0be83c86e8cace6abff714d3a9f55234c0b"
    )
    assert identity["implementation_commit"] == (
        "a654e1c1d97fa364ac32b6ccbf93070056066d21"
    )
    assert identity["record_commit_is_qualified_source"] is False
    assert set(record["issue_ids"]) == {
        "C1-10",
        "C1-11",
        "A2-24/C1-15",
        "C1-08",
        "V-23",
        "V-21",
    }


def test_s03_registry_reflects_the_r0008_independent_verdicts() -> None:
    record = _read(RECORD)
    review = _read(R0008_IMPORT)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert set(record["issue_ids"]) == set(review["reviewed_issue_ids"])
    for issue_id in review["verified_issue_ids"]:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0008"
    for issue_id in review["reopened_issue_ids"]:
        assert issues[issue_id]["status"] == "claimed_fixed"
        assert issues[issue_id]["fixed_in"] == _read(REWORK)["git_identity"][
            "implementation_commit"
        ]
    assert issues["V-24"]["status"] == "claimed_fixed"


def test_s03_successors_bind_exact_parent_and_current_bytes() -> None:
    record = _read(RECORD)
    parent = record["git_identity"]["development_parent"]
    successors = record["historical_successor_sources"]
    assert len(successors) == 79
    for relative, registration in successors.items():
        assert _git_sha256(parent, relative) == registration[
            "historical_sha256"
        ]
        assert _sha256(relative) == resolve_s03_successor(
            relative, registration["current_sha256"]
        )
        assert registration["historical_sha256"] != registration[
            "current_sha256"
        ]
        assert registration["reason"]


def test_s03_contract_is_structured_and_fail_closed() -> None:
    contract = _read(RECORD)["contract"]
    assert contract and all(contract.values())
    error = pssolver.SimulationInputError(
        "invalid input",
        application_family="example",
        stage="initial_condition",
        step=0,
        details={"value": float("nan")},
    )
    assert error.to_metadata()["details"]["value"] == "nan"
    assert isinstance(error, ValueError)


def test_s03_scope_does_not_expand_numerics_or_authority() -> None:
    record = _read(RECORD)
    scope = record["scope"]
    assert scope["public_error_contract_changed"] is True
    assert scope["failure_metadata_contract_changed"] is True
    for key, value in scope.items():
        if key not in {
            "public_error_contract_changed",
            "failure_metadata_contract_changed",
        }:
            assert value is False
    authorization = record["authorization"]
    assert authorization["local_implementation_authorized"] is True
    assert authorization["external_review_requested"] is True
    for key, value in authorization.items():
        if key not in {
            "local_implementation_authorized",
            "external_review_requested",
        }:
            assert value is False


def test_s03_registers_the_g04_v022_non_regression_gate() -> None:
    criterion = next(
        gate["criterion"]
        for gate in _read(ROADMAP)["completion_gates"]
        if gate["id"] == "G-04"
    )
    assert "Plane, Periodic, and Channel" in criterion
    assert "v0.2.2" in criterion
    assert "byte-identical" in criterion
    assert "tolerances registered before qualification" in criterion


def test_s03_new_sources_are_content_addressed() -> None:
    for relative, expected in _read(RECORD)["new_sources_sha256"].items():
        assert _sha256(relative) == resolve_s03_successor(relative, expected)


def test_s03_records_the_final_cpu_gate_without_new_skips() -> None:
    validation = _read(RECORD)["validation"]
    assert validation["full_cpu_suite"] == {
        "passed": 4299,
        "failed": 0,
        "skipped": 1,
        "xfailed": 0,
        "deselected": 7,
        "warnings": 6,
        "subtests_passed": 8,
        "allowed_skip": (
            "tests/test_analytic_periodic_defect_gas_2d.py::"
            "test_nematics3d_recovers_requested_defect_count_charges_and_positions"
        ),
    }
    assert validation["historical_successor_layers"] == {
        "first_layer_passed": 275,
        "second_layer_passed": 137,
        "failed": 0,
    }
