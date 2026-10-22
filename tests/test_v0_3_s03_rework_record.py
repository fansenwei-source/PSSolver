"""Evidence bindings for the R-0008 S-03 compatibility rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_02_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s03_rework_binds_the_implementation_and_r0008_scope() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_LOCAL_V0_3_S03_EXCEPTION_COMPATIBILITY_REWORK_CANDIDATE"
    )
    assert record["git_identity"]["implementation_commit"] == (
        "a10785853736b2458eec80680b9ced72a495775f"
    )
    review = record["governing_review"]
    assert review["id"] == "R-0008"
    assert set(review["verified_preserved"]) == {
        "C1-10",
        "C1-11",
        "V-23",
        "V-21",
    }
    assert set(review["reworked"]) == {"C1-08", "A2-24/C1-15"}
    assert review["new_finding_reworked"] == ["V-24"]


def test_s03_rework_registry_preserves_the_three_verified_fixes() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert {
        issue_id for issue_id, item in issues.items() if item["status"] == "claimed_fixed"
    } == {"A2-21/S2-12"}
    assert all(
        issues[issue_id]["status"] == "verified"
        for issue_id in ("V-22", "V-16", "V-13", "V-14", "C2-17", "V-25")
    )
    assert set(record["issue_status_after_rework"].values()) >= {
        "verified",
        "claimed_fixed",
    }
    for issue_id in ("C1-08", "A2-24/C1-15", "V-24"):
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == record["git_identity"][
            "implementation_commit"
        ]
        assert issues[issue_id]["verified_by"] == "R-0009"


def test_s03_rework_public_types_preserve_legacy_catch_boundaries() -> None:
    missing = pssolver.SimulationInputFileNotFoundError(
        "missing",
        application_family="test",
        stage="initial_condition",
        step=0,
    )
    unreadable = pssolver.SimulationInputIOError(
        "unreadable",
        application_family="test",
        stage="initial_condition",
        step=0,
    )
    assert isinstance(missing, pssolver.SimulationInputError)
    assert isinstance(missing, FileNotFoundError)
    assert isinstance(missing, OSError)
    assert isinstance(unreadable, pssolver.SimulationInputError)
    assert isinstance(unreadable, OSError)


def test_s03_rework_successors_are_forward_only_and_complete() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)
    for relative, expected in record["new_sources_sha256"].items():
        assert _sha256(ROOT / relative) == expected


def test_s03_rework_scope_and_authorization_are_fail_closed() -> None:
    record = _read(RECORD)
    assert all(record["contract"].values())
    assert record["scope"]["runtime_changed"] is True
    assert record["scope"]["public_error_contract_changed"] is True
    assert all(
        record["scope"][key] is False
        for key in (
            "checkpoint_schema_changed",
            "checkpoint_bytes_changed",
            "provider_identity_changed",
            "solver_equations_changed",
            "numerical_algorithms_changed",
            "test_thresholds_relaxed",
            "skip_or_xfail_added",
            "production_default_changed",
            "PSSolver_Control_modified",
            "nematics3d_modified",
            "HPCC_executed",
            "GPU_executed",
            "H100_executed",
            "push_executed",
        )
    )
    assert record["authorization"]["local_rework_authorized"] is True
    assert record["authorization"]["external_review_requested"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key not in {"local_rework_authorized", "external_review_requested"}
    )
