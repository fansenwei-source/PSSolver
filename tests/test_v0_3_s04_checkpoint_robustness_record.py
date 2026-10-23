"""Evidence bindings for the S-04 checkpoint robustness candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pssolver.functional import FunctionalCheckpointDurabilityError
from pssolver.io import CheckpointDurabilityError, CheckpointIOError

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes/PSSolver_v0_3_0_s03_successor_04_s04_checkpoint_robustness.json"
)
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s04_record_binds_the_implementation_and_issue_scope() -> None:
    record = _read(RECORD)
    assert record["git_identity"]["implementation_commit"] == (
        "468e06710e1b4881a557f0228dcb16be38791094"
    )
    assert record["roadmap"]["decision"] == "D-0010"
    assert record["roadmap"]["plan_step"] == "S-04"
    assert set(record["roadmap"]["issue_ids"]) == {
        "V-22",
        "V-16",
        "V-13",
        "V-14",
        "C2-17",
    }


def test_s04_registry_preserves_fix_ids_after_r0012_verification() -> None:
    record = _read(RECORD)
    registry = _read(REGISTRY)
    issues = {item["id"]: item for item in registry["issues"]}
    claimed = {
        issue_id
        for issue_id, item in issues.items()
        if item["status"] == "claimed_fixed"
    }
    historical_claims = set(record["roadmap"]["issue_ids"])
    assert claimed == {"A2-25", "N5", "V-31", "V-32"}
    assert issues["A2-21/S2-12"]["status"] == "verified"
    assert issues["A2-21/S2-12"]["verified_by"] == "R-0013"
    for issue_id in historical_claims:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == record["git_identity"][
            "implementation_commit"
        ]
        assert issues[issue_id]["verified_by"] == "R-0012"
    assert issues["V-25"]["status"] == "verified"
    assert issues["V-25"]["fixed_in"] == (
        "44a6fc3d88d94ccf939e78e23be3ffbac542def5"
    )
    assert issues["V-25"]["verified_by"] == "R-0012"
    for issue_id in ("C1-05", "C2-18", "S2-17", "V-27", "V-28", "V-29"):
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0016"
    for issue_id in ("A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30", "V-26", "V-30"):
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0018"


def test_s04_durability_error_preserves_the_public_io_hierarchy() -> None:
    core = CheckpointDurabilityError("durability not confirmed")
    functional = FunctionalCheckpointDurabilityError("durability not confirmed")
    assert isinstance(core, CheckpointIOError)
    assert isinstance(core, OSError)
    assert isinstance(functional, CheckpointDurabilityError)
    assert isinstance(functional, CheckpointIOError)
    assert isinstance(functional, OSError)


def test_s04_successors_are_forward_only_and_complete() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative,
            successor["historical_sha256"],
        ) == _sha256(ROOT / relative)
    for relative, expected in record["new_sources_sha256"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_s04_scope_and_authorization_are_fail_closed() -> None:
    record = _read(RECORD)
    contract = record["contract"]
    assert all(
        value
        for key, value in contract.items()
        if key not in {"checkpoint_schema_changed", "provider_identity_changed"}
    )
    assert contract["checkpoint_schema_changed"] is False
    assert contract["provider_identity_changed"] is False
    assert record["scope"]["runtime_changed"] is True
    assert record["scope"]["nfs_fallback_changed"] is True
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
    assert record["authorization"]["local_work_authorized"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_work_authorized"
    )


def test_s04_local_validation_passes_and_real_nfs_remains_separately_authorized() -> None:
    record = _read(RECORD)
    validation = record["validation"]
    assert record["classification"] == (
        "PASS_LOCAL_V0_3_S04_CHECKPOINT_ROBUSTNESS_PENDING_REAL_NFS"
    )
    assert validation["status"] == "pass_local_pending_real_nfs"
    assert validation["targeted_cpu"] == {
        "passed": 41,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert validation["full_cpu_suite"] == {
        "passed": 4340,
        "failed": 0,
        "skipped": 1,
        "deselected": 7,
        "subtests_passed": 8,
        "allowed_skip": "optional nematics3d dependency",
    }
    assert validation["n1_matrix"]["selection_mismatches"] == 0
    assert validation["n1_matrix"]["all_byte_identity_variants_passed"] is True
    assert validation["hpcc_real_nfs"] == "requires_separate_user_authorization"
    assert validation["d0008_environment_recorded"] is True
    assert validation["archive_complete"] is True
