"""Evidence bindings for the S-04 V-25 rework candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_06_s04_v25_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v25_record_binds_the_runtime_rework_and_s04_scope() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_LOCAL_V0_3_S04_V25_REWORK_PENDING_SHORT_REVIEW_AND_REAL_NFS"
    )
    assert record["git_identity"] == {
        "branch": "fix/v0.3-s04-checkpoint-robustness",
        "implementation_parent": "279a4aaec1f165122d140d7f1e53c96119292516",
        "implementation_commit": "44a6fc3d88d94ccf939e78e23be3ffbac542def5",
        "record_commit_is_qualified_source": False,
        "pushed": False,
    }
    assert record["roadmap"]["plan_step"] == "S-04"
    assert set(record["roadmap"]["issue_ids"]) == {
        "V-22",
        "V-16",
        "V-13",
        "V-14",
        "C2-17",
        "V-25",
    }


def test_v25_and_prior_s04_fixes_are_verified_without_identity_changes() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert issues["V-25"]["status"] == "verified"
    assert issues["V-25"]["fixed_in"] == record["git_identity"][
        "implementation_commit"
    ]
    assert issues["V-25"]["verified_by"] == "R-0012"
    for issue_id in record["preserved_prior_claims"]:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == (
            "468e06710e1b4881a557f0228dcb16be38791094"
        )
        assert issues[issue_id]["verified_by"] == "R-0012"


def test_v25_successors_are_forward_only_and_complete() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)
    for relative, expected in record["new_sources_sha256"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_v25_contract_is_fail_closed_and_does_not_expand_scope() -> None:
    record = _read(RECORD)
    contract = record["contract"]
    assert contract["absent_target_after_failed_claim_is_definite"] is True
    assert contract["definite_failure_removes_staging"] is True
    assert contract["existing_unclaimable_target_remains_ambiguous"] is True
    assert contract["ambiguous_error_names_preserved_staging"] is True
    assert contract["covered_mkdir_errnos"] == [
        "ENOSPC",
        "EDQUOT",
        "EROFS",
        "EACCES",
        "EIO",
    ]
    assert record["scope"]["runtime_changed"] is True
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
            "push_executed",
            "HPCC_executed",
            "GPU_executed",
            "H100_executed",
        )
    )


def test_v25_validation_still_requires_short_review_and_real_nfs() -> None:
    validation = _read(RECORD)["validation"]
    assert validation["runtime_targeted"] == {
        "passed": 40,
        "failed": 0,
        "skipped": 0,
    }
    assert validation["v25_errno_cases"] == {
        "passed": 5,
        "failed": 0,
        "staging_residue": 0,
    }
    assert validation["n1_matrix"]["selection_cases"] == 528
    assert validation["n1_matrix"]["selection_mismatches"] == 0
    assert validation["n1_matrix"]["all_byte_identity_variants_passed"] is True
    assert validation["short_review"] == "requires_R_0011"
    assert validation["HPCC_real_NFS"] == "requires_separate_post_R_0011_authorization"


def test_v25_record_authorizes_no_external_action() -> None:
    authorization = _read(RECORD)["authorization"]
    assert authorization["local_rework_authorized"] is True
    assert all(
        value is False
        for key, value in authorization.items()
        if key != "local_rework_authorized"
    )
