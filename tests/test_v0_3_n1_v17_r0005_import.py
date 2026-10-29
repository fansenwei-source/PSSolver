"""Evidence contract for importing the independent R-0005 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_n1_v17_r0005_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0005_import_binds_result_and_independent_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_R0005_DEV03_N1_V17_INDEPENDENT_REVIEW_IMPORT"
    )
    assert record["git_identity"]["sync_result_commit"] == (
        "06f290e7f609367579838b96f0da37b48edd2a30"
    )
    assert record["git_identity"]["qualified_source_commit"] == (
        "f45518ca34b7901d64c5ee43e2e739067fae8451"
    )
    result = record["sync_result"]
    assert result["id"] == "R-0005"
    assert result["result_sha256"] == (
        "5c5f01b5904fb6f80be678d819bdebd28e67d2014fe84cbbd083f07b2d1a76a8"
    )
    assert result["classification"] == "PASS"
    assert result["release_blockers"] == []
    assert result["registry_updates"] == []
    assert result["new_findings"] == []
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 2984
    assert archive["checksums_sha256"] == (
        "2a9509ea1b096f656130cbf863b295cfea58c9c65f1e2f460954261021825a07"
    )
    assert archive["complete_present"] is True


def test_r0005_import_preserves_every_claim_verdict() -> None:
    review = _read(RECORD)["review"]
    assert review["verdicts"] == {
        "V03-CKPT-1": "PASS",
        "V03-CKPT-2": "PASS",
        "V03-CKPT-3": "PASS",
        "V03-CKPT-4": "PASS",
        "V03-CKPT-5": "PASS",
        "V03-PLAN-1": "PASS",
    }
    assert review["eligible_for_non_force_push"] is True
    assert review["full_cpu_suite"] == {
        "passed": 4239,
        "failed": 0,
        "skipped": 0,
        "deselected": 7,
        "subtests_passed": 8,
    }


def test_r0005_import_binds_the_qualified_runtime_without_changing_it() -> None:
    record = _read(RECORD)
    assert record["review"]["version"] == "0.3.0.dev0"
    assert pssolver.__version__ == "0.4.0.dev0"
    expected = record["review"]["runtime_sha256"]["pssolver/io/checkpoint.py"]
    assert expected == (
        "1a90fe8edb640310c439d6b553f7de02237eccca368455b059565d2038bec0df"
    )
    assert _sha256(ROOT / "pssolver/io/checkpoint.py") == (
        resolve_s03_successor("pssolver/io/checkpoint.py", expected)
    )
    for key in (
        "provider_identity_unchanged",
        "checkpoint_schema_unchanged",
        "solver_equations_unchanged",
        "numerical_algorithms_unchanged",
        "production_defaults_unchanged",
    ):
        assert record["review"][key] is True
    assert record["review"]["history_rewritten"] is False


def test_r0005_import_does_not_change_the_registry() -> None:
    registry = _read(RECORD)["registry"]
    assert registry["sha256"] == (
        "ffdd4fe41ddc140857081febf634b28bed4e1d3345f9e434982ef11ec2f67cc2"
    )
    assert registry["issue_count"] == 28
    assert registry["modified_by_import"] is False
    assert registry["updates_imported"] == 0
    assert registry["new_findings_imported"] == 0


def test_r0005_import_records_but_does_not_apply_follow_up_tracking() -> None:
    tracking = _read(RECORD)["follow_up_tracking"]
    assert "0.3 plan" in tracking["next_registry_successor"]
    assert "system clock" in tracking["request_created_at_policy"]
    assert tracking["applied_by_this_import"] is False


def test_r0005_import_does_not_expand_scope_or_authority() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    for key, value in record["scope"].items():
        if key != "evidence_only":
            assert value is False
    assert all(value is False for value in record["authorization"].values())
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["git_identity"]["pushed"] is False
    assert record["next_sync_id"] == "R-0006"
