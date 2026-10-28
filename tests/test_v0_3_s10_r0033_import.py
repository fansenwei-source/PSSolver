"""Evidence-only binding for the R-0033 S-10 PASS import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_57_r0033_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0033.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0033_import_binds_the_exact_independent_review() -> None:
    record = _read(RECORD)
    assert record["plan_step"] == "S-10"
    assert record["classification"] == "PASS_R0033_S10_REVIEW_IMPORTED"
    assert record["sync_result"] == {
        "id": "R-0033",
        "path": "PSSolver_sync/results/R-0033.result.json",
        "sha256": "8b0f709dc37c1bb631b4efb9f82f47353fc2e06485e7b6c6ee7613536988bd49",
        "sync_commit": "805c580d6ed0cac7d06fdabedf175c5ace71c034",
        "classification": "PASS",
    }
    assert _sha256(RESULT) == record["sync_result"]["sha256"]
    archive = record["independent_archive"]
    root = Path(archive["path"])
    assert archive["complete_present"] is True
    assert (root / "COMPLETE").is_file()
    assert sum(1 for _ in (root / "checksums.sha256").open(encoding="utf-8")) == 37
    assert _sha256(root / "checksums.sha256") == archive["checksums_sha256"]
    assert _sha256(root / "final_report.md") == archive["final_report_sha256"]
    assert _sha256(root / "summary.json") == archive["summary_sha256"]


def test_r0033_import_verifies_only_v33_and_v43() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    expected = {
        "V-33": "61838af7910d955b27268abd4a79e4d32f235f58",
        "V-43": "61838af7910d955b27268abd4a79e4d32f235f58",
    }
    assert record["registry_import"]["verified"] == expected
    assert record["registry_import"]["new_findings"] == []
    for issue_id, fix in expected.items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == (
            "a54074013e5f0fb9bfe278baf944f805bf86eef4"
        )
        assert fix in issues[issue_id]["notes"]
        assert issues[issue_id]["verified_by"] == "R-0033"
    # The review record preserves the immutable R-0033 boundary.  The live
    # registry is intentionally allowed to advance through the later S-09
    # closure rather than being pinned to the pre-R-0031 state here.
    assert record["registry_import"]["verified_by"] == "R-0033"


def test_r0033_import_is_a_forward_only_complete_successor() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        live = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == live
        assert resolve_s03_successor(relative, successor["current_sha256"]) == live


def test_r0033_import_preserves_the_s09_and_external_action_boundary() -> None:
    record = _read(RECORD)
    scope = record["scope"]
    assert scope["evidence_only"] is True
    assert all(value is False for key, value in scope.items() if key != "evidence_only")
    authorization = record["authorization"]
    assert authorization["local_evidence_import"] is True
    assert all(
        value is False
        for key, value in authorization.items()
        if key != "local_evidence_import"
    )
    assert record["reserved_request"] == "R-0031"
    assert record["next_request"] == "R-0034"
