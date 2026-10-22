"""Evidence bindings for importing the independent R-0013 S-05 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_11_r0013_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0013_import_binds_the_pass_result_and_complete_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_R0013_S05_REVIEW_IMPORTED"
    assert record["git_identity"]["result_commit"] == (
        "44b3e2d78f291764e207b26375878540607975d1"
    )
    assert record["sync_result"] == {
        "id": "R-0013",
        "path": (
            "/home/fansenwei/Desktop/Develop/PSSolver_sync/"
            "results/R-0013.result.json"
        ),
        "sha256": "4eed1d24794a6918db9c474f9f9877e57d69a657ec1a1153cd5e5b22a42fd222",
        "classification": "PASS",
        "release_blockers": [],
    }
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 27
    assert archive["checksums_sha256"] == (
        "a386dcc7c3580979ecc0b342453774fa840213aa67e577d9ba4a89365de6e8b2"
    )
    assert archive["complete_present"] is True


def test_r0013_import_verifies_s05_and_registers_both_findings() -> None:
    imported = _read(RECORD)["registry_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    verified = imported["verified"]
    assert verified == {
        "id": "A2-21/S2-12",
        "status": "verified",
        "fixed_in": "65ef1080cb0215131efe90143863de7014fe4b5b",
        "verified_by": "R-0013",
    }
    assert all(issues[verified["id"]][key] == value for key, value in verified.items())
    assert imported["new_findings"] == [
        {
            "id": "V-26",
            "severity": "low",
            "status": "registered",
            "target_release": "0.3",
            "plan_step": "S-07",
        },
        {
            "id": "V-27",
            "severity": "low",
            "status": "registered",
            "target_release": "0.3",
            "plan_step": "S-06",
        },
    ]
    for finding in imported["new_findings"]:
        issue = issues[finding["id"]]
        assert issue["severity"] == finding["severity"]
        assert issue["status"] == finding["status"]
        assert issue["target_release"] == finding["target_release"]
        assert finding["plan_step"] in issue["notes"]


def test_r0013_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0013_import_is_evidence_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    validation = record["validation"]
    assert validation["archive_checksums_entries"] == 9
    assert len(validation["archive_checksums_sha256"]) == 64
    assert validation["archive_complete_present"] is True
    assert validation["environment_clone_pruned"] is True
    assert validation["focused_pytest"]["passed"] == 49
    assert validation["full_cpu_suite"]["passed"] == 4435
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert record["authorization"]["local_import_authorized"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_import_authorized"
    )
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["git_identity"]["pushed"] is False
    assert record["next_plan_step"] == "S-06"
    assert record["next_request"] == "R-0014"
