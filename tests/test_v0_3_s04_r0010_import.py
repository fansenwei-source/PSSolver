"""Evidence bindings for importing the R-0010 S-04 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_05_r0010_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0010_import_binds_the_rework_result_and_complete_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "REWORK_REQUIRED_R0010_S04_REVIEW_IMPORTED"
    assert record["git_identity"]["result_commit"] == (
        "d91f0204541b743be9c8b1e66ff91de0d25293c2"
    )
    assert record["sync_result"] == {
        "id": "R-0010",
        "path": (
            "/home/fansenwei/Desktop/Develop/PSSolver_sync/"
            "results/R-0010.result.json"
        ),
        "sha256": "6dba6baa3dc234a19a6b392010f1e8b837ae92125287918f2186d9a14bd8c004",
        "classification": "REWORK_REQUIRED",
        "release_blockers": [],
        "registry_updates": [],
    }
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 7248
    assert archive["checksums_sha256"] == (
        "9dce922447a605c94c7e82a75267c21cd5dba13a9ce784569fd61dff7a322e8b"
    )
    assert archive["complete_present"] is True


def test_r0010_registers_only_v25_and_preserves_the_five_claims() -> None:
    record = _read(RECORD)["review_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert set(record["preserved_claimed_fixed"]) == {
        "V-22",
        "V-16",
        "V-13",
        "V-14",
        "C2-17",
    }
    for issue_id in record["preserved_claimed_fixed"]:
        assert issues[issue_id]["status"] == "claimed_fixed"
        assert issues[issue_id]["fixed_in"] == (
            "468e06710e1b4881a557f0228dcb16be38791094"
        )
        assert "verified_by" not in issues[issue_id]
    assert record["registered_new_findings"] == ["V-25"]
    # The immutable R-0010 import records the initial registered state; the
    # later S-04 V-25 successor may advance it to claimed_fixed.
    assert record["V-25"]["status"] == "registered"
    assert issues["V-25"]["status"] == "claimed_fixed"
    assert issues["V-25"]["fixed_in"] == (
        "44a6fc3d88d94ccf939e78e23be3ffbac542def5"
    )
    assert issues["V-25"]["target_release"] == "0.3"


def test_r0010_import_is_a_forward_only_successor_layer() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0010_import_is_evidence_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
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
    assert record["next_request"] == "R-0011"
