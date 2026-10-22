"""Evidence bindings for importing the R-0012 final S-04 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_08_r0012_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0012_import_binds_the_pass_result_and_complete_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_R0012_S04_FINAL_REVIEW_IMPORTED"
    assert record["git_identity"]["result_commit"] == (
        "63c8eb6ce96a4a1d6d615bd270d436b5132ab4a8"
    )
    assert record["sync_result"] == {
        "id": "R-0012",
        "path": (
            "/home/fansenwei/Desktop/Develop/PSSolver_sync/"
            "results/R-0012.result.json"
        ),
        "sha256": "6b32eb48d782540c96995eb5791f84f2197b4b8f7f0bb833190c88f2cc67e609",
        "classification": "PASS",
        "new_findings": [],
    }
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 32
    assert archive["checksums_sha256"] == (
        "633ce4b9c215bd8f3a96e4560ee9aaea2c4e7e4ed6af1e66e165d9c5cb56d472"
    )
    assert archive["complete_present"] is True


def test_r0012_verifies_all_six_s04_items_with_exact_fix_ids() -> None:
    record = _read(RECORD)["registry_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    first_fix = set(record["fixed_at_468e067"])
    second_fix = set(record["fixed_at_44a6fc3"])
    assert first_fix == {"V-22", "V-16", "V-13", "V-14", "C2-17"}
    assert second_fix == {"V-25"}
    for issue_id in first_fix:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == (
            "468e06710e1b4881a557f0228dcb16be38791094"
        )
        assert issues[issue_id]["verified_by"] == "R-0012"
    assert issues["V-25"]["status"] == "verified"
    assert issues["V-25"]["fixed_in"] == (
        "44a6fc3d88d94ccf939e78e23be3ffbac542def5"
    )
    assert issues["V-25"]["verified_by"] == "R-0012"
    assert record["s04_complete"] is True
    assert record["eligible_for_d0009_integration_draft"] is True


def test_r0012_binds_the_final_hpcc_archive_without_new_execution() -> None:
    evidence = _read(RECORD)["qualified_hpcc_evidence"]
    assert evidence["checksums_entries"] == 25
    assert evidence["checksums_sha256"] == (
        "afd7e2d7d45d68935c23ed4110ca65fbc1ec96791ef98b4dbefe9254097af3ce"
    )
    assert evidence["complete_present"] is True
    assert evidence["helper_execution_count"] == 1
    assert evidence["automatic_retry"] is False
    assert evidence["slurm_used"] is False
    assert evidence["gpu_used"] is False
    assert evidence["h100_used"] is False


def test_r0012_successors_are_forward_only_and_complete() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0012_import_is_evidence_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    validation = record["validation"]
    assert validation["environment_clone_source"].endswith(
        "pssolver_v0_2_1_claude_acceptance_20261020_v1/a2/venv_wheel"
    )
    assert validation["environment_clone_pruned"] is True
    assert validation["sys_path"][1] == str(ROOT)
    assert validation["focused_pytest"] == "55 passed"
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
    assert record["next_request"] == "R-0013"
