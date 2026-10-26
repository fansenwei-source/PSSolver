"""Evidence bindings for importing the passing R-0024 S-09 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_38_r0024_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0024.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0024_import_binds_the_exact_passing_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_R0024_S09_FINAL_PRE_H100_REVIEW_IMPORTED"
    assert record["git_identity"]["reviewed_candidate"] == (
        "f01f591a570f3c827e0e199aaf905e0efd0c5af0"
    )
    assert _sha256(RESULT) == record["sync_result"]["sha256"] == (
        "2b172299be82ac1141badd4bbf099b8865a467926122cabb2a330d564037f968"
    )
    assert record["sync_result"]["classification"] == "PASS"
    assert record["sync_result"]["new_findings"] == ["V-38"]
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 3787
    assert archive["checksums_sha256"] == (
        "ec2aea1e573ed077047576cff091ede8fb7a4c6849c6399d434558f97c634a1f"
    )
    assert archive["complete_present"] is True


def test_r0024_import_applies_only_the_reviewed_registry_updates() -> None:
    record = _read(RECORD)["registry_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["verified"] == {
        "V-35": "85ba3aef75548d549d9db453ca4a5ab604b17a9d",
        "V-36": "85ba3aef75548d549d9db453ca4a5ab604b17a9d",
        "V-37": "87cf34cf4e17096a701f5cc5bc81076d14a9d237",
    }
    for issue_id, fixed_in in record["verified"].items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id]["verified_by"] == "R-0024"
    assert record["registered"] == ["V-38"]
    assert issues["V-38"]["status"] == "verified"
    assert issues["V-38"]["fixed_in"] == (
        "07b0ea69af08f20f2183262b39fde8a356cff029"
    )
    assert issues["V-38"]["verified_by"] == "R-0025"
    assert issues["V-38"]["source_report"] == "PSSolver_sync/results/R-0024.result.json"
    assert issues["V-38"]["target_release"] == "0.3"
    assert issues["A3-18"]["status"] == "verified"
    assert issues["A3-18"]["verified_by"] == "R-0025"
    assert issues["NEW-14"]["status"] == "verified"
    assert issues["NEW-14"]["fixed_in"] == "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    assert issues["NEW-14"]["verified_by"] == "R-0027"


def test_r0024_import_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert successor["current_sha256"]
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0024_import_preserves_frozen_h100_inputs_and_authorizes_nothing() -> None:
    record = _read(RECORD)
    assert record["frozen_h100_inputs"] == {
        "baseline_wheel_sha256": "a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e",
        "candidate_wheel_sha256": "4003645025da6d81317eeb7261ccf606aef7f60f20d072518f8baacf56cd4aa8",
        "consumer_wheel_sha256": "f7ae4c1f5f3c73e0be540b5e4ec5e05a9eef662867a149325462835820ae94ef",
        "plan_sha256": "d123e8cf6e529e4780d8be49478628c4e1553d655ae244acf9860845cd65f55e",
        "job_script_sha256": "34ed9a7e6d43acdcaa344218b396e71a439ca21da7b9073843eb1b7587fde855",
        "draft_sha256": "bb908866978eb00d8829792f1cb2869b6e1617eb1a88e6b10bbadd24d0293a45",
    }
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_import_authorized"
    )
    assert record["next_request"] == "R-0025"
