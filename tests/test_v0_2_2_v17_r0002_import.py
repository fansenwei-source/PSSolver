"""Evidence contract for importing the independent R-0002 V-17 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_2_v17_r0002_import.json"
RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_2.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
DEV_INTEGRATION_RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_n1_v17_checkpoint_integration.json"
)
S01_RECORD = ROOT / "notes/PSSolver_v0_3_0_s01_registry_migration.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _issues() -> dict[str, dict[str, object]]:
    return {item["id"]: item for item in _read(REGISTRY)["issues"]}


def test_r0002_import_binds_result_and_independent_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_R0002_V17_INDEPENDENT_REVIEW_IMPORT"
    assert record["sync_result"] == {
        "id": "R-0002",
        "path": (
            "/home/fansenwei/Desktop/Develop/PSSolver_sync/"
            "results/R-0002.result.json"
        ),
        "sha256": "8837d79a34997a762cca26f967df3396ea87b5208afc39a744fe2c6f9f32c93d",
        "classification": "PASS",
    }
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 32253
    assert archive["checksums_sha256"] == (
        "418f298d2dfcc81f99c0d3574f7c9c5d153a6c4483cc6f1732c050365980b9a8"
    )
    assert archive["complete_present"] is True


def test_r0002_import_is_a_forward_only_successor() -> None:
    record = _read(RECORD)
    release_successors = _read(RELEASE_RECORD)["historical_successor_sources"]
    development_successors = _read(DEV_INTEGRATION_RECORD)[
        "historical_successor_sources"
    ]
    s01_successors = _read(S01_RECORD)["historical_successor_sources"]
    parent = record["git_identity"]["evidence_parent"]
    assert parent == "6bd8e491dde4a72dd1dc29035c2310d565990070"
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        historical = subprocess.run(
            ["git", "show", f"{parent}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == registration[
            "historical_sha256"
        ]
        expected = registration["current_sha256"]
        successor = release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == expected:
            expected = successor["current_sha256"]
        successor = development_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == expected:
            expected = successor["current_sha256"]
        successor = s01_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == expected:
            expected = successor["current_sha256"]
        assert _sha256(relative) == resolve_s03_successor(relative, expected)
        assert registration["reason"]


def test_r0002_registry_updates_are_exact() -> None:
    record = _read(RECORD)["registry_import"]
    s01 = _read(S01_RECORD)["registry"]
    registry = _read(REGISTRY)
    issues = _issues()
    assert record["issue_count"] == 28
    assert registry["issue_count"] == 39
    assert record["status_counts"] == {
        "open": 1,
        "registered": 20,
        "verified": 1,
        "deferred": 2,
        "accepted": 1,
        "coverage_gap": 3,
    }
    assert registry["status_counts"] == {
        "open": 1,
        "registered": 1,
        "claimed_fixed": 0,
        "verified": 33,
        "deferred": 2,
        "accepted": 1,
        "coverage_gap": 1,
    }
    verified = record["verified"]
    assert issues["V-17"]["status"] == verified["status"] == "verified"
    assert issues["V-17"]["fixed_in"] == verified["fixed_in"]
    assert issues["V-17"]["verified_by"] == verified["verified_by"] == "R-0002"
    for expected in record["new_findings"]:
        actual = issues[expected["id"]]
        for key, value in expected.items():
            if key == "target_release" and expected["id"] in set(
                s01["changed_target_ids"]
            ):
                assert value == "0.2.x"
                assert actual[key] == "0.3"
            elif key == "status" and expected["id"] == "V-23":
                assert value == "registered"
                assert actual[key] == "verified"
                assert actual["fixed_in"] == (
                    "a654e1c1d97fa364ac32b6ccbf93070056066d21"
                )
                assert actual["verified_by"] == "R-0008"
            elif key == "status" and expected["id"] == "V-22":
                assert value == "registered"
                assert actual[key] == "verified"
                assert actual["fixed_in"] == (
                    "468e06710e1b4881a557f0228dcb16be38791094"
                )
                assert actual["verified_by"] == "R-0012"
            else:
                assert actual[key] == value
    for issue_id, decision in record["decision_bindings"].items():
        assert issues[issue_id]["decision"] == decision == "D-0007"


def test_r0002_import_does_not_expand_authority_or_runtime_scope() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    for key, value in record["scope"].items():
        if key != "evidence_only":
            assert value is False
    assert all(value is False for value in record["authorization"].values())
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["git_identity"]["pushed"] is False
