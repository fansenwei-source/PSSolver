"""Forward bindings for the S-13 cleanup and S-14 correction successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_67_s13_s14_corrections.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_67_binds_s13_and_the_exact_correction_source() -> None:
    record = _read(RECORD)
    assert record["classification"] == "S13_COMPLETE_S14_CORRECTIONS_CLAIMED"
    assert record["plan_step"] == "S-14"
    assert record["decision"] == "D-0014"
    assert record["git_identity"] == {
        "base_commit": "9f2874458d9293f972ab89c78097db2fd55e5dc8",
        "correction_source_commit": "eee4487658eada196ce3916cb08bd646da4993c6",
        "branch": "dev/0.4",
        "record_commit_is_correction_source": False,
        "pushed": False,
    }
    assert record["s13_archive"] == {
        "path": "/home/fansenwei/pssolver_v0_4_s13_workspace_cleanup_20261029_v1",
        "classification": "PASS_S13_WORKSPACE_CLEANUP",
        "checksums_entries": 13,
        "checksums_sha256": (
            "261108f0a2a473e89de0fb76bd49fb670e14d56be4b1d4ad4893cef5e37ae269"
        ),
        "pruned_sha256": (
            "13bb65414c8f40172da876e034f961a8cd0f8b9c7039c52b655fdb606033ce88"
        ),
        "complete_present": True,
    }


def test_successor_67_records_only_the_three_claims_and_two_forward_corrections() -> None:
    record = _read(RECORD)
    assert record["registry_claims"] == {
        issue_id: {
            "status": "claimed_fixed",
            "fixed_in": "eee4487658eada196ce3916cb08bd646da4993c6",
        }
        for issue_id in ("V-47", "V-48", "V-49")
    }
    corrections = record["forward_corrections"]
    assert corrections["successor_65"]["rewritten"] is False
    assert corrections["successor_65"]["s13_pruning_recorded"] is True
    assert corrections["successor_66"]["rewritten"] is False
    assert corrections["successor_66"]["G-01_through_G-08"] == (
        "PASS_WITH_D-0013_BOUNDED_EXCEPTION_FOR_G-06"
    )
    assert corrections["successor_66"]["sha256"] == (
        "6db6fbd7d90bdf364e0908bdd7ce26e312e6f7377e667e31f8ea1dba278c312d"
    )


def test_successor_67_advances_every_existing_changed_source_forward_only() -> None:
    record = _read(RECORD)
    for relative, registration in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, registration["historical_sha256"]
        ) == _sha256(ROOT / relative)
        recorded_current = subprocess.run(
            [
                "git",
                "show",
                f"0cbc7994ae7daf8c32d9cf845a634da6798e89e5:{relative}",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(recorded_current).hexdigest() == registration[
            "current_sha256"
        ]
        historical = subprocess.run(
            ["git", "show", f"9f2874458d9293f972ab89c78097db2fd55e5dc8:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == registration[
            "historical_sha256"
        ]


def test_successor_67_registry_is_claimed_without_runtime_or_external_authority() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    for issue_id in ("V-47", "V-48", "V-49"):
        assert issues[issue_id]["status"] in {"claimed_fixed", "verified"}
        assert issues[issue_id]["fixed_in"] == (
            "eee4487658eada196ce3916cb08bd646da4993c6"
        )
    record = _read(RECORD)
    assert record["scope"] == {
        "records_and_documentation_only": True,
        "runtime_changed": False,
        "packaging_changed": False,
        "version_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "historical_record_rewritten": False,
    }
    assert all(value is False for value in record["external_authorization"].values())
