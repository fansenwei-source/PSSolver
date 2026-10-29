"""Forward-only import contract for the independent R-0042 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_68_r0042_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
PREDECESSOR = (
    ROOT / "notes/PSSolver_v0_3_0_s03_successor_67_s13_s14_corrections.json"
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0042_result_and_independent_archive_are_exact() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_R0042_S13_S14_REVIEW"
    assert record["plan_steps"] == ["S-13", "S-14"]
    assert record["git_identity"] == {
        "reviewed_candidate": "0cbc7994ae7daf8c32d9cf845a634da6798e89e5",
        "correction_source": "eee4487658eada196ce3916cb08bd646da4993c6",
        "branch": "dev/0.4",
        "record_commit_is_reviewed_candidate": False,
        "pushed": False,
    }
    assert record["sync_result"] == {
        "id": "R-0042",
        "path": "PSSolver_sync/results/R-0042.result.json",
        "sha256": (
            "754cac81578be9fe61fdda745eff9e03d106b4ac1e65ef3722b2bc82db052616"
        ),
        "sync_commit": "ca9e168edc45c2d8d4c8104ec375363fe065ef24",
        "classification": "PASS",
    }
    assert record["independent_archive"] == {
        "path": "/home/fansenwei/pssolver_v0_4_claude_s13_s14_review_20261029_v1",
        "checksums_entries": 38,
        "checksums_sha256": (
            "248f43c488e47358e7718d5166b21083d0d685ae355b9d82d2e8138e4e23621c"
        ),
        "complete_present": True,
        "final_report_sha256": (
            "abc5686c358dd39f77345e46ecdbf3da4af0e0ea0920fa0c9eb778090dac5286"
        ),
        "summary_sha256": (
            "9d39fbdc9b5386d7f94085fcbb8b56edcd8ad04e6372b23f6631341237564c4e"
        ),
    }


def test_r0042_verifies_the_three_s14_items_at_the_exact_fix_source() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    expected = {
        issue_id: {
            "status": "verified",
            "fixed_in": "eee4487658eada196ce3916cb08bd646da4993c6",
            "verified_by": "R-0042",
        }
        for issue_id in ("V-47", "V-48", "V-49")
    }
    assert record["registry_updates"] == expected
    for issue_id, identity in expected.items():
        assert {
            key: issues[issue_id][key]
            for key in ("status", "fixed_in", "verified_by")
        } == identity


def test_r0042_advances_changed_sources_without_rewriting_successor_67() -> None:
    record = _read(RECORD)
    predecessor = record["immutable_predecessor"]
    assert predecessor == {
        "path": "notes/PSSolver_v0_3_0_s03_successor_67_s13_s14_corrections.json",
        "sha256": (
            "51f9519060cd228172a7ff7cd5ae600e73315d1e5030910d9eb2b8a42c58c62c"
        ),
        "rewritten": False,
    }
    assert _sha256(PREDECESSOR) == predecessor["sha256"]
    for relative, registration in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == registration["current_sha256"]
        assert resolve_s03_successor(relative, registration["historical_sha256"]) == (
            registration["current_sha256"]
        )
        historical = subprocess.run(
            [
                "git",
                "show",
                f"0cbc7994ae7daf8c32d9cf845a634da6798e89e5:{relative}",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == registration[
            "historical_sha256"
        ]


def test_r0042_import_is_records_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["new_sources"] == {
        "notes/PSSolver_v0_3_0_s03_successor_68_r0042_import_zh.md": (
            "d6a4253cffabd13bf8ebad6b30daf2c900b0cd410b25cb89a6f89f1b4231cd1e"
        ),
        "tests/test_v0_4_s14_r0042_import.py": _sha256(Path(__file__)),
    }
    assert record["scope"] == {
        "records_and_tests_only": True,
        "runtime_changed": False,
        "packaging_changed": False,
        "version_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "historical_record_rewritten": False,
    }
    assert record["authorization"] == {
        "local_import": True,
        "push": False,
        "tag": False,
        "github_release": False,
        "pypi": False,
        "hpcc_or_slurm": False,
    }
    assert record["next_action"] == "separately_authorized_dev_0_4_push"
