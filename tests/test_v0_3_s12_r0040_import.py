"""Contract for importing the independent R-0040 final-candidate acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_65_r0040_import.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0040_identity_and_independent_archive_are_exact() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_R0040_V0_3_0_FINAL_CANDIDATE_ACCEPTANCE"
    )
    assert record["git_identity"] == {
        "release_source": "47473880534a66f6da940db82122c79cda48a86b",
        "candidate_evidence_commit": "a34d14a09e00d78dbb4e77e30a1abbe4d2374687",
        "record_commit_is_release_source": False,
        "pushed": False,
    }
    assert record["sync_result"] == {
        "id": "R-0040",
        "path": "PSSolver_sync/results/R-0040.result.json",
        "sha256": "a79a210662561fb741ea8612e9bb84a70056695da50bd8cece6c695d49202147",
        "sync_commit": "a6a187feca37479af1500003abbc1285df07385d",
        "classification": "PASS",
    }
    assert record["independent_archive"] == {
        "path": "/home/fansenwei/pssolver_v0_3_0_claude_final_candidate_acceptance_20261028_v1",
        "checksums_entries": 86,
        "checksums_sha256": "d0fbe4744fe565f6bdb876447e618a5f197c40046b7168a797f4cf109d084f3c",
        "complete_present": True,
        "final_report_sha256": "7c5a5402bba2aefda7b56ac3c74769a1faf6ea05c3d4743d062a8f90fdadbbd0",
        "summary_sha256": "a7cd53dd3eb91fd5bdfef1365968b46d586beb22b6252238e1232ae787a99d3c",
    }


def test_r0040_release_assets_are_exact() -> None:
    assert _read(RECORD)["release_assets"] == [
        {
            "name": "pssolver-0.3.0-py3-none-any.whl",
            "path": "/home/fansenwei/pssolver_v0_3_0_package_4747388_20261028_v3/package/dist/pssolver-0.3.0-py3-none-any.whl",
            "sha256": "8e2bd0c7941220f6c7c9b4cba80d9802e1495db626ad1b330a2e1681645187d5",
            "size": 646909,
        },
        {
            "name": "pssolver-0.3.0.tar.gz",
            "path": "/home/fansenwei/pssolver_v0_3_0_package_4747388_20261028_v3/package/dist/pssolver-0.3.0.tar.gz",
            "sha256": "ec46471f5c35fc56089255155508b8a23a6d71ee81488d6b4aec933faed4d68d",
            "size": 1411145,
        },
    ]


def test_superseded_attempts_are_preserved_without_evidence_claims() -> None:
    record = _read(RECORD)
    attempts = record["superseded_local_attempts"]
    assert [item["commit"] for item in attempts] == [
        "d9d346f9db3ae0fde92c4bb8271e652d960348b1",
        "f8977a02cd97302e6c7c0542c6c072e1e6690760",
    ]
    assert [item["failed_test"] for item in attempts] == [
        "tests/test_v0_3_s11_r0038_import.py::test_successor_62_binds_every_changed_historical_source",
        "tests/test_v0_3_s11_r0039_import.py::test_r0039_leaves_the_rc1_source_records_unchanged",
    ]
    assert all(item["complete_present"] is False for item in attempts)
    assert all(item["preservation"] == "preserved_unchanged" for item in attempts)
    assert all(item["qualification_evidence"] is False for item in attempts)

    temporary = record["superseded_temporary_build_directories"]
    assert [Path(item["path"]).name for item in temporary] == [
        "pssolver-v030-build-4BWFPv",
        "pssolver-v030-build-v2-CILAZY",
        "pssolver-v030-sdist-rebuild-l0mzwY",
    ]
    assert all(item["preservation"] == "preserved_unchanged" for item in temporary)
    assert all(item["qualification_evidence"] is False for item in temporary)


def test_r0040_does_not_rewrite_hash_bound_source_records() -> None:
    for relative, expected in _read(RECORD)["immutable_source_records"].items():
        assert _sha256(ROOT / relative) == expected


def test_r0040_closes_only_final_candidate_acceptance() -> None:
    record = _read(RECORD)
    assert record["completion_gates"] == {
        "G-01": "PASS",
        "G-05": "PASS",
        "G-09_rc1_candidate_acceptance": "PASS_R0038",
        "G-09_rc1_post_release_acceptance": "PASS_R0039",
        "G-09_final_candidate_acceptance": "PASS_R0040",
        "G-09_final_publication": "not_started",
        "G-09_final_post_release_acceptance": "not_started",
    }
    assert record["registry_updates"] == []
    assert record["new_findings"] == []


def test_r0040_import_is_records_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"] == {
        "records_and_tests_only": True,
        "runtime_changed": False,
        "packaging_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "tag_or_release_modified": False,
    }
    assert record["authorization"] == {
        "local_import": True,
        "push": False,
        "tag": False,
        "github_release": False,
        "pypi": False,
        "hpcc_or_slurm": False,
    }
    assert record["historical_successor_sources"] == {}
    assert record["next_action"] == "separately_authorized_final_publication"
