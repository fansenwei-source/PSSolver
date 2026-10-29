"""Contract for importing the independent R-0041 final post-release acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_66_r0041_import.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0041_identity_and_independent_archive_are_exact() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_R0041_V0_3_0_FINAL_POST_RELEASE_ACCEPTANCE"
    )
    assert record["git_identity"] == {
        "release_source": "47473880534a66f6da940db82122c79cda48a86b",
        "release_evidence_tip": "f961689a2905a64cf7d8665cd491abe0b7a9b1e6",
        "record_commit_is_release_source": False,
        "pushed": False,
    }
    assert record["sync_result"] == {
        "id": "R-0041",
        "path": "PSSolver_sync/results/R-0041.result.json",
        "sha256": "ee20c59726f16227c2c424b02dc4689968d4c1158db346efde4bd8f3b851a965",
        "sync_commit": "aac441e9fc426ff1ddfb86ac7ef664da480d2827",
        "classification": "PASS",
    }
    assert record["independent_archive"] == {
        "path": "/home/fansenwei/pssolver_v0_3_0_claude_post_release_acceptance_20261028_v1",
        "checksums_entries": 73,
        "checksums_sha256": "c3649b4a9779337b518c26be513fc3e1f4afbb7289e989aea28231fd0eacd31d",
        "complete_present": True,
        "final_report_sha256": "d8d82a5109a26f3cd1894eb629c1fa1cab4cc7a195793bb803d65787ae0dc320",
        "summary_sha256": "245c99ac39c4b1245f7772ad8b70a77c7864cddd6bc8e9cec44effdbe45db367",
    }


def test_r0041_published_release_and_assets_are_exact() -> None:
    release = _read(RECORD)["published_release"]
    assert release["release_id"] == 427402710
    assert release["node_id"] == "RE_kwDOQ5Io-s4ZeaXW"
    assert release["tag"] == "v0.3.0"
    assert release["tag_object"] == "f14c921b7114e792d0bc2af81a045e156eb89f4e"
    assert release["peeled_commit"] == "47473880534a66f6da940db82122c79cda48a86b"
    assert release["draft"] is False
    assert release["prerelease"] is False
    assert release["latest"] is True
    assert release["assets"] == [
        {
            "name": "pssolver-0.3.0-py3-none-any.whl",
            "asset_id": 687111271,
            "node_id": "RA_kwDOQ5Io-s4o9Hxn",
            "size": 646909,
            "digest": "sha256:8e2bd0c7941220f6c7c9b4cba80d9802e1495db626ad1b330a2e1681645187d5",
        },
        {
            "name": "pssolver-0.3.0.tar.gz",
            "asset_id": 687111272,
            "node_id": "RA_kwDOQ5Io-s4o9Hxo",
            "size": 1411145,
            "digest": "sha256:ec46471f5c35fc56089255155508b8a23a6d71ee81488d6b4aec933faed4d68d",
        },
    ]


def test_r0041_action_chain_and_completion_gates_are_exact() -> None:
    record = _read(RECORD)
    assert record["publication_action_chain"] == [
        {
            "id": "A-0051",
            "sync_commit": "1c5ed2f4f8e1454b99b0c71d1a46be1b6cac6761",
            "action": "publish_release_branches",
        },
        {
            "id": "A-0052",
            "sync_commit": "5f060a0054d810e602041cc03fbbbb8c70223016",
            "action": "publish_annotated_tag",
        },
        {
            "id": "A-0053",
            "sync_commit": "522c7090b66ae0b14f2cce7294b0a55a919af116",
            "action": "publish_github_release",
        },
    ]
    gates = record["completion_gates"]
    assert gates["G-09_final_candidate_acceptance"] == "PASS_R0040"
    assert gates["G-09_final_publication"] == "PASS_A0053"
    assert gates["G-09_final_post_release_acceptance"] == "PASS_R0041"
    assert gates["G-09"] == "PASS"
    assert gates["G-01_through_G-09"] == "PASS"


def test_r0041_does_not_rewrite_hash_bound_source_records() -> None:
    for relative, expected in _read(RECORD)["immutable_source_records"].items():
        assert _sha256(ROOT / relative) == expected


def test_r0041_import_is_records_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["historical_successor_sources"] == {}
    assert record["registry_updates"] == []
    assert record["new_findings"] == []
    assert record["release_blockers"] == []
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
    assert record["next_action"] == (
        "user_decision_on_main_fast_forward_and_maintenance_branch_policy"
    )
