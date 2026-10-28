"""Contract for importing the independent R-0039 rc1 post-release acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_63_r0039_import.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0039_identity_and_independent_archive_are_exact() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_R0039_V0_3_0RC1_INDEPENDENT_POST_RELEASE_ACCEPTANCE"
    )
    assert record["sync_result"] == {
        "id": "R-0039",
        "path": "PSSolver_sync/results/R-0039.result.json",
        "sha256": "c2b09bce66f67e9cea3cc2830b00d70e2a388148b08130db978ace01f38f939a",
        "sync_commit": "4ded6ead72ebcaee1b5b3b7923dc5363e83b2a82",
        "classification": "PASS",
    }
    assert record["independent_archive"] == {
        "path": "/home/fansenwei/pssolver_v0_3_0rc1_claude_post_release_acceptance_20261028_v1",
        "checksums_entries": 76,
        "checksums_sha256": "7ae463d728156195fb92d1750a39a0c89c36d8e9378a62f12c3ca7077e3b0612",
        "complete_present": True,
        "final_report_sha256": "7837206cb54ae85bf65691b8c3162fe66b9ff70d950843527685838861fa0382",
        "summary_sha256": "3f32f12ab23b4fc8cc361f27cf0635f900a91ba46039f1f2639eb91f0e54e5a1",
    }


def test_r0039_release_identity_is_exact_and_unchanged() -> None:
    assert _read(RECORD)["release_identity"] == {
        "release_id": 427202656,
        "tag": "v0.3.0rc1",
        "tag_object": "3fb935027c3e308e3d41593161e9486e5c736b34",
        "tag_commit": "c031cdae646edc5b202b05238db1eb513b255805",
        "published_at": "2026-10-28T21:19:11Z",
        "prerelease": True,
        "draft": False,
        "asset_ids": [686374469, 686374464],
        "keep_published_unchanged": True,
    }


def test_first_publication_directory_is_only_the_release_body_source() -> None:
    source = _read(RECORD)["first_publication_directory"]
    assert source["release_notes_sha256"] == (
        "3ec58dc33449677f20857ac4c2c11c097703a3916e40ec586ffaac7eda7e436c"
    )
    assert source["role"] == "source of the GitHub Release body"
    assert source["preserve_unchanged"] is True
    assert source["success_evidence"] is False


def test_r0039_leaves_the_rc1_source_records_unchanged() -> None:
    for relative, expected in _read(RECORD)["immutable_source_records"].items():
        assert _sha256(ROOT / relative) == expected


def test_r0039_closes_only_s11_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["completion_gates"] == {
        "S-11": "PASS",
        "G-09_rc1_candidate_acceptance": "PASS_R0038",
        "G-09_rc1_post_release_acceptance": "PASS_R0039",
        "final_v0_3_0_S-12": "not_started",
    }
    assert record["registry_updates"] == []
    assert record["new_findings"] == []
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
    assert record["next_plan_step"] == "S-12"
