"""Evidence contract for importing the independent R-0004 acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_2_r0004_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0004_import_binds_result_and_independent_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_R0004_V0_2_2_POST_RELEASE_ACCEPTANCE_IMPORT"
    )
    assert record["git_identity"]["sync_result_commit"] == (
        "22e605a507fcb3c1be64876a3304c232dd862b13"
    )
    result = record["sync_result"]
    assert result["id"] == "R-0004"
    assert result["result_sha256"] == (
        "a51174da348ce6e0e434ed81ce36535fb323116894616bcf0295ecf6b4b3af2a"
    )
    assert result["classification"] == "PASS"
    assert result["release_blockers"] == []
    assert result["registry_updates"] == []
    assert result["new_findings"] == []
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 2086
    assert archive["checksums_sha256"] == (
        "c0aea7f24fd92f6ef5cc22a8874759812ab9194f6cb9448a5b08553d12f479a7"
    )
    assert archive["complete_present"] is True


def test_r0004_import_binds_exact_release_identity() -> None:
    release = _read(RECORD)["release_acceptance"]
    assert release["tag"] == "v0.2.2"
    assert release["tag_object"] == (
        "283fc61091c3cf90871ebb2a10e405e22b399a6f"
    )
    assert release["tag_commit"] == (
        "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
    )
    assert release["release_id"] == 419183819
    assert release["draft"] is False
    assert release["prerelease"] is False
    assert release["remain_published_unchanged"] is True
    assert release["assets"] == {
        "pssolver-0.2.2-py3-none-any.whl": {
            "asset_id": 661436530,
            "size": 801349,
            "sha256": (
                "a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e"
            ),
        },
        "pssolver-0.2.2.tar.gz": {
            "asset_id": 661436527,
            "size": 1410291,
            "sha256": (
                "ad8237e67478f7328c5152981ea4d888f1df64e1af7227f07ba0a0f070b02b39"
            ),
        },
    }


def test_r0004_import_preserves_all_verdicts_and_test_counts() -> None:
    release = _read(RECORD)["release_acceptance"]
    assert release["verdicts"] == {
        "V022-PR1": "PASS",
        "V022-PR2": "PASS",
        "V022-PR3": "PASS",
        "V022-PR4": "PASS",
        "V022-PR5": "PASS",
        "V022-PR6": "PASS",
        "h100": "NOT_RUN",
    }
    assert release["focused_installed_wheel_tests"] == {
        "passed": 16,
        "failed": 0,
    }
    assert release["full_cpu_suite"] == {
        "passed": 4179,
        "failed": 0,
        "skipped": 0,
        "deselected": 7,
        "subtests_passed": 8,
    }
    for key in (
        "online_assets_byte_identical_to_qualified_assets",
        "tag_source_sdist_wheel_checkpoint_bytes_identical",
        "checkpoint_trees_identical_to_v0_2_1",
        "provider_identity_unchanged",
    ):
        assert release[key] is True
    assert release["h100_required"] is False


def test_r0004_import_records_only_evidence_successors() -> None:
    branch = _read(RECORD)["release_branch"]
    assert branch["remote_head_at_acceptance"] == (
        "9bfa78e3e5d8a140627a151c2e8305019dab0e2e"
    )
    assert branch["evidence_only_successors_after_release_source"] == [
        "d607c31eda2de4662c5d775ca5fb233721c17d82",
        "d0a23a46bb77c9cc9b47e5df4e3f646ab7169d73",
        "9bfa78e3e5d8a140627a151c2e8305019dab0e2e",
    ]
    assert branch["runtime_or_packaging_files_changed_by_successors"] is False


def test_r0004_import_does_not_change_the_registry() -> None:
    registry = _read(RECORD)["registry"]
    assert registry["sha256"] == (
        "ffdd4fe41ddc140857081febf634b28bed4e1d3345f9e434982ef11ec2f67cc2"
    )
    assert registry["issue_count"] == 28
    assert registry["modified_by_import"] is False
    assert registry["updates_imported"] == 0
    assert registry["new_findings_imported"] == 0


def test_r0004_import_does_not_expand_scope_or_authority() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    for key, value in record["scope"].items():
        if key != "evidence_only":
            assert value is False
    assert all(value is False for value in record["authorization"].values())
    assert record["git_identity"]["record_commit_is_release_source"] is False
    assert record["git_identity"]["pushed"] is False
    assert record["next_sync_id"] == "R-0005"
