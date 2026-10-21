"""Evidence contract for importing the independent R-0003 acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_2_r0003_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0003_import_binds_the_sync_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_R0003_V0_2_2_CANDIDATE_ACCEPTANCE_IMPORT"
    )
    assert record["git_identity"]["sync_result_commit"] == (
        "a72fb4a01e9047ee2f5167954c2a286d59bc1f4a"
    )
    result = record["sync_result"]
    assert result["id"] == "R-0003"
    assert result["result_sha256"] == (
        "0f68e73a6d3cd130d44a2f41dfca9867702570049ec2c5f670a70dc13a03c91a"
    )
    assert result["classification"] == "PASS"
    assert result["release_blockers"] == []
    assert result["registry_updates"] == []
    assert result["new_findings"] == []
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 32515
    assert archive["checksums_sha256"] == (
        "a8f649d5d6a4354028a87ff690dc1325d5306ceca3395332ff1db9ef0813a6c6"
    )
    assert archive["complete_present"] is True


def test_r0003_import_binds_target_baseline_and_artifacts() -> None:
    candidate = _read(RECORD)["candidate_acceptance"]
    assert candidate["target_commit"] == (
        "d0a23a46bb77c9cc9b47e5df4e3f646ab7169d73"
    )
    assert candidate["baseline_tag"] == "v0.2.1"
    assert candidate["baseline_tag_object"] == (
        "daf1270b1f046ac767045cbbe536fd185599d83f"
    )
    assert candidate["baseline_commit"] == (
        "ca9c6729670f07639e7877881c777d1653f4a592"
    )
    assert candidate["artifacts"] == {
        "pssolver-0.2.2-py3-none-any.whl": {
            "size": 801349,
            "sha256": (
                "a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e"
            ),
        },
        "pssolver-0.2.2.tar.gz": {
            "size": 1410291,
            "sha256": (
                "ad8237e67478f7328c5152981ea4d888f1df64e1af7227f07ba0a0f070b02b39"
            ),
        },
    }


def test_r0003_import_preserves_the_exact_verdicts() -> None:
    candidate = _read(RECORD)["candidate_acceptance"]
    assert candidate["verdicts"] == {
        "V022-C1": "PASS",
        "V022-C2": "PASS",
        "V022-C3": "PASS",
        "V022-C4": "COVERED_BY_EVIDENCE",
        "V022-C5": "PASS",
        "full-cpu-suite": "PASS",
        "evidence-records": "PASS",
        "h100": "NOT_RUN",
    }
    assert candidate["full_cpu_suite"] == {
        "passed": 4191,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
    }
    for key in (
        "runtime_delta_is_v17_plus_version_only",
        "checkpoint_trees_byte_identical_to_v0_2_1",
        "provider_identity_unchanged",
        "control_provider_pins_accepted",
        "installed_wheel_gate_passed",
        "real_nfs_gate_covered_by_existing_evidence",
        "eligible_for_separately_authorized_release_actions",
    ):
        assert candidate[key] is True
    assert candidate["h100_required"] is False


def test_r0003_import_does_not_change_the_issue_registry() -> None:
    record = _read(RECORD)["registry"]
    assert _sha256(REGISTRY) == record["sha256"] == (
        "ffdd4fe41ddc140857081febf634b28bed4e1d3345f9e434982ef11ec2f67cc2"
    )
    assert record["issue_count"] == 28
    assert record["modified_by_import"] is False
    assert record["updates_imported"] == 0
    assert record["new_findings_imported"] == 0


def test_r0003_import_binds_reports_and_codex_evidence() -> None:
    record = _read(RECORD)
    archive = record["independent_archive"]
    assert archive["final_report_sha256"] == (
        "96dbad417b401692ea3ba37ad634c517a084899284588817e5163aa2748aef4c"
    )
    assert archive["summary_sha256"] == (
        "cdcccd1bcb072c343835647312c0befe5945881d57e313e046eb1b4e66efb121"
    )
    codex = record["cross_checked_codex_evidence"]
    assert codex["checksums_entries"] == 11
    assert codex["checksums_sha256"] == (
        "62bdd3cbb0acb9ba2bfb62b0ca920d730fa0b7f420240b3fa4a1bdc5bc4f2731"
    )


def test_r0003_import_does_not_expand_scope_or_authority() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    for key, value in record["scope"].items():
        if key != "evidence_only":
            assert value is False
    assert all(value is False for value in record["authorization"].values())
    assert record["git_identity"]["record_commit_is_release_source"] is False
    assert record["git_identity"]["pushed"] is False
    assert record["next_sync_id"] == "R-0004"
