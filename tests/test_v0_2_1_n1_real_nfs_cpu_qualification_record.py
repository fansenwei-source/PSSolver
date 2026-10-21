"""Evidence contract for the v0.2.1 N-1 real-NFS CPU qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_1_n1_real_nfs_cpu_qualification.json"
PREDECESSOR = ROOT / "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_n1_real_nfs_cpu_record_binds_the_exact_qualified_source():
    record = _record()
    assert record["classification"] == "PASS_V0_2_1_N1_REAL_NFS_CPU_QUALIFICATION"
    assert record["qualified_source"] == {
        "branch": "maint/v0.2",
        "commit": "15670eb264a44c6707aeb1695b74e6cdbb34f611",
        "implementation_commit": "927dc89aa63bcee9b03248f52662389bc947c47a",
        "package_version_during_qualification": "0.2.0",
        "record_commit_is_qualified_source": False,
        "requires_cpu_requalification": False,
    }


def test_n1_real_nfs_cpu_record_does_not_rewrite_the_local_candidate_record():
    predecessor = _record()["predecessor_record"]
    assert predecessor["path"] == PREDECESSOR.relative_to(ROOT).as_posix()
    assert predecessor["sha256"] == _sha256(PREDECESSOR)
    assert predecessor["sha256"] == (
        "8d84221555cfc4d982f681c07ed83ad3b630747d5cac633e1b140e71bd5b577f"
    )
    assert predecessor["rewritten"] is False


def test_n1_real_nfs_cpu_record_binds_installed_wheel_results():
    installed = _record()["installed_wheel"]
    assert installed["source_archive_sha256"] == (
        "a349d61791770cc62caee0dd08f28292dd81fc0f35003c57eed909ff4303e68f"
    )
    assert installed["wheel_sha256"] == (
        "8068a1d916a8c7939068b53bb496fe768e7e150f58a8c2d3d367b85ecf77849f"
    )
    assert installed["installed_with_no_deps"] is True
    assert installed["pip_check"] == "PASS"
    assert installed["source_shadow_import"] is False
    assert installed["n1_fault_injection"] == {
        "passed": 13,
        "failed": 0,
        "skipped": 0,
    }


def test_n1_real_nfs_cpu_record_binds_both_single_client_nfs_gates():
    nfs = _record()["real_nfs_single_client"]
    assert set(nfs) == {"home", "scratch1", "provider_identity", "temporary_residue_count"}
    for location in ("home", "scratch1"):
        gate = nfs[location]
        assert gate["classification"] == "PASS_V0_2_1_N1_REAL_NFS_SINGLE_CLIENT"
        assert gate["filesystem_type"] == "0x6969"
        assert gate["raw_rename_noreplace_errno"] == 22
        assert gate["directory_fsync_passed"] is True
        assert gate["functional_checkpoint_exact_restore"] is True
        assert gate["collision_rejected_without_modification"] is True
        assert gate["fast_fallback_tree_bytes_equal"] is True
    assert nfs["provider_identity"] == {
        "canonical_sha256": "2d36ab290d053f4580ccd4c93974db6794ec709575179d62c002e32fb5b499e9",
        "unchanged_on_home": True,
        "unchanged_on_scratch1": True,
    }
    assert nfs["temporary_residue_count"] == 0


def test_n1_real_nfs_cpu_record_binds_cross_client_exactly_one_winner():
    cross = _record()["real_nfs_cross_client"]
    assert cross["classification"] == "PASS_V0_2_1_N1_REAL_NFS_CROSS_CLIENT"
    assert cross["slurm_job_id"] == 10902631
    assert cross["state"] == "COMPLETED"
    assert cross["exit_code"] == "0:0"
    assert set(cross["nodes"]) == {"compute-9-0", "compute-9-1"}
    assert cross["success_count"] == 1
    assert cross["collision_count"] == 1
    assert cross["collision_exception"] == "CheckpointExistsError"
    assert cross["temporary_residue_count"] == 0
    assert cross["report_sha256"] == (
        "ed92cdc66afb22f498f7ddaaf25cab3f8a7c044748c24c2d961462d040ca76a8"
    )


def test_n1_real_nfs_cpu_record_binds_external_manifests():
    evidence = _record()["external_evidence"]
    assert evidence["control_manifest_entries"] == evidence["control_manifest_passed"] == 29
    assert evidence["control_manifest_sha256"] == (
        "9748645f3690e9681d2ba5ea7cf0f85d81aadce4ca92a80aca3d3cd19213822b"
    )
    assert evidence["scratch_manifest_entries"] == evidence["scratch_manifest_passed"] == 7
    assert evidence["scratch_manifest_sha256"] == (
        "5c06f84993367c856f9f72d7979494315fc0efbc1f3e4160606dcfdce303d031"
    )


def test_n1_real_nfs_cpu_record_preserves_scope_and_stays_fail_closed():
    record = _record()
    assert not any(record["scope"].values())
    authorization = record["authorization"]
    assert authorization["installed_wheel_and_real_nfs_cpu_complete"] is True
    assert authorization["n1_complete"] is False
    assert authorization["eligible_for_small_h100_checkpoint_qualification"] is True
    assert authorization["formal_h100_submission_count"] == 0
    assert authorization["h100_authorized"] is False
    assert authorization["h100_submitted"] is False
    assert authorization["automatic_retry"] is False
    assert authorization["eligible_for_push"] is False
    assert authorization["version_bumped_to_0_2_1"] is False
    assert authorization["tag_created"] is False
    assert authorization["release_created"] is False
    assert authorization["pypi_published"] is False
