"""Evidence contract for the v0.2.1 N-1 cumulative qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_1_n1_h100_qualification.json"
CPU_RECORD = ROOT / "notes/PSSolver_v0_2_1_n1_real_nfs_cpu_qualification.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_n1_h100_record_binds_the_exact_qualified_source_and_wheel():
    record = _record()
    assert record["classification"] == "PASS_V0_2_1_N1_CUMULATIVE_QUALIFICATION"
    assert record["qualified_source"] == {
        "branch": "maint/v0.2",
        "commit": "15670eb264a44c6707aeb1695b74e6cdbb34f611",
        "implementation_commit": "927dc89aa63bcee9b03248f52662389bc947c47a",
        "cpu_evidence_commit": "80cac090dadc1b205a7ecaf655d0bf027caf0b1d",
        "package_version_during_qualification": "0.2.0",
        "record_commit_is_qualified_source": False,
        "requires_h100_requalification": False,
    }
    assert record["installed_wheel"]["sha256"] == (
        "8068a1d916a8c7939068b53bb496fe768e7e150f58a8c2d3d367b85ecf77849f"
    )


def test_n1_h100_record_preserves_the_cpu_qualification_record():
    reused = _record()["reused_cpu_qualification"]
    assert reused["record"] == CPU_RECORD.relative_to(ROOT).as_posix()
    assert reused["record_sha256"] == _sha256(CPU_RECORD)
    assert reused["record_sha256"] == (
        "52f98d88c0a18b24db8ea14cf004e280216d17269f0d76a79039ce8d9363eafa"
    )
    assert reused["classification"] == "PASS_V0_2_1_N1_REAL_NFS_CPU_QUALIFICATION"
    assert reused["rewritten"] is False


def test_n1_h100_record_binds_the_only_job_and_environment():
    job = _record()["h100_job"]
    assert job["job_id"] == 10902665
    assert job["state"] == "COMPLETED"
    assert job["exit_code"] == "0:0"
    assert job["node"] == "gpu-h100-4-0"
    assert job["gpu"] == "NVIDIA H100 PCIe"
    assert job["visible_cuda_device_count"] == 1
    assert job["allocated_device"] == "cuda:0"
    assert job["requeue"] == 0
    assert job["restarts"] == 0
    assert job["formal_h100_submission_count"] == 1
    assert job["automatic_retry"] is False
    assert job["second_job_submitted"] is False
    environment = _record()["environment_gate"]
    assert environment["tf32_matmul"] is False
    assert environment["tf32_cudnn"] is False
    assert environment["source_shadow_import"] is False
    assert environment["workspace_filesystem_type"] == "0x6969"


def test_n1_h100_record_binds_both_checkpoint_restart_gates():
    gates = _record()["checkpoint_restart_gates"]
    assert set(gates) == {"periodic", "channel"}
    for family in ("periodic", "channel"):
        gate = gates[family]
        assert gate["classification"] == "PASS_V0_2_1_N1_H100_CHECKPOINT_RESTART"
        assert gate["continuous_steps"] == 4
        assert gate["split"] == "2+checkpoint+restart+2"
        assert gate["checkpoint_import_exact"] is True
        assert gate["state_byte_identical"] is True
        assert gate["q_velocity_pressure_byte_identical"] is True
        assert gate["finite"] is True
        assert gate["collision_rejected"] is True
        assert gate["provider_identity_unchanged"] is True
        assert gate["temporary_residue_count"] == 0


def test_n1_h100_record_binds_complete_external_manifests():
    evidence = _record()["external_evidence"]
    assert evidence["control_manifest_entries"] == evidence["control_manifest_passed"] == 25
    assert evidence["control_manifest_sha256"] == (
        "694043a8a8f01dba5506482d32273d1cc65ef2ad9fc4290a446d6002f1b26e0c"
    )
    assert evidence["scratch_manifest_entries"] == evidence["scratch_manifest_passed"] == 18
    assert evidence["scratch_manifest_sha256"] == (
        "2ec118f3c089a25377cbccad3da8d363bbb30d0cee61f42049d250f1076e6742"
    )
    assert evidence["complete_sha256"] == (
        "fa0aabdebf4847d99b5447d07cf286c2c6db075b5798ccecccf50ca29698d864"
    )


def test_n1_h100_record_closes_n1_without_expanding_scope():
    record = _record()
    assert not any(record["scope"].values())
    authorization = record["authorization"]
    assert authorization["qualification_complete"] is True
    assert authorization["n1_complete"] is True
    assert authorization["eligible_for_v0_2_1_finalization"] is True
    assert authorization["eligible_for_push"] is False
    assert authorization["version_bumped_to_0_2_1"] is False
    assert authorization["tag_created"] is False
    assert authorization["release_created"] is False
    assert authorization["pypi_published"] is False
