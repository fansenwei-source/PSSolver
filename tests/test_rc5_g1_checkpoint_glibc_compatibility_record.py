"""Evidence contract for the rc5 G-1 glibc checkpoint compatibility fix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_checkpoint_glibc_compatibility.json"
)
RC6_R1 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
RC6_R2 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
RC6_R3 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
RC6_R4 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"
RC6_R5 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json"
RC6_R6 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"
RC6_FREEZE = ROOT / "notes/PSSolver_v0_2_0rc6_source_freeze.json"
FINAL_RELEASE = ROOT / "notes/pssolver_v0_2_0.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_glibc_compatibility_record_binds_implementation_and_tests():
    record = _record()
    rc6_successors = json.loads(RC6_R1.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r2_successors = json.loads(RC6_R2.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r3_successors = json.loads(RC6_R3.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r4_successors = json.loads(RC6_R4.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r5_successors = json.loads(RC6_R5.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r6_successors = json.loads(RC6_R6.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_freeze_successors = json.loads(RC6_FREEZE.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    final_release_successors = json.loads(FINAL_RELEASE.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    assert record["classification"] == (
        "PASS_RC5_G1_CHECKPOINT_GLIBC_2_17_CPU_COMPATIBILITY_CANDIDATE"
    )
    for relative, expected in record["new_sources"].items():
        assert _sha256(relative) == expected
    for relative, successor in record["historical_successor_sources"].items():
        expected = successor["current_sha256"]
        rc6_successor = rc6_successors.get(relative)
        accepted = (
            []
            if rc6_successor is None
            else rc6_successor.get(
                "accepted_predecessor_sha256",
                [rc6_successor["historical_sha256"]],
            )
        )
        if rc6_successor is not None and expected in accepted:
            expected = rc6_successor["current_sha256"]
        rc6_successor = rc6_r2_successors.get(relative)
        if (
            rc6_successor is not None
            and rc6_successor["historical_sha256"] == expected
        ):
            expected = rc6_successor["current_sha256"]
        rc6_successor = rc6_r3_successors.get(relative)
        if (
            rc6_successor is not None
            and rc6_successor["historical_sha256"] == expected
        ):
            expected = rc6_successor["current_sha256"]
        rc6_successor = rc6_r4_successors.get(relative)
        if (
            rc6_successor is not None
            and rc6_successor["historical_sha256"] == expected
        ):
            expected = rc6_successor["current_sha256"]
        rc6_successor = rc6_r5_successors.get(relative)
        if (
            rc6_successor is not None
            and rc6_successor["historical_sha256"] == expected
        ):
            expected = rc6_successor["current_sha256"]
        rc6_successor = rc6_r6_successors.get(relative)
        if (
            rc6_successor is not None
            and rc6_successor["historical_sha256"] == expected
        ):
            expected = rc6_successor["current_sha256"]
        rc6_successor = rc6_freeze_successors.get(relative)
        if (
            rc6_successor is not None
            and rc6_successor["historical_sha256"] == expected
        ):
            expected = rc6_successor["current_sha256"]
        final_successor = final_release_successors.get(relative)
        if (
            final_successor is not None
            and final_successor["historical_sha256"] == expected
        ):
            expected = final_successor["current_sha256"]
        assert _sha256(relative) == expected
        assert successor["historical_sha256"] != successor["current_sha256"]
        assert successor["reason"]


def test_glibc_compatibility_record_preserves_atomic_no_overwrite():
    contract = _record()["implementation_contract"]
    assert contract["preferred_backend"] == "libc.renameat2"
    assert contract["compatibility_backend"] == (
        "libc.syscall(__NR_renameat2=316)"
    )
    assert contract["compatibility_architectures"] == ["amd64", "x86_64"]
    assert contract["flags"] == "RENAME_NOREPLACE"
    assert contract["atomic_no_overwrite_preserved"] is True
    assert contract["exists_then_rename_fallback_added"] is False
    assert contract["unknown_architecture_fails_closed"] is True
    assert contract["kernel_enosys_fails_closed"] is True


def test_glibc_compatibility_record_binds_the_hpcc_probe():
    trigger = _record()["trigger"]
    assert trigger["job_id"] == "10885517"
    assert trigger["glibc"] == "2.17"
    assert trigger["libc_wrapper_present"] is False
    assert trigger["raw_syscall_probe"] == {
        "architecture": "x86_64",
        "syscall_number": 316,
        "flags": 1,
        "result": 0,
        "errno": 0,
        "source_renamed_to_target": True,
    }


def test_glibc_compatibility_record_binds_fresh_installed_wheel_scope():
    evidence = _record()["test_evidence"]["installed_wheel_hpcc_cpu"]
    assert evidence["classification"] == (
        "PASS_RC5_G1_GLIBC_2_17_INSTALLED_WHEEL_CPU_SMOKE_WITH_FILESYSTEM_SCOPE"
    )
    assert evidence["glibc"] == "2.17"
    assert evidence["renameat2_wrapper_present"] is False
    assert evidence["tmp_atomic_publication"] == "PASS"
    assert evidence["tmp_racing_empty_target_rejected"] is True
    assert evidence["home_nfs"] == "FAIL_CLOSED_EINVAL"
    assert evidence["scratch1_nfs"] == "FAIL_CLOSED_EINVAL"
    assert evidence["source_shadow_import"] is False
    assert evidence["pip_check"] == "PASS"
    assert evidence["checksum_verification"] == "11/11 PASS"
    assert evidence["checksum_manifest_sha256"] == (
        "0ed5ebd4761129b5ff772b223e532acf7fecdc5f0ef21f04672cc248607fc9c9"
    )


def test_glibc_compatibility_record_distinguishes_source_and_record_commits():
    identity = _record()["git_identity"]
    assert identity == {
        "branch": "fix/v0.2.0rc5",
        "qualified_source_commit": "dabeb9a487a82700ddd364c4233941a175808a3a",
        "record_commit_is_qualified_source": False,
        "pushed": False,
    }


def test_glibc_compatibility_record_keeps_release_authority_closed():
    record = _record()
    assert record["qualification_complete"] is False
    assert record["g1_complete"] is False
    assert not any(record["scope"].values())
    assert record["next_gate"] == (
        "request_explicit_push_and_single_h100_recovery_authorization_"
        "without_automatic_retry"
    )
