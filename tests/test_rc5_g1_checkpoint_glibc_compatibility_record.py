"""Evidence contract for the rc5 G-1 glibc checkpoint compatibility fix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_checkpoint_glibc_compatibility.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_glibc_compatibility_record_binds_implementation_and_tests():
    record = _record()
    assert record["classification"] == (
        "PASS_RC5_G1_CHECKPOINT_GLIBC_2_17_CPU_COMPATIBILITY_CANDIDATE"
    )
    for relative, expected in record["new_sources"].items():
        assert _sha256(relative) == expected
    for relative, successor in record["historical_successor_sources"].items():
        assert _sha256(relative) == successor["current_sha256"]
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


def test_glibc_compatibility_record_keeps_release_authority_closed():
    record = _record()
    assert record["qualification_complete"] is False
    assert record["g1_complete"] is False
    assert not any(record["scope"].values())
    assert record["next_gate"].startswith("build_and_validate")
