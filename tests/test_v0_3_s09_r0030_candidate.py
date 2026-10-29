"""Bindings for the R-0030 S-09 A-p diagnostic rework candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s09_ap_revert_proxy_r0030_candidate.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0030_binds_unchanged_runtime_wheels_and_reworked_support() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "READY_R0030_S09_AP_DIAGNOSTIC_REWORK_REVIEW"
    )
    assert record["git_identity"]["runtime_commit"] == (
        "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    )
    assert record["git_identity"]["qualification_support_commit"] == (
        "616bc453d50040b20d1ef448cf2647358aeee1cc"
    )
    assert record["git_identity"]["support_changes_runtime"] is False
    assert record["git_identity"]["support_changes_wheels"] is False


def test_r0030_support_hashes_match_live_files() -> None:
    record = _read(RECORD)
    paths = {
        "analyzer": "benchmarks/analyze_v0_3_s09_ap_cache_diagnostic.py",
        "plan_source": "benchmarks/plan_v0_3_s09_ap_cache_diagnostic.py",
        "profile": "benchmarks/profile_v0_3_s09_ap_cache_diagnostic.py",
        "runner": "benchmarks/run_v0_3_s09_ap_cache_diagnostic.py",
        "child": "benchmarks/run_v0_3_s09_ap_cache_diagnostic_child.py",
        "cuda_snapshot": "benchmarks/run_v0_3_s09_ap_cache_memory_snapshot.py",
        "revert_proxy": "benchmarks/v0_3_s09_revert_pressure_proxy.py",
        "cpu_memory_probe": "benchmarks/run_v0_3_s09_ap_cache_cpu_memory_probe.py",
        "diagnostic_preflight": "benchmarks/run_v0_3_s09_ap_cache_diagnostic_preflight.py",
        "main_environment_helper": "benchmarks/run_v0_3_s09_environment.py",
        "job": "jobs/v0_3_s09_ap_cache_diagnostic_h100.sbatch",
        "authorization_draft": "notes/PSSolver_v0_3_0_s09_ap_cache_diagnostic_authorization_draft_zh.md",
        "plan_document": "notes/PSSolver_v0_3_0_s09_ap_cache_diagnostic_plan_zh.md",
    }
    assert {name: _sha256(ROOT / relative) for name, relative in paths.items()} == (
        record["support_hashes"]
    )


def test_r0030_artifacts_and_archive_are_exact() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_r0030_candidate.py::"
        "test_r0030_artifacts_and_archive_are_exact"
    )
    assert record["local_evidence"]["checksums_entries"] == 28
    assert len(record["local_evidence"]["checksums_sha256"]) == 64
    for name in (
        "baseline_wheel",
        "candidate_wheel",
        "support_archive",
        "historical_source_archive",
        "h100_plan",
        "cpu_plan",
    ):
        expected = record["artifacts"][name]
        assert expected["size"] > 0
        assert len(expected["sha256"]) == 64
    assert record["artifacts"]["support_archive_excludes_pssolver"] is True
    assert record["artifacts"]["support_archive_excludes_pssolver_control"] is True


def test_r0030_withdraws_cpu_peak_proof_and_authorizes_nothing() -> None:
    record = _read(RECORD)
    boundary = record["memory_evidence_boundary"]
    assert boundary["cpu_probe_reference_only"] is True
    assert boundary["cpu_probe_reproducible_at_one_pressure_field_resolution"] is False
    assert boundary["r0029_cpu_memory_parity_claim_withdrawn"] is True
    assert boundary["persistent_cache_vs_local_lifetime"] == (
        "pending_h100_allocator_snapshots"
    )
    assert record["proxy_fidelity"]["cpu_peak_memory_used_as_proof"] is False
    assert record["status"] == {
        "v41": "registered",
        "s09_complete": False,
        "qualification_complete": False,
    }
    assert record["authorization"] == {
        "push": False,
        "hpcc": False,
        "slurm": False,
        "h100": False,
        "automatic_retry": False,
        "production_default_changed": False,
    }
