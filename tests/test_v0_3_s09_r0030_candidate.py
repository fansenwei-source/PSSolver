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
    record = _read(RECORD)
    root = Path(record["local_evidence"]["path"])
    assert (root / "COMPLETE").is_file()
    assert sum(1 for _ in (root / "checksums.sha256").open(encoding="utf-8")) == 28
    assert _sha256(root / "checksums.sha256") == record["local_evidence"][
        "checksums_sha256"
    ]
    paths = {
        "baseline_wheel": root / "artifacts/pssolver-0.2.2-py3-none-any.whl",
        "candidate_wheel": root / "artifacts/pssolver-0.3.0.dev0-py3-none-any.whl",
        "support_archive": root / "artifacts/provider_support_658baca.tar",
        "historical_source_archive": root / "artifacts/provider_f94b9be.tar",
        "h100_plan": root / "plans/h100_plan.json",
        "cpu_plan": root / "plans/cpu_rehearsal_plan.json",
    }
    for name, path in paths.items():
        expected = record["artifacts"][name]
        assert path.stat().st_size == expected["size"]
        assert _sha256(path) == expected["sha256"]
    with tarfile.open(paths["support_archive"], "r") as archive:
        names = archive.getnames()
    assert not any("/pssolver/" in name or "/pssolver_control/" in name for name in names)


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
