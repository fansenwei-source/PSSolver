"""Bindings for the R-0029 exact-revert diagnostic candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s09_ap_revert_proxy_r0029_candidate.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0029_candidate_binds_unchanged_runtime_wheels_and_exact_proxy() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "READY_R0029_S09_AP_EXACT_REVERT_PROXY_REVIEW"
    )
    assert record["git_identity"]["runtime_commit"] == (
        "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    )
    assert record["git_identity"]["qualification_support_commit"] == (
        "b0314216d1943c5131097f80d0ec164fe1b06bd2"
    )
    assert record["git_identity"]["support_changes_runtime"] is False
    assert record["git_identity"]["support_changes_wheels"] is False
    proxy = record["revert_proxy"]
    assert proxy["historical_method_lines"] == 691
    assert proxy["historical_method_sha256"] == (
        "2ffb5673ebdb92525b8959ee014aa3178b91cdf37789bdbd03d3cf2a5654e8f3"
    )
    assert proxy["optional_fourth_role_added"] is False


def test_r0029_support_hashes_match_live_files() -> None:
    hashes = _read(RECORD)["support_hashes"]
    paths = {
        "analyzer": ROOT / "benchmarks/analyze_v0_3_s09_ap_cache_diagnostic.py",
        "plan_source": ROOT / "benchmarks/plan_v0_3_s09_ap_cache_diagnostic.py",
        "profile": ROOT / "benchmarks/profile_v0_3_s09_ap_cache_diagnostic.py",
        "cpu_memory_probe": ROOT / "benchmarks/run_v0_3_s09_ap_cache_cpu_memory_probe.py",
        "runner": ROOT / "benchmarks/run_v0_3_s09_ap_cache_diagnostic.py",
        "child": ROOT / "benchmarks/run_v0_3_s09_ap_cache_diagnostic_child.py",
        "cuda_snapshot": ROOT / "benchmarks/run_v0_3_s09_ap_cache_memory_snapshot.py",
        "revert_proxy": ROOT / "benchmarks/v0_3_s09_revert_pressure_proxy.py",
        "job": ROOT / "jobs/v0_3_s09_ap_cache_diagnostic_h100.sbatch",
        "authorization_draft": ROOT / "notes/PSSolver_v0_3_0_s09_ap_cache_diagnostic_authorization_draft_zh.md",
    }
    for name, expected in hashes.items():
        path = paths[name]
        assert resolve_s03_successor(
            str(path.relative_to(ROOT)), expected
        ) == _sha256(path)


def test_r0029_local_archive_and_cpu_claims_are_exact() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    evidence = record["local_evidence"]
    assert_pytest_migration(
        "tests/test_v0_3_s09_r0029_candidate.py::"
        "test_r0029_local_archive_and_cpu_claims_are_exact"
    )
    assert evidence["checksums_entries"] == 331
    for field in (
        "checksums_sha256",
        "final_report_sha256",
        "summary_sha256",
        "cpu_memory_comparison_sha256",
    ):
        assert len(evidence[field]) == 64
    assert evidence["temporary_venvs_pruned"] is True
    rehearsal = record["installed_wheel_cpu_rehearsal"]
    assert rehearsal["profile_count"] == 72
    assert rehearsal["all_profiles_finite"] is True
    assert rehearsal["all_revert_proxy_proofs_passed"] is True
    # R-0029 withdrew the single-run CPU peak comparison as a fidelity proof.
    # Preserve only the reproducible identity evidence from this historical record.
    memory = record["cpu_r512_memory"]
    assert memory["proxy_matches_historical_operator_calls"] is True
    assert memory["all_final_states_byte_identical"] is True


def test_r0029_contract_is_balanced_descriptive_and_unauthorized() -> None:
    record = _read(RECORD)
    contract = record["diagnostic_contract"]
    assert contract["profile_count"] == 72
    assert contract["all_six_role_permutations"] is True
    assert contract["each_pair_ordered_three_each_way"] is True
    assert contract["each_role_in_each_position_twice"] is True
    assert contract["single_trial_ratio_estimator"] == (
        "frozen_profile_center_six_fixed_block_means_median"
    )
    assert contract["qualification_complete"] is False
    assert contract["s09_complete"] is False
    assert record["auxiliary_cuda_snapshot"]["included_in_timing_statistics"] is False
    assert record["authorization"] == {
        "push": False,
        "hpcc": False,
        "slurm": False,
        "h100": False,
        "automatic_retry": False,
        "production_default_changed": False,
    }
