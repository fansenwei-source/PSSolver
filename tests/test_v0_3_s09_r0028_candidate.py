from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s09_ap_cache_diagnostic_r0028_candidate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0028_candidate_is_diagnostic_only_and_binds_frozen_wheels() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == "READY_R0028_S09_AP_CACHE_DIAGNOSTIC_REVIEW"
    assert record["plan_step"] == "S-09"
    assert record["decision"] == "D-0012"
    assert record["git_identity"]["runtime_commit"] == (
        "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    )
    assert record["artifacts"]["baseline_wheel_sha256"] == (
        "a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e"
    )
    assert record["artifacts"]["candidate_wheel_sha256"] == (
        "4003645025da6d81317eeb7261ccf606aef7f60f20d072518f8baacf56cd4aa8"
    )
    contract = record["diagnostic_contract"]
    assert contract["profile_count"] == 72
    assert contract["formal_samples_per_profile"] == 60
    assert contract["thresholds_used_for_qualification"] is False
    assert contract["qualification_complete"] is False
    assert contract["s09_complete"] is False


def test_r0028_support_hashes_match_checked_in_files() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    hashes = record["support_hashes"]
    paths = {
        "plan_source": ROOT / "benchmarks/plan_v0_3_s09_ap_cache_diagnostic.py",
        "profile": ROOT / "benchmarks/profile_v0_3_s09_ap_cache_diagnostic.py",
        "child": ROOT / "benchmarks/run_v0_3_s09_ap_cache_diagnostic_child.py",
        "runner": ROOT / "benchmarks/run_v0_3_s09_ap_cache_diagnostic.py",
        "analyzer": ROOT / "benchmarks/analyze_v0_3_s09_ap_cache_diagnostic.py",
        "job": ROOT / "jobs/v0_3_s09_ap_cache_diagnostic_h100.sbatch",
        "plan_document": ROOT / "notes/PSSolver_v0_3_0_s09_ap_cache_diagnostic_plan_zh.md",
        "authorization_draft": ROOT
        / "notes/PSSolver_v0_3_0_s09_ap_cache_diagnostic_authorization_draft_zh.md",
    }
    assert {name: _sha256(path) for name, path in paths.items()} == {
        name: hashes[name] for name in paths
    }


def test_r0028_cpu_rehearsal_proves_state_identity_and_cache_disable() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    rehearsal = record["installed_wheel_cpu_rehearsal"]
    assert rehearsal["source_shadow_import"] is False
    assert rehearsal["profile_count"] == 72
    assert rehearsal["all_profiles_finite"] is True
    assert rehearsal["all_cache_off_proofs_passed"] is True
    assert rehearsal["candidate_cache_on_off_initial_state_byte_identical"] is True
    assert rehearsal["candidate_cache_on_off_final_state_byte_identical"] is True
    assert rehearsal["performance_interpretation"].startswith("none")


def test_r0028_grants_no_external_authorization() -> None:
    authorization = json.loads(RECORD.read_text(encoding="utf-8"))["authorization"]
    assert authorization == {
        "push": False,
        "hpcc": False,
        "slurm": False,
        "h100": False,
        "automatic_retry": False,
        "production_default_changed": False,
    }
