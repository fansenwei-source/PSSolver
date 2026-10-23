from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_29_s09_cpu_6x60_stop.json"


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s09_6x60_stopped_after_exactly_one_candidate_matrix():
    record = _record()
    assert record["classification"] == (
        "STOPPED_S09_CPU_6X60_AT_CHANNEL_FUNCTIONAL_TOLERANCE_PAIRED_LIMIT"
    )
    assert record["execution"] == {
        "self_control_runs": 1,
        "candidate_bound_runs": 1,
        "automatic_retry": False,
        "self_control_passed": True,
        "candidate_bound_passed": False,
        "remaining_local_gates_executed": False,
        "r0022_created": False,
        "h100_preparation_executed": False,
    }
    assert record["roadmap"]["s09_complete"] is False
    assert record["roadmap"]["request_created"] is False


def test_s09_6x60_self_control_passed_but_candidate_pair_failed():
    record = _record()
    assert record["self_control"]["failed_cells"] == 0
    assert record["self_control"]["maximum_paired_time_ratio"] <= 1.05
    failure = record["candidate_failure"]
    assert failure["family"] == "channel_functional"
    assert failure["mode"] == "relative_tolerance"
    assert failure["trial"] == 1
    assert failure["paired_time_ratio"] > 1.05
    assert failure["cell_median_time_ratio"] <= 1.03
    assert failure["optimization_cache_active_for_functional_path"] is False


def test_s09_6x60_contract_kept_every_sample_and_threshold():
    contract = _record()["measurement_contract"]
    assert contract["trials_per_cell"] == 6
    assert contract["samples_per_profile"] == 60
    assert contract["profile_center"] == "median_of_six_fixed_block_means"
    assert contract["all_samples_retained"] is True
    assert contract["trial_replacement_allowed"] is False
    assert contract["outlier_removal_allowed"] is False
    assert contract["thresholds_relaxed"] is False
    assert contract["median_time_ratio_max"] == 1.03
    assert contract["paired_time_ratio_max"] == 1.05


def test_s09_6x60_archive_and_matrices_are_content_addressed():
    evidence = _record()["evidence"]
    archive = Path(evidence["archive"])
    assert (archive / "COMPLETE").is_file()
    assert evidence["checksums_entries"] == sum(
        1 for _ in (archive / "checksums.sha256").open(encoding="utf-8")
    )
    assert evidence["checksums_sha256"] == _sha256(archive / "checksums.sha256")
    assert evidence["self_control_matrix_sha256"] == _sha256(
        archive / "results/self_control/matrix.json"
    )
    assert evidence["candidate_bound_matrix_sha256"] == _sha256(
        archive / "results/candidate_bound/matrix.json"
    )


def test_s09_6x60_sources_and_historical_failure_remain_bound():
    record = _record()
    for relative, expected in record["runtime_sources"].items():
        assert _sha256(ROOT / relative) == expected
    historical = record["historical_5x30_evidence"]
    assert historical["retained"] is True
    assert historical["failed_paired_time_ratio"] == 1.0822582930338371
    assert historical["retained_spike_seconds"] == 0.10759581700000001
    assert _sha256(Path(historical["archive"]) / "checksums.sha256") == historical["checksums_sha256"]


def test_s09_6x60_stop_does_not_authorize_external_actions():
    authorization = _record()["authorization"]
    for key in (
        "push_performed",
        "hpcc_executed",
        "slurm_submitted",
        "gpu_executed",
        "h100_submitted",
        "threshold_change",
        "sample_deletion",
        "production_default_changed",
    ):
        assert authorization[key] is False
