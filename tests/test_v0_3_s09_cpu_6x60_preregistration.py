from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s09_cpu_6x60_preregistration.json"


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_s09_6x60_contract_is_frozen_before_measurement():
    record = _record()
    assert record["classification"] == "PREREGISTERED_S09_CPU_6X60_MEASUREMENT_CONTRACT"
    assert record["decision"]["id"] == "Q-S09-PERF-02"
    assert record["decision"]["selected"] == "B"
    assert record["estimator"] == {
        "trials_per_cell": 6,
        "samples_per_profile": 60,
        "fixed_block_count": 6,
        "fixed_block_size": 10,
        "profile_center": "median_of_six_fixed_block_means",
        "alternating_pair_order": True,
        "all_samples_retained": True,
        "trial_replacement_allowed": False,
        "outlier_removal_allowed": False,
    }
    assert record["thresholds"]["relaxed"] is False
    assert record["execution_order"][0].startswith("v0.2.2 versus v0.2.2")
    assert record["stop_contract"]["rerun_until_pass_allowed"] is False
    runner = record["qualification_runner"]
    assert hashlib.sha256((ROOT / runner["path"]).read_bytes()).hexdigest() == runner["sha256"]


def test_s09_6x60_contract_preserves_the_failed_5x30_sample():
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    historical = _record()["historical_evidence_retained"]
    assert historical["five_by_thirty_results_retained"] is True
    assert historical["failed_ratio"] == 1.0822582930338371
    assert historical["failed_sample_seconds"] == 0.10759581700000001
    assert_pytest_migration(
        "tests/test_v0_3_s09_cpu_6x60_preregistration.py::"
        "test_s09_6x60_contract_preserves_the_failed_5x30_sample"
    )
    assert historical["checksums_sha256"] == (
        "679ba7adaceddceaebe55769b1d2bc81a7c4c47c680525465a9884ea482aa02d"
    )


def test_s09_6x60_preregistration_does_not_authorize_external_actions():
    authorization = _record()["authorization"]
    assert authorization["local_cpu_execution"] is True
    for key in ("push", "hpcc", "slurm", "gpu", "h100", "automatic_retry"):
        assert authorization[key] is False
