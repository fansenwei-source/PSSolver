"""Historical binding for the completed rc5 G-1 v12 instrumentation run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc5_g1_timing_instrumentation_v12_result.json"


def test_v12_result_binds_completed_job_and_immutable_archive():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == "ANALYSIS_ONLY_INSTRUMENTATION_EVIDENCE_COMPLETE"
    assert record["attribution_classification"] == (
        "NO_SINGLE_GC_REFRESH_OR_PRESSURE_TERMINATION_CAUSE_SUPPORTED"
    )
    assert record["qualified_source_commit"] == (
        "6c572e1a99ef61e3306c455618c02156b5493b4a"
    )
    assert record["job"] == {
        "job_id": "10886072",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:02:39",
        "node": "gpu-h100-4-0",
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "second_job_submitted": False,
    }
    archive = record["archive"]
    assert (archive["checksum_pass"], archive["checksum_fail"]) == (83, 0)
    assert archive["checksum_manifest_sha256"] == (
        "da7ddb92658572b76889123c56f08e7bdb6ef863b009409c42a1ab6379e58ee1"
    )
    assert archive["analysis_complete_present"] is True
    assert archive["qualification_complete_present"] is False


def test_v12_result_preserves_analysis_only_boundary_and_source_identity():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    evidence = record["evidence"]
    assert evidence["profiles_completed"] == evidence["profiles_validated"] == 12
    assert evidence["asynchronous_spikes"] == 42
    assert evidence["synchronized_spikes"] == 34
    assert evidence["pressure_diagnostic_records"] == 180
    assert evidence["single_root_cause_resolved"] is False
    scope = record["scope"]
    assert scope["formal_v10_classification_preserved"] == (
        "FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE"
    )
    assert scope["formal_requalification_performed"] is False
    assert scope["qualification_complete"] is False
    assert scope["g1_complete"] is False
    assert scope["thresholds_changed"] is False
    assert scope["outlier_removal_performed"] is False
    assert scope["trial_replacement_performed"] is False
    assert scope["production_default_changed"] is False
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
