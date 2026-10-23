from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_31_s09_transient_telemetry.json"


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_transient_telemetry_is_one_shot_and_all_samples_are_retained():
    record = _record()
    assert record["classification"] == (
        "PASS_S09_CPU_REFERENCE_PREFLIGHT_AFTER_ONE_SHOT_TRANSIENT_TELEMETRY"
    )
    assert record["result"]["execution_count"] == 1
    assert record["result"]["profiles_completed"] == 12
    assert record["result"]["all_samples_retained"] is True
    assert max(record["result"]["paired_wall_ratios"]) <= 1.05
    assert record["result"]["focused_median_ratio"] <= 1.03
    assert record["result"]["candidate_fix_required_before_h100"] is False


def test_transient_telemetry_preserves_old_failures_and_h100_limits():
    record = _record()
    assert record["preserved_failures"]["samples_deleted"] is False
    assert record["preserved_failures"]["verdicts_rewritten"] is False
    assert record["formal_h100_contract"] == {
        "median_time_ratio_max": 1.03,
        "paired_time_ratio_max": 1.05,
        "all_samples_retained": True,
        "changed": False,
    }


def test_transient_telemetry_archive_and_sources_are_content_addressed():
    record = _record()
    evidence = record["evidence"]
    archive = Path(evidence["archive"])
    manifest = archive / "checksums.sha256"
    assert (archive / "COMPLETE").is_file()
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 29
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == evidence["checksums_sha256"]
    for relative, expected in record["runtime_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_transient_telemetry_does_not_authorize_external_actions():
    record = _record()
    assert record["roadmap"]["remaining_local_gates_authorized"] is True
    assert record["roadmap"]["h100_submission_authorized"] is False
    for key in ("push_performed", "hpcc_executed", "slurm_submitted", "gpu_executed", "h100_submitted", "automatic_retry"):
        assert record["authorization"][key] is False
