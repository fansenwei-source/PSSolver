from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_30_s09_failure_analysis.json"


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s09_failure_analysis_uses_no_new_profiles_and_preserves_failure():
    record = _record()
    assert record["classification"] == "TRANSIENT_PROFILE_SLOWDOWN_SUPPORTED_CAUSE_UNRESOLVED"
    assert record["findings"]["new_profile_execution_count"] == 0
    assert record["adjudication"]["qualification_failure_remains_binding"] is True
    assert record["adjudication"]["s09_complete"] is False
    assert record["adjudication"]["r0022_created"] is False


def test_s09_failure_analysis_does_not_overclaim_causality():
    findings = _record()["findings"]
    assert findings["baseline_pressure_iterations"] == 27
    assert findings["candidate_pressure_iterations"] == 27
    assert findings["failed_profile_first_half_ratio"] > 1.05
    assert findings["failed_profile_second_half_ratio"] < 1.03
    assert findings["stable_acceptance_recompute_sufficient_to_explain_failure"] is False
    assert findings["single_outlier_explanation_supported"] is False
    assert findings["transient_first_half_slowdown_supported"] is True
    assert findings["unique_runtime_root_cause_supported"] is False


def test_s09_failure_analysis_archive_is_content_addressed():
    evidence = _record()["evidence"]
    archive = Path(evidence["archive"])
    assert (archive / "COMPLETE").is_file()
    assert evidence["checksums_entries"] == sum(
        1 for _ in (archive / "checksums.sha256").open(encoding="utf-8")
    )
    assert evidence["checksums_sha256"] == _sha256(archive / "checksums.sha256")
    assert evidence["analysis_sha256"] == _sha256(archive / "results/analysis.json")
    assert evidence["pytest_log_sha256"] == _sha256(archive / "logs/pytest.log")
    assert evidence["pytest_junit_sha256"] == _sha256(archive / "logs/pytest.junit.xml")


def test_s09_failure_analysis_source_and_authorization_are_bound():
    record = _record()
    for relative, expected in record["runtime_sources"].items():
        assert _sha256(ROOT / relative) == expected
    for key, value in record["authorization"].items():
        assert value is False, key
