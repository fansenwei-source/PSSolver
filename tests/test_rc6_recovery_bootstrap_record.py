"""Bind the RC6 recovery-v6 bootstrap root-separation record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_recovery_bootstrap_root_separation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_preserves_the_v5_failure_classification_and_scope():
    record = _record()
    failure = record["failure_evidence"]
    assert failure["job_id"] == "10901181"
    assert failure["failed_predicate"] == "P08"
    assert failure["classification"].startswith("FAIL_RC6_")
    assert failure["historical_classification_rewritten"] is False
    assert failure["scientific_execution_started"] is False
    assert failure["solver_failure"] is False
    assert failure["scientific_failure"] is False


def test_record_binds_exact_new_qualification_sources():
    record = _record()
    assert set(record["source_sha256"]) == {
        "benchmarks/prepare_rc6_recovery_bootstrap.py",
        "benchmarks/validate_rc6_recovery_bootstrap.py",
        "jobs/rc6_consumer_rebind_recovery_v6.sbatch",
        "tests/test_rc6_recovery_bootstrap.py",
    }
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected


def test_record_requires_distinct_source_free_roots_and_installed_pssolver():
    correction = _record()["correction"]
    assert correction["strategy"] == (
        "distinct_source_free_script_root_and_benchmarks_path_shim"
    )
    assert correction["script_root"]["pssolver_source_allowed"] is False
    assert correction["import_root"]["contains_exactly"] == ["benchmarks"]
    assert correction["import_root"]["pssolver_source_allowed"] is False
    assert correction["pssolver_import_must_resolve_under_driver_purelib"] is True
    assert correction["all_stage_script_hashes_must_match_frozen_plan"] is True
    assert correction["historical_job_script_modified"] is False


def test_record_is_qualification_only_and_does_not_authorize_push_or_h100():
    record = _record()
    assert record["classification"] == (
        "READY_RC6_RECOVERY_V6_BOOTSTRAP_ROOT_SEPARATION_H100_NOT_AUTHORIZED"
    )
    scope = record["scope"]
    assert scope["qualification_support_only"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key != "qualification_support_only"
    )
    assert all(value is False for value in record["authorization"].values())
