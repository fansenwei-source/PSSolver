"""Bind the RC6 nvcc-bootstrap recovery record and its qualification-only scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_nvcc_bootstrap_recovery.json"
HISTORICAL = ROOT / "notes/PSSolver_v0_2_0rc6_qualification_support.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_recovery_record_binds_failure_without_reclassifying_it_as_scientific():
    failed = _record()["failed_job"]
    assert failed["job_id"] == "10900727"
    assert failed["classification"] == "FAIL_RC6_SINGLE_H100_QUALIFICATION_EXTERNAL_NVCC_BOOTSTRAP"
    assert failed["node_local_tmp_preflight_passed"] is True
    assert failed["scientific_timestep_started"] is False
    assert failed["solver_failure"] is False
    assert failed["scientific_failure"] is False


def test_recovery_record_preserves_history_and_binds_exact_successor_sources():
    record = _record()
    historical = record["historical_record"]
    assert historical["rewritten"] is False
    assert _sha256(HISTORICAL) == historical["sha256"]
    for relative, expected in record["recovery_sources"].items():
        assert _sha256(ROOT / relative) == expected
    successors = record["historical_successor_sources"]
    assert set(successors) == {"jobs/rc6_single_h100_qualification.sbatch"}
    registration = successors["jobs/rc6_single_h100_qualification.sbatch"]
    assert registration["predecessor_sha256"] == (
        json.loads(HISTORICAL.read_text(encoding="utf-8"))["support_files"]
        ["jobs/rc6_single_h100_qualification.sbatch"]
    )
    assert registration["successor_sha256"] == _sha256(
        ROOT / "jobs/rc6_single_h100_qualification.sbatch"
    )


def test_recovery_record_is_qualification_only_and_has_no_submission_authority():
    record = _record()
    assert record["classification"] == "READY_RC6_NVCC_BOOTSTRAP_RECOVERY_SUPPORT_H100_NOT_AUTHORIZED"
    scope = record["scope"]
    assert scope["qualification_only"] is True
    assert all(value is False for key, value in scope.items() if key != "qualification_only")
    authorization = record["authorization"]
    assert authorization["push_authorized"] is False
    assert authorization["recovery_h100_submission_authorized"] is False
    assert authorization["automatic_retry"] is False
    assert authorization["tag_authorized"] is False
    assert authorization["release_authorized"] is False
