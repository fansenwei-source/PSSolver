"""Static gates for the RC4.4.7 bounded-operator recovery candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_pressure_bounded_operator_recovery_v11.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_v11_binds_failed_h100_evidence_and_analysis_only_input():
    record = _record()
    assert record["phase"] == (
        "RC4.4.7-bounded-operator-recovery-v11-provider"
    )
    assert record["status"] == "LOCAL_PROVIDER_CANDIDATE_H100_NOT_AUTHORIZED"
    assert record["failed_v10_evidence"]["job_id"] == "10878393"
    assert record["failed_v10_evidence"]["scientific_failure"] is False
    assert record["failed_v10_evidence"]["performance_failure"] is True
    assert record["analysis_only_input"]["checksums_sha256"] == (
        "91cc97425d258876843efb15492fd4c2f2ab962d2930b7aec687de19c6f8adf8"
    )


def test_v11_preserves_pressure_and_failure_contracts():
    repair = _record()["repair"]
    for field in (
        "breakdown_definition_changed",
        "breakdown_threshold_changed",
        "exact_fallback_removed",
        "nonfinite_failure_changed",
        "fixed_work_changed",
        "relative_tolerance_changed",
        "pressure_tensor_update_order_changed",
        "diagnostics_schema_changed",
        "model_equations_changed",
        "production_default_changed",
    ):
        assert repair[field] is False


def test_v11_successor_sources_are_exact_and_content_addressed():
    successors = _record()["successor_sources"]
    assert set(successors) == {
        "pssolver/linear_solvers/stokes/channel_no_slip.py",
        "tests/test_rc4_pressure_policy_implementation.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert registration["current_sha256"] == _sha256(relative)
        assert registration["introduced_by_phase"] == (
            "RC4.4.7-bounded-operator-recovery-v11-provider"
        )
        assert registration["reason"]


def test_v11_local_screen_is_nonqualifying_and_h100_remains_unauthorized():
    record = _record()
    inventory = record["local_validation"]["operator_inventory_RTX3060Ti"]
    assert inventory["qualification_evidence"] is False
    assert inventory["candidate_vdot_operations_per_step"] == 0
    assert inventory["candidate_gt_operations_per_step"] == 0
    assert inventory["candidate_real_operations_per_step"] <= 108
    assert record["local_validation"]["H100_requalified"] is False
    assert record["authorization"] == {
        "push_authorized": False,
        "single_h100_submission_authorized": False,
        "automatic_retry_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
    }
    assert record["PSSolver_Control_modified"] is False
    assert record["nematics3d_modified"] is False
