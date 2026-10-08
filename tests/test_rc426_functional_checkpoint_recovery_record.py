"""Frozen CPU recovery record for the RC4.2.6 functional identity defect."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc426_functional_checkpoint_recovery.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc426_recovery_binds_failure_and_parent_records():
    record = _record()
    assert record["status"] == "ready_for_h100_recovery"
    assert record["classification"] == (
        "PASS_RC4_2_6_FUNCTIONAL_CHECKPOINT_DEVICE_NEUTRAL_"
        "IDENTITY_CPU_RECOVERY"
    )
    assert record["git_identity"]["implementation_parent"] == (
        "949058d97f1beb12774fd981fe9e0c49adeebdfb"
    )
    for name in ("planning_record", "predecessor_record"):
        relative = record["git_identity"][name]
        assert _sha256(relative) == record["git_identity"][f"{name}_sha256"]
    failure = record["failure_evidence"]
    assert failure["job_id"] == "10861281"
    assert failure["failed_cell"] == "X11"
    assert failure["formal_h100_submission_count"] == 1
    assert failure["scientific_failure"] is False
    assert failure["manifest_entries"] == 596
    assert failure["manifest_sha256"] == (
        "fe98d2b8184591e8ed0dc222628be91a5795ec12b73e9d9e9689c541fc668967"
    )


def test_rc426_recovery_registers_exact_historical_successors():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "pssolver/functional/channel_checkpoint.py",
        "pssolver/functional/checkpoint_identity.py",
        "pssolver/functional/periodic_checkpoint.py",
        "tests/test_phase9_p93_replay_checkpoint_bridge.py",
        "tests/test_rc4_functional_checkpoint_identity.py",
    }
    for relative, registration in successors.items():
        assert registration["introduced_by_phase"] == "RC4.2.6 recovery"
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(relative) == registration["current_sha256"]


def test_rc426_recovery_contract_is_device_neutral_but_fail_closed():
    contract = _record()["identity_contract"]
    assert contract["device_and_run_provenance_excluded_before_digest"] is True
    assert contract["filtered_execution_digest_recomputed"] is True
    assert contract["functional_run_provenance_retained_for_audit"] is True
    assert contract["periodic_carrier_identity_matches_bridge_provenance"] is True
    assert contract["production_runtime_identity_used_as_current_functional_gate"] is False
    assert contract["scientific_discretization_derivative_layout_and_backend_gated"] is True
    assert contract["target_tensor_load_before_identity_acceptance"] is False
    compatibility = _record()["backward_compatibility"]
    assert compatibility["current_bridge_format_version_changed"] is False
    assert compatibility["pre_recovery_current_v3_execution_digest_normalized"] is True
    assert compatibility["raw_source_identity_digest_verified_before_normalization"] is True
    assert compatibility["production_checkpoint_runtime_identity_gate_preserved"] is True


def test_rc426_recovery_scope_does_not_claim_h100_or_completion():
    record = _record()
    assert record["cpu_validation"] == {
        "cuda_only_deselected": 7,
        "failed": 0,
        "full_passed": 2926,
        "subtests_passed": 8,
        "targeted_passed": 93,
        "xfailed": 0,
    }
    scope = record["scope"]
    assert scope["functional_checkpoint_compatibility_semantics_changed"] is True
    for key in (
        "PSSolver_Control_modified",
        "functional_API_version_changed",
        "nematics3d_modified",
        "production_checkpoint_acceptance_broadened",
        "production_default_changed",
        "runtime_hot_path_changed",
        "solver_numerics_changed",
    ):
        assert scope[key] is False
    authorization = record["authorization"]
    assert authorization["eligible_for_rc4_2_6_h100_recovery"] is True
    assert authorization["rc4_2_6_complete"] is False
    assert authorization["h100_submission_authorized"] is False
    assert authorization["automatic_merge_authorized"] is False
