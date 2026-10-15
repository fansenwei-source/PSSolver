"""RC4.5.8 final cumulative closure-record tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc4_rc458_h100_closure.json"
RELEASE = ROOT / "notes" / "pssolver_v0_2_0_rc4.json"
POST_RELEASE = ROOT / "notes" / "PSSolver_v0_2_0rc4_post_release_findings.json"
RC5_W1 = ROOT / "notes" / "PSSolver_v0_2_0rc5_w1_documentation.json"
RC5_R51 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
RC5_R52 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R53 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
RC5_R55 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
RC5_R56 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC5_R57 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RC5_R58 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
RC5_R59 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
RC5_R510 = ROOT / "notes" / (
    "PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_rc458_closure_binds_exact_provider_consumer_and_job_identities():
    record = _record()
    assert record["classification"] == (
        "PASS_RC4_5_8_SINGLE_H100_TF32_CUMULATIVE_CLOSURE_"
        "WITH_COUNTER_AND_RESERVED_MEMORY_REDUCER_RECOVERY"
    )
    assert record["qualification_complete"] is True
    assert record["rc4_5_8_complete"] is True
    assert record["identities"] == {
        "provider_baseline_commit": "51f47e9f06bd3e019d785305bff47c80f5519b10",
        "provider_runtime_candidate_commit": (
            "a195bd05992bc6553c72c400587d2b07afe722d0"
        ),
        "qualification_commit": "cffae1f0eb07775dbb76b1548480db3049a04f37",
        "consumer_qualified_source_commit": (
            "b8539e0ec4c137a6c13746f6832e26e74be24515"
        ),
        "provider_branch": "fix/v0.2.0rc4-audit",
        "provider_remote_tip_at_registration": (
            "cffae1f0eb07775dbb76b1548480db3049a04f37"
        ),
    }
    assert record["h100_execution"]["job_id"] == 10879606
    assert record["h100_execution"]["scientific_profile_matrix_completed"] is True
    assert record["h100_execution"]["scientific_failure"] is False


def test_rc458_closure_preserves_failed_layers_and_binds_final_complete_archive():
    layers = _record()["evidence_layers"]
    assert layers["h100_recovery_v2"]["classification_preserved"].startswith("FAIL_")
    assert layers["h100_recovery_v2"]["checksums_verified"] == 3371
    assert layers["counter_recovery_v3"]["classification_preserved"].startswith(
        "FAIL_"
    )
    assert layers["counter_recovery_v3"]["checksums_verified"] == 16
    final = layers["memory_recovery_v4"]
    assert final["checksums_verified"] == 10
    assert final["checksums_sha256"] == (
        "d8e20c6422e5919aae6eee254eb971b60329622d4d473454fb69946b63231249"
    )
    assert final["complete_present"] is True
    assert final["new_h100_submission_count"] == 0
    assert final["new_scientific_timestep_count"] == 0


def test_rc458_closure_records_every_cumulative_gate_without_relaxing_limits():
    record = _record()
    gates = record["cumulative_gates"]
    assert gates["cuda_only"] is True
    assert gates["tf32_scope_and_caller_state_restoration"] is True
    assert gates["consumer_installed_wheel_h100"] is True
    assert gates["profile_matrix"] == "96/96"
    assert gates["runtime_groups"] == "16/16"
    assert gates["transform_counter_layer"] == "96/96"
    assert gates["runtime_fallback"] is False
    assert gates["compile_fallback"] is False
    assert gates["graph_breaks"] == 0
    assert gates["maximum_reserved_memory_group_peak_ratio"] == 1.0
    counters = record["contract_recoveries"]["transform_counter"]
    assert counters["raw_channel_forward_calls"] == 663
    assert counters["scheduled_refresh_forward_calls"] == 3
    assert counters["base_forward_per_step"] == 11.0
    assert counters["threshold_changed"] is False
    memory = record["contract_recoveries"]["reserved_memory"]
    assert memory["ratio_limit"] == 1.03
    assert memory["threshold_changed"] is False
    assert memory["peak_allocated_gate_changed"] is False
    assert memory["raw_profiles_modified"] is False


def test_rc458_closure_only_authorizes_release_preparation():
    assert _record()["closure"] == {
        "b11_closed_verified": True,
        "rc4_5_complete": True,
        "rc4_complete": True,
        "eligible_for_rc4_release_preparation": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
        "functional_api_version_changed": False,
        "checkpoint_schema_changed": False,
        "PSSolver_Control_modified_by_registration": False,
        "nematics3d_modified": False,
        "tag_created": False,
        "package_published": False,
    }


def test_rc458_closure_registers_exact_historical_source_successors():
    successors = _record()["historical_successor_sources"]
    release_successors = json.loads(RELEASE.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    post_release_successors = json.loads(POST_RELEASE.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_w1_successors = json.loads(RC5_W1.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r51_successors = json.loads(RC5_R51.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r52_successors = json.loads(RC5_R52.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r53_successors = json.loads(RC5_R53.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r55_successors = json.loads(RC5_R55.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r56_successors = json.loads(RC5_R56.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r57_successors = json.loads(RC5_R57.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r58_successors = json.loads(RC5_R58.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r59_successors = json.loads(RC5_R59.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r510_successors = json.loads(RC5_R510.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    assert set(successors) == {
        "tests/test_rc451_cross_runtime_adjudication.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc3_known_issues_matrix.py",
    }
    for relative, registration in successors.items():
        resolved = registration["current_sha256"]
        release = release_successors.get(relative)
        if release is not None and release["historical_sha256"] == resolved:
            resolved = release["current_sha256"]
        post_release = post_release_successors.get(relative)
        if (
            post_release is not None
            and post_release["historical_sha256"] == resolved
        ):
            resolved = post_release["current_sha256"]
        rc5_w1 = rc5_w1_successors.get(relative)
        if rc5_w1 is not None and rc5_w1["historical_sha256"] == resolved:
            resolved = rc5_w1["current_sha256"]
        rc5_r51 = rc5_r51_successors.get(relative)
        if rc5_r51 is not None and rc5_r51["historical_sha256"] == resolved:
            resolved = rc5_r51["current_sha256"]
        rc5_r52 = rc5_r52_successors.get(relative)
        if rc5_r52 is not None and rc5_r52["historical_sha256"] == resolved:
            resolved = rc5_r52["current_sha256"]
        rc5_r53 = rc5_r53_successors.get(relative)
        if rc5_r53 is not None and rc5_r53["historical_sha256"] == resolved:
            resolved = rc5_r53["current_sha256"]
        rc5_r55 = rc5_r55_successors.get(relative)
        if rc5_r55 is not None and rc5_r55["historical_sha256"] == resolved:
            resolved = rc5_r55["current_sha256"]
        rc5_r56 = rc5_r56_successors.get(relative)
        if rc5_r56 is not None and rc5_r56["historical_sha256"] == resolved:
            resolved = rc5_r56["current_sha256"]
        rc5_r57 = rc5_r57_successors.get(relative)
        if rc5_r57 is not None and rc5_r57["historical_sha256"] == resolved:
            resolved = rc5_r57["current_sha256"]
        rc5_r58 = rc5_r58_successors.get(relative)
        if rc5_r58 is not None and rc5_r58["historical_sha256"] == resolved:
            resolved = rc5_r58["current_sha256"]
        rc5_r59 = rc5_r59_successors.get(relative)
        if rc5_r59 is not None and rc5_r59["historical_sha256"] == resolved:
            resolved = rc5_r59["current_sha256"]
        rc5_r510 = rc5_r510_successors.get(relative)
        if rc5_r510 is not None and rc5_r510["historical_sha256"] == resolved:
            resolved = rc5_r510["current_sha256"]
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolved
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert registration["introduced_by_phase"] == "RC4.5.8-finalization"
