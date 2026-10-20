"""Static closure record for R5-10 Channel checkpoint observation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
R5_3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
)
R5_11_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
R5_12_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)
R5_13_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
G1_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"
G1_GLIBC_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_checkpoint_glibc_compatibility.json"
)
RC6_R1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
)
RC6_R2_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
)
RC6_R3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
)
RC6_R4_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"
)
RC6_R5_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json"
)
RC6_R6_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"
)
RC6_FREEZE_RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_source_freeze.json"
FINAL_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_0.json"
V021_N1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _r512_successor(relative: str, historical: str) -> str:
    resolved = historical
    for path in (
        R5_12_RECORD,
        R5_13_RECORD,
        G1_RECORD,
        G1_GLIBC_RECORD,
        RC6_R1_RECORD,
        RC6_R2_RECORD,
        RC6_R3_RECORD,
        RC6_R4_RECORD,
        RC6_R5_RECORD,
        RC6_R6_RECORD,
        RC6_FREEZE_RECORD,
        FINAL_RELEASE_RECORD,
        V021_N1_RECORD,
    ):
        registration = json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        accepted = (
            []
            if registration is None
            else registration.get(
                "accepted_predecessor_sha256",
                [registration["historical_sha256"]],
            )
        )
        if registration is not None and resolved in accepted:
            resolved = registration["current_sha256"]
    return resolved


def test_r510_record_binds_implementation_and_exact_sources():
    record = _record()
    successors = json.loads(R5_11_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    assert record["git_identity"]["branch"] == "fix/v0.2.0rc5"
    assert record["git_identity"]["implementation_commit"] == (
        "5d8941686ba641d1750ddf5cd5bc9ecdfb35666c"
    )
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["git_identity"]["pushed"] is False
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            registration = successors.get(relative)
            if (
                registration is not None
                and registration["historical_sha256"] == expected
            ):
                expected = registration["current_sha256"]
            assert _sha256(relative) == _r512_successor(relative, expected)


def test_r510_preserves_the_r53_new10_adjudication():
    record = _record()
    r53 = json.loads(R5_3_RECORD.read_text(encoding="utf-8"))
    assert record["new_10_disposition"] == {
        "source_work_item": "R5-3",
        "complete_stress_channel_rc1_even": (
            "reject_pre_rc4_1_forward_nyquist"
        ),
        "complete_stress_channel_rc1_odd": (
            "reject_exact_restart_because_B2_incompressibility_dynamics_changed"
        ),
        "state_only_migration_added": False,
        "new_checkpoint_schema_change": False,
    }
    assert r53["released_checkpoint_matrix"]["new_10_adjudication"] == (
        "rc1 complete-stress Channel checkpoints are not trajectory-equivalent "
        "exact restarts; this batch chooses explicit rejection rather than "
        "state-only migration"
    )


def test_r510_checkpoint_observation_contract_is_explicit():
    contract = _record()["new_11_contract"]
    for key in (
        "checkpoint_keeps_synchronized_pressure_guess",
        "live_runtime_restores_pre_observation_pressure_guess",
        "restoration_occurs_on_observation_failure",
        "checkpoint_schedule_is_live_trajectory_invariant",
        "legacy_and_compiled_p7_paths_covered",
        "same_runtime_restart_remains_byte_identical",
        "complete_stress_channel_behavior_unchanged",
    ):
        assert contract[key] is True


def test_r510_successor_registry_is_forward_only_and_exact():
    successors = _record()["historical_successor_sources"]
    r5_11_successors = json.loads(R5_11_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    assert set(successors) == {
        "pssolver/workflows/channel_checkpoint.py",
        "tests/test_phase7_channel_runtime_facade.py",
        "tests/test_rc451_cross_runtime_adjudication.py",
        "tests/test_rc453_checkpoint_transactionality.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc458_h100_closure_record.py",
        "tests/test_rc4_cross_runtime_robustness_plan.py",
        "tests/test_rc4_post_release_findings.py",
        "tests/test_rc5_legacy_functional_checkpoint_record.py",
        "tests/test_rc5_r52_finite_workflow_record.py",
        "tests/test_rc5_r53_channel_checkpoint_record.py",
        "tests/test_rc5_r55_tf32_state_restore.py",
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_r57_device_normalization_record.py",
        "tests/test_rc5_r58_checkpoint_integrity_record.py",
        "tests/test_rc5_r59_pressure_postcondition_record.py",
        "tests/test_rc5_w1_documentation.py",
        "tests/test_release_v0_2.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        expected = registration["current_sha256"]
        successor = r5_11_successors.get(relative)
        if (
            successor is not None
            and successor["historical_sha256"] == expected
        ):
            expected = successor["current_sha256"]
        assert _sha256(relative) == _r512_successor(relative, expected)
        assert registration["introduced_by_phase"] == "R5-10"
        assert registration["reason"]


def test_r510_scope_and_completion_are_fail_closed():
    record = _record()
    scope = record["scope"]
    assert scope["checkpoint_capture_semantics_changed"] is True
    for key in (
        "scientific_equations_changed",
        "pressure_solver_algorithm_changed",
        "checkpoint_schema_changed",
        "production_default_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "H100_required",
        "H100_executed",
        "push_executed",
    ):
        assert scope[key] is False

    completed = record["status"] == "implementation_complete_cpu_verified"
    if completed:
        assert record["classification"] == (
            "PASS_R5_10_CHANNEL_CHECKPOINT_OBSERVATION_CPU_CLOSURE"
        )
        assert record["git_identity"]["qualified_source_commit"] == (
            record["git_identity"]["implementation_commit"]
        )
        assert record["installed_wheel_evidence"] is not None
        assert record["authorization"]["r5_10_cpu_closure_complete"] is True
        assert record["authorization"]["eligible_for_next_work_item"] is True
    else:
        assert record["status"] == "implementation_complete_validation_pending"
        assert record["classification"].startswith("PENDING_")
        assert record["git_identity"]["qualified_source_commit"] is None
        assert record["installed_wheel_evidence"] is None
        assert record["authorization"]["r5_10_cpu_closure_complete"] is False
