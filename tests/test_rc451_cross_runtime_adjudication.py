"""Contracts for the analysis-only RC4.5.1 residual adjudication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc451_cross_runtime_adjudication.json"
CONTRACT = ROOT / "notes/PSSolver_v0_2_0rc4_rc452_public_robustness_contract.json"
MATRIX = ROOT / "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json"
RC453 = ROOT / "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
RC454 = ROOT / "notes/PSSolver_v0_2_0rc4_rc454_boundary_semantics.json"
RC455 = ROOT / "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
RC456 = ROOT / "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
POST_RELEASE = ROOT / "notes/PSSolver_v0_2_0rc4_post_release_findings.json"
RC5_W1 = ROOT / "notes/PSSolver_v0_2_0rc5_w1_documentation.json"
RC5_R51 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
RC5_R52 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R53 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
RC5_R55 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
RC5_R56 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC5_R57 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RC5_R58 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
RC5_R510 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
RC5_R511 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
RC5_R513 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_successor_sha256(relative: str, historical: str) -> str:
    resolved = historical
    for record_path in (
        RC453,
        RC454,
        RC455,
        RC456,
        POST_RELEASE,
        RC5_W1,
        RC5_R51,
        RC5_R52,
        RC5_R53,
        RC5_R55,
        RC5_R56,
        RC5_R57,
        RC5_R58,
        RC5_R510,
        RC5_R511,
        RC5_R513,
    ):
        registration = _read(record_path).get(
            "historical_successor_sources", {}
        ).get(relative)
        if registration is None:
            continue
        if registration["historical_sha256"] == resolved:
            resolved = registration["current_sha256"]
    return resolved


def test_rc451_is_analysis_only_and_runs_all_ten_frozen_probes():
    record = _read(RECORD)
    assert record["schema_version"] == 1
    assert record["stage"] == "RC4.5.1"
    assert record["classification"] == (
        "RC4_5_1_RESIDUAL_ADJUDICATION_COMPLETE"
    )
    assert record["analysis_only"] is True
    assert record["h100_job_submitted"] is False
    assert record["runtime_source_modified"] is False
    assert record["production_default_changed"] is False
    assert record["probe_count"] == 10
    assert set(record["probes"]) == {
        "C01",
        "C02",
        "C03",
        "C04",
        "B01",
        "B02",
        "V01",
        "V02",
        "V03",
        "T01",
    }


def test_rc451_outcomes_cover_exactly_the_nine_residuals():
    record = _read(RECORD)
    outcomes = record["issue_outcomes"]
    assert set(outcomes) == {
        "B7",
        "B9",
        "B11",
        "B13",
        "B15",
        "B17",
        "B21",
        "B24",
        "B27",
    }
    assert outcomes["B21"]["outcome"] == "PUBLIC_BOUND_REQUIRED"
    assert all(
        value["outcome"] == "REPRODUCED_REPAIR_REQUIRED"
        for issue, value in outcomes.items()
        if issue != "B21"
    )
    assert outcomes["B11"]["h100_followup_required"] is True
    assert record["summary"]["next_stage"] == (
        "RC4.5.2_checkpoint_transactionality"
    )


def test_rc451_checkpoint_probes_preserve_positive_gates_and_find_residuals():
    probes = _read(RECORD)["probes"]
    c01 = probes["C01"]
    assert all(
        value["resealed_unknown_top_level_key_accepted"] is True
        for value in c01["families"].values()
    )
    assert c01["missing_current_provenance"]["raised"] is True

    c02 = probes["C02"]
    assert all(
        value["type"] == "ValueError"
        for value in c02["production_filename_path_checks"].values()
    )
    assert c02["target_mutation_before_payload_open"] is False

    c03 = probes["C03"]
    assert c03["functional_modes"] == {
        "channel": "0o700",
        "periodic": "0o755",
    }
    assert c03["cleanup"]["target_exists"] is False
    assert c03["cleanup"]["residual_entries"] == []
    assert all(
        value["uses_atomic_replace"] is True
        for value in c03["writer_source_inventory"].values()
    )

    c04 = probes["C04"]
    observed = c04["observed"]
    assert observed["existing_target"]["type"] == (
        "FunctionalCheckpointExistsError"
    )
    assert observed["missing_target"]["type"] == (
        "FunctionalCheckpointNotFoundError"
    )
    assert observed["future_format"]["type"] == (
        "FunctionalCheckpointCompatibilityError"
    )
    assert observed["payload_tamper"]["type"] == (
        "FunctionalCheckpointIntegrityError"
    )
    assert observed["parent_is_file"]["type"] == "FileExistsError"
    assert c04["parent_filesystem_error_translated"] is False


def test_rc451_boundary_probes_separate_physics_from_public_support():
    probes = _read(RECORD)["probes"]
    anchoring = probes["B01"]
    assert anchoring["outward"]["raised"] is False
    assert anchoring["opposite"]["type"] == "ValueError"
    assert anchoring["q_byte_identical_under_sign_flip"] is True
    assert anchoring["diagnostic_reports_expected_and_observed_vectors"] is False

    robin = probes["B02"]
    assert robin["largest_passing_scale"] == 1.0e14
    assert robin["smallest_failing_scale"] == 1.0e15
    assert robin["recommended_public_bound"]["upper_inclusive"] == 1.0e14


def test_rc451_validation_and_diagnostics_residuals_are_black_box_visible():
    probes = _read(RECORD)["probes"]
    devices = probes["V01"]
    assert devices["invalid_device_accepted"] is True
    assert devices["indexed_cpu_token_accepted"] is True
    assert all(row["compile_accepted"] for row in devices["plane_device_matrix"])

    capability = probes["V02"]
    assert capability["invalid_device_uses_public_compilation_error"] is True
    assert capability["plane_invalid_options_use_public_compilation_error"] is False
    assert capability["plane_invalid_pointwise_exception"]["type"] == "ValueError"
    assert capability["plane_invalid_reuse_exception"]["type"] == "TypeError"

    diagnostics = probes["V03"]
    assert diagnostics["all_field_sets_equal"] is False
    assert diagnostics["plane_pressure_iterations_dtype"] == "float64"
    assert diagnostics["complete_channel_pressure_iterations_dtype"] == "int64"


def test_rc451_tf32_probe_restores_the_analysis_process_not_the_callee():
    tf32 = _read(RECORD)["probes"]["T01"]
    assert tf32["callee_restored_caller_state"] is False
    assert tf32["analysis_harness_restored_initial_state"] is True
    assert tf32["invalid_request"]["type"] == "ValueError"
    assert tf32["cuda_nested_runtime_matrix"] == "deferred_to_h100"


def test_rc451_record_is_bound_to_every_reviewed_source():
    provenance = _read(RECORD)["provenance"]
    assert len(provenance["git_head"]) == 40
    assert provenance["torch"]
    assert provenance["numpy"]
    for relative, expected in provenance["source_sha256"].items():
        actual = _sha256(ROOT / relative)
        assert _resolve_successor_sha256(relative, expected) == actual


def test_known_issue_matrix_registers_rc451_and_later_scoped_closures():
    matrix = _read(MATRIX)
    progress = matrix["progress"]
    assert progress["rc4_5_1_analysis_authorized"] is True
    assert progress["rc4_5_1_analysis_complete"] is True
    assert progress["rc4_5_1_classification"] == (
        "RC4_5_1_RESIDUAL_ADJUDICATION_COMPLETE"
    )
    assert progress["rc4_5_1_analysis_record_sha256"] == _sha256(RECORD)
    assert progress["eligible_for_rc4_5_2_contract"] is True
    assert progress["rc4_5_2_contract_complete"] is True
    assert progress["rc4_5_2_contract_record_sha256"] == _sha256(CONTRACT)
    assert progress["eligible_for_rc4_5_3_implementation"] is True
    assert progress["rc4_5_4_complete"] is True
    assert progress["eligible_for_rc4_5_5_implementation"] is True
    assert progress["rc4_5_implementation_authorized"] is True
    issues = {item["id"]: item for item in matrix["items"]}
    assert all(
        issues[issue]["state"] == "partial_residual"
        for issue in ("B7", "B24")
    )
    assert issues["B27"]["state"] == "closed_verified"
    assert all(
        issues[issue]["state"] == "closed_verified"
        for issue in ("B9", "B21")
    )
    assert issues["B11"]["state"] == "partial_residual"
    assert progress["rc4_5_8_h100_complete"] is True
    assert progress["rc4_5_complete"] is True
    assert progress["rc4_complete"] is True
    assert progress["eligible_for_rc4_release_preparation"] is True
    assert all(
        issues[issue]["state"] == "partial_residual"
        for issue in ("B13", "B17")
    )
    assert issues["B15"]["state"] == "closed_verified"
    assert matrix["post_release"]["classification"] == "FAIL"
