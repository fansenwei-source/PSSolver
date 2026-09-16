"""CPU contracts for the Stage Q.6.1 native-handoff design review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pssolver.experimental._shadow_support import file_sha256
from pssolver.experimental.stage_q61_design import (
    STAGE_Q61_ARCHITECTURE_DECISION,
    STAGE_Q61_CLASSIFICATION,
    STAGE_Q61_REQUIRED_SOURCES,
    analysis_main,
    analyze_stage_q61_native_handoff_design,
)


def _write_q6(tmp_path: Path) -> tuple[Path, str]:
    specifications = {
        "algebraic.nematic_stress.dependencies": (
            "inverse_projected",
            "_StressSolver",
            2,
            20,
            1_318_912_000,
            1.462,
        ),
        "explicit_rhs.dependencies": (
            "inverse_projected",
            "LegacyExplicitRHSAdapter",
            2,
            9,
            593_510_400,
            0.694,
        ),
        "explicit_rhs.outputs": (
            "forward_projected",
            "projected semi-implicit integrator",
            1,
            5,
            327_680_000,
            0.366,
        ),
    }
    rows = []
    for source, (
        direction,
        consumer,
        batches,
        components,
        materialized_bytes,
        milliseconds,
    ) in specifications.items():
        rows.append(
            {
                "source": source,
                "measurement": {
                    "copy_cat_batches_per_step": batches,
                    "copy_cat_components_per_step": components,
                    "copy_cat_materialized_output_bytes_per_step": (
                        materialized_bytes
                    ),
                    "mean_milliseconds_per_step": milliseconds,
                },
                "handoff": {"direction": direction},
                "tensor_copy": {"present": True},
                "view_or_alias": {
                    "zero_copy_currently_exercised": False
                },
                "transform_input_layout": "boundary-signature packed leading axis",
                "consumer_required_layout": consumer,
            }
        )
    report = {
        "qualification_stage": "Q.6",
        "classification": "DATA_MOVEMENT_MAP_COMPLETE",
        "architecture_decision": (
            "boundary_signature_native_handoff_design_review"
        ),
        "data_movement_map": rows,
        "aggregate_measured_evidence": {
            "source_count": 3,
            "mean_milliseconds_per_step": 2.522,
            "mean_fraction_of_timestep": 0.03445,
            "copy_cat_materialized_output_bytes_per_step": 2_240_102_400,
            "no_speedup_prediction_is_made": True,
        },
        "cross_source_findings": {
            "copy_cat_is_present_at_every_remaining_source": True,
            "projected_transforms_remain_required_numerical_operations": True,
            "boundary_signatures_must_remain_separate": True,
        },
        "stage_q61_design_boundary": {
            "design_target": "boundary_signature_native_handoff_contract",
            "may_execute_solver": False,
            "may_implement_candidate": False,
            "may_change_production_default": False,
        },
        "eligible_for_stage_q61_design": True,
        "eligible_for_stage_q61_candidate_implementation": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "changes_equations": False,
        "changes_runtime_implementation": False,
    }
    path = tmp_path / "q6.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, file_sha256(path)


def _analyze(tmp_path: Path) -> dict[str, object]:
    path, digest = _write_q6(tmp_path)
    return analyze_stage_q61_native_handoff_design(
        stage_q6_report=path,
        expected_stage_q6_sha256=digest,
    )


def test_stage_q61_freezes_one_shared_design_and_bounded_candidate(tmp_path):
    report = _analyze(tmp_path)

    assert report["qualification_stage"] == "Q.6.1"
    assert report["classification"] == STAGE_Q61_CLASSIFICATION
    assert report["architecture_decision"] == STAGE_Q61_ARCHITECTURE_DECISION
    assert [row["source"] for row in report["source_adaptation_plans"]] == list(
        STAGE_Q61_REQUIRED_SOURCES
    )
    assert report["design_resolution"][
        "all_three_sources_must_be_implemented_together"
    ] is True
    assert report["design_resolution"][
        "single_source_candidate_remains_closed"
    ] is True
    assert report["eligible_for_stage_q62_candidate_implementation"] is True
    assert report["eligible_for_production_promotion"] is False
    assert report["changes_runtime_implementation"] is False


def test_stage_q61_contract_forbids_copy_workspace_and_silent_fallback(tmp_path):
    report = _analyze(tmp_path)
    rules = report["native_segment_contract"]["scheduler_rules"]

    assert rules["may_concatenate_native_segments"] is False
    assert rules["may_copy_into_workspace"] is False
    assert rules["may_pack_incompatible_boundary_signatures"] is False
    assert rules["silent_fallback_to_copy_cat"] is False
    candidate = report["stage_q62_candidate"]
    assert candidate["fallback_policy"] == "fail_closed_no_copy_cat_fallback"
    assert candidate["qualification_gates"][
        "all_three_sources_copy_cat_batches_per_step"
    ] == 0


def test_stage_q61_exposes_transform_fragmentation_as_primary_risk(tmp_path):
    report = _analyze(tmp_path)

    assert report["design_resolution"][
        "transform_batch_fragmentation_is_primary_risk"
    ] is True
    assert report["stage_q62_candidate"]["qualification_gates"][
        "performance_must_be_measured_not_predicted"
    ] is True
    assert report["measured_evidence_context"][
        "not_a_predicted_speedup"
    ] is True


def test_stage_q61_rejects_changed_q6_identity(tmp_path):
    path, digest = _write_q6(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 differs"):
        analyze_stage_q61_native_handoff_design(
            stage_q6_report=path,
            expected_stage_q6_sha256=digest,
        )


def test_stage_q61_rejects_incomplete_source_map(tmp_path):
    path, _ = _write_q6(tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["data_movement_map"].pop()
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="complete three-source map"):
        analyze_stage_q61_native_handoff_design(
            stage_q6_report=path,
            expected_stage_q6_sha256=file_sha256(path),
        )


def test_stage_q61_rejects_existing_zero_copy_or_missing_copy(tmp_path):
    path, _ = _write_q6(tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["data_movement_map"][0]["tensor_copy"]["present"] = False
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="is not designable"):
        analyze_stage_q61_native_handoff_design(
            stage_q6_report=path,
            expected_stage_q6_sha256=file_sha256(path),
        )


def test_stage_q61_cli_writes_once(tmp_path):
    path, digest = _write_q6(tmp_path)
    output = tmp_path / "q61.json"
    argv = [
        "--stage-q6-report",
        str(path),
        "--expected-stage-q6-sha256",
        digest,
        "--output",
        str(output),
    ]

    assert analysis_main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["classification"] == (
        STAGE_Q61_CLASSIFICATION
    )
    with pytest.raises(FileExistsError):
        analysis_main(argv)
