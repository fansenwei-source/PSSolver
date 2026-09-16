"""CPU contracts for post-Q.4 Stage Q.5 target selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pssolver.experimental._shadow_support import file_sha256
from pssolver.experimental.stage_q5_diagnostics import (
    STAGE_Q5_PRIMARY_TARGET,
    analysis_main,
    analyze_stage_q5_post_q4_retargeting,
)


def _write(path: Path, report: dict[str, object]) -> tuple[Path, str]:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, file_sha256(path)


def _reports(tmp_path: Path) -> dict[str, tuple[Path, str]]:
    q2_path = tmp_path / "q2.json"
    q2 = {
        "qualification_stage": "Q.2",
        "classification": "TARGET_REVIEW_COMPLETE",
        "authoritative_throughput": {
            "production_mean_milliseconds_per_step": 51.0,
            "candidate_mean_milliseconds_per_step": 55.2,
            "residual_gap_milliseconds_per_step": 4.2,
        },
        "nested_nonadditive_evidence": {
            "regions": [
                {
                    "region": "transform_inverse",
                    "delta_milliseconds_per_step": 0.94,
                }
            ]
        },
        "operator_kernel_evidence": {
            "largest_positive_operator_deltas": [
                {
                    "name": "aten::cat",
                    "device_microseconds_delta_per_step": 3700.0,
                },
                {
                    "name": "aten::copy_",
                    "device_microseconds_delta_per_step": 1900.0,
                },
            ],
            "largest_positive_kernel_deltas": [
                {
                    "name": "CatArrayBatchedCopy_contig",
                    "device_microseconds_delta_per_step": 3800.0,
                }
            ],
        },
        "target_review": {
            "primary_target": "algebraic_runtime_orchestration",
            "explicit_rhs_line_closed": True,
            "transform_forward_line_closed": True,
            "projected_materialization_line_remains_closed": True,
        },
        "eligible_for_stage_q2_candidate_design": True,
        "eligible_for_stage_q2_candidate_implementation": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
    }
    q2_path, q2_sha = _write(q2_path, q2)

    q3_path, q3_sha = _write(
        tmp_path / "q3.json",
        {
            "qualification_stage": "Q.3",
            "classification": "ALGEBRAIC_TARGET_REFINEMENT_COMPLETE",
            "candidate_design": {
                "selected_candidate": "preplanned_algebraic_batch_workspace"
            },
            "eligible_for_stage_q4_candidate_implementation": True,
            "eligible_for_production_promotion": False,
            "production_default_changed": False,
            "inputs": [
                {
                    "kind": "stage_q2_report",
                    "path": str(q2_path),
                    "sha256": q2_sha,
                }
            ],
        },
    )

    assembly = {
        "workspace_allocated_bytes": 3000,
        "workspace_active_count": 0,
        "retained_tensor_references": 0,
    }
    q4_path, q4_sha = _write(
        tmp_path / "q4.json",
        {
            "qualification_stage": "Q.4",
            "classification": "C_rejected",
            "architecture_decision": "preplanned_algebraic_batch_workspace",
            "eligible_for_stage_q5_decision": False,
            "eligible_for_production_promotion": False,
            "production_default_changed": False,
            "stage_q3_evidence": {"sha256": q3_sha},
            "performance": {
                "baseline_mean_timestep_seconds": 0.0565,
                "candidate_mean_timestep_seconds": 0.0566,
                "mean_timestep_ratio_candidate_over_baseline": 1.001,
                "baseline_peak_allocated_bytes": 9000,
                "candidate_peak_allocated_bytes": 12000,
            },
            "workspace": {"candidate": [assembly, assembly, assembly]},
            "gates": {
                "numerical_equivalence": True,
                "workspace_lifecycle": True,
                "r320_performance_improvement": False,
                "r320_memory_non_regression": False,
                "candidate_safety_non_regression": False,
            },
        },
    )

    common_gates = {
        "numerical_equivalence": True,
        "zero_copy_batch_assembly_exercised": True,
        "r320_performance_improvement": False,
        "candidate_safety_non_regression": True,
    }
    o431_path, o431_sha = _write(
        tmp_path / "o431.json",
        {
            "qualification_stage": "O.4.3.1",
            "classification": "B_neutral",
            "architecture_decision": (
                "opportunistic_natural_storage_views_without_republication"
            ),
            "eligible_for_stage_o44_decision": False,
            "eligible_for_production_promotion": False,
            "production_default_changed": False,
            "gates": common_gates,
            "scales": {
                "r320": {
                    "mean_timestep_ratio_candidate_over_baseline": 1.0001
                }
            },
        },
    )
    o433_path, o433_sha = _write(
        tmp_path / "o433.json",
        {
            "qualification_stage": "O.4.3.3",
            "classification": "B_neutral",
            "architecture_decision": (
                "producer_owned_boundary_packed_h_and_stress"
            ),
            "eligible_for_stage_o44_decision": False,
            "eligible_for_production_promotion": False,
            "production_default_changed": False,
            "prior_storage_evidence": {
                "stage_o431": {"sha256": o431_sha}
            },
            "gates": common_gates,
            "scales": {
                "r320": {
                    "mean_timestep_ratio_candidate_over_baseline": 0.983
                }
            },
        },
    )
    copy_rows = [
        {
            "source": "algebraic.nematic_stress.dependencies",
            "mean_milliseconds_per_step": 1.46,
            "mean_fraction_of_timestep": 0.02,
            "copy_cat_materialized_output_bytes_per_step": 1300,
        },
        {
            "source": "explicit_rhs.dependencies",
            "mean_milliseconds_per_step": 0.69,
            "mean_fraction_of_timestep": 0.0095,
            "copy_cat_materialized_output_bytes_per_step": 590,
        },
    ]
    o434_path, o434_sha = _write(
        tmp_path / "o434.json",
        {
            "qualification_stage": "O.4.3.4",
            "classification": "DIAGNOSTIC_COMPLETE",
            "architecture_decision": "remaining_materialization_attribution",
            "stage_o433_evidence": {"sha256": o433_sha},
            "remaining_copy_sources": copy_rows,
            "minimum_screening_fraction_of_timestep": 0.03,
            "dominant_source_meets_screening_signal": False,
            "eligible_for_target_selection_review": True,
            "eligible_for_new_optimization_candidate": False,
            "eligible_for_stage_o44_decision": False,
            "eligible_for_production_promotion": False,
            "production_default_changed": False,
        },
    )
    return {
        "q2": (q2_path, q2_sha),
        "q3": (q3_path, q3_sha),
        "q4": (q4_path, q4_sha),
        "o431": (o431_path, o431_sha),
        "o433": (o433_path, o433_sha),
        "o434": (o434_path, o434_sha),
    }


def _analyze(inputs: dict[str, tuple[Path, str]]) -> dict[str, object]:
    kwargs = {}
    for stage, (path, digest) in inputs.items():
        kwargs[f"stage_{stage}_report"] = path
        kwargs[f"expected_stage_{stage}_sha256"] = digest
    return analyze_stage_q5_post_q4_retargeting(**kwargs)


def test_stage_q5_closes_workspace_and_selects_diagnostic_only(tmp_path):
    report = _analyze(_reports(tmp_path))

    assert report["qualification_stage"] == "Q.5"
    assert report["classification"] == "POST_Q4_RETARGETING_COMPLETE"
    assert report["primary_target"] == STAGE_Q5_PRIMARY_TARGET
    assert report["q4_workspace_outcome"][
        "peak_allocated_increase_bytes"
    ] == 3000
    assert report["q4_workspace_outcome"][
        "workspace_bytes_over_peak_allocated_increase"
    ] == pytest.approx(1.0)
    assert report["ranked_diagnostic_targets"][0]["rank"] == 1
    assert report["eligible_for_stage_q6_diagnostic_design"] is True
    assert report["eligible_for_stage_q6_candidate_implementation"] is False
    assert report["stage_q6_scope"]["may_execute_solver"] is False
    assert report["eligible_for_production_promotion"] is False
    assert report["production_default_changed"] is False


def test_stage_q5_rejects_changed_input_identity(tmp_path):
    inputs = _reports(tmp_path)
    path, _ = inputs["q4"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 differs"):
        _analyze(inputs)


def test_stage_q5_rejects_q4_without_memory_regression(tmp_path):
    inputs = _reports(tmp_path)
    path, _ = inputs["q4"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["gates"]["r320_memory_non_regression"] = True
    inputs["q4"] = _write(path, report)

    with pytest.raises(ValueError, match="Q.4 evidence contract differs"):
        _analyze(inputs)


def test_stage_q5_cli_writes_once(tmp_path):
    inputs = _reports(tmp_path)
    output = tmp_path / "stage_q5.json"
    argv = []
    for stage, (path, digest) in inputs.items():
        argv.extend(
            [
                f"--stage-{stage}-report",
                str(path),
                f"--expected-stage-{stage}-sha256",
                digest,
            ]
        )
    argv.extend(["--output", str(output)])

    assert analysis_main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8"))[
        "classification"
    ] == "POST_Q4_RETARGETING_COMPLETE"
    with pytest.raises(FileExistsError):
        analysis_main(argv)
