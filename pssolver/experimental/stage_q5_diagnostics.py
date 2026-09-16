"""Post-Q.4 target selection from frozen Plane performance evidence.

Stage Q.5 is deliberately analysis-only.  It closes mechanisms that have
already been measured, separates allocation reuse from data movement, and
selects one diagnostic target for the next architecture stage.  It never
constructs a solver or authorizes a production change.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import math
from pathlib import Path

from ._shadow_support import file_sha256
from .h100_shadow_qualification import _write_new_json
from .stage_q2_diagnostics import _read_unchanged_json


STAGE_Q5_CLASSIFICATION = "POST_Q4_RETARGETING_COMPLETE"
STAGE_Q5_PRIMARY_TARGET = (
    "end_to_end_algebraic_data_movement_and_consumer_layout"
)


def _finite(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{description} must be finite")
    return float(value)


def _nonnegative(value: object, description: str) -> float:
    result = _finite(value, description)
    if result < 0.0:
        raise ValueError(f"{description} must be nonnegative")
    return result


def _load_frozen_report(
    path: str | Path,
    *,
    expected_sha256: str,
    description: str,
) -> tuple[Path, dict[str, object]]:
    resolved, digest, report = _read_unchanged_json(path, description)
    if digest != expected_sha256:
        raise ValueError(f"{description} SHA-256 differs")
    return resolved, report


def _require(condition: object, message: str) -> None:
    if condition is not True:
        raise ValueError(message)


def _stage_q2_contract(report: Mapping[str, object]) -> None:
    try:
        valid = bool(
            report["qualification_stage"] == "Q.2"
            and report["classification"] == "TARGET_REVIEW_COMPLETE"
            and report["target_review"]["primary_target"]
            == "algebraic_runtime_orchestration"
            and report["target_review"]["explicit_rhs_line_closed"] is True
            and report["target_review"]["transform_forward_line_closed"]
            is True
            and report["target_review"][
                "projected_materialization_line_remains_closed"
            ]
            is True
            and report["eligible_for_stage_q2_candidate_design"] is True
            and report["eligible_for_stage_q2_candidate_implementation"]
            is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
        )
    except (KeyError, TypeError):
        valid = False
    _require(valid, "Stage Q.5 Q.2 evidence contract differs")


def _stage_q3_contract(
    report: Mapping[str, object], expected_q2_sha256: str
) -> None:
    try:
        q2_inputs = [
            value
            for value in report["inputs"]
            if value.get("kind") == "stage_q2_report"
        ]
        valid = bool(
            report["qualification_stage"] == "Q.3"
            and report["classification"]
            == "ALGEBRAIC_TARGET_REFINEMENT_COMPLETE"
            and report["candidate_design"]["selected_candidate"]
            == "preplanned_algebraic_batch_workspace"
            and report["eligible_for_stage_q4_candidate_implementation"]
            is True
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
            and len(q2_inputs) == 1
            and q2_inputs[0]["sha256"] == expected_q2_sha256
        )
    except (KeyError, TypeError):
        valid = False
    _require(valid, "Stage Q.5 Q.3 evidence contract differs")


def _stage_q4_contract(
    report: Mapping[str, object], expected_q3_sha256: str
) -> None:
    try:
        valid = bool(
            report["qualification_stage"] == "Q.4"
            and report["classification"] == "C_rejected"
            and report["architecture_decision"]
            == "preplanned_algebraic_batch_workspace"
            and report["stage_q3_evidence"]["sha256"]
            == expected_q3_sha256
            and report["eligible_for_stage_q5_decision"] is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
            and report["gates"]["numerical_equivalence"] is True
            and report["gates"]["workspace_lifecycle"] is True
            and report["gates"]["r320_performance_improvement"] is False
            and report["gates"]["r320_memory_non_regression"] is False
            and report["gates"]["candidate_safety_non_regression"] is False
        )
    except (KeyError, TypeError):
        valid = False
    _require(valid, "Stage Q.5 Q.4 evidence contract differs")


def _stage_o431_contract(report: Mapping[str, object]) -> None:
    try:
        valid = bool(
            report["qualification_stage"] == "O.4.3.1"
            and report["classification"] == "B_neutral"
            and report["architecture_decision"]
            == "opportunistic_natural_storage_views_without_republication"
            and report["eligible_for_stage_o44_decision"] is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
            and report["gates"]["numerical_equivalence"] is True
            and report["gates"]["zero_copy_batch_assembly_exercised"]
            is True
            and report["gates"]["r320_performance_improvement"] is False
            and report["gates"]["candidate_safety_non_regression"] is True
        )
    except (KeyError, TypeError):
        valid = False
    _require(valid, "Stage Q.5 O.4.3.1 evidence contract differs")


def _stage_o433_contract(
    report: Mapping[str, object], expected_o431_sha256: str
) -> None:
    try:
        valid = bool(
            report["qualification_stage"] == "O.4.3.3"
            and report["classification"] == "B_neutral"
            and report["architecture_decision"]
            == "producer_owned_boundary_packed_h_and_stress"
            and report["prior_storage_evidence"]["stage_o431"]["sha256"]
            == expected_o431_sha256
            and report["eligible_for_stage_o44_decision"] is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
            and report["gates"]["numerical_equivalence"] is True
            and report["gates"]["zero_copy_batch_assembly_exercised"]
            is True
            and report["gates"]["r320_performance_improvement"] is False
            and report["gates"]["candidate_safety_non_regression"] is True
        )
    except (KeyError, TypeError):
        valid = False
    _require(valid, "Stage Q.5 O.4.3.3 evidence contract differs")


def _stage_o434_contract(
    report: Mapping[str, object], expected_o433_sha256: str
) -> None:
    try:
        valid = bool(
            report["qualification_stage"] == "O.4.3.4"
            and report["classification"] == "DIAGNOSTIC_COMPLETE"
            and report["architecture_decision"]
            == "remaining_materialization_attribution"
            and report["stage_o433_evidence"]["sha256"]
            == expected_o433_sha256
            and report["dominant_source_meets_screening_signal"] is False
            and report["eligible_for_target_selection_review"] is True
            and report["eligible_for_new_optimization_candidate"] is False
            and report["eligible_for_stage_o44_decision"] is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
        )
    except (KeyError, TypeError):
        valid = False
    _require(valid, "Stage Q.5 O.4.3.4 evidence contract differs")


def _r320_ratio(report: Mapping[str, object], description: str) -> float:
    try:
        value = report["scales"]["r320"][
            "mean_timestep_ratio_candidate_over_baseline"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{description} lacks the R320 ratio") from exc
    return _nonnegative(value, f"{description} R320 ratio")


def _q2_nested_delta(
    report: Mapping[str, object], region: str
) -> float:
    try:
        rows = report["nested_nonadditive_evidence"]["regions"]
        row = next(value for value in rows if value["region"] == region)
        return _finite(
            row["delta_milliseconds_per_step"],
            f"Stage Q.2 {region} delta",
        )
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError(f"Stage Q.2 lacks {region!r} evidence") from exc


def _positive_event(
    report: Mapping[str, object], section: str, name: str
) -> float:
    try:
        rows = report["operator_kernel_evidence"][section]
        row = next(value for value in rows if value["name"] == name)
        return _nonnegative(
            row["device_microseconds_delta_per_step"] / 1000.0,
            f"Stage Q.2 {name} delta",
        )
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError(f"Stage Q.2 lacks {name!r} evidence") from exc


def _workspace_summary(report: Mapping[str, object]) -> dict[str, object]:
    try:
        trials = report["workspace"]["candidate"]
        allocated = [
            int(value["workspace_allocated_bytes"]) for value in trials
        ]
        active = [int(value["workspace_active_count"]) for value in trials]
        retained = [
            int(value["retained_tensor_references"]) for value in trials
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("Stage Q.4 workspace evidence is invalid") from exc
    if not allocated or min(allocated) <= 0:
        raise ValueError("Stage Q.4 workspace allocation evidence is empty")
    return {
        "maximum_workspace_allocated_bytes": max(allocated),
        "maximum_workspace_active_count_after_step": max(active),
        "maximum_retained_tensor_references_after_step": max(retained),
    }


def _remaining_copy_summary(
    report: Mapping[str, object],
) -> dict[str, object]:
    try:
        rows = report["remaining_copy_sources"]
        threshold = _nonnegative(
            report["minimum_screening_fraction_of_timestep"],
            "Stage O.4.3.4 screening fraction",
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Stage O.4.3.4 copy evidence is invalid") from exc
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("Stage O.4.3.4 has no remaining copy sources")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Stage O.4.3.4 copy row is invalid")
        normalized.append(
            {
                "source": str(row["source"]),
                "mean_milliseconds_per_step": _nonnegative(
                    row["mean_milliseconds_per_step"],
                    "remaining copy time",
                ),
                "mean_fraction_of_timestep": _nonnegative(
                    row["mean_fraction_of_timestep"],
                    "remaining copy fraction",
                ),
                "copy_cat_materialized_output_bytes_per_step": int(
                    row["copy_cat_materialized_output_bytes_per_step"]
                ),
            }
        )
    return {
        "sources": normalized,
        "aggregate_measured_fraction_of_timestep": math.fsum(
            float(row["mean_fraction_of_timestep"]) for row in normalized
        ),
        "largest_single_source_fraction_of_timestep": max(
            float(row["mean_fraction_of_timestep"]) for row in normalized
        ),
        "single_source_screening_fraction": threshold,
        "fractions_are_measured_subregions_not_additive_speedup": True,
    }


def analyze_stage_q5_post_q4_retargeting(
    *,
    stage_q2_report: str | Path,
    expected_stage_q2_sha256: str,
    stage_q3_report: str | Path,
    expected_stage_q3_sha256: str,
    stage_q4_report: str | Path,
    expected_stage_q4_sha256: str,
    stage_o431_report: str | Path,
    expected_stage_o431_sha256: str,
    stage_o433_report: str | Path,
    expected_stage_o433_sha256: str,
    stage_o434_report: str | Path,
    expected_stage_o434_sha256: str,
) -> dict[str, object]:
    """Close rejected routes and select one analysis-only Stage Q.6 target."""

    inputs: dict[str, tuple[Path, dict[str, object]]] = {}
    for key, path, digest, description in (
        ("stage_q2", stage_q2_report, expected_stage_q2_sha256, "Stage Q.2 report"),
        ("stage_q3", stage_q3_report, expected_stage_q3_sha256, "Stage Q.3 report"),
        ("stage_q4", stage_q4_report, expected_stage_q4_sha256, "Stage Q.4 report"),
        (
            "stage_o431",
            stage_o431_report,
            expected_stage_o431_sha256,
            "Stage O.4.3.1 report",
        ),
        (
            "stage_o433",
            stage_o433_report,
            expected_stage_o433_sha256,
            "Stage O.4.3.3 report",
        ),
        (
            "stage_o434",
            stage_o434_report,
            expected_stage_o434_sha256,
            "Stage O.4.3.4 report",
        ),
    ):
        inputs[key] = _load_frozen_report(
            path,
            expected_sha256=digest,
            description=description,
        )

    q2 = inputs["stage_q2"][1]
    q3 = inputs["stage_q3"][1]
    q4 = inputs["stage_q4"][1]
    o431 = inputs["stage_o431"][1]
    o433 = inputs["stage_o433"][1]
    o434 = inputs["stage_o434"][1]
    _stage_q2_contract(q2)
    _stage_q3_contract(q3, expected_stage_q2_sha256)
    _stage_q4_contract(q4, expected_stage_q3_sha256)
    _stage_o431_contract(o431)
    _stage_o433_contract(o433, expected_stage_o431_sha256)
    _stage_o434_contract(o434, expected_stage_o433_sha256)

    throughput = q2["authoritative_throughput"]
    residual_gap_ms = _nonnegative(
        throughput["residual_gap_milliseconds_per_step"],
        "Stage Q.2 residual gap",
    )
    production_ms = _nonnegative(
        throughput["production_mean_milliseconds_per_step"],
        "Stage Q.2 production time",
    )
    candidate_ms = _nonnegative(
        throughput["candidate_mean_milliseconds_per_step"],
        "Stage Q.2 candidate time",
    )
    if not (production_ms > 0.0 and candidate_ms > production_ms):
        raise ValueError("Stage Q.5 requires a positive residual gap")

    q4_performance = q4["performance"]
    q4_baseline_peak = int(q4_performance["baseline_peak_allocated_bytes"])
    q4_candidate_peak = int(q4_performance["candidate_peak_allocated_bytes"])
    if q4_candidate_peak <= q4_baseline_peak:
        raise ValueError("Stage Q.5 expects the Q.4 memory regression")
    workspace = _workspace_summary(q4)
    memory_increase = q4_candidate_peak - q4_baseline_peak
    remaining_copy = _remaining_copy_summary(o434)

    q4_ratio = _nonnegative(
        q4_performance["mean_timestep_ratio_candidate_over_baseline"],
        "Stage Q.4 mean timestep ratio",
    )
    o431_ratio = _r320_ratio(o431, "Stage O.4.3.1")
    o433_ratio = _r320_ratio(o433, "Stage O.4.3.3")

    closed_routes = [
        {
            "route": "preallocated_batch_workspace",
            "disposition": "rejected_and_closed",
            "reason": (
                "allocation reuse preserved numerical results and lifecycle "
                "safety but did not remove tensor data movement"
            ),
            "mean_timestep_ratio_candidate_over_baseline": q4_ratio,
            "peak_allocated_increase_bytes": memory_increase,
            "workspace_bytes_over_peak_allocated_increase": (
                int(workspace["maximum_workspace_allocated_bytes"])
                / memory_increase
            ),
        },
        {
            "route": "opportunistic_natural_storage_views",
            "disposition": "closed_as_performance_neutral",
            "mean_timestep_ratio_candidate_over_baseline": o431_ratio,
        },
        {
            "route": "producer_owned_boundary_packed_h_and_stress",
            "disposition": "safe_evidence_only_not_standalone_candidate",
            "mean_timestep_ratio_candidate_over_baseline": o433_ratio,
            "measured_improvement_fraction": 1.0 - o433_ratio,
        },
        {
            "route": "single_remaining_copy_source",
            "disposition": "closed_below_screening_threshold",
            "largest_source_fraction_of_timestep": remaining_copy[
                "largest_single_source_fraction_of_timestep"
            ],
            "screening_fraction_of_timestep": remaining_copy[
                "single_source_screening_fraction"
            ],
        },
        {
            "route": "explicit_rhs_reimplementation",
            "disposition": "remains_closed",
            "reason": "Stage Q.2 measured it as non-material",
        },
        {
            "route": "projected_materialization_reopening",
            "disposition": "remains_closed",
            "reason": "Stage Q.2 and prior Stage O evidence already closed it",
        },
    ]

    inverse_delta_ms = _q2_nested_delta(q2, "transform_inverse")
    ranked_targets = [
        {
            "rank": 1,
            "target": STAGE_Q5_PRIMARY_TARGET,
            "next_action": "diagnostic_design_only",
            "rationale": (
                "Q.4 separated allocation reuse from unavoidable copies; "
                "O.4.3.3 showed that true producer-side copy elimination is "
                "safe but insufficient alone; O.4.3.4 leaves multiple "
                "producer/consumer materialization boundaries."
            ),
            "authoritative_residual_gap_milliseconds_per_step": residual_gap_ms,
            "remaining_copy_evidence": remaining_copy,
        },
        {
            "rank": 2,
            "target": "inverse_transform_interaction_inside_algebraic_path",
            "next_action": "retain_as_nested_secondary_diagnostic",
            "delta_milliseconds_per_step": inverse_delta_ms,
            "nonadditive_to_primary_gap": True,
        },
        {
            "rank": 3,
            "target": "operator_kernel_fragmentation_after_compilation",
            "next_action": "measure_with_primary_target_not_separately",
            "aten_cat_delta_milliseconds_per_step": _positive_event(
                q2, "largest_positive_operator_deltas", "aten::cat"
            ),
            "aten_copy_delta_milliseconds_per_step": _positive_event(
                q2, "largest_positive_operator_deltas", "aten::copy_"
            ),
            "cat_kernel_delta_milliseconds_per_step": _positive_event(
                q2,
                "largest_positive_kernel_deltas",
                "CatArrayBatchedCopy_contig",
            ),
            "durations_overlap_and_are_not_additive": True,
        },
    ]

    return {
        "schema_version": 1,
        "qualification_stage": "Q.5",
        "classification": STAGE_Q5_CLASSIFICATION,
        "architecture_decision": "post_q4_evidence_retargeting",
        "authoritative_residual": {
            "production_mean_milliseconds_per_step": production_ms,
            "candidate_mean_milliseconds_per_step": candidate_ms,
            "residual_gap_milliseconds_per_step": residual_gap_ms,
            "candidate_over_production_ratio": candidate_ms / production_ms,
        },
        "q4_workspace_outcome": {
            "classification": q4["classification"],
            "mean_timestep_ratio_candidate_over_baseline": q4_ratio,
            "baseline_peak_allocated_bytes": q4_baseline_peak,
            "candidate_peak_allocated_bytes": q4_candidate_peak,
            "peak_allocated_increase_bytes": memory_increase,
            "workspace_bytes_over_peak_allocated_increase": (
                int(workspace["maximum_workspace_allocated_bytes"])
                / memory_increase
            ),
            **workspace,
            "allocation_reuse_did_not_eliminate_data_movement": True,
        },
        "closed_routes": closed_routes,
        "ranked_diagnostic_targets": ranked_targets,
        "primary_target": STAGE_Q5_PRIMARY_TARGET,
        "stage_q6_scope": {
            "action": "define_one_read_only_end_to_end_data_movement_map",
            "must_cover_sources": [
                row["source"] for row in remaining_copy["sources"]
            ],
            "must_distinguish": [
                "allocation",
                "tensor_copy",
                "view_or_alias",
                "transform_input_layout",
                "consumer_required_layout",
                "operator_launch_fragmentation",
            ],
            "may_reuse_existing_profiles": True,
            "may_execute_solver": False,
            "may_implement_candidate": False,
            "may_change_production_default": False,
        },
        "interpretation_constraints": {
            "operator_and_kernel_durations_may_overlap": True,
            "nested_transform_time_is_not_additive": True,
            "remaining_copy_source_fractions_are_not_predicted_speedups": True,
            "producer_packing_gain_is_not_additive_to_q2_gap": True,
            "changes_equations": False,
            "changes_runtime_implementation": False,
            "changes_production_default": False,
        },
        "eligible_for_stage_q6_diagnostic_design": True,
        "eligible_for_stage_q6_candidate_implementation": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "inputs": [
            {
                "kind": key,
                "path": str(value[0]),
                "sha256": file_sha256(value[0]),
            }
            for key, value in inputs.items()
        ],
    }


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select the post-Q.4 Plane diagnostic target."
    )
    for stage in ("q2", "q3", "q4", "o431", "o433", "o434"):
        parser.add_argument(f"--stage-{stage}-report", type=Path, required=True)
        parser.add_argument(f"--expected-stage-{stage}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_q5_post_q4_retargeting(
        stage_q2_report=args.stage_q2_report,
        expected_stage_q2_sha256=args.expected_stage_q2_sha256,
        stage_q3_report=args.stage_q3_report,
        expected_stage_q3_sha256=args.expected_stage_q3_sha256,
        stage_q4_report=args.stage_q4_report,
        expected_stage_q4_sha256=args.expected_stage_q4_sha256,
        stage_o431_report=args.stage_o431_report,
        expected_stage_o431_sha256=args.expected_stage_o431_sha256,
        stage_o433_report=args.stage_o433_report,
        expected_stage_o433_sha256=args.expected_stage_o433_sha256,
        stage_o434_report=args.stage_o434_report,
        expected_stage_o434_sha256=args.expected_stage_o434_sha256,
    )
    _write_new_json(args.output, report)
    return 0


__all__ = [
    "STAGE_Q5_CLASSIFICATION",
    "STAGE_Q5_PRIMARY_TARGET",
    "analysis_main",
    "analyze_stage_q5_post_q4_retargeting",
]
