"""Stage Q.3 refinement of the Plane algebraic-orchestration target.

The analysis consumes frozen Stage Q.1 candidate profiles and the completed
Stage Q.2 report.  It does not execute a solver.  Its only purpose is to map
the already-observed algebraic gap to solver, batch-assembly, publication, and
transform subregions before one bounded Stage Q.4 candidate is implemented.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import math
from pathlib import Path

from ._shadow_support import file_sha256
from .h100_shadow_qualification import _write_new_json
from .stage_q2_diagnostics import (
    STAGE_Q2_CANDIDATE_ROLE,
    STAGE_Q2_PROFILE_TRIALS,
    _profile_identity,
    _raw_regions,
    _read_unchanged_json,
    _validate_profile,
)


STAGE_Q3_PROFILE_TRIALS = STAGE_Q2_PROFILE_TRIALS
STAGE_Q3_MINIMUM_MATERIAL_MILLISECONDS = 0.25


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


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = math.fsum(values) / len(values)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in values)
        / (len(values) - 1)
    )


def _validate_q2_report(
    path: str | Path,
) -> tuple[Path, str, dict[str, object]]:
    resolved, digest, report = _read_unchanged_json(
        path,
        "Stage Q.2 residual-gap report",
    )
    try:
        valid = (
            report["qualification_stage"] == "Q.2"
            and report["classification"] == "TARGET_REVIEW_COMPLETE"
            and report["target_review"]["primary_target"]
            == "algebraic_runtime_orchestration"
            and report["target_review"]["primary_source_region"]
            == "algebraic"
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
    if not valid:
        raise ValueError("Stage Q.2 report does not authorize target refinement")
    return resolved, digest, report


def _candidate_profile_inputs(
    report: Mapping[str, object],
) -> set[tuple[Path, str]]:
    result: set[tuple[Path, str]] = set()
    inputs = report.get("inputs")
    if not isinstance(inputs, Sequence):
        raise ValueError("Stage Q.2 report inputs are invalid")
    for item in inputs:
        if not isinstance(item, Mapping) or item.get("kind") != "profile":
            continue
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("Stage Q.2 profile identity is invalid")
        result.add((Path(path).expanduser().resolve(), digest))
    return result


def _region_statistics(
    reports: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    inventories = []
    for report in reports:
        inventories.append(
            set(_raw_regions(report, STAGE_Q2_CANDIDATE_ROLE))
        )
    if not inventories or any(
        inventory != inventories[0] for inventory in inventories[1:]
    ):
        raise ValueError("Stage Q.3 semantic region inventories differ")

    rows = []
    for name in sorted(inventories[0]):
        milliseconds = []
        calls_per_step = []
        for report in reports:
            regions = _raw_regions(report, STAGE_Q2_CANDIDATE_ROLE)
            region = regions[name]
            if not isinstance(region, Mapping):
                raise ValueError(f"Stage Q.3 region {name!r} is invalid")
            steps = _nonnegative(
                report["matched_semantic_regions"]["semantic_steps"],
                "Stage Q.3 semantic steps",
            )
            if steps <= 0.0:
                raise ValueError("Stage Q.3 semantic steps must be positive")
            milliseconds.append(
                1000.0
                * _nonnegative(
                    region["total_seconds"],
                    f"Stage Q.3 region {name} total seconds",
                )
                / steps
            )
            calls_per_step.append(
                _nonnegative(
                    region["calls"],
                    f"Stage Q.3 region {name} calls",
                )
                / steps
            )
        rows.append(
            {
                "region": name,
                "mean_milliseconds_per_step": (
                    math.fsum(milliseconds) / len(milliseconds)
                ),
                "sample_std_milliseconds_per_step": _sample_std(milliseconds),
                "mean_calls_per_step": (
                    math.fsum(calls_per_step) / len(calls_per_step)
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["mean_milliseconds_per_step"]),
            str(row["region"]),
        )
    )
    return rows


def _row_by_name(
    rows: Sequence[Mapping[str, object]],
    name: str,
) -> Mapping[str, object] | None:
    return next((row for row in rows if row["region"] == name), None)


def _positive_event_delta_milliseconds(
    q2_report: Mapping[str, object],
    *,
    section: str,
    predicate,
) -> float:
    evidence = q2_report["operator_kernel_evidence"]
    if not isinstance(evidence, Mapping):
        raise ValueError("Stage Q.2 operator/kernel evidence is invalid")
    rows = evidence.get(section)
    if not isinstance(rows, Sequence):
        raise ValueError(f"Stage Q.2 evidence section {section!r} is invalid")
    values = []
    for row in rows:
        if not isinstance(row, Mapping) or not predicate(str(row.get("name", ""))):
            continue
        values.append(
            max(
                0.0,
                _finite(
                    row["device_microseconds_delta_per_step"],
                    f"Stage Q.2 {section} event delta",
                )
                / 1000.0,
            )
        )
    return max(values, default=0.0)


def _q2_nested_delta(
    q2_report: Mapping[str, object],
    region_name: str,
) -> float:
    rows = q2_report["nested_nonadditive_evidence"]["regions"]
    if not isinstance(rows, Sequence):
        raise ValueError("Stage Q.2 nested evidence is invalid")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("region") == region_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage Q.2 nested region {region_name!r} is missing")
    return _finite(
        matches[0]["delta_milliseconds_per_step"],
        f"Stage Q.2 {region_name} delta",
    )


def _candidate_design(
    *,
    assembly_sum_ms: float,
    largest_assembly_ms: float,
    publication_ms: float,
    cat_delta_ms: float,
    copy_clone_delta_ms: float,
    inverse_delta_ms: float,
) -> dict[str, object]:
    material = STAGE_Q3_MINIMUM_MATERIAL_MILLISECONDS
    if (
        assembly_sum_ms >= material
        and largest_assembly_ms >= material
        and max(cat_delta_ms, copy_clone_delta_ms) >= material
    ):
        selected = "preplanned_algebraic_batch_workspace"
        rationale = (
            "source-attributed batch assembly and global copy/cat deltas "
            "coincide inside the algebraic region"
        )
    elif inverse_delta_ms >= material:
        selected = "inverse_transform_batch_scheduling"
        rationale = "inverse-transform excess remains the only material signal"
    elif publication_ms >= material:
        selected = "direct_spectral_handoff"
        rationale = "algebraic spectral publication is the largest bounded signal"
    else:
        selected = None
        rationale = "no single bounded mechanism meets the materiality gate"

    return {
        "selected_candidate": selected,
        "rationale": rationale,
        "materiality_threshold_milliseconds_per_step": material,
        "scope": (
            None
            if selected is None
            else {
                "geometry": "PlaneSlab",
                "model": "BerisEdwardsPlaneCoupledModel",
                "runtime": "separated_canary_only",
                "mechanism_count": 1,
                "preserve_algebraic_solver_boundaries": True,
                "preserve_boundary_signatures": True,
                "preserve_transform_order_and_count_initially": True,
                "preserve_equations_and_time_integrator": True,
                "retain_tensor_values_across_timesteps": False,
                "production_default_change": False,
            }
        ),
        "forbidden_scope": [
            "explicit_rhs_reoptimization",
            "projected_materialization_reopening",
            "generic_solver_default_change",
            "channel_runtime_change",
            "cross_timestep_physical_state_cache",
        ],
    }


def analyze_stage_q3_algebraic_target(
    candidate_profile_paths: Sequence[str | Path],
    *,
    stage_q2_report_path: str | Path,
) -> dict[str, object]:
    """Refine the Stage Q.2 algebraic target without executing a solver."""

    if len(candidate_profile_paths) != STAGE_Q3_PROFILE_TRIALS:
        raise ValueError("Stage Q.3 requires exactly three candidate profiles")
    resolved_paths = tuple(
        Path(path).expanduser().resolve() for path in candidate_profile_paths
    )
    if len(set(resolved_paths)) != STAGE_Q3_PROFILE_TRIALS:
        raise ValueError("Stage Q.3 requires three distinct candidate profiles")

    q2_path, q2_digest, q2_report = _validate_q2_report(
        stage_q2_report_path
    )
    loaded = tuple(
        _validate_profile(path, role=STAGE_Q2_CANDIDATE_ROLE)
        for path in candidate_profile_paths
    )
    reports = tuple(report for _, _, report in loaded)
    if len({_profile_identity(report) for report in reports}) != 1:
        raise ValueError("Stage Q.3 candidate profile identities differ")
    q2_profile_inputs = _candidate_profile_inputs(q2_report)
    for path, digest, _ in loaded:
        if (path, digest) not in q2_profile_inputs:
            raise ValueError(
                "Stage Q.3 candidate profile is not frozen by Stage Q.2"
            )

    rows = _region_statistics(reports)
    algebraic_total = _row_by_name(rows, "timestep.algebraic_update")
    if algebraic_total is None:
        raise ValueError("Stage Q.3 lacks timestep.algebraic_update")
    algebraic_ms = float(algebraic_total["mean_milliseconds_per_step"])
    if algebraic_ms <= 0.0:
        raise ValueError("Stage Q.3 algebraic time must be positive")

    solver_rows = [
        row
        for row in rows
        if str(row["region"]).startswith("algebraic.")
        and str(row["region"]).endswith(".solve")
    ]
    assembly_rows = [
        row
        for row in rows
        if str(row["region"]).startswith(
            "projected_batch_assembly.source.algebraic."
        )
    ]
    publication = _row_by_name(rows, "algebraic.publish_spectral")
    publication_ms = (
        0.0
        if publication is None
        else float(publication["mean_milliseconds_per_step"])
    )
    solver_sum_ms = math.fsum(
        float(row["mean_milliseconds_per_step"]) for row in solver_rows
    )
    assembly_sum_ms = math.fsum(
        float(row["mean_milliseconds_per_step"]) for row in assembly_rows
    )
    largest_assembly_ms = max(
        (
            float(row["mean_milliseconds_per_step"])
            for row in assembly_rows
        ),
        default=0.0,
    )

    cat_delta_ms = _positive_event_delta_milliseconds(
        q2_report,
        section="largest_positive_operator_deltas",
        predicate=lambda name: name == "aten::cat",
    )
    copy_clone_delta_ms = max(
        _positive_event_delta_milliseconds(
            q2_report,
            section="largest_positive_operator_deltas",
            predicate=lambda name: name in {"aten::copy_", "aten::clone"},
        ),
        _positive_event_delta_milliseconds(
            q2_report,
            section="largest_positive_kernel_deltas",
            predicate=lambda name: "CatArrayBatchedCopy" in name,
        ),
    )
    inverse_delta_ms = _q2_nested_delta(q2_report, "transform_inverse")
    design = _candidate_design(
        assembly_sum_ms=assembly_sum_ms,
        largest_assembly_ms=largest_assembly_ms,
        publication_ms=publication_ms,
        cat_delta_ms=cat_delta_ms,
        copy_clone_delta_ms=copy_clone_delta_ms,
        inverse_delta_ms=inverse_delta_ms,
    )

    def with_share(row: Mapping[str, object]) -> dict[str, object]:
        return {
            **dict(row),
            "share_of_candidate_algebraic_time": (
                float(row["mean_milliseconds_per_step"]) / algebraic_ms
            ),
            "nested_or_subregion_not_additive_to_parent": True,
        }

    return {
        "schema_version": 1,
        "qualification_stage": "Q.3",
        "classification": "ALGEBRAIC_TARGET_REFINEMENT_COMPLETE",
        "architecture_decision": "one_bounded_stage_q4_candidate_design",
        "candidate_profile_trials": STAGE_Q3_PROFILE_TRIALS,
        "candidate_algebraic_milliseconds_per_step": algebraic_ms,
        "algebraic_solver_regions": [with_share(row) for row in solver_rows],
        "algebraic_solver_region_sum_milliseconds_per_step": solver_sum_ms,
        "batch_assembly_source_regions": [
            with_share(row) for row in assembly_rows
        ],
        "batch_assembly_source_sum_milliseconds_per_step": assembly_sum_ms,
        "algebraic_publication_region": (
            None if publication is None else with_share(publication)
        ),
        "supporting_excess_signals": {
            "aten_cat_delta_milliseconds_per_step": cat_delta_ms,
            "copy_clone_or_cat_kernel_delta_milliseconds_per_step": (
                copy_clone_delta_ms
            ),
            "inverse_transform_nested_delta_milliseconds_per_step": (
                inverse_delta_ms
            ),
            "signals_are_not_additive": True,
        },
        "candidate_design": design,
        "interpretation_constraints": {
            "subregions_may_be_nested": True,
            "subregion_sums_are_not_authoritative_throughput": True,
            "stage_q2_noninstrumented_gap_remains_authoritative": True,
            "projected_materialization_line_remains_closed": True,
            "explicit_rhs_line_remains_closed": True,
            "changes_equations": False,
            "changes_runtime_implementation": False,
            "changes_production_default": False,
        },
        "eligible_for_stage_q4_candidate_implementation": (
            design["selected_candidate"] is not None
        ),
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "all_semantic_regions": rows,
        "inputs": [
            {
                "kind": "stage_q2_report",
                "path": str(q2_path),
                "sha256": q2_digest,
            }
        ]
        + [
            {"kind": "candidate_profile", "path": str(path), "sha256": digest}
            for path, digest, _ in loaded
        ],
    }


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refine the Plane Stage Q.3 algebraic target."
    )
    parser.add_argument(
        "--candidate-profile",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--stage-q2-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_q3_algebraic_target(
        args.candidate_profile,
        stage_q2_report_path=args.stage_q2_report,
    )
    _write_new_json(args.output, report)
    return 0


__all__ = [
    "STAGE_Q3_MINIMUM_MATERIAL_MILLISECONDS",
    "STAGE_Q3_PROFILE_TRIALS",
    "analysis_main",
    "analyze_stage_q3_algebraic_target",
]
