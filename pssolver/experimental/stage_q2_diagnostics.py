"""Residual-gap attribution for the Plane Stage Q.2 target review.

This module is deliberately analysis-only.  It compares three frozen
production profiles with three frozen compiled-explicit-RHS canary profiles.
Authoritative throughput, disjoint top-level semantic regions, and nested
transform observations remain separate so overlapping timings are never
double-counted.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

from ._shadow_support import file_sha256
from .h100_shadow_qualification import _write_new_json
from .stage_o4_qualification import _load_json
from .stage_p_diagnostics import _aggregate_entries, _entry_delta


STAGE_Q2_PROFILE_TRIALS = 3
STAGE_Q2_PRODUCTION_ROLE = "legacy_production"
STAGE_Q2_CANDIDATE_ROLE = "separated_canary"
STAGE_Q2_SHAPE = (320, 320, 80)

_TOP_LEVEL_REGIONS = (
    "algebraic",
    "explicit_rhs",
    "spectral_update",
    "dynamic_inverse",
    "spectral_refresh",
)
_TARGET_BY_REGION = {
    "algebraic": "algebraic_runtime_orchestration",
    "explicit_rhs": "explicit_rhs_execution",
    "spectral_update": "spectral_update",
    "dynamic_inverse": "dynamic_inverse_or_materialization",
    "spectral_refresh": "spectral_refresh",
}
_MATCHED_KEYS = {
    "total": "total_milliseconds_per_step",
    "algebraic": "algebraic_milliseconds_per_step",
    "explicit_rhs": "explicit_rhs_milliseconds_per_step",
    "spectral_update": "spectral_update_milliseconds_per_step",
    "dynamic_inverse": "dynamic_inverse_milliseconds_per_step",
    "spectral_refresh": "spectral_refresh_milliseconds_per_step",
    "transform_total": "transform_milliseconds_per_step",
}
_MEASUREMENT_IDENTITY_KEYS = (
    "shape",
    "lengths",
    "dtype",
    "dt",
    "dealias_rule",
    "spectral_storage",
    "hermitian_axis",
    "projected_transform_execution",
    "transform_execution_order",
    "spectral_refresh_interval",
    "pointwise_execution",
    "warmup_steps",
    "throughput_steps",
    "operator_audit_steps",
    "semantic_steps",
)
_ENVIRONMENT_IDENTITY_KEYS = (
    "device_name",
    "compute_capability",
    "torch",
    "cuda_runtime",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "float32_matmul_precision",
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


def _read_unchanged_json(
    path: str | Path,
    description: str,
) -> tuple[Path, str, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    digest = file_sha256(resolved)
    report = _load_json(resolved, description)
    if file_sha256(resolved) != digest:
        raise RuntimeError(f"{description} changed while it was read")
    return resolved, digest, report


def _validate_profile(
    path: str | Path,
    *,
    role: str,
) -> tuple[Path, str, dict[str, object]]:
    resolved, digest, report = _read_unchanged_json(
        path,
        f"Stage Q.2 {role} profile",
    )
    try:
        configuration = report["configuration"]
        throughput = report["throughput"]
        matched = report["matched_semantic_regions"]
        audit = report["operator_kernel_audit"]
        contract = report["measurement_contract"]
        valid = (
            report["classification"] == "OPERATOR_KERNEL_PROFILE_COMPLETE"
            and report["runtime_role"] == role
            and report["finite"] is True
            and tuple(configuration["shape"]) == STAGE_Q2_SHAPE
            and configuration["dtype"] == "float64"
            and throughput["instrumentation"] == "outer_cuda_events_only"
            and matched["nested_regions_overlap"] is True
            and matched["separate_from_authoritative_throughput"] is True
            and audit["operator_entries_complete"] is True
            and audit["kernel_entries_complete"] is True
            and audit["bounded_aggregate_only"] is True
            and audit["raw_trace_retained"] is False
            and contract[
                "throughput_operator_and_semantic_windows_are_separate"
            ]
            is True
            and contract[
                "throughput_uses_internal_semantic_instrumentation"
            ]
            is False
            and contract["raw_trace_retained"] is False
            and contract["changes_equations"] is False
            and contract["changes_production_default"] is False
            and report["production_default_changed"] is False
            and report["eligible_for_production_promotion"] is False
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"Stage Q.2 {role} profile is incomplete")

    _nonnegative(
        throughput["mean_timestep_seconds"],
        f"Stage Q.2 {role} mean timestep",
    )
    for label, key in _MATCHED_KEYS.items():
        _nonnegative(
            matched[key],
            f"Stage Q.2 {role} semantic region {label}",
        )
    if int(report.get("dynamo_delta", {}).get("graph_breaks") or 0) != 0:
        raise ValueError(f"Stage Q.2 {role} profile contains a graph break")
    return resolved, digest, report


def _profile_identity(report: Mapping[str, object]) -> str:
    configuration = report["configuration"]
    if not isinstance(configuration, Mapping):
        raise ValueError("Stage Q.2 profile configuration is invalid")
    environment = report.get("environment", {})
    if not isinstance(environment, Mapping):
        raise ValueError("Stage Q.2 profile environment is invalid")
    measurement_configuration = {
        key: configuration.get(key) for key in _MEASUREMENT_IDENTITY_KEYS
    }
    measurement_environment = {
        key: environment.get(key) for key in _ENVIRONMENT_IDENTITY_KEYS
    }
    identity = {
        "configuration": measurement_configuration,
        "environment": measurement_environment,
        "production_metadata_sha256": report["production_metadata_sha256"],
        "production_initial_q_sha256": report[
            "production_initial_q_sha256"
        ],
        "stage_o_closure_sha256": report["stage_o_closure_sha256"],
    }
    return json.dumps(identity, allow_nan=False, sort_keys=True)


def _mean_path(
    reports: Sequence[Mapping[str, object]],
    *path: str,
) -> float:
    values = []
    for report in reports:
        value: object = report
        for name in path:
            if not isinstance(value, Mapping) or name not in value:
                raise ValueError(f"Stage Q.2 profile path {path!r} is invalid")
            value = value[name]
        values.append(_nonnegative(value, ".".join(path)))
    return math.fsum(values) / len(values)


def _raw_regions(report: Mapping[str, object], role: str) -> Mapping[str, object]:
    raw = report["raw_semantic_regions"]
    if not isinstance(raw, Mapping):
        raise ValueError("Stage Q.2 raw semantic regions are invalid")
    if role == STAGE_Q2_PRODUCTION_ROLE:
        if "regions" in raw:
            raise ValueError("legacy production profile uses wrapper schema")
        return raw
    if role == STAGE_Q2_CANDIDATE_ROLE:
        regions = raw.get("regions")
        if not isinstance(regions, Mapping):
            raise ValueError("separated candidate profile lacks wrapper schema")
        direct_region_keys = {
            "transform_forward",
            "transform_inverse",
            "static_fields",
            "q_nonlinear",
        }
        if direct_region_keys.intersection(raw):
            raise ValueError("separated candidate profile mixes semantic schemas")
        return regions
    raise ValueError(f"unsupported Stage Q.2 runtime role {role!r}")


def _nested_region_mean_milliseconds(
    reports: Sequence[Mapping[str, object]],
    *,
    role: str,
    legacy_name: str,
    separated_name: str,
) -> float:
    values = []
    region_name = (
        legacy_name if role == STAGE_Q2_PRODUCTION_ROLE else separated_name
    )
    for report in reports:
        regions = _raw_regions(report, role)
        region = regions.get(region_name)
        if not isinstance(region, Mapping):
            raise ValueError(
                f"Stage Q.2 {role} is missing semantic region {region_name}"
            )
        semantic_steps = _nonnegative(
            report["matched_semantic_regions"]["semantic_steps"],
            f"Stage Q.2 {role} semantic steps",
        )
        if semantic_steps <= 0.0:
            raise ValueError("Stage Q.2 semantic steps must be positive")
        total_seconds = _nonnegative(
            region["total_seconds"],
            f"Stage Q.2 {role} {region_name} total seconds",
        )
        values.append(1000.0 * total_seconds / semantic_steps)
    return math.fsum(values) / len(values)


def _semantic_means(
    reports: Sequence[Mapping[str, object]],
    *,
    role: str,
) -> dict[str, float]:
    means = {
        label: _mean_path(reports, "matched_semantic_regions", key)
        for label, key in _MATCHED_KEYS.items()
    }
    means["transform_forward"] = _nested_region_mean_milliseconds(
        reports,
        role=role,
        legacy_name="transform_forward",
        separated_name="transform.forward",
    )
    means["transform_inverse"] = _nested_region_mean_milliseconds(
        reports,
        role=role,
        legacy_name="transform_inverse",
        separated_name="transform.inverse",
    )
    return means


def _category_means(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    names = sorted(
        {
            str(name)
            for report in reports
            for name in report["operator_kernel_audit"]["category_totals"]
        }
    )
    result: dict[str, dict[str, float]] = {}
    for name in names:
        metrics = {}
        for metric in (
            "operator_calls_per_step",
            "kernel_calls_per_step",
            "kernel_device_microseconds_per_step",
        ):
            values = []
            for report in reports:
                category = report["operator_kernel_audit"][
                    "category_totals"
                ].get(name, {})
                values.append(
                    _nonnegative(
                        category.get(metric, 0.0),
                        f"Stage Q.2 category {name} {metric}",
                    )
                )
            metrics[metric] = math.fsum(values) / len(values)
        result[name] = metrics
    return result


def _category_deltas(
    production: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    rows = []
    for name in sorted(set(production) | set(candidate)):
        left = production.get(name, {})
        right = candidate.get(name, {})
        row: dict[str, object] = {"category": name}
        for metric in (
            "operator_calls_per_step",
            "kernel_calls_per_step",
            "kernel_device_microseconds_per_step",
        ):
            p_value = float(left.get(metric, 0.0))
            c_value = float(right.get(metric, 0.0))
            row[f"production_{metric}"] = p_value
            row[f"candidate_{metric}"] = c_value
            row[f"delta_{metric}"] = c_value - p_value
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["delta_kernel_device_microseconds_per_step"]),
            str(row["category"]),
        )
    )
    return rows


def _target_ranking(
    region_rows: Sequence[Mapping[str, object]],
    *,
    authoritative_gap_ms: float,
) -> list[dict[str, object]]:
    minimum_material_delta = max(0.1, 0.05 * authoritative_gap_ms)
    rows = []
    for row in region_rows:
        delta = float(row["delta_milliseconds_per_step"])
        if delta <= 0.0:
            continue
        region = str(row["region"])
        rows.append(
            {
                "target": _TARGET_BY_REGION[region],
                "source_region": region,
                "observed_delta_milliseconds_per_step": delta,
                "share_of_authoritative_gap": (
                    delta / authoritative_gap_ms
                    if authoritative_gap_ms > 0.0
                    else None
                ),
                "material": delta >= minimum_material_delta,
                "evidence_kind": "disjoint_top_level_semantic_region",
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["observed_delta_milliseconds_per_step"]),
            str(row["target"]),
        )
    )
    return rows


def analyze_stage_q2_residual_gap(
    production_profile_paths: Sequence[str | Path],
    candidate_profile_paths: Sequence[str | Path],
    *,
    supporting_artifact_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Attribute the residual A/C gap without changing runtime code."""

    if len(production_profile_paths) != STAGE_Q2_PROFILE_TRIALS or len(
        candidate_profile_paths
    ) != STAGE_Q2_PROFILE_TRIALS:
        raise ValueError("Stage Q.2 requires exactly three profiles per runtime")
    input_paths = tuple(
        Path(path).expanduser().resolve()
        for path in (*production_profile_paths, *candidate_profile_paths)
    )
    if len(set(input_paths)) != 2 * STAGE_Q2_PROFILE_TRIALS:
        raise ValueError("Stage Q.2 requires six distinct profile paths")

    production_loaded = tuple(
        _validate_profile(path, role=STAGE_Q2_PRODUCTION_ROLE)
        for path in production_profile_paths
    )
    candidate_loaded = tuple(
        _validate_profile(path, role=STAGE_Q2_CANDIDATE_ROLE)
        for path in candidate_profile_paths
    )
    production = tuple(report for _, _, report in production_loaded)
    candidate = tuple(report for _, _, report in candidate_loaded)
    all_reports = (*production, *candidate)
    if len({_profile_identity(report) for report in all_reports}) != 1:
        raise ValueError("Stage Q.2 production/candidate profile identities differ")

    production_throughput_ms = 1000.0 * _mean_path(
        production,
        "throughput",
        "mean_timestep_seconds",
    )
    candidate_throughput_ms = 1000.0 * _mean_path(
        candidate,
        "throughput",
        "mean_timestep_seconds",
    )
    if production_throughput_ms <= 0.0 or candidate_throughput_ms <= 0.0:
        raise ValueError("Stage Q.2 throughput must be positive")
    authoritative_gap_ms = candidate_throughput_ms - production_throughput_ms

    production_semantic = _semantic_means(
        production,
        role=STAGE_Q2_PRODUCTION_ROLE,
    )
    candidate_semantic = _semantic_means(
        candidate,
        role=STAGE_Q2_CANDIDATE_ROLE,
    )
    top_level_rows = [
        {
            "region": region,
            "production_milliseconds_per_step": production_semantic[region],
            "candidate_milliseconds_per_step": candidate_semantic[region],
            "delta_milliseconds_per_step": (
                candidate_semantic[region] - production_semantic[region]
            ),
            "disjoint_with_other_top_level_regions": True,
        }
        for region in _TOP_LEVEL_REGIONS
    ]
    semantic_total_delta = (
        candidate_semantic["total"] - production_semantic["total"]
    )
    top_level_delta_sum = math.fsum(
        float(row["delta_milliseconds_per_step"])
        for row in top_level_rows
    )
    nested_transform_rows = [
        {
            "region": region,
            "production_milliseconds_per_step": production_semantic[region],
            "candidate_milliseconds_per_step": candidate_semantic[region],
            "delta_milliseconds_per_step": (
                candidate_semantic[region] - production_semantic[region]
            ),
            "nested_overlapping_evidence": True,
            "included_in_additive_gap_attribution": False,
        }
        for region in (
            "transform_forward",
            "transform_inverse",
            "transform_total",
        )
    ]

    production_categories = _category_means(production)
    candidate_categories = _category_means(candidate)
    category_deltas = _category_deltas(
        production_categories,
        candidate_categories,
    )
    production_operators = _aggregate_entries(production, "operators")
    candidate_operators = _aggregate_entries(candidate, "operators")
    production_kernels = _aggregate_entries(production, "kernels")
    candidate_kernels = _aggregate_entries(candidate, "kernels")
    operator_deltas = _entry_delta(production_operators, candidate_operators)
    kernel_deltas = _entry_delta(production_kernels, candidate_kernels)

    target_ranking = _target_ranking(
        top_level_rows,
        authoritative_gap_ms=authoritative_gap_ms,
    )
    primary = next(
        (row for row in target_ranking if row["material"] is True),
        None,
    )
    forward_delta = next(
        float(row["delta_milliseconds_per_step"])
        for row in nested_transform_rows
        if row["region"] == "transform_forward"
    )
    inverse_delta = next(
        float(row["delta_milliseconds_per_step"])
        for row in nested_transform_rows
        if row["region"] == "transform_inverse"
    )

    supporting_loaded = tuple(
        (
            Path(path).expanduser().resolve(),
            file_sha256(Path(path).expanduser().resolve()),
        )
        for path in supporting_artifact_paths
    )
    return {
        "schema_version": 1,
        "qualification_stage": "Q.2",
        "classification": "TARGET_REVIEW_COMPLETE",
        "architecture_decision": "residual_production_candidate_gap_attribution",
        "trial_count_per_runtime": STAGE_Q2_PROFILE_TRIALS,
        "authoritative_throughput": {
            "production_mean_milliseconds_per_step": production_throughput_ms,
            "candidate_mean_milliseconds_per_step": candidate_throughput_ms,
            "candidate_over_production_ratio": (
                candidate_throughput_ms / production_throughput_ms
            ),
            "residual_gap_milliseconds_per_step": authoritative_gap_ms,
            "source": "noninstrumented_outer_cuda_event_window",
        },
        "additive_semantic_attribution": {
            "regions": top_level_rows,
            "top_level_delta_sum_milliseconds_per_step": top_level_delta_sum,
            "semantic_total_delta_milliseconds_per_step": semantic_total_delta,
            "semantic_total_reconciliation_error_milliseconds_per_step": (
                top_level_delta_sum - semantic_total_delta
            ),
            "authoritative_minus_semantic_gap_milliseconds_per_step": (
                authoritative_gap_ms - semantic_total_delta
            ),
            "separate_instrumented_window": True,
        },
        "nested_nonadditive_evidence": {
            "regions": nested_transform_rows,
            "transform_forward_materially_changed": abs(forward_delta) >= 0.25,
            "transform_inverse_materially_changed": abs(inverse_delta) >= 0.25,
            "must_not_be_added_to_top_level_attribution": True,
        },
        "operator_kernel_evidence": {
            "category_deltas": category_deltas,
            "largest_positive_operator_deltas": [
                row
                for row in operator_deltas
                if float(row["device_microseconds_delta_per_step"]) > 0.0
            ][:25],
            "largest_positive_kernel_deltas": [
                row
                for row in kernel_deltas
                if float(row["device_microseconds_delta_per_step"]) > 0.0
            ][:25],
            "bounded_aggregate_only": True,
            "durations_may_overlap": True,
            "not_authoritative_throughput": True,
        },
        "target_review": {
            "ranked_targets": target_ranking,
            "primary_target": None if primary is None else primary["target"],
            "primary_source_region": (
                None if primary is None else primary["source_region"]
            ),
            "candidate_design_scope": (
                None
                if primary is None
                else "one_bounded_runtime_mechanism_only"
            ),
            "explicit_rhs_line_closed": all(
                float(row["delta_milliseconds_per_step"]) <= 0.25
                for row in top_level_rows
                if row["region"] == "explicit_rhs"
            ),
            "transform_forward_line_closed": abs(forward_delta) < 0.25,
            "projected_materialization_line_remains_closed": True,
        },
        "interpretation_constraints": {
            "top_level_semantic_regions_are_disjoint": True,
            "nested_transform_regions_overlap_top_level_regions": True,
            "nested_transform_deltas_are_not_subtracted_from_residual": True,
            "semantic_and_throughput_windows_are_separate": True,
            "operator_kernel_durations_are_supporting_evidence_only": True,
            "changes_equations": False,
            "changes_runtime_implementation": False,
            "changes_production_default": False,
        },
        "eligible_for_stage_q2_candidate_design": primary is not None,
        "eligible_for_stage_q2_candidate_implementation": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "inputs": [
            {"kind": "profile", "path": str(path), "sha256": digest}
            for path, digest, _ in (*production_loaded, *candidate_loaded)
        ]
        + [
            {
                "kind": "supporting_artifact",
                "path": str(path),
                "sha256": digest,
            }
            for path, digest in supporting_loaded
        ],
    }


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the Plane Stage Q.2 residual performance gap."
    )
    parser.add_argument(
        "--production-profile",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--candidate-profile",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--supporting-artifact",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_q2_residual_gap(
        args.production_profile,
        args.candidate_profile,
        supporting_artifact_paths=args.supporting_artifact,
    )
    _write_new_json(args.output, report)
    return 0


__all__ = [
    "STAGE_Q2_CANDIDATE_ROLE",
    "STAGE_Q2_PRODUCTION_ROLE",
    "STAGE_Q2_PROFILE_TRIALS",
    "STAGE_Q2_SHAPE",
    "analysis_main",
    "analyze_stage_q2_residual_gap",
]
