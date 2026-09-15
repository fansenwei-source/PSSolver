"""Stage O.4.3.4 attribution of remaining projected-batch materializations.

This stage is measurement-only.  It profiles the unchanged O.4.3.3 candidate,
attributes scheduler assembly work to semantic callers, and ranks the remaining
copy/cat sites.  It cannot authorize a new layout or a production default.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics

from .h100_shadow_qualification import _write_new_json
from .stage_o4_qualification import _load_json, _sha256
from .stage_o43_qualification import profile_stage_o433_h100_runtime


STAGE_O434_SHAPE = (320, 320, 80)
STAGE_O434_PROFILE_TRIALS = 3
STAGE_O434_MINIMUM_SCREENING_FRACTION = 0.03


def _finite_nonnegative(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def summarize_projected_batch_source_attribution(
    profile: Mapping[str, object],
) -> dict[str, object]:
    """Normalize scheduler counters and semantic timings per measured step."""

    try:
        steps = int(profile["configuration"]["profile_steps"])
        assembly = profile["batch_assembly_diagnostics"]
        sources = assembly["source_attribution"]
        regions = profile["semantic_regions"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("profile lacks Stage O.4.3.4 attribution inputs") from exc
    if steps <= 0 or not isinstance(sources, Mapping) or not isinstance(
        regions, Mapping
    ):
        raise ValueError("profile attribution inputs are invalid")
    if assembly.get("schema_version") != 2:
        raise ValueError("scheduler attribution schema is not version 2")

    normalized: dict[str, dict[str, object]] = {}
    for source, counters in sorted(sources.items()):
        if not isinstance(source, str) or not isinstance(counters, Mapping):
            raise ValueError("scheduler source attribution is malformed")
        timing_region = counters.get("timing_region")
        timing = regions.get(timing_region)
        if timing is None:
            total_seconds = 0.0
            timing_calls = 0
            peak_allocated = 0
            peak_reserved = 0
        elif isinstance(timing, Mapping):
            total_seconds = _finite_nonnegative(
                timing.get("total_seconds"),
                f"{source} total_seconds",
            )
            timing_calls = int(timing.get("calls", 0))
            peak_allocated = int(
                timing.get("peak_allocated_bytes_observed", 0)
            )
            peak_reserved = int(
                timing.get("peak_reserved_bytes_observed", 0)
            )
        else:
            raise ValueError(f"{source} timing region is malformed")
        if timing_calls < 0 or peak_allocated < 0 or peak_reserved < 0:
            raise ValueError(f"{source} timing counters are invalid")

        values: dict[str, int] = {}
        for mode in ("singleton", "copy_cat", "contiguous_view"):
            for quantity in (
                "batches",
                "components",
                "logical_input_bytes",
            ):
                key = f"{mode}_{quantity}"
                value = counters.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"{source} {key} is invalid")
                values[key] = value
        for mode in ("copy_cat", "contiguous_view"):
            key = f"{mode}_materialized_output_bytes"
            value = counters.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{source} {key} is invalid")
            values[key] = value
        if values["contiguous_view_materialized_output_bytes"] != 0:
            raise ValueError("a contiguous view cannot materialize output bytes")
        if counters.get("retained_tensor_references") != 0:
            raise ValueError("source attribution retained tensor references")
        fallback = counters.get("fallback_reasons")
        if not isinstance(fallback, Mapping) or any(
            not isinstance(name, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for name, count in fallback.items()
        ):
            raise ValueError(f"{source} fallback counters are invalid")

        normalized[source] = {
            "timing_region": timing_region,
            "timing_calls": timing_calls,
            "milliseconds_per_profile_step": (
                1000.0 * total_seconds / steps
            ),
            "peak_allocated_bytes_observed": peak_allocated,
            "peak_reserved_bytes_observed": peak_reserved,
            **{
                f"{key}_per_profile_step": value / steps
                for key, value in values.items()
            },
            "fallback_reasons": dict(sorted(fallback.items())),
            "retained_tensor_references": 0,
        }

    unattributed = normalized.get("unattributed")
    unattributed_batches = 0.0
    if unattributed is not None:
        unattributed_batches = sum(
            float(unattributed[f"{mode}_batches_per_profile_step"])
            for mode in ("singleton", "copy_cat", "contiguous_view")
        )
    total_copy_bytes = sum(
        float(value["copy_cat_materialized_output_bytes_per_profile_step"])
        for value in normalized.values()
    )
    total_milliseconds = sum(
        float(value["milliseconds_per_profile_step"])
        for value in normalized.values()
    )
    for value in normalized.values():
        copy_bytes = float(
            value["copy_cat_materialized_output_bytes_per_profile_step"]
        )
        milliseconds = float(value["milliseconds_per_profile_step"])
        value["copy_materialized_byte_fraction"] = (
            copy_bytes / total_copy_bytes if total_copy_bytes > 0.0 else 0.0
        )
        value["assembly_time_fraction"] = (
            milliseconds / total_milliseconds
            if total_milliseconds > 0.0
            else 0.0
        )
    return {
        "schema_version": 1,
        "profile_steps": steps,
        "sources": normalized,
        "total_copy_cat_materialized_output_bytes_per_step": total_copy_bytes,
        "total_measured_assembly_milliseconds_per_step": total_milliseconds,
        "unattributed_batches_per_step": unattributed_batches,
        "all_sources_attributed": unattributed_batches == 0.0,
        "retained_tensor_references": 0,
    }


def profile_stage_o434_h100_runtime(
    production_directory: str | Path,
    *,
    warmup_steps: int = 10,
    profile_steps: int = 20,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Profile the unchanged O.4.3.3 candidate with source attribution."""

    report = profile_stage_o433_h100_runtime(
        production_directory,
        role="candidate",
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        expected_gpu_name=expected_gpu_name,
    )
    result = dict(report)
    result.update(
        qualification_stage="O.4.3.4",
        classification="ATTRIBUTION_PROFILE_COMPLETE",
        measurement_role="candidate_attribution",
        architecture_decision="remaining_materialization_attribution",
        source_attribution=summarize_projected_batch_source_attribution(
            report
        ),
        eligible_for_stage_o44_decision=False,
        eligible_for_new_optimization_candidate=False,
        eligible_for_production_promotion=False,
        production_default_changed=False,
    )
    return result


def _validated_profile(path: str | Path) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    report = _load_json(resolved, "Stage O.4.3.4 profile")
    try:
        valid = (
            report["qualification_stage"] == "O.4.3.4"
            and report["classification"] == "ATTRIBUTION_PROFILE_COMPLETE"
            and report["measurement_role"] == "candidate_attribution"
            and tuple(report["configuration"]["shape"]) == STAGE_O434_SHAPE
            and report["finite"] is True
            and report["production_default_changed"] is False
            and report["eligible_for_production_promotion"] is False
            and report["source_attribution"]["all_sources_attributed"] is True
            and report["source_attribution"]["retained_tensor_references"] == 0
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Stage O.4.3.4 profile is incomplete")
    return resolved, report


def _require_neutral_o433_evidence(
    path: str | Path,
    expected_sha256: str,
) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    if _sha256(resolved) != expected_sha256:
        raise ValueError("Stage O.4.3.3 evidence SHA-256 differs")
    evidence = _load_json(resolved, "Stage O.4.3.3 evidence")
    try:
        valid = (
            evidence["qualification_stage"] == "O.4.3.3"
            and evidence["classification"] == "B_neutral"
            and evidence["architecture_decision"]
            == "producer_owned_boundary_packed_h_and_stress"
            and evidence["eligible_for_stage_o44_decision"] is False
            and evidence["eligible_for_production_promotion"] is False
            and evidence["production_default_changed"] is False
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("input is not the frozen neutral O.4.3.3 evidence")
    return resolved, evidence


def analyze_stage_o434_diagnostics(
    profile_paths: Sequence[str | Path],
    *,
    stage_o433_report: str | Path,
    expected_stage_o433_sha256: str,
) -> dict[str, object]:
    """Aggregate three R320 attribution profiles without authorizing a change."""

    paths_and_profiles = tuple(_validated_profile(path) for path in profile_paths)
    if len(paths_and_profiles) != STAGE_O434_PROFILE_TRIALS:
        raise ValueError("Stage O.4.3.4 requires exactly three profiles")
    evidence_path, _ = _require_neutral_o433_evidence(
        stage_o433_report,
        expected_stage_o433_sha256,
    )
    paths = tuple(path for path, _ in paths_and_profiles)
    profiles = tuple(profile for _, profile in paths_and_profiles)
    identities = {
        (
            profile["production_metadata_sha256"],
            profile["production_initial_q_sha256"],
            json.dumps(
                profile["configuration"],
                allow_nan=False,
                sort_keys=True,
            ),
        )
        for profile in profiles
    }
    if len(identities) != 1:
        raise ValueError("Stage O.4.3.4 profile identities differ")
    source_sets = {
        tuple(profile["source_attribution"]["sources"])
        for profile in profiles
    }
    if len(source_sets) != 1:
        raise ValueError("Stage O.4.3.4 source sets differ")

    rows: list[dict[str, object]] = []
    for source in next(iter(source_sets)):
        trials = tuple(
            profile["source_attribution"]["sources"][source]
            for profile in profiles
        )
        stable_fields = (
            "singleton_batches_per_profile_step",
            "singleton_components_per_profile_step",
            "copy_cat_batches_per_profile_step",
            "copy_cat_components_per_profile_step",
            "copy_cat_materialized_output_bytes_per_profile_step",
            "contiguous_view_batches_per_profile_step",
            "contiguous_view_components_per_profile_step",
            "contiguous_view_materialized_output_bytes_per_profile_step",
        )
        for field in stable_fields:
            if len({trial[field] for trial in trials}) != 1:
                raise ValueError(
                    f"Stage O.4.3.4 source {source!r} has unstable {field}"
                )
        timings = tuple(
            _finite_nonnegative(
                trial["milliseconds_per_profile_step"],
                f"{source} milliseconds_per_profile_step",
            )
            for trial in trials
        )
        copy_bytes = float(
            trials[0]["copy_cat_materialized_output_bytes_per_profile_step"]
        )
        timestep_fractions = tuple(
            float(trial["milliseconds_per_profile_step"])
            / (
                1000.0
                * float(profile["throughput"]["mean_timestep_seconds"])
            )
            for trial, profile in zip(trials, profiles, strict=True)
        )
        rows.append(
            {
                "source": source,
                "mean_milliseconds_per_step": statistics.fmean(timings),
                "median_milliseconds_per_step": statistics.median(timings),
                "sample_standard_deviation_milliseconds_per_step": (
                    statistics.stdev(timings)
                ),
                "mean_fraction_of_timestep": statistics.fmean(
                    timestep_fractions
                ),
                "copy_cat_batches_per_step": trials[0][
                    "copy_cat_batches_per_profile_step"
                ],
                "copy_cat_components_per_step": trials[0][
                    "copy_cat_components_per_profile_step"
                ],
                "copy_cat_materialized_output_bytes_per_step": copy_bytes,
                "contiguous_view_batches_per_step": trials[0][
                    "contiguous_view_batches_per_profile_step"
                ],
                "contiguous_view_components_per_step": trials[0][
                    "contiguous_view_components_per_profile_step"
                ],
                "fallback_reasons": trials[0]["fallback_reasons"],
                "retained_tensor_references": 0,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["mean_milliseconds_per_step"]),
            str(row["source"]),
        )
    )
    remaining = [
        row for row in rows if row["copy_cat_batches_per_step"] > 0.0
    ]
    dominant = remaining[0] if remaining else None
    screening_signal = bool(
        dominant is not None
        and dominant["mean_fraction_of_timestep"]
        >= STAGE_O434_MINIMUM_SCREENING_FRACTION
    )
    return {
        "schema_version": 1,
        "qualification_stage": "O.4.3.4",
        "classification": "DIAGNOSTIC_COMPLETE",
        "architecture_decision": "remaining_materialization_attribution",
        "stage_o433_evidence": {
            "path": str(evidence_path),
            "sha256": expected_stage_o433_sha256,
            "classification_preserved": "B_neutral",
        },
        "profile_paths": [str(path) for path in paths],
        "profile_sha256": [_sha256(path) for path in paths],
        "configuration": profiles[0]["configuration"],
        "source_ranking": rows,
        "remaining_copy_sources": remaining,
        "dominant_remaining_copy_source": dominant,
        "minimum_screening_fraction_of_timestep": (
            STAGE_O434_MINIMUM_SCREENING_FRACTION
        ),
        "dominant_source_meets_screening_signal": screening_signal,
        "all_sources_attributed": True,
        "counter_structure_stable_across_trials": True,
        "retained_tensor_references": 0,
        "eligible_for_target_selection_review": True,
        "eligible_for_new_optimization_candidate": False,
        "eligible_for_stage_o44_decision": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "interpretation": (
            "review the dominant measured source before defining any new "
            "candidate; this diagnostic never authorizes a layout change"
        ),
    }


def profile_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile Stage O.4.3.4 materialization attribution"
    )
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = profile_stage_o434_h100_runtime(
        args.production_reference_dir,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    return 0


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Stage O.4.3.4 materialization attribution"
    )
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--stage-o433-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o433-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_o434_diagnostics(
        args.profile,
        stage_o433_report=args.stage_o433_report,
        expected_stage_o433_sha256=args.expected_stage_o433_sha256,
    )
    _write_new_json(args.output, report)
    return 0


__all__ = [
    "STAGE_O434_MINIMUM_SCREENING_FRACTION",
    "STAGE_O434_PROFILE_TRIALS",
    "STAGE_O434_SHAPE",
    "analyze_stage_o434_diagnostics",
    "analysis_main",
    "profile_main",
    "profile_stage_o434_h100_runtime",
    "summarize_projected_batch_source_attribution",
]
