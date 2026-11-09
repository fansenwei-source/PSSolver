"""Operator/kernel attribution for the Plane production--canary gap.

Stage P is diagnostic only.  It measures non-instrumented throughput in a
separate window from bounded PyTorch operator/kernel profiling and semantic
timing, then compares three independent profiles per runtime.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

from ._shadow_support import file_sha256, require_sha256
from .h100_shadow_qualification import _write_new_json
from .stage_o4_qualification import _load_json


STAGE_P_SHAPE = (320, 320, 80)
STAGE_P_PROFILE_TRIALS = 3
STAGE_P_WARMUP_STEPS = 10
STAGE_P_THROUGHPUT_STEPS = 20
STAGE_P_OPERATOR_AUDIT_STEPS = 3
STAGE_P_SEMANTIC_STEPS = 3
STAGE_P_MAXIMUM_DISTINCT_EVENTS = 2048
STAGE_P_RUNTIME_ROLES = ("legacy_production", "separated_canary")
STAGE_P_TRANSFORM_MILLISECONDS_KEY = "transform_milliseconds_per_step"


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


def _event_number(event: object, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return _finite(value, f"profiler event {name}")
    return 0.0


def _event_device_name(event: object) -> str:
    device = getattr(event, "device_type", "cpu")
    name = getattr(device, "name", None)
    if isinstance(name, str):
        return name.lower()
    text = str(device).lower()
    return text.rsplit(".", maxsplit=1)[-1]


def _event_category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("fft", "cufft", "dct", "dst")):
        return "spectral_transform"
    if any(
        token in lowered
        for token in ("triton", "inductor", "compiledfunction", "compiled region")
    ):
        return "compiled_or_fused"
    if any(token in lowered for token in ("reduce", "sum", "mean", "norm")):
        return "reduction"
    if any(token in lowered for token in ("copy", "cat", "stack", "clone")):
        return "movement_or_assembly"
    if any(token in lowered for token in ("cuda", "memcpy", "memset")):
        return "cuda_runtime_or_memory"
    return "other"


def _ranked_entries(
    entries: Mapping[str, Mapping[str, object]],
    *,
    time_key: str,
) -> list[dict[str, object]]:
    result = [
        {"name": name, **dict(values)} for name, values in entries.items()
    ]
    result.sort(
        key=lambda item: (
            -float(item[time_key]),
            -float(item["calls_per_step"]),
            str(item["name"]),
        )
    )
    return result


def summarize_operator_kernel_events(
    events: Sequence[object],
    key_averages: Sequence[object],
    *,
    steps: int,
    maximum_distinct_events: int = STAGE_P_MAXIMUM_DISTINCT_EVENTS,
) -> dict[str, object]:
    """Reduce a bounded PyTorch trace without retaining profiler objects."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        not isinstance(maximum_distinct_events, int)
        or isinstance(maximum_distinct_events, bool)
        or maximum_distinct_events <= 0
    ):
        raise ValueError("maximum_distinct_events must be positive")

    operator_totals: dict[str, dict[str, object]] = {}
    for event in key_averages:
        if _event_device_name(event) != "cpu":
            continue
        name = str(getattr(event, "key", getattr(event, "name", "")))
        if not name:
            continue
        count = _event_number(event, "count")
        operator_totals[name] = {
            "calls_per_step": count / steps,
            "self_cpu_microseconds_per_step": _event_number(
                event,
                "self_cpu_time_total",
            )
            / steps,
            "device_microseconds_per_step": _event_number(
                event,
                "device_time_total",
                "cuda_time_total",
            )
            / steps,
            "self_device_microseconds_per_step": _event_number(
                event,
                "self_device_time_total",
                "self_cuda_time_total",
            )
            / steps,
            "category": _event_category(name),
        }

    kernel_accumulator: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0.0, "device_microseconds": 0.0}
    )
    for event in events:
        if _event_device_name(event) == "cpu":
            continue
        name = str(getattr(event, "name", getattr(event, "key", "")))
        if not name:
            continue
        kernel_accumulator[name]["calls"] += max(
            1.0,
            _event_number(event, "count"),
        )
        kernel_accumulator[name]["device_microseconds"] += _event_number(
            event,
            "device_time_total",
            "cuda_time_total",
        )
    kernel_totals = {
        name: {
            "calls_per_step": values["calls"] / steps,
            "device_microseconds_per_step": (
                values["device_microseconds"] / steps
            ),
            "category": _event_category(name),
        }
        for name, values in kernel_accumulator.items()
    }

    if len(operator_totals) > maximum_distinct_events:
        raise RuntimeError("operator trace exceeds the bounded event limit")
    if len(kernel_totals) > maximum_distinct_events:
        raise RuntimeError("kernel trace exceeds the bounded event limit")

    operator_entries = _ranked_entries(
        operator_totals,
        time_key="device_microseconds_per_step",
    )
    kernel_entries = _ranked_entries(
        kernel_totals,
        time_key="device_microseconds_per_step",
    )
    runtime_entries = [
        entry
        for entry in operator_entries
        if str(entry["name"]).lower().startswith(("cuda", "hip"))
    ]
    compiler_entries = [
        entry
        for entry in operator_entries
        if entry["category"] == "compiled_or_fused"
    ]

    category_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "operator_calls_per_step": 0.0,
            "kernel_calls_per_step": 0.0,
            "kernel_device_microseconds_per_step": 0.0,
        }
    )
    for entry in operator_entries:
        category_totals[str(entry["category"])][
            "operator_calls_per_step"
        ] += float(entry["calls_per_step"])
    for entry in kernel_entries:
        category = category_totals[str(entry["category"])]
        category["kernel_calls_per_step"] += float(entry["calls_per_step"])
        category["kernel_device_microseconds_per_step"] += float(
            entry["device_microseconds_per_step"]
        )

    return {
        "schema_version": 1,
        "steps": steps,
        "bounded_aggregate_only": True,
        "raw_trace_retained": False,
        "operator_entries_complete": True,
        "kernel_entries_complete": True,
        "operators": {
            "distinct_names": len(operator_entries),
            "total_calls_per_step": math.fsum(
                float(entry["calls_per_step"]) for entry in operator_entries
            ),
            "entries": operator_entries,
        },
        "kernels": {
            "distinct_names": len(kernel_entries),
            "total_launches_per_step": math.fsum(
                float(entry["calls_per_step"]) for entry in kernel_entries
            ),
            "total_device_microseconds_per_step": math.fsum(
                float(entry["device_microseconds_per_step"])
                for entry in kernel_entries
            ),
            "entries": kernel_entries,
        },
        "cuda_runtime": {
            "calls_per_step": math.fsum(
                float(entry["calls_per_step"]) for entry in runtime_entries
            ),
            "entries": runtime_entries,
        },
        "compiler_markers": {
            "calls_per_step": math.fsum(
                float(entry["calls_per_step"]) for entry in compiler_entries
            ),
            "entries": compiler_entries,
        },
        "category_totals": {
            name: dict(values) for name, values in sorted(category_totals.items())
        },
    }


def _validated_profile(
    path: str | Path,
    *,
    role: str,
    expected_closure_sha256: str,
) -> tuple[Path, str, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    digest = file_sha256(resolved)
    report = _load_json(resolved, f"Stage P {role} profile")
    if file_sha256(resolved) != digest:
        raise RuntimeError(f"Stage P {role} profile changed while it was read")
    try:
        config = report["configuration"]
        audit = report["operator_kernel_audit"]
        contract = report["measurement_contract"]
        valid = (
            report["qualification_stage"] == "P"
            and report["classification"]
            == "OPERATOR_KERNEL_PROFILE_COMPLETE"
            and report["runtime_role"] == role
            and report["stage_o_closure_sha256"] == expected_closure_sha256
            and tuple(config["shape"]) == STAGE_P_SHAPE
            and config["warmup_steps"] == STAGE_P_WARMUP_STEPS
            and config["throughput_steps"] == STAGE_P_THROUGHPUT_STEPS
            and config["operator_audit_steps"]
            == STAGE_P_OPERATOR_AUDIT_STEPS
            and config["semantic_steps"] == STAGE_P_SEMANTIC_STEPS
            and config["dtype"] == "float64"
            and report["finite"] is True
            and report["production_default_changed"] is False
            and report["eligible_for_production_promotion"] is False
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
            and contract["changes_runtime_implementation"] is False
            and contract["changes_production_default"] is False
            and report["eligible_for_stage_q_optimization_candidate"] is False
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"Stage P {role} profile is incomplete")
    return resolved, digest, report


def _aggregate_entries(
    reports: Sequence[Mapping[str, object]],
    section: str,
) -> dict[str, dict[str, float | str]]:
    by_name: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for report in reports:
        entries = report["operator_kernel_audit"][section]["entries"]
        if not isinstance(entries, Sequence):
            raise ValueError(f"Stage P {section} entries are invalid")
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("name"), str
            ):
                raise ValueError(f"Stage P {section} entry is invalid")
            by_name[str(entry["name"])].append(entry)
    result = {}
    trial_count = len(reports)
    for name, values in by_name.items():
        calls = [
            _nonnegative(
                value["calls_per_step"],
                f"Stage P {section} {name} calls_per_step",
            )
            for value in values
        ]
        times = [
            _nonnegative(
                value["device_microseconds_per_step"],
                f"Stage P {section} {name} device time",
            )
            for value in values
        ]
        categories = {str(value["category"]) for value in values}
        if len(categories) != 1:
            raise ValueError(f"Stage P {section} category changed for {name}")
        result[name] = {
            "category": categories.pop(),
            "mean_calls_per_step": math.fsum(calls) / trial_count,
            "mean_device_microseconds_per_step": (
                math.fsum(times) / trial_count
            ),
        }
    return result


def _entry_delta(
    production: Mapping[str, Mapping[str, float | str]],
    canary: Mapping[str, Mapping[str, float | str]],
) -> list[dict[str, object]]:
    rows = []
    for name in sorted(set(production) | set(canary)):
        left = production.get(name, {})
        right = canary.get(name, {})
        p_calls = float(left.get("mean_calls_per_step", 0.0))
        c_calls = float(right.get("mean_calls_per_step", 0.0))
        p_time = float(left.get("mean_device_microseconds_per_step", 0.0))
        c_time = float(right.get("mean_device_microseconds_per_step", 0.0))
        rows.append(
            {
                "name": name,
                "category": str(
                    right.get("category", left.get("category", "other"))
                ),
                "production_calls_per_step": p_calls,
                "canary_calls_per_step": c_calls,
                "call_delta_per_step": c_calls - p_calls,
                "production_device_microseconds_per_step": p_time,
                "canary_device_microseconds_per_step": c_time,
                "device_microseconds_delta_per_step": c_time - p_time,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["device_microseconds_delta_per_step"]),
            -float(row["call_delta_per_step"]),
            str(row["name"]),
        )
    )
    return rows


def _mean_profile_value(
    reports: Sequence[Mapping[str, object]],
    *path: str,
) -> float:
    values = []
    for report in reports:
        value: object = report
        for name in path:
            if not isinstance(value, Mapping):
                raise ValueError(f"Stage P profile path {path!r} is invalid")
            value = value[name]
        values.append(_nonnegative(value, ".".join(path)))
    return math.fsum(values) / len(values)


def analyze_stage_p_diagnostics(
    production_profile_paths: Sequence[str | Path],
    canary_profile_paths: Sequence[str | Path],
    *,
    expected_stage_o_closure_sha256: str,
) -> dict[str, object]:
    """Compare matched Stage P profiles without selecting an optimization."""

    require_sha256(
        expected_stage_o_closure_sha256,
        "Stage O closure SHA-256",
    )
    if len(production_profile_paths) != STAGE_P_PROFILE_TRIALS or len(
        canary_profile_paths
    ) != STAGE_P_PROFILE_TRIALS:
        raise ValueError("Stage P requires exactly three profiles per runtime")
    input_paths = tuple(
        Path(path).expanduser().resolve()
        for path in (*production_profile_paths, *canary_profile_paths)
    )
    if len(set(input_paths)) != 2 * STAGE_P_PROFILE_TRIALS:
        raise ValueError("Stage P requires six distinct profile paths")
    production_loaded = tuple(
        _validated_profile(
            path,
            role="legacy_production",
            expected_closure_sha256=expected_stage_o_closure_sha256,
        )
        for path in production_profile_paths
    )
    canary_loaded = tuple(
        _validated_profile(
            path,
            role="separated_canary",
            expected_closure_sha256=expected_stage_o_closure_sha256,
        )
        for path in canary_profile_paths
    )
    production = tuple(report for _, _, report in production_loaded)
    canary = tuple(report for _, _, report in canary_loaded)
    all_reports = (*production, *canary)
    identities = {
        json.dumps(
            {
                "configuration": report["configuration"],
                "production_metadata_sha256": report[
                    "production_metadata_sha256"
                ],
                "production_initial_q_sha256": report[
                    "production_initial_q_sha256"
                ],
                "stage_o_closure_sha256": report[
                    "stage_o_closure_sha256"
                ],
            },
            allow_nan=False,
            sort_keys=True,
        )
        for report in all_reports
    }
    if len(identities) != 1:
        raise ValueError("Stage P production/canary profile identities differ")
    if any(
        int(report["dynamo_delta"]["graph_breaks"] or 0) != 0
        for report in all_reports
    ):
        raise ValueError("Stage P profile contains a graph break")

    production_mean = _mean_profile_value(
        production,
        "throughput",
        "mean_timestep_seconds",
    )
    canary_mean = _mean_profile_value(
        canary,
        "throughput",
        "mean_timestep_seconds",
    )
    if production_mean <= 0.0 or canary_mean <= 0.0:
        raise ValueError("Stage P throughput must be positive")

    production_operators = _aggregate_entries(production, "operators")
    canary_operators = _aggregate_entries(canary, "operators")
    production_kernels = _aggregate_entries(production, "kernels")
    canary_kernels = _aggregate_entries(canary, "kernels")
    operator_deltas = _entry_delta(production_operators, canary_operators)
    kernel_deltas = _entry_delta(production_kernels, canary_kernels)

    production_launches = _mean_profile_value(
        production,
        "operator_kernel_audit",
        "kernels",
        "total_launches_per_step",
    )
    canary_launches = _mean_profile_value(
        canary,
        "operator_kernel_audit",
        "kernels",
        "total_launches_per_step",
    )
    production_transform_ms = _mean_profile_value(
        production,
        "matched_semantic_regions",
        STAGE_P_TRANSFORM_MILLISECONDS_KEY,
    )
    canary_transform_ms = _mean_profile_value(
        canary,
        "matched_semantic_regions",
        STAGE_P_TRANSFORM_MILLISECONDS_KEY,
    )
    production_runtime_calls = _mean_profile_value(
        production,
        "operator_kernel_audit",
        "cuda_runtime",
        "calls_per_step",
    )
    canary_runtime_calls = _mean_profile_value(
        canary,
        "operator_kernel_audit",
        "cuda_runtime",
        "calls_per_step",
    )
    production_compiled_calls = _mean_profile_value(
        production,
        "operator_kernel_audit",
        "compiler_markers",
        "calls_per_step",
    )
    canary_compiled_calls = _mean_profile_value(
        canary,
        "operator_kernel_audit",
        "compiler_markers",
        "calls_per_step",
    )

    return {
        "schema_version": 1,
        "qualification_stage": "P",
        "classification": "DIAGNOSTIC_COMPLETE",
        "architecture_decision": (
            "production_canary_operator_kernel_gap_attribution"
        ),
        "trial_count_per_runtime": STAGE_P_PROFILE_TRIALS,
        "stage_o_closure_sha256": expected_stage_o_closure_sha256,
        "throughput": {
            "production_mean_timestep_seconds": production_mean,
            "canary_mean_timestep_seconds": canary_mean,
            "canary_over_production_ratio": canary_mean / production_mean,
            "gap_milliseconds_per_step": 1000.0
            * (canary_mean - production_mean),
            "noninstrumented_window": True,
        },
        "gap_signals": {
            "kernel_launches": {
                "production_per_step": production_launches,
                "canary_per_step": canary_launches,
                "delta_per_step": canary_launches - production_launches,
            },
            "cuda_runtime_calls": {
                "production_per_step": production_runtime_calls,
                "canary_per_step": canary_runtime_calls,
                "delta_per_step": canary_runtime_calls
                - production_runtime_calls,
            },
            "compiled_marker_calls": {
                "production_per_step": production_compiled_calls,
                "canary_per_step": canary_compiled_calls,
                "delta_per_step": canary_compiled_calls
                - production_compiled_calls,
            },
            "semantic_transform_time": {
                "production_milliseconds_per_step": production_transform_ms,
                "canary_milliseconds_per_step": canary_transform_ms,
                "delta_milliseconds_per_step": canary_transform_ms
                - production_transform_ms,
                "separate_instrumented_window": True,
            },
        },
        "positive_operator_device_time_deltas": [
            row
            for row in operator_deltas
            if float(row["device_microseconds_delta_per_step"]) > 0.0
        ],
        "positive_kernel_device_time_deltas": [
            row
            for row in kernel_deltas
            if float(row["device_microseconds_delta_per_step"]) > 0.0
        ],
        "all_operator_deltas": operator_deltas,
        "all_kernel_deltas": kernel_deltas,
        "interpretation_constraints": {
            "throughput_profile_and_semantic_windows_are_separate": True,
            "operator_kernel_profile_is_not_authoritative_throughput": True,
            "kernel_durations_may_overlap": True,
            "semantic_regions_are_nested": True,
            "automatic_target_selection_forbidden": True,
            "materialization_layout_line_remains_closed": True,
        },
        "eligible_for_stage_q_target_review": True,
        "eligible_for_stage_q_optimization_candidate": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "inputs": [
            {"path": str(path), "sha256": digest}
            for path, digest, _ in (*production_loaded, *canary_loaded)
        ],
    }


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Stage P production/canary operator profiles."
    )
    parser.add_argument(
        "--production-profile",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--canary-profile",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--expected-stage-o-closure-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_p_diagnostics(
        args.production_profile,
        args.canary_profile,
        expected_stage_o_closure_sha256=(
            args.expected_stage_o_closure_sha256
        ),
    )
    _write_new_json(args.output, report)
    return 0


__all__ = [
    "STAGE_P_MAXIMUM_DISTINCT_EVENTS",
    "STAGE_P_OPERATOR_AUDIT_STEPS",
    "STAGE_P_PROFILE_TRIALS",
    "STAGE_P_RUNTIME_ROLES",
    "STAGE_P_SEMANTIC_STEPS",
    "STAGE_P_SHAPE",
    "STAGE_P_THROUGHPUT_STEPS",
    "STAGE_P_TRANSFORM_MILLISECONDS_KEY",
    "STAGE_P_WARMUP_STEPS",
    "analysis_main",
    "analyze_stage_p_diagnostics",
    "summarize_operator_kernel_events",
]
