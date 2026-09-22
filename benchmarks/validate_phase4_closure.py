"""Validate immutable P4.6 H100 SBDF2/modal-block profiler evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


CONTINUOUS = "A_continuous"
REBOUND = "B_rebound"
VARIANTS = (CONTINUOUS, REBOUND)
EXPECTED_POINT_COUNTS = (131_072, 1_048_576)


class ClosureEvidenceError(RuntimeError):
    """Raised when P4.6 evidence violates the frozen closure contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ClosureEvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_evidence(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ClosureEvidenceError(f"non-finite JSON constant: {value}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClosureEvidenceError(f"could not load evidence: {exc}") from exc
    if not isinstance(payload, dict):
        raise ClosureEvidenceError("evidence must be a JSON object")
    return payload


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClosureEvidenceError(f"{description} must be an object")
    return value


def _list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise ClosureEvidenceError(f"{description} must be an array")
    return value


def _finite(value: object, description: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise ClosureEvidenceError(f"{description} must be {qualifier}")
    return float(value)


def _exact(value: object, expected: object, description: str) -> None:
    if value != expected:
        raise ClosureEvidenceError(
            f"{description} mismatch: observed={value!r}, expected={expected!r}"
        )


def _close(observed: float, expected: float, description: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ClosureEvidenceError(
            f"{description} mismatch: observed={observed}, expected={expected}"
        )


def _validate_memory(record: dict[str, Any], description: str) -> dict[str, float]:
    memory = _mapping(record.get("memory"), f"{description}.memory")
    keys = (
        "starting_allocated_bytes",
        "starting_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
    )
    parsed = {
        key: _finite(memory.get(key), f"{description}.memory.{key}")
        for key in keys
    }
    if any(value < 0.0 for value in parsed.values()):
        raise ClosureEvidenceError(f"{description}.memory contains negative bytes")
    _close(
        parsed["incremental_peak_allocated_bytes"],
        parsed["peak_allocated_bytes"] - parsed["starting_allocated_bytes"],
        f"{description}.incremental allocated",
    )
    _close(
        parsed["incremental_peak_reserved_bytes"],
        parsed["peak_reserved_bytes"] - parsed["starting_reserved_bytes"],
        f"{description}.incremental reserved",
    )
    return parsed


def _validate_trials(
    row: dict[str, Any],
    variant: str,
) -> tuple[list[float], list[dict[str, float]]]:
    records = _list(
        _mapping(row.get("trial_records"), "trial_records").get(variant),
        f"trial_records.{variant}",
    )
    if len(records) != 3:
        raise ClosureEvidenceError(f"{variant} must contain three trials")
    expected_orders = [1, 2, 1] if variant == CONTINUOUS else [2, 1, 2]
    timings = []
    memories = []
    for index, raw in enumerate(records):
        record = _mapping(raw, f"{variant} trial")
        _exact(record.get("trial"), index + 1, f"{variant} trial identity")
        _exact(record.get("order"), expected_orders[index], f"{variant} order")
        _exact(record.get("workspace_pointer_stable"), True, "workspace stable")
        _exact(record.get("finite"), True, "finite")
        _exact(record.get("workspace_allocated_tensor_count"), 7, "workspace count")
        _exact(record.get("implementation"), "closed_form_2x2", "implementation")
        timings.append(
            _finite(
                record.get("milliseconds_per_step"),
                f"{variant} milliseconds_per_step",
                positive=True,
            )
        )
        memories.append(_validate_memory(record, f"{variant} trial {index + 1}"))
    return timings, memories


def _validate_aggregate(
    row: dict[str, Any],
    variant: str,
    timings: list[float],
) -> dict[str, float]:
    aggregate = _mapping(
        _mapping(row.get("aggregate"), "aggregate").get(variant),
        f"aggregate.{variant}",
    )
    expected = {
        "mean_milliseconds_per_step": statistics.fmean(timings),
        "median_milliseconds_per_step": statistics.median(timings),
        "sample_standard_deviation_milliseconds_per_step": statistics.stdev(
            timings
        ),
    }
    _exact(aggregate.get("trials"), 3, f"aggregate.{variant}.trials")
    for key, value in expected.items():
        _close(
            _finite(aggregate.get(key), f"aggregate.{variant}.{key}"),
            value,
            f"aggregate.{variant}.{key}",
        )
    return expected


def _validate_row(
    row: dict[str, Any],
    point_count: int,
    *,
    maximum_timing_ratio: float,
    maximum_memory_ratio: float,
) -> dict[str, object]:
    _exact(row.get("point_count"), point_count, "point_count")
    timings: dict[str, list[float]] = {}
    memories: dict[str, list[dict[str, float]]] = {}
    aggregates: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        timings[variant], memories[variant] = _validate_trials(row, variant)
        aggregates[variant] = _validate_aggregate(
            row,
            variant,
            timings[variant],
        )

    comparisons = _list(row.get("comparisons"), "comparisons")
    if len(comparisons) != 3:
        raise ClosureEvidenceError("comparisons must contain three trials")
    for index, raw in enumerate(comparisons):
        comparison = _mapping(raw, "comparison")
        _exact(comparison.get("trial"), index + 1, "comparison trial")
        _exact(comparison.get("physical_byte_identical"), True, "physical identity")
        _exact(comparison.get("spectrum_byte_identical"), True, "spectrum identity")
        _exact(comparison.get("physical_relative_l2"), 0.0, "physical relative L2")
        _exact(comparison.get("physical_linf"), 0.0, "physical Linf")

    median_ratio = (
        aggregates[REBOUND]["median_milliseconds_per_step"]
        / aggregates[CONTINUOUS]["median_milliseconds_per_step"]
    )
    _close(
        _finite(
            row.get("rebound_over_continuous_median_ratio"),
            "median ratio",
            positive=True,
        ),
        median_ratio,
        "median ratio",
    )
    if median_ratio > maximum_timing_ratio:
        raise ClosureEvidenceError("rebound timing exceeds the closure gate")

    paired = _list(
        row.get("paired_rebound_over_continuous_ratios"),
        "paired ratios",
    )
    expected_paired = [
        rebound / continuous
        for continuous, rebound in zip(
            timings[CONTINUOUS], timings[REBOUND], strict=True
        )
    ]
    if len(paired) != 3:
        raise ClosureEvidenceError("paired ratios must contain three trials")
    for observed, expected in zip(paired, expected_paired, strict=True):
        _close(
            _finite(observed, "paired ratio", positive=True),
            expected,
            "paired ratio",
        )

    allocated_ratio = max(
        memory["peak_allocated_bytes"] for memory in memories[REBOUND]
    ) / max(memory["peak_allocated_bytes"] for memory in memories[CONTINUOUS])
    reserved_ratio = max(
        memory["peak_reserved_bytes"] for memory in memories[REBOUND]
    ) / max(memory["peak_reserved_bytes"] for memory in memories[CONTINUOUS])
    if allocated_ratio > maximum_memory_ratio or reserved_ratio > maximum_memory_ratio:
        raise ClosureEvidenceError("rebound memory exceeds the closure gate")
    return {
        "point_count": point_count,
        "median_timing_ratio": median_ratio,
        "peak_allocated_ratio": allocated_ratio,
        "peak_reserved_ratio": reserved_ratio,
        "byte_identical": True,
    }


def validate_evidence(
    payload: dict[str, Any],
    *,
    expected_commit: str,
) -> dict[str, object]:
    _exact(payload.get("schema_version"), 2, "schema version")
    _exact(
        payload.get("identity"),
        "p4_6_combined_sbdf2_modal_block_profile",
        "identity",
    )
    measurement = _mapping(
        payload.get("memory_measurement"),
        "memory_measurement",
    )
    _exact(
        measurement.get("role_live_sets_isolated"),
        True,
        "role live-set isolation",
    )
    _exact(
        measurement.get("comparison_snapshots_device"),
        "cpu",
        "comparison snapshot device",
    )
    _exact(
        measurement.get("unused_allocator_cache_cleared_before_timing"),
        True,
        "allocator cache isolation",
    )
    _exact(
        measurement.get("peak_scope"),
        "current_role_only",
        "memory peak scope",
    )
    config = _mapping(payload.get("config"), "config")
    _exact(config.get("point_counts"), list(EXPECTED_POINT_COUNTS), "point counts")
    _exact(config.get("device"), "cuda", "device")
    _exact(config.get("dtype"), "float64", "dtype")
    _exact(config.get("dt"), 0.005, "dt")
    _exact(config.get("warmup_steps"), 5, "warmup steps")
    _exact(config.get("measured_steps"), 30, "measured steps")
    _exact(config.get("trials"), 3, "trials")
    maximum_timing_ratio = _finite(
        config.get("maximum_timing_ratio"),
        "maximum timing ratio",
        positive=True,
    )
    maximum_memory_ratio = _finite(
        config.get("maximum_memory_ratio"),
        "maximum memory ratio",
        positive=True,
    )
    _exact(maximum_timing_ratio, 1.10, "maximum timing ratio")
    _exact(maximum_memory_ratio, 1.10, "maximum memory ratio")

    environment = _mapping(payload.get("environment"), "environment")
    _exact(environment.get("git_head"), expected_commit, "git head")
    _exact(environment.get("cuda_available"), True, "CUDA available")
    _exact(environment.get("requested_device"), "cuda", "requested device")
    if "H100" not in str(environment.get("device_name")):
        raise ClosureEvidenceError("profile was not measured on an H100")
    _exact(environment.get("tf32_matmul_effective"), False, "matmul TF32")
    _exact(environment.get("tf32_cudnn_effective"), False, "cuDNN TF32")

    rows = _list(payload.get("rows"), "rows")
    if len(rows) != len(EXPECTED_POINT_COUNTS):
        raise ClosureEvidenceError("row count is incomplete")
    validated_rows = [
        _validate_row(
            _mapping(raw, "row"),
            point_count,
            maximum_timing_ratio=maximum_timing_ratio,
            maximum_memory_ratio=maximum_memory_ratio,
        )
        for raw, point_count in zip(rows, EXPECTED_POINT_COUNTS, strict=True)
    ]

    convergence = _mapping(payload.get("convergence"), "convergence")
    _exact(convergence.get("passed"), True, "convergence passed")
    minimum_order = _finite(
        convergence.get("minimum_observed_l2_order"),
        "minimum observed order",
    )
    if minimum_order < 1.8:
        raise ClosureEvidenceError("SBDF2 convergence order is below 1.8")
    orders = _list(convergence.get("pairwise_l2_orders"), "pairwise orders")
    if len(orders) != 3 or any(
        _finite(value, "pairwise order") < 1.8 for value in orders
    ):
        raise ClosureEvidenceError("pairwise convergence orders are invalid")

    eligibility = _mapping(payload.get("eligibility"), "eligibility")
    _exact(eligibility.get("profile_complete"), True, "profile complete")
    _exact(
        eligibility.get("production_default_changed"),
        False,
        "production default",
    )
    _exact(eligibility.get("phase_5_authorized"), False, "Phase 5 authority")
    return {
        "schema_version": 2,
        "classification": "PASS_P4_6_H100_PROFILE",
        "expected_commit": expected_commit,
        "rows": validated_rows,
        "minimum_observed_l2_order": minimum_order,
        "gates": {
            "h100_identity": True,
            "tf32_disabled": True,
            "finite": True,
            "workspace_identity": True,
            "workspace_tensor_count": 7,
            "continuous_rebound_byte_identity": True,
            "performance": True,
            "memory": True,
            "memory_role_isolation": True,
            "convergence": True,
        },
        "eligibility": {
            "profile_passed": True,
            "phase_4_complete": False,
            "phase_5_authorized": False,
            "production_default_changed": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite")
    return args


def main() -> None:
    args = parse_args()
    result = validate_evidence(
        load_evidence(args.input),
        expected_commit=args.expected_commit,
    )
    serialized = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
