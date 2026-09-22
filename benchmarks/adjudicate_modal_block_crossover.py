"""Adjudicate existing P4.4 H100 operator-contract crossover evidence.

The adjudicator is analysis-only.  It never imports or executes a modal
operator, never selects an automatic implementation, and never launches GPU
work.  It recomputes the registered numerical, timing, trial-completeness, and
memory gates from one immutable profiler JSON before authorizing P4.5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "7619c58c931eda89ff6613936851d8e7be1f52b9"
EXPECTED_MODE_COUNTS = (
    131_072,
    262_144,
    524_288,
    1_048_576,
    2_097_152,
    4_194_304,
)
REFERENCE = "A_torch_linalg"
CANDIDATE = "B_closed_form_2x2"
NUMERICAL_TOLERANCE = 1.0e-12
MAXIMUM_TIMING_RATIO = 1.05
MAXIMUM_MEMORY_RATIO = 1.10


class EvidenceError(RuntimeError):
    """Raised when immutable profiler evidence violates the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_evidence(path: Path) -> dict[str, Any]:
    """Load one finite JSON object while rejecting duplicate keys."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON constant: {value}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"could not load evidence: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceError("profiler evidence must be a JSON object")
    return payload


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{description} must be an object")
    return value


def _list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceError(f"{description} must be an array")
    return value


def _finite(value: object, description: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise EvidenceError(f"{description} must be {qualifier}")
    return float(value)


def _exact(value: object, expected: object, description: str) -> None:
    if value != expected:
        raise EvidenceError(
            f"{description} mismatch: observed={value!r}, expected={expected!r}"
        )


def _close(observed: float, expected: float, description: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise EvidenceError(
            f"{description} mismatch: observed={observed}, expected={expected}"
        )


def _trial_values(
    row: dict[str, Any],
    variant: str,
) -> list[float]:
    records = _list(
        _mapping(row.get("trial_records"), "trial_records").get(variant),
        f"trial_records.{variant}",
    )
    if len(records) != 3:
        raise EvidenceError(f"{variant} must contain exactly three trials")
    parsed: list[tuple[int, float]] = []
    for record in records:
        item = _mapping(record, f"{variant} trial")
        trial = item.get("trial")
        order = item.get("order")
        if trial not in {1, 2, 3} or order not in {1, 2}:
            raise EvidenceError(f"{variant} trial or balanced order is invalid")
        parsed.append(
            (
                int(trial),
                _finite(
                    item.get("mean_milliseconds"),
                    f"{variant} mean_milliseconds",
                    positive=True,
                ),
            )
        )
    if {trial for trial, _ in parsed} != {1, 2, 3}:
        raise EvidenceError(f"{variant} trial identities are incomplete")
    return [value for _, value in sorted(parsed)]


def _validate_aggregate(
    row: dict[str, Any],
    variant: str,
    values: list[float],
) -> dict[str, float | int]:
    aggregate = _mapping(
        _mapping(row.get("aggregate"), "aggregate").get(variant),
        f"aggregate.{variant}",
    )
    expected = {
        "trials": 3,
        "mean_milliseconds": statistics.fmean(values),
        "median_milliseconds": statistics.median(values),
        "sample_standard_deviation_milliseconds": statistics.stdev(values),
    }
    _exact(aggregate.get("trials"), 3, f"{variant} aggregate trials")
    for key in (
        "mean_milliseconds",
        "median_milliseconds",
        "sample_standard_deviation_milliseconds",
    ):
        _close(
            _finite(aggregate.get(key), f"{variant} aggregate {key}"),
            float(expected[key]),
            f"{variant} aggregate {key}",
        )
    return expected


def _validate_memory(row: dict[str, Any]) -> dict[str, float]:
    memory = _mapping(row.get("memory"), "memory")
    parsed: dict[str, dict[str, float]] = {}
    keys = (
        "starting_allocated_bytes",
        "starting_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
    )
    for variant in (REFERENCE, CANDIDATE):
        record = _mapping(memory.get(variant), f"memory.{variant}")
        parsed[variant] = {
            key: _finite(record.get(key), f"memory.{variant}.{key}")
            for key in keys
        }
        if any(value < 0.0 for value in parsed[variant].values()):
            raise EvidenceError(f"memory.{variant} contains a negative byte count")
        if (
            parsed[variant]["peak_allocated_bytes"]
            < parsed[variant]["starting_allocated_bytes"]
            or parsed[variant]["peak_reserved_bytes"]
            < parsed[variant]["starting_reserved_bytes"]
        ):
            raise EvidenceError(f"memory.{variant} peak precedes its starting value")
        _close(
            parsed[variant]["incremental_peak_allocated_bytes"],
            parsed[variant]["peak_allocated_bytes"]
            - parsed[variant]["starting_allocated_bytes"],
            f"memory.{variant}.incremental_peak_allocated_bytes",
        )
        _close(
            parsed[variant]["incremental_peak_reserved_bytes"],
            parsed[variant]["peak_reserved_bytes"]
            - parsed[variant]["starting_reserved_bytes"],
            f"memory.{variant}.incremental_peak_reserved_bytes",
        )

    reference = parsed[REFERENCE]
    candidate = parsed[CANDIDATE]
    allocated_ratio = (
        candidate["peak_allocated_bytes"] / reference["peak_allocated_bytes"]
    )
    reserved_ratio = (
        candidate["peak_reserved_bytes"] / reference["peak_reserved_bytes"]
    )
    if allocated_ratio > MAXIMUM_MEMORY_RATIO:
        raise EvidenceError("candidate peak allocated memory exceeds the gate")
    if reserved_ratio > MAXIMUM_MEMORY_RATIO:
        raise EvidenceError("candidate peak reserved memory exceeds the gate")
    return {
        "candidate_over_reference_peak_allocated_ratio": allocated_ratio,
        "candidate_over_reference_peak_reserved_ratio": reserved_ratio,
    }


def _validate_row(row: dict[str, Any], mode_count: int) -> dict[str, object]:
    _exact(row.get("mode_count"), mode_count, "row mode_count")
    correctness = _mapping(row.get("correctness"), "correctness")
    numerical = {}
    for key in ("relative_l2", "linf", "residual_relative_l2"):
        value = _finite(correctness.get(key), f"correctness.{key}")
        if value > NUMERICAL_TOLERANCE:
            raise EvidenceError(
                f"mode_count={mode_count} correctness.{key} exceeds tolerance"
            )
        numerical[key] = value
    _exact(correctness.get("finite"), True, "correctness finite")
    _exact(
        correctness.get("reference_workspace_pointer_stable"),
        True,
        "reference workspace pointer stability",
    )
    _exact(
        correctness.get("candidate_workspace_pointer_stable"),
        True,
        "candidate workspace pointer stability",
    )

    reference_values = _trial_values(row, REFERENCE)
    candidate_values = _trial_values(row, CANDIDATE)
    reference_aggregate = _validate_aggregate(row, REFERENCE, reference_values)
    candidate_aggregate = _validate_aggregate(row, CANDIDATE, candidate_values)
    median_ratio = (
        float(candidate_aggregate["median_milliseconds"])
        / float(reference_aggregate["median_milliseconds"])
    )
    _close(
        _finite(
            row.get("candidate_over_reference_median_ratio"),
            "candidate_over_reference_median_ratio",
            positive=True,
        ),
        median_ratio,
        "candidate_over_reference_median_ratio",
    )
    paired = _list(
        row.get("paired_candidate_over_reference_ratios"),
        "paired_candidate_over_reference_ratios",
    )
    expected_paired = [
        candidate / reference
        for reference, candidate in zip(
            reference_values,
            candidate_values,
            strict=True,
        )
    ]
    if len(paired) != 3:
        raise EvidenceError("paired timing ratios must contain exactly three values")
    for index, (observed, expected) in enumerate(
        zip(paired, expected_paired, strict=True),
        start=1,
    ):
        _close(
            _finite(observed, f"paired ratio {index}", positive=True),
            expected,
            f"paired ratio {index}",
        )
    if median_ratio > MAXIMUM_TIMING_RATIO:
        raise EvidenceError(
            f"mode_count={mode_count} median performance ratio exceeds the gate"
        )
    if any(value > MAXIMUM_TIMING_RATIO for value in expected_paired):
        raise EvidenceError(
            f"mode_count={mode_count} has a paired trial exceeding the gate"
        )

    return {
        "mode_count": mode_count,
        **numerical,
        "reference_median_milliseconds": reference_aggregate[
            "median_milliseconds"
        ],
        "candidate_median_milliseconds": candidate_aggregate[
            "median_milliseconds"
        ],
        "candidate_over_reference_median_ratio": median_ratio,
        "paired_candidate_over_reference_ratios": expected_paired,
        **_validate_memory(row),
    }


def adjudicate_evidence(
    payload: dict[str, Any],
    *,
    input_sha256: str,
) -> dict[str, Any]:
    """Recompute the corrected operator-contract qualification."""

    _exact(payload.get("schema_version"), 1, "schema_version")
    _exact(
        payload.get("identity"),
        "p4_4_modal_block_crossover_profile",
        "identity",
    )
    _exact(
        payload.get("scope"),
        "evidence_only_no_production_selection",
        "scope",
    )
    config = _mapping(payload.get("config"), "config")
    _exact(config.get("mode_counts"), list(EXPECTED_MODE_COUNTS), "mode_counts")
    _exact(config.get("device"), "cuda", "config.device")
    _exact(config.get("dtype"), "complex128", "config.dtype")
    _exact(config.get("alpha"), 4.0, "config.alpha")
    _exact(config.get("warmup"), 5, "config.warmup")
    _exact(config.get("repeats"), 30, "config.repeats")
    _exact(config.get("trials"), 3, "config.trials")
    _exact(config.get("seed"), 20260922, "config.seed")
    _exact(
        config.get("maximum_acceptable_median_ratio"),
        MAXIMUM_TIMING_RATIO,
        "config.maximum_acceptable_median_ratio",
    )

    environment = _mapping(payload.get("environment"), "environment")
    _exact(environment.get("git_head"), EXPECTED_COMMIT, "environment.git_head")
    if not str(environment.get("device", "")).startswith("cuda"):
        raise EvidenceError("environment.device must be CUDA")
    _exact(
        environment.get("device_name"),
        "NVIDIA H100 PCIe",
        "environment.device_name",
    )
    _exact(
        environment.get("tf32_matmul_effective"),
        False,
        "TF32 matmul state",
    )
    _exact(
        environment.get("tf32_cudnn_effective"),
        False,
        "TF32 cuDNN state",
    )

    rows = _list(payload.get("rows"), "rows")
    if len(rows) != len(EXPECTED_MODE_COUNTS):
        raise EvidenceError("rows do not contain the six registered mode counts")
    results = [
        _validate_row(_mapping(row, "row"), mode_count)
        for row, mode_count in zip(rows, EXPECTED_MODE_COUNTS, strict=True)
    ]

    crossover = _mapping(payload.get("crossover"), "crossover")
    _exact(
        crossover.get("recommended_switch_at_mode_count"),
        EXPECTED_MODE_COUNTS[0],
        "raw recommended switch",
    )
    _exact(
        crossover.get("eligible_for_auto_policy_implementation"),
        True,
        "raw profiler eligibility",
    )
    _exact(
        crossover.get("production_default_changed"),
        False,
        "raw production default state",
    )
    _exact(
        crossover.get("p4_5_authorized"),
        False,
        "raw P4.5 state",
    )

    return {
        "schema_version": 1,
        "classification": (
            "PASS_P4_4_MODAL_BLOCK_H100_CONTRACT_EQUIVALENT"
        ),
        "input": {
            "identity": payload["identity"],
            "sha256": input_sha256,
            "git_head": environment["git_head"],
            "device_name": environment["device_name"],
        },
        "contract_correction": {
            "authoritative_scope": "complete_operator_contract",
            "superseded_gate": "reproduce_non_equivalent_bare_kernel_low_endpoint",
            "old_low_endpoint_direction_required": False,
            "reason": (
                "the earlier reference and the complete operator-contract "
                "reference measured different execution scopes"
            ),
            "auto_policy_required": False,
        },
        "gates": {
            "registered_mode_counts": list(EXPECTED_MODE_COUNTS),
            "numerical_tolerance": NUMERICAL_TOLERANCE,
            "maximum_timing_ratio": MAXIMUM_TIMING_RATIO,
            "maximum_memory_ratio": MAXIMUM_MEMORY_RATIO,
            "all_rows_passed": True,
            "all_paired_trials_passed": True,
            "all_memory_gates_passed": True,
            "finite_and_workspace_gates_passed": True,
        },
        "rows": results,
        "eligibility": {
            "p4_4_h100_qualified": True,
            "eligible_for_p4_5": True,
            "auto_policy_implemented": False,
            "phase_4_complete": False,
            "phase_5_authorized": False,
            "production_default_changed": False,
            "separated_canary_promoted": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def main() -> None:
    args = parse_args()
    input_sha256 = _sha256(args.input)
    try:
        result = adjudicate_evidence(
            load_evidence(args.input),
            input_sha256=input_sha256,
        )
        exit_code = 0
    except EvidenceError as error:
        result = {
            "schema_version": 1,
            "classification": "FAIL_P4_4_CONTRACT_EQUIVALENT_ADJUDICATION",
            "input": {"sha256": input_sha256},
            "error": str(error),
            "eligibility": {
                "p4_4_h100_qualified": False,
                "eligible_for_p4_5": False,
                "phase_4_complete": False,
                "phase_5_authorized": False,
                "production_default_changed": False,
            },
        }
        exit_code = 1
    serialized = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
