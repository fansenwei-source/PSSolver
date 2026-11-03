"""D-0019 comparison rules for S-19 qualification support, not runtime.

H failures invalidate the execution and stop subsequent stages. N failures
are finite numerical differences: qualification remains failed, but Control
and performance stages may still run. No tolerance is introduced here.
"""

from __future__ import annotations

import math
from typing import Any


class HardQualificationError(RuntimeError):
    """Identity, environment, device, fallback, non-finite or crash failure."""


def require_execution(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    """Reject missing or different identity facts; never infer success."""
    for key, value in expected.items():
        if key not in observed or observed[key] != value:
            raise HardQualificationError(f"execution identity mismatch: {key}")
    for key in ("finite", "installed_wheel_only", "cache_initially_empty"):
        if observed.get(key) is not True:
            raise HardQualificationError(f"execution gate failed: {key}")
    if observed.get("fallback_used") is not False:
        raise HardQualificationError("execution gate failed: fallback_used")
    if observed.get("returncode") != 0:
        raise HardQualificationError("execution gate failed: child process")


def _json_number(value: float) -> float | str:
    # Exact zero denominators and extreme finite differences remain visible;
    # JSON NaN/Infinity literals are never emitted and never treated as PASS.
    return value if math.isfinite(value) else "Infinity"


def compare_field(left: Any, right: Any, *, field: str, step: int) -> dict[str, Any]:
    """Compare every stored field, including signed-zero and payload bytes."""
    import numpy as np

    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise HardQualificationError(f"array schema mismatch: {field}/{step}")
    if left.dtype.kind not in "fciub" or left.size == 0:
        raise HardQualificationError(f"unsupported array: {field}/{step}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise HardQualificationError(f"non-finite array: {field}/{step}")
    identical = left.tobytes(order="C") == right.tobytes(order="C")
    wide = np.complex128 if left.dtype.kind == "c" else np.float64
    with np.errstate(over="ignore", invalid="raise", divide="ignore"):
        delta = np.abs(right.astype(wide) - left.astype(wide))
        reference = np.abs(left.astype(wide))
        zero_difference = (reference == 0) & (delta != 0)
        relative = np.divide(delta, reference, out=np.zeros_like(delta), where=reference != 0)
        max_relative = math.inf if zero_difference.any() else float(relative.max())
    return {
        "field": field,
        "step": step,
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "bitwise_equal": identical,
        "classification": "PASS" if identical else "N",
        "max_absolute_difference": _json_number(float(delta.max())),
        "max_relative_difference": _json_number(max_relative),
        "relative_definition": "abs(candidate-reference)/abs(reference); unequal zero reference => Infinity",
        "unequal_zero_reference_elements": int(zero_difference.sum()),
        "tolerance": 0,
    }


def compare_trace(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate exact step/field coverage and report all finite differences."""
    if not left or len(left) != len(right):
        raise HardQualificationError("trace step coverage mismatch")
    rows = []
    for index, (reference, candidate) in enumerate(zip(left, right)):
        if reference.get("step") != index or candidate.get("step") != index:
            raise HardQualificationError("trace step identity mismatch")
        for key in ("runtime_identity", "configuration_identity", "progress"):
            if key not in reference or key not in candidate or reference[key] != candidate[key]:
                raise HardQualificationError(f"trace identity mismatch: {key}/{index}")
        a, b = reference.get("fields"), candidate.get("fields")
        if not isinstance(a, dict) or not isinstance(b, dict) or not a or list(a) != list(b):
            raise HardQualificationError(f"trace field coverage mismatch: {index}")
        rows.extend(compare_field(a[field], b[field], field=field, step=index) for field in a)
    failed = any(row["classification"] == "N" for row in rows)
    return {
        "classification": "N" if failed else "PASS",
        "qualification_passed": not failed,
        "continue_control_and_profiles": True,
        "step_count_including_initial": len(left),
        "fields": rows,
        "nonzero_tolerance_authorized": False,
    }


def final_verdict(stages: list[dict[str, Any]], *, formal_h100: bool) -> dict[str, Any]:
    """Keep an N failure latched even if later stages pass."""
    if not stages or any(row.get("classification") not in ("PASS", "N", "H") for row in stages):
        raise HardQualificationError("stage classification missing or invalid")
    hard = any(row["classification"] == "H" for row in stages)
    numerical = any(row["classification"] == "N" for row in stages)
    passed = not (hard or numerical)
    return {
        "classification": ("FAIL_H" if hard else "FAIL_N") if not passed else (
            "PASS_S19_H100" if formal_h100 else "PASS_CPU_REHEARSAL_ONLY"
        ),
        "qualification_passed": passed and formal_h100,
        "cpu_rehearsal_passed": passed and not formal_h100,
        "formal_h100_contract": formal_h100,
        "no_threshold_exception": True,
        "stages": stages,
    }
