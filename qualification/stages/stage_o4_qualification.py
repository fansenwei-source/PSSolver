"""Bounded Stage O.4 qualification for the dual-path Plane workflow.

This module is analysis-only.  It compares completed workflow artifacts and
aggregates separately generated H100 profiles; it never constructs a solver or
changes the production runtime default.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

from .shadow_metadata import plane_beris_edwards_production_signature


STAGE_O4_RELATIVE_L2_TOLERANCE = 1.0e-10
STAGE_O4_PROFILE_TRIALS = 3
STAGE_O4_MAXIMUM_MEAN_TIMESTEP_RATIO = 1.03
STAGE_O4_MAXIMUM_PAIRED_TIMESTEP_RATIO = 1.05
STAGE_O4_MAXIMUM_MEMORY_RATIO = 1.03

_FIELDS = ("Q", "u", "p")
_RUNTIME_PATHS = (
    "legacy_production",
    "separated_canary",
    "compiled_v2",
)
_ARRAY_PATTERN = re.compile(r"^(Q|u|p)_(\d+)\.npy$")
_LIFECYCLE_KEYS = (
    "physical_materializations",
    "on_demand_physical_materializations",
    "physical_materialization_batches",
    "batched_physical_components",
    "singleton_materialization_batches",
    "maximum_materialization_batch_size",
    "physical_island_prefetches",
    "physical_island_requested_components",
    "unmaterialized_published_components",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must contain a JSON object")
    return dict(value)


def _write_new_json(path: str | Path, value: Mapping[str, object]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite JSON output: {target}")
    target.write_text(
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _completed_metadata(
    directory: Path,
    *,
    expected_runtime_path: str,
) -> tuple[dict[str, object], str, dict[str, object]]:
    if expected_runtime_path not in _RUNTIME_PATHS:
        raise ValueError("expected runtime path is invalid")
    if not directory.is_dir():
        raise FileNotFoundError(f"workflow directory is missing: {directory}")
    metadata_path = directory / "metadata.json"
    complete_path = directory / "COMPLETE"
    metadata = _load_json(metadata_path, "workflow metadata")
    if not complete_path.is_file() or complete_path.read_text(encoding="utf-8") != (
        "complete\n"
    ):
        raise ValueError(f"workflow COMPLETE marker is invalid: {directory}")
    if complete_path.stat().st_mtime_ns < metadata_path.stat().st_mtime_ns:
        raise ValueError("workflow COMPLETE marker was not written last")
    try:
        runtime = metadata["runtime_selection"]
        workflow = metadata["workflow"]
        if metadata["status"] != "complete":
            raise ValueError("workflow metadata is not complete")
        if runtime["effective"] != expected_runtime_path:
            raise ValueError("workflow runtime path differs from expectation")
        if workflow["runtime_path"] != expected_runtime_path:
            raise ValueError("workflow metadata runtime path is inconsistent")
        if workflow["completion_marker_order"] != "metadata_then_COMPLETE":
            raise ValueError("workflow completion ordering contract is absent")
    except (KeyError, TypeError) as exc:
        raise ValueError("workflow metadata lacks the Stage O.4 contract") from exc
    signature = plane_beris_edwards_production_signature(metadata)
    return metadata, _sha256(metadata_path), signature


def _available_steps(directory: Path, field: str) -> tuple[int, ...]:
    steps = []
    for path in directory.glob(f"{field}_*.npy"):
        match = _ARRAY_PATTERN.fullmatch(path.name)
        if match is not None and match.group(1) == field:
            steps.append(int(match.group(2)))
    return tuple(sorted(steps))


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right,
        dtype=np.float64,
    )
    numerator = float(np.linalg.norm(difference.ravel()))
    denominator = max(
        float(np.linalg.norm(np.asarray(left).ravel())),
        float(np.linalg.norm(np.asarray(right).ravel())),
        1.0e-300,
    )
    return numerator / denominator


def _array_record(
    left_path: Path,
    right_path: Path,
    *,
    field: str,
    step: int,
) -> dict[str, object]:
    left_sha_before = _sha256(left_path)
    right_sha_before = _sha256(right_path)
    left = np.load(left_path, allow_pickle=False)
    right = np.load(right_path, allow_pickle=False)
    if left.shape != right.shape:
        raise ValueError(f"array shape differs for {field}_{step}")
    if left.dtype != right.dtype or left.dtype != np.dtype("float64"):
        raise ValueError(f"array dtype differs or is not float64 for {field}_{step}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"array contains NaN or Inf for {field}_{step}")
    raw_relative_l2 = _relative_l2(left, right)
    raw_linf = float(np.max(np.abs(left - right)))
    record: dict[str, object] = {
        "field": field,
        "step": step,
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "left_sha256": left_sha_before,
        "right_sha256": right_sha_before,
        "byte_identical": left_sha_before == right_sha_before,
        "raw_relative_l2": raw_relative_l2,
        "raw_linf": raw_linf,
        "gate_relative_l2": raw_relative_l2,
    }
    if field == "p":
        left_demeaned = left - np.mean(left, dtype=np.float64)
        right_demeaned = right - np.mean(right, dtype=np.float64)
        record["demeaned_relative_l2"] = _relative_l2(
            left_demeaned,
            right_demeaned,
        )
        record["demeaned_linf"] = float(
            np.max(np.abs(left_demeaned - right_demeaned))
        )
        record["gate_relative_l2"] = record["demeaned_relative_l2"]
    if _sha256(left_path) != left_sha_before or _sha256(right_path) != right_sha_before:
        raise RuntimeError(f"array changed while comparing {field}_{step}")
    return record


def compare_plane_stage_o4_workflows(
    left_directory: str | Path,
    right_directory: str | Path,
    *,
    left_runtime_path: str,
    right_runtime_path: str,
    expected_steps: Sequence[int],
    comparison_role: str,
    relative_l2_tolerance: float = STAGE_O4_RELATIVE_L2_TOLERANCE,
    require_byte_identity: bool = False,
    require_initial_q_identity: bool = False,
    allow_extra_steps: bool = False,
) -> dict[str, object]:
    """Compare two complete Stage O.3 workflow outputs without modifying them."""

    left_dir = Path(left_directory).expanduser().resolve()
    right_dir = Path(right_directory).expanduser().resolve()
    steps = tuple(expected_steps)
    if (
        not steps
        or any(not isinstance(step, int) or isinstance(step, bool) or step < 0 for step in steps)
        or tuple(sorted(set(steps))) != steps
    ):
        raise ValueError("expected_steps must be unique increasing non-negative integers")
    if not isinstance(comparison_role, str) or not comparison_role:
        raise ValueError("comparison_role must not be empty")
    if (
        not isinstance(relative_l2_tolerance, (int, float))
        or not math.isfinite(relative_l2_tolerance)
        or relative_l2_tolerance <= 0.0
    ):
        raise ValueError("relative_l2_tolerance must be positive and finite")

    left_metadata, left_metadata_sha, left_signature = _completed_metadata(
        left_dir,
        expected_runtime_path=left_runtime_path,
    )
    right_metadata, right_metadata_sha, right_signature = _completed_metadata(
        right_dir,
        expected_runtime_path=right_runtime_path,
    )
    if left_signature != right_signature:
        raise ValueError("workflow scientific signatures differ")
    for field in _FIELDS:
        left_steps = _available_steps(left_dir, field)
        right_steps = _available_steps(right_dir, field)
        if allow_extra_steps:
            if not set(steps).issubset(left_steps):
                raise ValueError(f"left {field} frame set lacks expected steps")
            if not set(steps).issubset(right_steps):
                raise ValueError(f"right {field} frame set lacks expected steps")
        else:
            if left_steps != steps:
                raise ValueError(f"left {field} frame set differs from expected steps")
            if right_steps != steps:
                raise ValueError(f"right {field} frame set differs from expected steps")

    arrays = [
        _array_record(
            left_dir / f"{field}_{step}.npy",
            right_dir / f"{field}_{step}.npy",
            field=field,
            step=step,
        )
        for step in steps
        for field in _FIELDS
    ]
    maximum_error = max(float(record["gate_relative_l2"]) for record in arrays)
    numerical_gate = maximum_error <= relative_l2_tolerance
    byte_identity_gate = (
        not require_byte_identity
        or all(record["byte_identical"] is True for record in arrays)
    )
    q0_records = [
        record for record in arrays if record["field"] == "Q" and record["step"] == 0
    ]
    initial_q_identity_gate = (
        not require_initial_q_identity
        or len(q0_records) == 1
        and q0_records[0]["byte_identical"] is True
    )
    accepted = numerical_gate and byte_identity_gate and initial_q_identity_gate
    return {
        "schema_version": 1,
        "qualification_stage": "O.4",
        "comparison_role": comparison_role,
        "classification": "PASS" if accepted else "FAIL",
        "left": {
            "directory": str(left_dir),
            "runtime_path": left_runtime_path,
            "metadata_sha256": left_metadata_sha,
            "completed_steps": left_metadata.get("completed_steps"),
        },
        "right": {
            "directory": str(right_dir),
            "runtime_path": right_runtime_path,
            "metadata_sha256": right_metadata_sha,
            "completed_steps": right_metadata.get("completed_steps"),
        },
        "scientific_signature": left_signature,
        "expected_steps": list(steps),
        "array_count": len(arrays),
        "arrays": arrays,
        "relative_l2_tolerance": float(relative_l2_tolerance),
        "maximum_gate_relative_l2": maximum_error,
        "numerical_gate": numerical_gate,
        "require_byte_identity": require_byte_identity,
        "byte_identity_gate": byte_identity_gate,
        "require_initial_q_identity": require_initial_q_identity,
        "initial_q_identity_gate": initial_q_identity_gate,
        "allow_extra_steps": allow_extra_steps,
        "production_default_changed": False,
    }


def _validated_comparison(
    path: str | Path,
    *,
    role: str,
    expected_steps: Sequence[int],
    require_exact: bool,
) -> tuple[dict[str, object], Path]:
    resolved = Path(path).expanduser().resolve()
    report = _load_json(resolved, f"{role} comparison")
    try:
        arrays = report["arrays"]
        expected_pairs = {
            (field, step) for step in expected_steps for field in _FIELDS
        }
        actual_pairs = {
            (record["field"], record["step"]) for record in arrays
        }
        record_errors = [float(record["gate_relative_l2"]) for record in arrays]
        valid = (
            report["qualification_stage"] == "O.4"
            and report["comparison_role"] == role
            and report["classification"] == "PASS"
            and report["expected_steps"] == list(expected_steps)
            and report["array_count"] == len(expected_steps) * len(_FIELDS)
            and report["relative_l2_tolerance"]
            == STAGE_O4_RELATIVE_L2_TOLERANCE
            and report["maximum_gate_relative_l2"]
            <= STAGE_O4_RELATIVE_L2_TOLERANCE
            and report["numerical_gate"] is True
            and report["production_default_changed"] is False
            and isinstance(report["scientific_signature"], Mapping)
            and isinstance(arrays, list)
            and len(arrays) == len(expected_pairs)
            and actual_pairs == expected_pairs
            and all(
                math.isfinite(error)
                and error <= STAGE_O4_RELATIVE_L2_TOLERANCE
                for error in record_errors
            )
            and max(record_errors) == report["maximum_gate_relative_l2"]
        )
        if require_exact:
            valid = valid and report["require_byte_identity"] is True and report[
                "byte_identity_gate"
            ] is True and all(
                record["byte_identical"] is True for record in arrays
            )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{role} comparison violates the fixed O.4 gate")
    return report, resolved


def _profile_configuration(profile: Mapping[str, object], *, role: str) -> dict[str, object]:
    try:
        if role == "legacy":
            config = profile["config"]
            if config.get("timing_scope") != "whole_timestep":
                raise ValueError(
                    "legacy O.4 throughput must use whole-timestep-only timing"
                )
        else:
            config = profile["configuration"]
        return {
            "shape": list(config["shape"]),
            "lengths": list(config["lengths"]),
            "dtype": config["dtype"],
            "dt": float(config["dt"]),
            "dealias_rule": config["dealias_rule"],
            "projected_transform_execution": config[
                "projected_transform_execution"
            ],
            "spectral_storage": config["spectral_storage"],
            "spectral_refresh_interval": config["spectral_refresh_interval"],
            "warmup_steps": config["warmup_steps"],
            "profile_steps": config["profile_steps"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} profile configuration is incomplete") from exc


def _profile_measurement(
    profile: Mapping[str, object],
    *,
    role: str,
    expected_gpu_name: str,
) -> tuple[float, int, int, float, float, str, Mapping[str, object] | None]:
    configuration = _profile_configuration(profile, role=role)
    expected = {
        "shape": [128, 128, 32],
        "lengths": [100.0, 100.0, 20.0],
        "dtype": "float64",
        "dt": 0.005,
        "dealias_rule": "cubic_half",
        "projected_transform_execution": "truncated",
        "spectral_storage": "hermitian_half",
        "spectral_refresh_interval": 2,
        "warmup_steps": 10,
        "profile_steps": 20,
    }
    if configuration != expected:
        raise ValueError(f"{role} profile violates the fixed O.4 configuration")
    try:
        environment = profile["environment"]
        if (
            environment["cuda_available"] is not True
            or expected_gpu_name.lower() not in environment["device_name"].lower()
            or environment["cuda_matmul_allow_tf32"] is not False
        ):
            raise ValueError(f"{role} profile GPU provenance is incompatible")
        throughput = profile["throughput"]
        memory = profile["memory"]
        mean = float(throughput["mean_timestep_seconds"])
        allocated = int(memory["peak_allocated_bytes"])
        reserved = int(memory["peak_reserved_bytes"])
        if role == "legacy":
            steps = int(profile["config"]["profile_steps"])
            forward = float(profile["timings"]["transform_forward"]["calls"]) / steps
            inverse = float(profile["timings"]["transform_inverse"]["calls"]) / steps
            initial_q_sha = profile["profile_input"]["initial_q_sha256"]
            lifecycle = None
        else:
            if (
                profile["qualification_stage"] != "N.4"
                or profile["classification"] != "PROFILE_COMPLETE"
                or profile["mode"] != "candidate"
                or profile["finite"] is not True
                or profile["configuration_authority"]
                != "unified_execution_policy"
                or profile["algebraic_execution_policy"]["mode"]
                != "batched_physical_islands"
            ):
                raise ValueError("canary profile identity is incompatible")
            audit = profile["transform_call_audit"]
            forward = float(audit["forward_calls_per_step"])
            inverse = float(audit["inverse_calls_per_step"])
            initial_q_sha = profile["production_initial_q_sha256"]
            lifecycle = audit
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{role} profile is incomplete") from exc
    if (
        not math.isfinite(mean)
        or mean <= 0.0
        or allocated <= 0
        or reserved <= 0
        or not math.isfinite(forward)
        or not math.isfinite(inverse)
        or not isinstance(initial_q_sha, str)
        or len(initial_q_sha) != 64
    ):
        raise ValueError(f"{role} profile contains invalid measurements")
    return mean, allocated, reserved, forward, inverse, initial_q_sha, lifecycle


def analyze_stage_o4_qualification(
    six_step_comparison: str | Path,
    hundred_step_comparison: str | Path,
    legacy_restart_comparison: str | Path,
    canary_restart_comparison: str | Path,
    legacy_profile_paths: Sequence[str | Path],
    canary_profile_paths: Sequence[str | Path],
    *,
    expected_gpu_name: str = "H100",
    maximum_mean_timestep_ratio: float = STAGE_O4_MAXIMUM_MEAN_TIMESTEP_RATIO,
    maximum_paired_timestep_ratio: float = STAGE_O4_MAXIMUM_PAIRED_TIMESTEP_RATIO,
    maximum_memory_ratio: float = STAGE_O4_MAXIMUM_MEMORY_RATIO,
) -> dict[str, object]:
    """Apply every frozen Stage O.4 science and H100 non-regression gate."""

    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")
    for value, description in (
        (maximum_mean_timestep_ratio, "maximum_mean_timestep_ratio"),
        (maximum_paired_timestep_ratio, "maximum_paired_timestep_ratio"),
        (maximum_memory_ratio, "maximum_memory_ratio"),
    ):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{description} must be positive and finite")

    six, six_path = _validated_comparison(
        six_step_comparison,
        role="cross_runtime_six_step",
        expected_steps=tuple(range(7)),
        require_exact=False,
    )
    hundred, hundred_path = _validated_comparison(
        hundred_step_comparison,
        role="cross_runtime_hundred_step",
        expected_steps=(0, 100),
        require_exact=False,
    )
    legacy_restart, legacy_restart_path = _validated_comparison(
        legacy_restart_comparison,
        role="legacy_same_backend_restart",
        expected_steps=(6,),
        require_exact=True,
    )
    canary_restart, canary_restart_path = _validated_comparison(
        canary_restart_comparison,
        role="canary_same_backend_restart",
        expected_steps=(6,),
        require_exact=True,
    )
    if six.get("initial_q_identity_gate") is not True or hundred.get(
        "initial_q_identity_gate"
    ) is not True:
        raise ValueError("cross-runtime comparisons lack exact initial-Q identity")
    for report in (six, hundred):
        q0 = [
            record
            for record in report["arrays"]
            if record["field"] == "Q" and record["step"] == 0
        ]
        if (
            len(q0) != 1
            or q0[0]["left_sha256"] != q0[0]["right_sha256"]
            or q0[0]["byte_identical"] is not True
        ):
            raise ValueError("cross-runtime initial-Q record is not exact")
    if six["scientific_signature"] != hundred["scientific_signature"]:
        raise ValueError("six-step and hundred-step scientific signatures differ")

    legacy_paths = tuple(Path(path).expanduser().resolve() for path in legacy_profile_paths)
    canary_paths = tuple(Path(path).expanduser().resolve() for path in canary_profile_paths)
    if len(legacy_paths) != STAGE_O4_PROFILE_TRIALS or len(canary_paths) != (
        STAGE_O4_PROFILE_TRIALS
    ):
        raise ValueError("O.4 requires exactly three profiles per runtime path")
    legacy_profiles = [_load_json(path, "legacy profile") for path in legacy_paths]
    canary_profiles = [_load_json(path, "canary profile") for path in canary_paths]
    legacy_values = [
        _profile_measurement(
            profile,
            role="legacy",
            expected_gpu_name=expected_gpu_name,
        )
        for profile in legacy_profiles
    ]
    canary_values = [
        _profile_measurement(
            profile,
            role="canary",
            expected_gpu_name=expected_gpu_name,
        )
        for profile in canary_profiles
    ]
    profile_q_hashes = {value[5] for value in (*legacy_values, *canary_values)}
    six_q0 = next(
        record["left_sha256"]
        for record in six["arrays"]
        if record["field"] == "Q" and record["step"] == 0
    )
    initial_q_identity_gate = profile_q_hashes == {six_q0}

    legacy_times = [value[0] for value in legacy_values]
    canary_times = [value[0] for value in canary_values]
    paired_ratios = [
        candidate / reference
        for reference, candidate in zip(legacy_times, canary_times, strict=True)
    ]
    mean_legacy = math.fsum(legacy_times) / len(legacy_times)
    mean_canary = math.fsum(canary_times) / len(canary_times)
    mean_ratio = mean_canary / mean_legacy
    paired_gate = sum(
        ratio <= maximum_paired_timestep_ratio for ratio in paired_ratios
    ) >= 2
    performance_gate = mean_ratio <= maximum_mean_timestep_ratio and paired_gate
    allocated_ratio = max(value[1] for value in canary_values) / max(
        value[1] for value in legacy_values
    )
    reserved_ratio = max(value[2] for value in canary_values) / max(
        value[2] for value in legacy_values
    )
    memory_gate = (
        allocated_ratio <= maximum_memory_ratio
        and reserved_ratio <= maximum_memory_ratio
    )
    transform_count_gate = (
        [value[3] for value in legacy_values]
        == [value[3] for value in canary_values]
        and [value[4] for value in legacy_values]
        == [value[4] for value in canary_values]
    )

    canary_lifecycles = [value[6] for value in canary_values]
    if any(not isinstance(value, Mapping) for value in canary_lifecycles):
        raise ValueError("canary profiles lack lifecycle audits")
    materializations = [value["physical_materialization"] for value in canary_lifecycles]
    lifecycle_identity = all(
        all(left.get(key) == right.get(key) for key in _LIFECYCLE_KEYS)
        for left, right in zip(materializations[1:], materializations[:-1], strict=True)
    )
    lifecycle_valid = lifecycle_identity and all(
        lifecycle["representation_reuse"].get("retained_pairs_after_generation") == 0
        and materialization.get("on_demand_physical_materializations") == 0
        and materialization.get("physical_materialization_batches", math.inf)
        < materialization.get("physical_materializations", 0)
        and lifecycle["execution_policy"].get("mode")
        == "batched_physical_islands"
        for lifecycle, materialization in zip(
            canary_lifecycles,
            materializations,
            strict=True,
        )
    )

    gates = {
        "six_step_numerical": True,
        "hundred_step_numerical": True,
        "legacy_same_backend_restart_exact": True,
        "canary_same_backend_restart_exact": True,
        "initial_q_identity": initial_q_identity_gate,
        "transform_count_identity": transform_count_gate,
        "canary_lifecycle_consistency": lifecycle_valid,
        "performance_non_regression": performance_gate,
        "memory_non_regression": memory_gate,
    }
    accepted = all(gates.values())
    failures = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1,
        "qualification_stage": "O.4",
        "classification": "A_recommended" if accepted else "B_neutral",
        "eligible_for_stage_o5_decision": accepted,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "gate_failures": failures,
        "gates": gates,
        "trajectory": {
            "six_step_report": str(six_path),
            "six_step_report_sha256": _sha256(six_path),
            "hundred_step_report": str(hundred_path),
            "hundred_step_report_sha256": _sha256(hundred_path),
            "legacy_restart_report": str(legacy_restart_path),
            "legacy_restart_report_sha256": _sha256(legacy_restart_path),
            "canary_restart_report": str(canary_restart_path),
            "canary_restart_report_sha256": _sha256(canary_restart_path),
            "maximum_gate_relative_l2": max(
                float(six["maximum_gate_relative_l2"]),
                float(hundred["maximum_gate_relative_l2"]),
            ),
            "relative_l2_tolerance": STAGE_O4_RELATIVE_L2_TOLERANCE,
        },
        "profile_count_per_runtime": STAGE_O4_PROFILE_TRIALS,
        "legacy": {
            "mean_timestep_seconds": mean_legacy,
            "median_timestep_seconds": statistics.median(legacy_times),
            "peak_allocated_bytes": max(value[1] for value in legacy_values),
            "peak_reserved_bytes": max(value[2] for value in legacy_values),
            "forward_calls_per_step": legacy_values[0][3],
            "inverse_calls_per_step": legacy_values[0][4],
        },
        "canary": {
            "mean_timestep_seconds": mean_canary,
            "median_timestep_seconds": statistics.median(canary_times),
            "peak_allocated_bytes": max(value[1] for value in canary_values),
            "peak_reserved_bytes": max(value[2] for value in canary_values),
            "forward_calls_per_step": canary_values[0][3],
            "inverse_calls_per_step": canary_values[0][4],
            "lifecycle": materializations[0],
        },
        "comparison": {
            "mean_timestep_ratio_canary_over_legacy": mean_ratio,
            "median_paired_timestep_ratio": statistics.median(paired_ratios),
            "paired_timestep_ratios": paired_ratios,
            "peak_allocated_ratio_canary_over_legacy": allocated_ratio,
            "peak_reserved_ratio_canary_over_legacy": reserved_ratio,
            "maximum_mean_timestep_ratio": maximum_mean_timestep_ratio,
            "maximum_paired_timestep_ratio": maximum_paired_timestep_ratio,
            "maximum_memory_ratio": maximum_memory_ratio,
        },
        "profile_inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (*legacy_paths, *canary_paths)
        ],
    }


def comparison_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Stage O.4 workflows")
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-runtime-path", choices=_RUNTIME_PATHS, required=True)
    parser.add_argument("--right-runtime-path", choices=_RUNTIME_PATHS, required=True)
    parser.add_argument("--expected-step", type=int, action="append", required=True)
    parser.add_argument("--comparison-role", required=True)
    parser.add_argument(
        "--relative-l2-tolerance",
        type=float,
        default=STAGE_O4_RELATIVE_L2_TOLERANCE,
    )
    parser.add_argument("--require-byte-identity", action="store_true")
    parser.add_argument("--require-initial-q-identity", action="store_true")
    parser.add_argument("--allow-extra-steps", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_plane_stage_o4_workflows(
        args.left_dir,
        args.right_dir,
        left_runtime_path=args.left_runtime_path,
        right_runtime_path=args.right_runtime_path,
        expected_steps=args.expected_step,
        comparison_role=args.comparison_role,
        relative_l2_tolerance=args.relative_l2_tolerance,
        require_byte_identity=args.require_byte_identity,
        require_initial_q_identity=args.require_initial_q_identity,
        allow_extra_steps=args.allow_extra_steps,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["classification"] == "PASS" else 1


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage O.4 evidence")
    parser.add_argument("--six-step-comparison", type=Path, required=True)
    parser.add_argument("--hundred-step-comparison", type=Path, required=True)
    parser.add_argument("--legacy-restart-comparison", type=Path, required=True)
    parser.add_argument("--canary-restart-comparison", type=Path, required=True)
    parser.add_argument("--legacy-profile", type=Path, action="append", required=True)
    parser.add_argument("--canary-profile", type=Path, action="append", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_o4_qualification(
        args.six_step_comparison,
        args.hundred_step_comparison,
        args.legacy_restart_comparison,
        args.canary_restart_comparison,
        args.legacy_profile,
        args.canary_profile,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["classification"] == "A_recommended" else 1


__all__ = [
    "STAGE_O4_MAXIMUM_MEAN_TIMESTEP_RATIO",
    "STAGE_O4_MAXIMUM_MEMORY_RATIO",
    "STAGE_O4_MAXIMUM_PAIRED_TIMESTEP_RATIO",
    "STAGE_O4_PROFILE_TRIALS",
    "STAGE_O4_RELATIVE_L2_TOLERANCE",
    "analyze_stage_o4_qualification",
    "analysis_main",
    "compare_plane_stage_o4_workflows",
    "comparison_main",
]
