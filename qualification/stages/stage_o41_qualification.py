"""Prospective Stage O.4.1 contract correction and memory attribution.

Stage O.4 remains immutable evidence.  This analyzer binds that report, then
applies architecture-aware transform and phase-aware CUDA-memory gates to new
R128/R320 profiles.  It never constructs a solver or changes a runtime default.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics

from .stage_o4_qualification import _load_json, _sha256, _write_new_json


STAGE_O41_PROFILE_TRIALS = 3
STAGE_O41_DIAGNOSTIC_SHAPE = (128, 128, 32)
STAGE_O41_DECISION_SHAPE = (320, 320, 80)
STAGE_O41_MAXIMUM_MEAN_TIMESTEP_RATIO = 1.03
STAGE_O41_MAXIMUM_PAIRED_TIMESTEP_RATIO = 1.05
STAGE_O41_MAXIMUM_MEMORY_RATIO = 1.03
STAGE_O41_RETAINED_GROWTH_FLOOR_BYTES = 16 * 1024 * 1024
STAGE_O41_RETAINED_GROWTH_FRACTION = 0.01

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
_WINDOW_KEYS = (
    "start_allocated_bytes",
    "start_reserved_bytes",
    "end_allocated_bytes",
    "end_reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "retained_allocated_growth_bytes",
    "retained_reserved_growth_bytes",
)


def _positive_finite(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{description} must be positive and finite")
    return result


def _validated_stage_o4_report(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected Stage O.4 SHA-256 must contain 64 characters")
    if _sha256(resolved) != expected_sha256:
        raise ValueError("Stage O.4 report SHA-256 differs from the frozen input")
    report = _load_json(resolved, "Stage O.4 qualification")
    required_passes = {
        "six_step_numerical",
        "hundred_step_numerical",
        "legacy_same_backend_restart_exact",
        "canary_same_backend_restart_exact",
        "initial_q_identity",
        "canary_lifecycle_consistency",
        "performance_non_regression",
    }
    try:
        gates = report["gates"]
        failures = set(report["gate_failures"])
        valid = (
            report["qualification_stage"] == "O.4"
            and report["classification"] == "B_neutral"
            and report["eligible_for_stage_o5_decision"] is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
            and isinstance(gates, Mapping)
            and all(gates[name] is True for name in required_passes)
            and gates["transform_count_identity"] is False
            and gates["memory_non_regression"] is False
            and failures
            == {"transform_count_identity", "memory_non_regression"}
            and float(report["trajectory"]["maximum_gate_relative_l2"])
            <= float(report["trajectory"]["relative_l2_tolerance"])
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("input is not the frozen Stage O.4 B_neutral evidence")
    return resolved, report


def _profile_configuration(
    profile: Mapping[str, object],
    *,
    role: str,
) -> dict[str, object]:
    try:
        config = profile["config"] if role == "legacy" else profile["configuration"]
        if role == "legacy" and config["timing_scope"] != "whole_timestep":
            raise ValueError("legacy profile must use whole-timestep timing")
        return {
            "shape": tuple(int(value) for value in config["shape"]),
            "lengths": tuple(float(value) for value in config["lengths"]),
            "dtype": config["dtype"],
            "dt": float(config["dt"]),
            "dealias_rule": config["dealias_rule"],
            "projected_transform_execution": config[
                "projected_transform_execution"
            ],
            "spectral_storage": config["spectral_storage"],
            "spectral_refresh_interval": config["spectral_refresh_interval"],
            "warmup_steps": int(config["warmup_steps"]),
            "profile_steps": int(config["profile_steps"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} profile configuration is incomplete") from exc


def _memory_window(
    profile: Mapping[str, object],
    *,
    role: str,
    name: str,
) -> dict[str, int]:
    try:
        attribution = profile["memory_attribution"]
        if (
            attribution["schema_version"] != 1
            or attribution["allocator_scope"]
            != "current_process_cuda_allocator"
            or attribution["peak_reset_after_warmup"] is not True
            or attribution["observation_peak_reset_after_timestep_window"]
            is not True
        ):
            raise ValueError("memory attribution contract differs")
        raw = attribution["windows"][name]
        result = {key: int(raw[key]) for key in _WINDOW_KEYS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} {name} memory window is incomplete") from exc
    if any(value < 0 for value in result.values()):
        raise ValueError(f"{role} {name} memory window contains a negative value")
    if result["peak_allocated_bytes"] < max(
        result["start_allocated_bytes"],
        result["end_allocated_bytes"],
    ):
        raise ValueError(f"{role} {name} allocated-memory peak is invalid")
    if result["peak_reserved_bytes"] < max(
        result["start_reserved_bytes"],
        result["end_reserved_bytes"],
    ):
        raise ValueError(f"{role} {name} reserved-memory peak is invalid")
    return result


def _profile_measurement(
    profile: Mapping[str, object],
    *,
    role: str,
    expected_shape: tuple[int, int, int],
    expected_gpu_name: str,
) -> dict[str, object]:
    configuration = _profile_configuration(profile, role=role)
    expected = {
        "shape": expected_shape,
        "lengths": (100.0, 100.0, 20.0),
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
        raise ValueError(f"{role} profile violates the fixed O.4.1 configuration")
    try:
        environment = profile["environment"]
        if (
            environment["cuda_available"] is not True
            or expected_gpu_name.lower() not in environment["device_name"].lower()
            or environment["cuda_matmul_allow_tf32"] is not False
        ):
            raise ValueError("GPU identity differs")
        mean = _positive_finite(
            profile["throughput"]["mean_timestep_seconds"],
            f"{role} mean timestep",
        )
        if role == "legacy":
            steps = int(profile["config"]["profile_steps"])
            forward = float(profile["timings"]["transform_forward"]["calls"]) / steps
            inverse = float(profile["timings"]["transform_inverse"]["calls"]) / steps
            initial_q_sha256 = profile["profile_input"]["initial_q_sha256"]
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
                raise ValueError("canary identity differs")
            audit = profile["transform_call_audit"]
            forward = float(audit["forward_calls_per_step"])
            inverse = float(audit["inverse_calls_per_step"])
            initial_q_sha256 = profile["production_initial_q_sha256"]
            lifecycle = audit
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{role} profile is incomplete") from exc
    if (
        not math.isfinite(forward)
        or forward <= 0.0
        or not math.isfinite(inverse)
        or inverse <= 0.0
        or not isinstance(initial_q_sha256, str)
        or len(initial_q_sha256) != 64
    ):
        raise ValueError(f"{role} profile has invalid transform or input identity")
    return {
        "configuration": configuration,
        "mean_timestep_seconds": mean,
        "forward_calls_per_step": forward,
        "inverse_calls_per_step": inverse,
        "initial_q_sha256": initial_q_sha256,
        "timestep_memory": _memory_window(
            profile,
            role=role,
            name="timestep",
        ),
        "observation_memory": _memory_window(
            profile,
            role=role,
            name="observation",
        ),
        "lifecycle": lifecycle,
    }


def _stable(values: Sequence[float]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def _lifecycle_gate(measurements: Sequence[Mapping[str, object]]) -> bool:
    try:
        audits = [measurement["lifecycle"] for measurement in measurements]
        if any(not isinstance(audit, Mapping) for audit in audits):
            return False
        materializations = [audit["physical_materialization"] for audit in audits]
        reuse = [audit["representation_reuse"] for audit in audits]
    except (KeyError, TypeError):
        return False
    if any(not isinstance(value, Mapping) for value in (*materializations, *reuse)):
        return False
    return all(
        all(left.get(key) == right.get(key) for key in _LIFECYCLE_KEYS)
        for left, right in zip(materializations[1:], materializations[:-1])
    ) and all(
        reuse_value.get("retained_pairs_after_generation") == 0
        and materialization.get("on_demand_physical_materializations") == 0
        and materialization.get("physical_materialization_batches", math.inf)
        < materialization.get("physical_materializations", 0)
        for reuse_value, materialization in zip(
            reuse,
            materializations,
            strict=True,
        )
    )


def _maximum_ratio(
    candidate: Sequence[Mapping[str, object]],
    legacy: Sequence[Mapping[str, object]],
    *,
    window: str,
    key: str,
) -> float:
    reference = max(int(value[window][key]) for value in legacy)
    if reference <= 0:
        raise ValueError(f"legacy {window} {key} must be positive")
    return max(int(value[window][key]) for value in candidate) / reference


def _growth_gate(measurements: Sequence[Mapping[str, object]]) -> bool:
    for measurement in measurements:
        window = measurement["timestep_memory"]
        start = int(window["start_allocated_bytes"])
        allowance = max(
            STAGE_O41_RETAINED_GROWTH_FLOOR_BYTES,
            math.ceil(start * STAGE_O41_RETAINED_GROWTH_FRACTION),
        )
        if int(window["retained_allocated_growth_bytes"]) > allowance:
            return False
    return True


def _shape_analysis(
    legacy_profiles: Sequence[Mapping[str, object]],
    canary_profiles: Sequence[Mapping[str, object]],
    *,
    shape: tuple[int, int, int],
    expected_gpu_name: str,
) -> dict[str, object]:
    if len(legacy_profiles) != STAGE_O41_PROFILE_TRIALS or len(
        canary_profiles
    ) != STAGE_O41_PROFILE_TRIALS:
        raise ValueError("O.4.1 requires exactly three profiles per runtime and shape")
    legacy = [
        _profile_measurement(
            profile,
            role="legacy",
            expected_shape=shape,
            expected_gpu_name=expected_gpu_name,
        )
        for profile in legacy_profiles
    ]
    canary = [
        _profile_measurement(
            profile,
            role="canary",
            expected_shape=shape,
            expected_gpu_name=expected_gpu_name,
        )
        for profile in canary_profiles
    ]
    if len(
        {value["initial_q_sha256"] for value in (*legacy, *canary)}
    ) != 1:
        raise ValueError("legacy and canary profiles do not share one initial Q")

    legacy_times = [float(value["mean_timestep_seconds"]) for value in legacy]
    canary_times = [float(value["mean_timestep_seconds"]) for value in canary]
    paired_ratios = [
        candidate / reference
        for reference, candidate in zip(legacy_times, canary_times, strict=True)
    ]
    mean_legacy = math.fsum(legacy_times) / len(legacy_times)
    mean_canary = math.fsum(canary_times) / len(canary_times)
    mean_ratio = mean_canary / mean_legacy

    legacy_forward = [float(value["forward_calls_per_step"]) for value in legacy]
    legacy_inverse = [float(value["inverse_calls_per_step"]) for value in legacy]
    canary_forward = [float(value["forward_calls_per_step"]) for value in canary]
    canary_inverse = [float(value["inverse_calls_per_step"]) for value in canary]
    transform_plan_stability = all(
        _stable(values)
        for values in (
            legacy_forward,
            legacy_inverse,
            canary_forward,
            canary_inverse,
        )
    )
    transform_nonincrease = (
        canary_forward[0] <= legacy_forward[0]
        and canary_inverse[0] <= legacy_inverse[0]
        and (
            canary_forward[0] < legacy_forward[0]
            or canary_inverse[0] < legacy_inverse[0]
        )
    )

    ratios = {
        "timestep_peak_allocated": _maximum_ratio(
            canary,
            legacy,
            window="timestep_memory",
            key="peak_allocated_bytes",
        ),
        "timestep_peak_reserved": _maximum_ratio(
            canary,
            legacy,
            window="timestep_memory",
            key="peak_reserved_bytes",
        ),
        "observation_peak_allocated": _maximum_ratio(
            canary,
            legacy,
            window="observation_memory",
            key="peak_allocated_bytes",
        ),
        "observation_peak_reserved": _maximum_ratio(
            canary,
            legacy,
            window="observation_memory",
            key="peak_reserved_bytes",
        ),
        "post_observation_allocated": _maximum_ratio(
            canary,
            legacy,
            window="observation_memory",
            key="end_allocated_bytes",
        ),
    }
    return {
        "shape": list(shape),
        "profile_count_per_runtime": STAGE_O41_PROFILE_TRIALS,
        "initial_q_identity": True,
        "legacy": {
            "mean_timestep_seconds": mean_legacy,
            "median_timestep_seconds": statistics.median(legacy_times),
            "forward_calls_per_step": legacy_forward[0],
            "inverse_calls_per_step": legacy_inverse[0],
        },
        "canary": {
            "mean_timestep_seconds": mean_canary,
            "median_timestep_seconds": statistics.median(canary_times),
            "forward_calls_per_step": canary_forward[0],
            "inverse_calls_per_step": canary_inverse[0],
        },
        "comparison": {
            "mean_timestep_ratio_canary_over_legacy": mean_ratio,
            "paired_timestep_ratios": paired_ratios,
            "median_paired_timestep_ratio": statistics.median(paired_ratios),
            "memory_ratios_canary_over_legacy": ratios,
        },
        "gates": {
            "transform_plan_stability": transform_plan_stability,
            "transform_nonincrease_with_strict_reduction": transform_nonincrease,
            "canary_lifecycle_consistency": _lifecycle_gate(canary),
            "post_warmup_timestep_retained_growth": _growth_gate(
                (*legacy, *canary)
            ),
        },
    }


def analyze_stage_o41_qualification(
    stage_o4_report: str | Path,
    legacy_r128_profile_paths: Sequence[str | Path],
    canary_r128_profile_paths: Sequence[str | Path],
    legacy_r320_profile_paths: Sequence[str | Path],
    canary_r320_profile_paths: Sequence[str | Path],
    *,
    expected_stage_o4_sha256: str,
    expected_gpu_name: str = "H100",
    maximum_mean_timestep_ratio: float = STAGE_O41_MAXIMUM_MEAN_TIMESTEP_RATIO,
    maximum_paired_timestep_ratio: float = STAGE_O41_MAXIMUM_PAIRED_TIMESTEP_RATIO,
    maximum_memory_ratio: float = STAGE_O41_MAXIMUM_MEMORY_RATIO,
) -> dict[str, object]:
    """Apply the prospectively frozen O.4.1 contract to two profile scales."""

    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")
    maximum_mean_timestep_ratio = _positive_finite(
        maximum_mean_timestep_ratio,
        "maximum_mean_timestep_ratio",
    )
    maximum_paired_timestep_ratio = _positive_finite(
        maximum_paired_timestep_ratio,
        "maximum_paired_timestep_ratio",
    )
    maximum_memory_ratio = _positive_finite(
        maximum_memory_ratio,
        "maximum_memory_ratio",
    )
    o4_path, o4_report = _validated_stage_o4_report(
        stage_o4_report,
        expected_sha256=expected_stage_o4_sha256,
    )

    def load(paths: Sequence[str | Path], description: str):
        resolved = tuple(Path(path).expanduser().resolve() for path in paths)
        identities = tuple(_sha256(path) for path in resolved)
        return resolved, identities, [
            _load_json(path, description) for path in resolved
        ]

    l128_paths, l128_sha, l128 = load(
        legacy_r128_profile_paths,
        "legacy R128 profile",
    )
    c128_paths, c128_sha, c128 = load(
        canary_r128_profile_paths,
        "canary R128 profile",
    )
    l320_paths, l320_sha, l320 = load(
        legacy_r320_profile_paths,
        "legacy R320 profile",
    )
    c320_paths, c320_sha, c320 = load(
        canary_r320_profile_paths,
        "canary R320 profile",
    )
    profile_paths = (*l128_paths, *c128_paths, *l320_paths, *c320_paths)
    profile_sha256 = (*l128_sha, *c128_sha, *l320_sha, *c320_sha)
    if len(set(profile_paths)) != 4 * STAGE_O41_PROFILE_TRIALS:
        raise ValueError("O.4.1 profile inputs must be distinct files")
    diagnostic = _shape_analysis(
        l128,
        c128,
        shape=STAGE_O41_DIAGNOSTIC_SHAPE,
        expected_gpu_name=expected_gpu_name,
    )
    decision = _shape_analysis(
        l320,
        c320,
        shape=STAGE_O41_DECISION_SHAPE,
        expected_gpu_name=expected_gpu_name,
    )

    decision_ratios = decision["comparison"]
    paired = decision_ratios["paired_timestep_ratios"]
    performance_gate = (
        decision_ratios["mean_timestep_ratio_canary_over_legacy"]
        <= maximum_mean_timestep_ratio
        and sum(value <= maximum_paired_timestep_ratio for value in paired) >= 2
    )
    memory_ratios = decision_ratios["memory_ratios_canary_over_legacy"]
    memory_gate = all(
        value <= maximum_memory_ratio for value in memory_ratios.values()
    )
    inherited_science_gate = all(
        o4_report["gates"][name] is True
        for name in (
            "six_step_numerical",
            "hundred_step_numerical",
            "legacy_same_backend_restart_exact",
            "canary_same_backend_restart_exact",
            "initial_q_identity",
        )
    )
    architecture_gates = all(diagnostic["gates"].values()) and all(
        decision["gates"].values()
    )
    gates = {
        "inherited_o4_science_and_restart": inherited_science_gate,
        "architecture_aware_transform_and_lifecycle": architecture_gates,
        "r320_performance_non_regression": performance_gate,
        "r320_phase_aware_memory_non_regression": memory_gate,
    }
    accepted = all(gates.values())
    if _sha256(o4_path) != expected_stage_o4_sha256 or any(
        _sha256(path) != identity
        for path, identity in zip(profile_paths, profile_sha256, strict=True)
    ):
        raise RuntimeError("O.4.1 input changed while it was being analyzed")
    return {
        "schema_version": 1,
        "qualification_stage": "O.4.1",
        "classification": "A_recommended" if accepted else "B_neutral",
        "eligible_for_stage_o5_decision": accepted,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "stage_o4_original_classification_unchanged": True,
        "stage_o4_evidence": {
            "path": str(o4_path),
            "sha256": expected_stage_o4_sha256,
            "classification": "B_neutral",
            "gate_failures": list(o4_report["gate_failures"]),
        },
        "contract_correction": {
            "r128_role": "diagnostic_fixed_overhead_scale",
            "r320_role": "production_decision_scale",
            "cross_runtime_transform_identity_required": False,
            "per_runtime_transform_plan_stability_required": True,
            "canary_transform_nonincrease_required": True,
            "strict_transform_reduction_required": True,
            "phase_aware_memory_required": True,
        },
        "fixed_limits": {
            "maximum_mean_timestep_ratio": maximum_mean_timestep_ratio,
            "maximum_paired_timestep_ratio": maximum_paired_timestep_ratio,
            "minimum_paired_pass_count": 2,
            "maximum_memory_ratio": maximum_memory_ratio,
            "retained_growth_floor_bytes": (
                STAGE_O41_RETAINED_GROWTH_FLOOR_BYTES
            ),
            "retained_growth_fraction": STAGE_O41_RETAINED_GROWTH_FRACTION,
        },
        "gates": gates,
        "gate_failures": [name for name, passed in gates.items() if not passed],
        "r128_diagnostic": diagnostic,
        "r320_decision": decision,
        "profile_inputs": [
            {"path": str(path), "sha256": identity}
            for path, identity in zip(profile_paths, profile_sha256, strict=True)
        ],
    }


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage O.4.1 evidence")
    parser.add_argument("--stage-o4-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o4-sha256", required=True)
    parser.add_argument(
        "--legacy-r128-profile", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--canary-r128-profile", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--legacy-r320-profile", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--canary-r320-profile", type=Path, action="append", required=True
    )
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_o41_qualification(
        args.stage_o4_report,
        args.legacy_r128_profile,
        args.canary_r128_profile,
        args.legacy_r320_profile,
        args.canary_r320_profile,
        expected_stage_o4_sha256=args.expected_stage_o4_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["classification"] == "A_recommended" else 1


__all__ = [
    "STAGE_O41_DECISION_SHAPE",
    "STAGE_O41_DIAGNOSTIC_SHAPE",
    "analyze_stage_o41_qualification",
    "analysis_main",
]
