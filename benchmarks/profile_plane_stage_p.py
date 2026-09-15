"""Profile the Plane production or separated runtime for Stage P.

Authoritative throughput, operator/kernel events, and nested semantic regions
are measured in separate windows so profiler instrumentation is never treated
as production timing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from benchmarks.profile_beris_edwards_timestep import (
    ProfileConfig,
    RegionTimer,
    _build_solver,
    _counter_delta,
    _dynamo_counter_snapshot,
)
from pssolver.execution import AlgebraicExecutionPolicy
from pssolver.experimental._shadow_support import file_sha256
from pssolver.experimental.h100_shadow_qualification import (
    _cuda_identity,
    _timing_summary,
    _write_new_json,
)
from pssolver.experimental.plane_shadow_driver import (
    ProductionPlaneReference,
    build_h100_plane_shadow_runtime_from_production_metadata,
    load_h100_production_plane_reference,
)
from pssolver.experimental.stage_o_closure import (
    build_stage_o_closure_decision,
    stage_o_closure_identity_sha256,
)
from pssolver.experimental.stage_p_diagnostics import (
    STAGE_P_MAXIMUM_DISTINCT_EVENTS,
    STAGE_P_OPERATOR_AUDIT_STEPS,
    STAGE_P_RUNTIME_ROLES,
    STAGE_P_SEMANTIC_STEPS,
    STAGE_P_SHAPE,
    STAGE_P_THROUGHPUT_STEPS,
    STAGE_P_TRANSFORM_MILLISECONDS_KEY,
    STAGE_P_WARMUP_STEPS,
    summarize_operator_kernel_events,
)
from pssolver.experimental.shadow_metadata import (
    plane_beris_edwards_production_signature,
)


@dataclass(slots=True)
class _RoleExecutor:
    owner: object
    fields: object
    step: Callable[[], None]
    enable_semantic: Callable[[], None]
    semantic_snapshot: Callable[[], Mapping[str, object]]


def _production_config(
    reference: ProductionPlaneReference,
    *,
    warmup_steps: int,
    profile_steps: int,
) -> ProfileConfig:
    signature = plane_beris_edwards_production_signature(reference.metadata)
    solver = signature["solver"]
    numerical = signature["numerics"]
    return ProfileConfig(
        shape=tuple(solver["shape"]),
        lengths=tuple(solver["lengths"]),
        device="cuda",
        dtype=str(solver["real_dtype"]),
        dt=float(solver["dt"]),
        dealias_rule=str(numerical["dealias_rule"]),
        projected_transform_execution=str(
            numerical["projected_transform_execution"]
        ),
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        spectral_refresh_interval=numerical[
            "spectral_refresh_interval_steps"
        ],
        pressure_diagnostics=bool(
            reference.metadata["numerics"]["pressure_residual_diagnostics"]
        ),
        reuse_q_gradients=True,
        molecular_field_linear_space=str(
            numerical["molecular_field_linear_space"]
        ),
        stress_divergence_sum_space=str(
            numerical["stress_divergence_sum_space"]
        ),
        pointwise_execution=str(numerical["pointwise_execution"]),
        transform_execution_order=str(solver["transform_execution_order"]),
        spectral_storage=str(solver["spectral_storage"]),
        initial_q_path=str(reference.initial_q_path),
        timing_scope="all_regions",
    )


def _build_executor(
    role: str,
    reference: ProductionPlaneReference,
    *,
    expected_gpu_name: str,
    instrument_semantic: bool,
) -> _RoleExecutor:
    if role == "legacy_production":
        timer = RegionTimer(torch.device("cuda:0"))
        config = _production_config(
            reference,
            warmup_steps=STAGE_P_WARMUP_STEPS,
            profile_steps=STAGE_P_THROUGHPUT_STEPS,
        )
        solver = _build_solver(config, timer)
        return _RoleExecutor(
            owner=solver,
            fields=solver.model.fields,
            step=solver.integrator.step,
            enable_semantic=timer.reset,
            semantic_snapshot=timer.summarize,
        )
    if role != "separated_canary":
        raise ValueError(f"unsupported Stage P runtime role: {role!r}")
    runtime, comparison = (
        build_h100_plane_shadow_runtime_from_production_metadata(
            reference.metadata,
            expected_gpu_name=expected_gpu_name,
            device="cuda:0",
            enable_performance_instrumentation=instrument_semantic,
            algebraic_execution_policy=AlgebraicExecutionPolicy.batched(),
        )
    )
    comparison.require_compatible()
    initial_values = {
        name: value.to(device="cuda:0")
        for name, value in reference.initial_values.items()
    }
    runtime.reset(initial_values)
    recorder = runtime.performance_recorder

    def enable_semantic() -> None:
        if recorder is None:
            raise RuntimeError("Stage P canary lacks semantic instrumentation")
        recorder.reset()

    def semantic_snapshot() -> Mapping[str, object]:
        if recorder is None:
            raise RuntimeError("Stage P canary lacks semantic instrumentation")
        return recorder.snapshot()

    return _RoleExecutor(
        owner=runtime,
        fields=runtime.solver.fields,
        step=lambda: runtime.solver.run(1),
        enable_semantic=enable_semantic,
        semantic_snapshot=semantic_snapshot,
    )


def _fields_are_finite(fields: object) -> bool:
    return bool(torch.isfinite(fields.spatial).all().item()) and bool(
        torch.isfinite(fields.spectral).all().item()
    )


def _release_cuda_cache() -> None:
    gc.collect()
    torch.cuda.synchronize("cuda:0")
    torch.cuda.empty_cache()


def _run_steps(executor: _RoleExecutor, steps: int) -> None:
    for _ in range(steps):
        executor.step()


def _throughput_window(
    role: str,
    reference: ProductionPlaneReference,
    *,
    expected_gpu_name: str,
) -> tuple[dict[str, object], bool]:
    executor = _build_executor(
        role,
        reference,
        expected_gpu_name=expected_gpu_name,
        instrument_semantic=False,
    )
    _run_steps(executor, STAGE_P_WARMUP_STEPS)
    torch.cuda.synchronize("cuda:0")
    torch.cuda.reset_peak_memory_stats("cuda:0")
    events = []
    for _ in range(STAGE_P_THROUGHPUT_STEPS):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        executor.step()
        stop.record()
        events.append((start, stop))
    torch.cuda.synchronize("cuda:0")
    samples = tuple(
        start.elapsed_time(stop) / 1000.0 for start, stop in events
    )
    result = {
        **_timing_summary(samples),
        "instrumentation": "outer_cuda_events_only",
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda:0")),
    }
    finite = _fields_are_finite(executor.fields)
    del executor, events
    _release_cuda_cache()
    return result, finite


def _operator_kernel_window(
    role: str,
    reference: ProductionPlaneReference,
    *,
    expected_gpu_name: str,
) -> tuple[dict[str, object], dict[str, int | None], bool]:
    counters_before = _dynamo_counter_snapshot()
    executor = _build_executor(
        role,
        reference,
        expected_gpu_name=expected_gpu_name,
        instrument_semantic=False,
    )
    _run_steps(executor, STAGE_P_WARMUP_STEPS)
    torch.cuda.synchronize("cuda:0")
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        profile_memory=False,
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        for _ in range(STAGE_P_OPERATOR_AUDIT_STEPS):
            with torch.profiler.record_function("stage_p.timestep"):
                executor.step()
    torch.cuda.synchronize("cuda:0")
    summary = summarize_operator_kernel_events(
        tuple(profiler.events()),
        tuple(profiler.key_averages()),
        steps=STAGE_P_OPERATOR_AUDIT_STEPS,
        maximum_distinct_events=STAGE_P_MAXIMUM_DISTINCT_EVENTS,
    )
    counters_after = _dynamo_counter_snapshot()
    finite = _fields_are_finite(executor.fields)
    del profiler, executor
    _release_cuda_cache()
    return summary, _counter_delta(counters_after, counters_before), finite


def _region_total(
    regions: Mapping[str, object],
    names: Sequence[str],
) -> tuple[float, int]:
    seconds = 0.0
    calls = 0
    for name in names:
        raw = regions.get(name)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"semantic region {name!r} is malformed")
        seconds += float(raw["total_seconds"])
        calls += int(raw["calls"])
    if not math.isfinite(seconds) or seconds < 0.0 or calls < 0:
        raise ValueError("semantic region aggregate is invalid")
    return seconds, calls


def _matched_semantic_regions(
    role: str,
    raw: Mapping[str, object],
) -> dict[str, object]:
    regions = raw if role == "legacy_production" else raw["regions"]
    if not isinstance(regions, Mapping):
        raise ValueError("Stage P semantic regions are missing")
    names = {
        "legacy_production": {
            "total": ("whole_timestep",),
            "transform": ("transform_forward", "transform_inverse"),
            "algebraic": ("static_fields",),
            "explicit_rhs": ("q_nonlinear",),
            "spectral_update": ("imex_and_dealias",),
            "dynamic_inverse": ("dynamic_inverse",),
            "spectral_refresh": ("spectral_refresh",),
        },
        "separated_canary": {
            "total": ("timestep.total",),
            "transform": ("transform.forward", "transform.inverse"),
            "algebraic": ("timestep.algebraic_update",),
            "explicit_rhs": ("timestep.explicit_rhs",),
            "spectral_update": (
                "timestep.spectral_update",
                "timestep.dynamic_projection",
            ),
            "dynamic_inverse": ("timestep.dynamic_inverse",),
            "spectral_refresh": ("timestep.spectral_refresh",),
        },
    }[role]
    result: dict[str, object] = {
        "semantic_steps": STAGE_P_SEMANTIC_STEPS,
        "nested_regions_overlap": True,
        "separate_from_authoritative_throughput": True,
    }
    for label, region_names in names.items():
        seconds, calls = _region_total(regions, region_names)
        milliseconds_key = (
            STAGE_P_TRANSFORM_MILLISECONDS_KEY
            if label == "transform"
            else f"{label}_milliseconds_per_step"
        )
        result[milliseconds_key] = (
            1000.0 * seconds / STAGE_P_SEMANTIC_STEPS
        )
        result[f"{label}_calls_per_step"] = calls / STAGE_P_SEMANTIC_STEPS
    return result


def _semantic_window(
    role: str,
    reference: ProductionPlaneReference,
    *,
    expected_gpu_name: str,
) -> tuple[dict[str, object], Mapping[str, object], bool]:
    executor = _build_executor(
        role,
        reference,
        expected_gpu_name=expected_gpu_name,
        instrument_semantic=True,
    )
    _run_steps(executor, STAGE_P_WARMUP_STEPS)
    executor.enable_semantic()
    _run_steps(executor, STAGE_P_SEMANTIC_STEPS)
    raw = executor.semantic_snapshot()
    matched = _matched_semantic_regions(role, raw)
    finite = _fields_are_finite(executor.fields)
    del executor
    _release_cuda_cache()
    return matched, raw, finite


def profile_plane_stage_p(
    production_directory: str | Path,
    *,
    runtime_role: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Run three isolated diagnostic windows for one runtime role."""

    if runtime_role not in STAGE_P_RUNTIME_ROLES:
        raise ValueError(f"runtime_role must be one of {STAGE_P_RUNTIME_ROLES}")
    reference = load_h100_production_plane_reference(
        production_directory,
        expected_gpu_name=expected_gpu_name,
    )
    signature = plane_beris_edwards_production_signature(reference.metadata)
    if tuple(signature["solver"]["shape"]) != STAGE_P_SHAPE:
        raise ValueError(f"Stage P requires production shape {STAGE_P_SHAPE}")
    project_root = Path(__file__).parents[1]
    closure = build_stage_o_closure_decision(project_root)
    closure_sha256 = stage_o_closure_identity_sha256(closure)
    metadata_path = reference.directory / "metadata.json"
    metadata_sha_before = file_sha256(metadata_path)
    q_sha_before = file_sha256(reference.initial_q_path)
    environment = _cuda_identity(
        expected_gpu_name=expected_gpu_name,
        device=torch.device("cuda:0"),
    )

    throughput, throughput_finite = _throughput_window(
        runtime_role,
        reference,
        expected_gpu_name=expected_gpu_name,
    )
    audit, dynamo, audit_finite = _operator_kernel_window(
        runtime_role,
        reference,
        expected_gpu_name=expected_gpu_name,
    )
    matched, raw_semantic, semantic_finite = _semantic_window(
        runtime_role,
        reference,
        expected_gpu_name=expected_gpu_name,
    )
    if file_sha256(metadata_path) != metadata_sha_before or file_sha256(
        reference.initial_q_path
    ) != q_sha_before:
        raise RuntimeError("Stage P production reference changed during profiling")
    finite = throughput_finite and audit_finite and semantic_finite
    if not finite:
        raise RuntimeError("Stage P profile ended with non-finite fields")
    solver = signature["solver"]
    numerical = signature["numerics"]
    return {
        "schema_version": 1,
        "qualification_stage": "P",
        "classification": "OPERATOR_KERNEL_PROFILE_COMPLETE",
        "runtime_role": runtime_role,
        "runtime_variant": (
            "optimized_legacy_plane"
            if runtime_role == "legacy_production"
            else "stage_n41_separated_canary"
        ),
        "stage_o_closure_sha256": closure_sha256,
        "production_directory": str(reference.directory),
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "configuration": {
            "shape": list(solver["shape"]),
            "lengths": list(solver["lengths"]),
            "dtype": solver["real_dtype"],
            "dt": solver["dt"],
            "dealias_rule": numerical["dealias_rule"],
            "projected_transform_execution": numerical[
                "projected_transform_execution"
            ],
            "spectral_storage": solver["spectral_storage"],
            "transform_execution_order": solver["transform_execution_order"],
            "pointwise_execution": numerical["pointwise_execution"],
            "spectral_refresh_interval": numerical[
                "spectral_refresh_interval_steps"
            ],
            "warmup_steps": STAGE_P_WARMUP_STEPS,
            "throughput_steps": STAGE_P_THROUGHPUT_STEPS,
            "operator_audit_steps": STAGE_P_OPERATOR_AUDIT_STEPS,
            "semantic_steps": STAGE_P_SEMANTIC_STEPS,
        },
        "environment": environment,
        "throughput": throughput,
        "operator_kernel_audit": audit,
        "dynamo_delta": dynamo,
        "matched_semantic_regions": matched,
        "raw_semantic_regions": raw_semantic,
        "measurement_contract": {
            "throughput_operator_and_semantic_windows_are_separate": True,
            "throughput_uses_internal_semantic_instrumentation": False,
            "operator_profile_is_authoritative_throughput": False,
            "raw_trace_retained": False,
            "changes_equations": False,
            "changes_runtime_implementation": False,
            "changes_production_default": False,
        },
        "finite": True,
        "eligible_for_stage_q_optimization_candidate": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-role",
        choices=STAGE_P_RUNTIME_ROLES,
        required=True,
    )
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = profile_plane_stage_p(
        args.production_reference_dir,
        runtime_role=args.runtime_role,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
