"""Bounded H100 diagnostics for the qualified Stage M shadow runtime."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics

import torch

from ._shadow_support import file_sha256
from .h100_shadow_qualification import (
    _cuda_identity,
    _cuda_step_samples,
    _require_positive_integer,
    _timing_summary,
    _write_new_json,
)
from .plane_shadow_driver import (
    build_h100_plane_shadow_runtime_from_production_metadata,
    load_h100_production_plane_reference,
)


_AUDITED_OPERATORS = frozenset(
    {
        "aten::_to_copy",
        "aten::cat",
        "aten::clone",
        "aten::contiguous",
        "aten::copy_",
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::stack",
        "aten::to",
    }
)
_TOP_LEVEL_REGIONS = (
    "timestep.algebraic_update",
    "timestep.explicit_rhs",
    "timestep.spectral_update",
    "timestep.dynamic_projection",
    "timestep.dynamic_inverse",
    "timestep.spectral_refresh",
)


def _event_integer(event: object, name: str) -> int:
    value = getattr(event, name, 0)
    if value is None:
        return 0
    return int(value)


def summarize_audited_operators(events: Sequence[object]) -> dict[str, object]:
    """Reduce selected allocation and movement operators to stable counters."""

    totals = {
        name: {
            "calls": 0,
            "self_cpu_memory_bytes": 0,
            "self_device_memory_bytes": 0,
        }
        for name in sorted(_AUDITED_OPERATORS)
    }
    for event in events:
        name = str(getattr(event, "key", ""))
        if name not in totals:
            continue
        entry = totals[name]
        entry["calls"] += _event_integer(event, "count")
        entry["self_cpu_memory_bytes"] += _event_integer(
            event,
            "self_cpu_memory_usage",
        )
        device_memory = getattr(event, "self_device_memory_usage", None)
        if device_memory is None:
            device_memory = getattr(event, "self_cuda_memory_usage", 0)
        entry["self_device_memory_bytes"] += int(device_memory or 0)
    return {
        "schema_version": 1,
        "operators": totals,
        "total_selected_calls": sum(
            int(value["calls"]) for value in totals.values()
        ),
    }


def _operator_audit(runtime, steps: int) -> dict[str, object]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if runtime.context.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        runtime.solver.run(steps)
    if runtime.context.device.type == "cuda":
        torch.cuda.synchronize(runtime.context.device)
    return summarize_audited_operators(tuple(profiler.key_averages()))


def profile_stage_n_h100_shadow(
    production_directory: str | Path,
    *,
    warmup_steps: int,
    profile_steps: int,
    operator_audit_steps: int,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Measure semantic phases and selected tensor operators on one H100."""

    _require_positive_integer(warmup_steps, "warmup_steps")
    _require_positive_integer(profile_steps, "profile_steps")
    _require_positive_integer(operator_audit_steps, "operator_audit_steps")
    reference = load_h100_production_plane_reference(
        production_directory,
        expected_gpu_name=expected_gpu_name,
    )
    requested_device = torch.device("cuda")
    environment = _cuda_identity(
        expected_gpu_name=expected_gpu_name,
        device=requested_device,
    )
    runtime, comparison = (
        build_h100_plane_shadow_runtime_from_production_metadata(
            reference.metadata,
            expected_gpu_name=expected_gpu_name,
            device=requested_device,
            enable_performance_instrumentation=True,
        )
    )
    recorder = runtime.performance_recorder
    if recorder is None:
        raise RuntimeError("Stage N runtime lacks performance instrumentation")
    initial_values = {
        name: value.to(device=runtime.context.device)
        for name, value in reference.initial_values.items()
    }

    runtime.reset(initial_values)
    _cuda_step_samples(runtime, warmup_steps)
    recorder.reset()
    torch.cuda.reset_peak_memory_stats(runtime.context.device)
    samples = _cuda_step_samples(runtime, profile_steps)
    semantic_regions = recorder.snapshot()
    peak_allocated = int(
        torch.cuda.max_memory_allocated(runtime.context.device)
    )
    peak_reserved = int(
        torch.cuda.max_memory_reserved(runtime.context.device)
    )

    runtime.reset(initial_values)
    _cuda_step_samples(runtime, warmup_steps)
    recorder.reset()
    operator_audit = _operator_audit(runtime, operator_audit_steps)
    recorder.reset()

    fields = runtime.solver.fields
    finite = bool(torch.isfinite(fields.spatial).all().item()) and bool(
        torch.isfinite(fields.spectral).all().item()
    )
    if not finite:
        raise RuntimeError("Stage N diagnostic ended with non-finite fields")
    signature = comparison.production_signature
    solver = signature["solver"]
    numerical = signature["numerics"]
    return {
        "schema_version": 1,
        "qualification_stage": "N",
        "measurement_role": "shadow_runtime_diagnostic",
        "classification": "DIAGNOSTIC_COMPLETE",
        "production_path_changed": False,
        "eligible_for_production_promotion": False,
        "production_directory": str(reference.directory),
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "environment": environment,
        "configuration": {
            "shape": solver["shape"],
            "lengths": solver["lengths"],
            "dtype": solver["real_dtype"],
            "dt": solver["dt"],
            "warmup_steps": warmup_steps,
            "profile_steps": profile_steps,
            "operator_audit_steps": operator_audit_steps,
            "dealias_rule": numerical["dealias_rule"],
            "projected_transform_execution": numerical[
                "projected_transform_execution"
            ],
            "spectral_storage": solver["spectral_storage"],
            "spectral_refresh_interval": numerical[
                "spectral_refresh_interval_steps"
            ],
        },
        "throughput": _timing_summary(samples),
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "semantic_regions": semantic_regions,
        "operator_audit": {
            **operator_audit,
            "steps": operator_audit_steps,
            "separate_from_semantic_timing": True,
        },
        "configuration_comparison": comparison.to_metadata(),
        "finite": True,
    }


def _load_diagnostic(path: str | Path) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Stage N diagnostic is missing: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Stage N diagnostic must contain a JSON object")
    return resolved, value


def analyze_stage_n_diagnostics(
    diagnostic_paths: Sequence[str | Path],
) -> dict[str, object]:
    """Aggregate independent diagnostics without declaring an optimization."""

    loaded = tuple(_load_diagnostic(path) for path in diagnostic_paths)
    if not loaded:
        raise ValueError("at least one Stage N diagnostic is required")
    reports = tuple(value for _, value in loaded)
    for report in reports:
        if (
            report.get("qualification_stage") != "N"
            or report.get("classification") != "DIAGNOSTIC_COMPLETE"
            or report.get("finite") is not True
        ):
            raise ValueError("Stage N diagnostic is incomplete")
        semantic = report.get("semantic_regions")
        if (
            not isinstance(semantic, Mapping)
            or semantic.get("enabled") is not True
        ):
            raise ValueError("Stage N semantic instrumentation was not enabled")
        regions = semantic.get("regions")
        if not isinstance(regions, Mapping):
            raise ValueError("Stage N semantic regions are missing")
        required_regions = {
            "timestep.total",
            "transform.forward",
            "transform.inverse",
            *_TOP_LEVEL_REGIONS,
        }
        missing_regions = tuple(sorted(required_regions - set(regions)))
        if missing_regions:
            raise ValueError(
                f"Stage N semantic regions are incomplete: {missing_regions!r}"
            )
        if not any(
            name.startswith("algebraic.") and name.endswith(".solve")
            for name in regions
        ):
            raise ValueError("Stage N algebraic solver regions are missing")
        audit = report.get("operator_audit")
        if not isinstance(audit, Mapping) or not isinstance(
            audit.get("operators"),
            Mapping,
        ):
            raise ValueError("Stage N operator audit is missing")
        if set(audit["operators"]) != _AUDITED_OPERATORS:
            raise ValueError("Stage N operator audit keys are inconsistent")
        _require_positive_integer(audit.get("steps"), "operator audit steps")
    identities = {
        json.dumps(
            {
                "configuration": report.get("configuration"),
                "production_metadata_sha256": report.get(
                    "production_metadata_sha256"
                ),
                "production_initial_q_sha256": report.get(
                    "production_initial_q_sha256"
                ),
            },
            allow_nan=False,
            sort_keys=True,
        )
        for report in reports
    }
    if len(identities) != 1:
        raise ValueError("Stage N diagnostic identities differ")

    region_names = set.intersection(
        *(
            set(report["semantic_regions"]["regions"])
            for report in reports
        )
    )
    region_summary = {}
    for name in sorted(region_names):
        values = [
            report["semantic_regions"]["regions"][name]
            for report in reports
        ]
        calls = {int(value["calls"]) for value in values}
        if len(calls) != 1:
            raise ValueError(f"Stage N region call counts differ for {name!r}")
        per_call = [float(value["mean_seconds_per_call"]) for value in values]
        fractions = [
            float(value["fraction_of_timestep_total"])
            for value in values
            if value["fraction_of_timestep_total"] is not None
        ]
        region_summary[name] = {
            "calls_per_trial": calls.pop(),
            "mean_seconds_per_call_across_trials": (
                math.fsum(per_call) / len(per_call)
            ),
            "median_seconds_per_call_across_trials": statistics.median(
                per_call
            ),
            "mean_fraction_of_timestep_total": (
                math.fsum(fractions) / len(fractions)
                if fractions
                else None
            ),
        }

    top_level = [
        (name, region_summary[name])
        for name in _TOP_LEVEL_REGIONS
        if name in region_summary
    ]
    top_level.sort(
        key=lambda item: float(
            item[1]["mean_fraction_of_timestep_total"] or 0.0
        ),
        reverse=True,
    )
    algebraic = [
        (name, value)
        for name, value in region_summary.items()
        if name.startswith("algebraic.") and name.endswith(".solve")
    ]
    algebraic.sort(
        key=lambda item: float(item[1]["mean_seconds_per_call_across_trials"]),
        reverse=True,
    )

    operator_summary = {}
    for name in sorted(_AUDITED_OPERATORS):
        calls_per_step = []
        for report in reports:
            audit = report["operator_audit"]
            calls = audit["operators"][name]["calls"]
            calls_per_step.append(float(calls) / int(audit["steps"]))
        operator_summary[name] = {
            "mean_calls_per_step": math.fsum(calls_per_step)
            / len(calls_per_step),
            "median_calls_per_step": statistics.median(calls_per_step),
        }

    timestep_means = [
        float(report["throughput"]["mean_timestep_seconds"])
        for report in reports
    ]
    return {
        "schema_version": 1,
        "qualification_stage": "N",
        "classification": "DIAGNOSTIC_COMPLETE",
        "architecture_decision": "retain_shadow_without_promotion",
        "eligible_for_stage_n1_optimization_design": True,
        "eligible_for_production_promotion": False,
        "trial_count": len(reports),
        "mean_timestep_seconds_across_trials": (
            math.fsum(timestep_means) / len(timestep_means)
        ),
        "median_timestep_seconds_across_trials": statistics.median(
            timestep_means
        ),
        "semantic_regions": region_summary,
        "ranked_top_level_regions": [name for name, _ in top_level],
        "ranked_algebraic_solvers": [name for name, _ in algebraic],
        "operator_counts": operator_summary,
        "interpretation_constraints": {
            "semantic_regions_are_nested": True,
            "operator_audit_is_separate_from_semantic_timing": True,
            "no_production_timing_is_inferred": True,
            "no_optimization_is_authorized": True,
        },
        "inputs": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path, _ in loaded
        ],
    }


def _profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile Stage N semantic regions and tensor operators."
    )
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--operator-audit-steps", type=int, default=2)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def profile_main(argv: Sequence[str] | None = None) -> int:
    args = _profile_parser().parse_args(argv)
    report = profile_stage_n_h100_shadow(
        args.production_reference_dir,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        operator_audit_steps=args.operator_audit_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _analysis_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate bounded Stage N shadow diagnostics."
    )
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def analysis_main(argv: Sequence[str] | None = None) -> int:
    args = _analysis_parser().parse_args(argv)
    report = analyze_stage_n_diagnostics(args.profile)
    _write_new_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "analysis_main",
    "analyze_stage_n_diagnostics",
    "profile_main",
    "profile_stage_n_h100_shadow",
    "summarize_audited_operators",
]
