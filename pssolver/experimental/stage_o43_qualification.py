"""Stage O.4.3 packed-publication and zero-copy assembly qualification."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from pssolver.diagnostics import cuda_memory_snapshot, cuda_memory_window
from pssolver.execution import (
    AlgebraicExecutionPolicy,
    AlgebraicOutputPublicationPolicy,
)

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
from .projected_scheduler import ProjectedBatchAssemblyPolicy
from .shadow_run import ExperimentalPlaneShadowRun
from .stage_n1_qualification import _load_json
from .stage_n4_qualification import _initial_values
from .stage_o4_qualification import _sha256


STAGE_O43_PROFILE_TRIALS = 3
STAGE_O43_DIAGNOSTIC_SHAPE = (128, 128, 32)
STAGE_O43_DECISION_SHAPE = (320, 320, 80)
STAGE_O43_TRAJECTORY_STEPS = 100
STAGE_O43_RELATIVE_L2_TOLERANCE = 1.0e-10
STAGE_O43_MAXIMUM_MEAN_TIMESTEP_RATIO = 0.98
STAGE_O43_MAXIMUM_PAIRED_TIMESTEP_RATIO = 1.02
STAGE_O43_MAXIMUM_MEMORY_RATIO = 1.03

_ROLES = ("baseline", "candidate")
_FIELDS = ("Q", "u", "p")


@dataclass(frozen=True, slots=True)
class _StageO43Variant:
    """Internal contract for one storage-layout qualification variant."""

    qualification_stage: str
    baseline_output_policy: AlgebraicOutputPublicationPolicy
    baseline_batch_policy: ProjectedBatchAssemblyPolicy
    baseline_producer_output_layout: str
    candidate_output_policy: AlgebraicOutputPublicationPolicy
    candidate_batch_policy: ProjectedBatchAssemblyPolicy
    candidate_producer_output_layout: str
    architecture_decision: str
    metrics_filename: str
    structural_contract: str


_PACKED_PUBLICATION_VARIANT = _StageO43Variant(
    qualification_stage="O.4.3",
    baseline_output_policy=AlgebraicOutputPublicationPolicy.deferred_stack(),
    baseline_batch_policy=ProjectedBatchAssemblyPolicy.copy_cat(),
    baseline_producer_output_layout="component_mapping",
    candidate_output_policy=AlgebraicOutputPublicationPolicy.preallocated(),
    candidate_batch_policy=(
        ProjectedBatchAssemblyPolicy.contiguous_storage_view()
    ),
    candidate_producer_output_layout="component_mapping",
    architecture_decision="packed_generation_storage_with_safe_views",
    metrics_filename="stage_o43_metrics.json",
    structural_contract="scheduler_view_adoption",
)

_OPPORTUNISTIC_VIEW_VARIANT = _StageO43Variant(
    qualification_stage="O.4.3.1",
    baseline_output_policy=AlgebraicOutputPublicationPolicy.deferred_stack(),
    baseline_batch_policy=ProjectedBatchAssemblyPolicy.copy_cat(),
    baseline_producer_output_layout="component_mapping",
    candidate_output_policy=AlgebraicOutputPublicationPolicy.deferred_stack(),
    candidate_batch_policy=(
        ProjectedBatchAssemblyPolicy.contiguous_storage_view()
    ),
    candidate_producer_output_layout="component_mapping",
    architecture_decision=(
        "opportunistic_natural_storage_views_without_republication"
    ),
    metrics_filename="stage_o431_metrics.json",
    structural_contract="scheduler_view_adoption",
)

_PRODUCER_PACKED_VARIANT = _StageO43Variant(
    qualification_stage="O.4.3.3",
    baseline_output_policy=AlgebraicOutputPublicationPolicy.deferred_stack(),
    baseline_batch_policy=(
        ProjectedBatchAssemblyPolicy.contiguous_storage_view()
    ),
    baseline_producer_output_layout="component_mapping",
    candidate_output_policy=AlgebraicOutputPublicationPolicy.deferred_stack(),
    candidate_batch_policy=(
        ProjectedBatchAssemblyPolicy.contiguous_storage_view()
    ),
    candidate_producer_output_layout="boundary_packed",
    architecture_decision="producer_owned_boundary_packed_h_and_stress",
    metrics_filename="stage_o433_metrics.json",
    structural_contract="producer_owned_packing",
)


def _require_role(role: str) -> str:
    if role not in _ROLES:
        raise ValueError("role must be 'baseline' or 'candidate'")
    return role


def _policies(
    role: str,
    variant: _StageO43Variant = _PACKED_PUBLICATION_VARIANT,
):
    role = _require_role(role)
    if role == "baseline":
        return (
            variant.baseline_output_policy,
            variant.baseline_batch_policy,
            variant.baseline_producer_output_layout,
        )
    return (
        variant.candidate_output_policy,
        variant.candidate_batch_policy,
        variant.candidate_producer_output_layout,
    )


def _build_runtime(
    reference,
    *,
    role: str,
    expected_gpu_name: str,
    device: torch.device,
    instrument: bool,
    variant: _StageO43Variant = _PACKED_PUBLICATION_VARIANT,
):
    output_policy, batch_policy, producer_output_layout = _policies(
        role,
        variant,
    )
    return build_h100_plane_shadow_runtime_from_production_metadata(
        reference.metadata,
        expected_gpu_name=expected_gpu_name,
        device=device,
        enable_performance_instrumentation=instrument,
        algebraic_execution_policy=AlgebraicExecutionPolicy.batched(),
        algebraic_output_publication_policy=output_policy,
        projected_batch_assembly_policy=batch_policy,
        producer_output_layout=producer_output_layout,
    )


def _producer_output_snapshot(runtime) -> dict[str, object]:
    """Return tensor-free H/stress producer layout provenance."""

    policies = {}
    for resolved in runtime.resolved_algebraic_systems:
        name = resolved.system.name
        if name not in ("molecular_field", "nematic_stress"):
            continue
        observability = resolved.to_metadata()["observability"]
        numerical = dict(observability["numerical_policy"])
        policies[name] = {
            "producer_output_layout": numerical.get(
                "producer_output_layout",
                "component_mapping",
            ),
            "producer_storage_order": numerical.get(
                "producer_storage_order"
            ),
            "producer_packing": numerical.get("producer_packing"),
        }
    if set(policies) != {"molecular_field", "nematic_stress"}:
        raise RuntimeError("Stage O.4.3 producer provenance is incomplete")
    return policies


def _profile_stage_o43_variant_h100_runtime(
    production_directory: str | Path,
    *,
    role: str,
    warmup_steps: int = 10,
    profile_steps: int = 20,
    expected_gpu_name: str = "H100",
    variant: _StageO43Variant,
) -> dict[str, object]:
    """Profile one policy while retaining only scalar diagnostics."""

    role = _require_role(role)
    _require_positive_integer(warmup_steps, "warmup_steps")
    _require_positive_integer(profile_steps, "profile_steps")
    reference = load_h100_production_plane_reference(
        production_directory,
        expected_gpu_name=expected_gpu_name,
    )
    device = torch.device("cuda")
    environment = _cuda_identity(
        expected_gpu_name=expected_gpu_name,
        device=device,
    )
    runtime, comparison = _build_runtime(
        reference,
        role=role,
        expected_gpu_name=expected_gpu_name,
        device=device,
        instrument=True,
        variant=variant,
    )
    runtime.reset(_initial_values(reference, device))
    _cuda_step_samples(runtime, warmup_steps)
    torch.cuda.synchronize(device)
    start_memory = cuda_memory_snapshot(device)
    torch.cuda.reset_peak_memory_stats(device)
    recorder = runtime.performance_recorder
    if recorder is None:
        raise RuntimeError("Stage O.4.3 profile lacks instrumentation")
    recorder.reset()
    runtime.reset_projected_batch_assembly_diagnostics()
    samples = _cuda_step_samples(runtime, profile_steps)
    end_memory = cuda_memory_snapshot(device)
    peak_memory = cuda_memory_snapshot(device)
    memory = cuda_memory_window(start_memory, end_memory, peak_memory)
    regions = recorder.snapshot()["regions"]
    assembly = dict(runtime.projected_batch_assembly_diagnostics())
    fields = runtime.solver.fields
    finite = bool(torch.isfinite(fields.spatial).all().item()) and bool(
        torch.isfinite(fields.spectral).all().item()
    )
    if not finite:
        raise RuntimeError("Stage O.4.3 profile ended with non-finite fields")
    signature = comparison.production_signature
    return {
        "schema_version": 1,
        "qualification_stage": variant.qualification_stage,
        "classification": "PROFILE_COMPLETE",
        "measurement_role": role,
        "production_default_changed": False,
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "configuration": {
            "shape": signature["solver"]["shape"],
            "lengths": signature["solver"]["lengths"],
            "dtype": signature["solver"]["real_dtype"],
            "dt": signature["solver"]["dt"],
            "dealias_rule": signature["numerics"]["dealias_rule"],
            "projected_transform_execution": signature["numerics"][
                "projected_transform_execution"
            ],
            "spectral_storage": signature["solver"]["spectral_storage"],
            "spectral_refresh_interval": signature["numerics"][
                "spectral_refresh_interval_steps"
            ],
            "warmup_steps": warmup_steps,
            "profile_steps": profile_steps,
        },
        "configuration_comparison": comparison.to_metadata(),
        "algebraic_execution_policy": (
            runtime.algebraic_execution_policy.to_metadata()
        ),
        "algebraic_output_publication_policy": (
            runtime.algebraic_output_publication_policy.to_metadata()
        ),
        "projected_batch_assembly_policy": (
            runtime.projected_batch_assembly_policy.to_metadata()
        ),
        "explicit_rhs_execution": runtime.to_metadata()[
            "explicit_rhs_execution"
        ],
        "producer_outputs": _producer_output_snapshot(runtime),
        "batch_assembly_diagnostics": assembly,
        "throughput": _timing_summary(samples),
        "memory": memory,
        "semantic_regions": regions,
        "environment": environment,
        "finite": True,
    }


def profile_stage_o43_h100_runtime(
    production_directory: str | Path,
    *,
    role: str,
    warmup_steps: int = 10,
    profile_steps: int = 20,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Profile the original packed-publication Stage O.4.3 variant."""

    return _profile_stage_o43_variant_h100_runtime(
        production_directory,
        role=role,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        expected_gpu_name=expected_gpu_name,
        variant=_PACKED_PUBLICATION_VARIANT,
    )


def profile_stage_o431_h100_runtime(
    production_directory: str | Path,
    *,
    role: str,
    warmup_steps: int = 10,
    profile_steps: int = 20,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Profile safe natural-storage views without republishing outputs."""

    return _profile_stage_o43_variant_h100_runtime(
        production_directory,
        role=role,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        expected_gpu_name=expected_gpu_name,
        variant=_OPPORTUNISTIC_VIEW_VARIANT,
    )


def profile_stage_o433_h100_runtime(
    production_directory: str | Path,
    *,
    role: str,
    warmup_steps: int = 10,
    profile_steps: int = 20,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Profile producer-owned boundary-packed H and stress outputs."""

    return _profile_stage_o43_variant_h100_runtime(
        production_directory,
        role=role,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        expected_gpu_name=expected_gpu_name,
        variant=_PRODUCER_PACKED_VARIANT,
    )


def _run_stage_o43_variant_h100_trajectory(
    production_directory: str | Path,
    output_directory: str | Path,
    *,
    role: str,
    confirmed_steps: int = STAGE_O43_TRAJECTORY_STEPS,
    expected_gpu_name: str = "H100",
    variant: _StageO43Variant,
) -> dict[str, object]:
    """Run one bounded trajectory from an immutable production Q0."""

    role = _require_role(role)
    _require_positive_integer(confirmed_steps, "confirmed_steps")
    if confirmed_steps != STAGE_O43_TRAJECTORY_STEPS:
        raise ValueError("Stage O.4.3 trajectory must contain exactly 100 steps")
    reference = load_h100_production_plane_reference(
        production_directory,
        expected_gpu_name=expected_gpu_name,
    )
    device = torch.device("cuda")
    environment = _cuda_identity(
        expected_gpu_name=expected_gpu_name,
        device=device,
    )
    runtime, comparison = _build_runtime(
        reference,
        role=role,
        expected_gpu_name=expected_gpu_name,
        device=device,
        instrument=False,
        variant=variant,
    )
    run = ExperimentalPlaneShadowRun(
        runtime,
        output_directory,
        initial_values=_initial_values(reference, device),
        initial_condition_metadata={
            "name": "production_projected_q_snapshot",
            "source_run_directory": str(reference.directory),
            "source_step": 0,
            "source_metadata_sha256": reference.metadata_sha256,
            "source_q_file": reference.initial_q_path.name,
            "source_q_file_sha256": reference.initial_q_file_sha256,
            "source_projected_q_sha256": reference.metadata[
                "initial_condition"
            ]["projected_q_sha256"],
            "configuration_comparison": comparison.to_metadata(),
            "qualification_stage": variant.qualification_stage,
            "measurement_role": role,
        },
    )
    saved_steps = [run.save_observation().step]
    run.advance(confirmed_steps)
    saved_steps.append(run.save_observation().step)
    final = run.complete()
    result = {
        "schema_version": 1,
        "qualification_stage": variant.qualification_stage,
        "classification": "TRAJECTORY_COMPLETE",
        "measurement_role": role,
        "production_default_changed": False,
        "production_directory": str(reference.directory),
        "trajectory_directory": str(run.output_directory),
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "completed_steps": final.step,
        "saved_steps": saved_steps,
        "algebraic_output_publication_policy": (
            runtime.algebraic_output_publication_policy.to_metadata()
        ),
        "projected_batch_assembly_policy": (
            runtime.projected_batch_assembly_policy.to_metadata()
        ),
        "producer_outputs": _producer_output_snapshot(runtime),
        "environment": environment,
    }
    _write_new_json(run.output_directory / variant.metrics_filename, result)
    return result


def run_stage_o43_h100_trajectory(
    production_directory: str | Path,
    output_directory: str | Path,
    *,
    role: str,
    confirmed_steps: int = STAGE_O43_TRAJECTORY_STEPS,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Run the original packed-publication Stage O.4.3 trajectory."""

    return _run_stage_o43_variant_h100_trajectory(
        production_directory,
        output_directory,
        role=role,
        confirmed_steps=confirmed_steps,
        expected_gpu_name=expected_gpu_name,
        variant=_PACKED_PUBLICATION_VARIANT,
    )


def run_stage_o431_h100_trajectory(
    production_directory: str | Path,
    output_directory: str | Path,
    *,
    role: str,
    confirmed_steps: int = STAGE_O43_TRAJECTORY_STEPS,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Run the natural-storage-view Stage O.4.3.1 trajectory."""

    return _run_stage_o43_variant_h100_trajectory(
        production_directory,
        output_directory,
        role=role,
        confirmed_steps=confirmed_steps,
        expected_gpu_name=expected_gpu_name,
        variant=_OPPORTUNISTIC_VIEW_VARIANT,
    )


def run_stage_o433_h100_trajectory(
    production_directory: str | Path,
    output_directory: str | Path,
    *,
    role: str,
    confirmed_steps: int = STAGE_O43_TRAJECTORY_STEPS,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Run the producer-owned packing Stage O.4.3.3 trajectory."""

    return _run_stage_o43_variant_h100_trajectory(
        production_directory,
        output_directory,
        role=role,
        confirmed_steps=confirmed_steps,
        expected_gpu_name=expected_gpu_name,
        variant=_PRODUCER_PACKED_VARIANT,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_comparison(
    baseline: Path,
    candidate: Path,
    *,
    field: str,
    step: int,
) -> dict[str, object]:
    baseline_array = np.load(baseline, allow_pickle=False)
    candidate_array = np.load(candidate, allow_pickle=False)
    if (
        baseline_array.shape != candidate_array.shape
        or baseline_array.dtype != candidate_array.dtype
    ):
        raise ValueError("Stage O.4.3 trajectory array identity differs")
    if not np.isfinite(baseline_array).all() or not np.isfinite(
        candidate_array
    ).all():
        raise ValueError("Stage O.4.3 trajectory contains NaN or Inf")
    difference = candidate_array - baseline_array
    if field == "p":
        axes = tuple(range(baseline_array.ndim))
        baseline_gate = baseline_array - baseline_array.mean(axis=axes)
        candidate_gate = candidate_array - candidate_array.mean(axis=axes)
        difference_gate = candidate_gate - baseline_gate
    else:
        baseline_gate = baseline_array
        difference_gate = difference
    denominator = float(np.linalg.norm(baseline_gate.ravel()))
    numerator = float(np.linalg.norm(difference_gate.ravel()))
    relative_l2 = numerator / denominator if denominator else numerator
    return {
        "field": field,
        "step": step,
        "shape": list(baseline_array.shape),
        "dtype": str(baseline_array.dtype),
        "baseline_sha256": _file_sha256(baseline),
        "candidate_sha256": _file_sha256(candidate),
        "byte_identical": _file_sha256(baseline) == _file_sha256(candidate),
        "relative_l2": relative_l2,
        "linf": float(np.max(np.abs(difference_gate))),
        "pressure_demeaned": field == "p",
    }


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _analyze_stage_o43_variant_qualification(
    baseline_trajectory: str | Path,
    candidate_trajectory: str | Path,
    baseline_r128_profiles: Sequence[str | Path],
    candidate_r128_profiles: Sequence[str | Path],
    baseline_r320_profiles: Sequence[str | Path],
    candidate_r320_profiles: Sequence[str | Path],
    *,
    stage_o42_report: str | Path,
    expected_stage_o42_sha256: str,
    expected_gpu_name: str = "H100",
    variant: _StageO43Variant,
) -> dict[str, object]:
    """Apply numerical, structural, timing, and memory gates."""

    evidence_path = Path(stage_o42_report).expanduser().resolve()
    if _sha256(evidence_path) != expected_stage_o42_sha256:
        raise ValueError("Stage O.4.2 evidence SHA-256 differs")
    evidence = _load_json(evidence_path, "Stage O.4.2 evidence")
    if not (
        evidence.get("qualification_stage") == "O.4.2"
        and evidence.get("classification") == "DIAGNOSTIC_COMPLETE"
        and evidence.get("accounting_ready_for_optimization") is True
        and evidence.get("eligible_for_stage_o43_optimization_design") is True
        and evidence.get("production_default_changed") is False
    ):
        raise ValueError("Stage O.4.2 accounting gate is not closed")

    trajectory_dirs = {
        "baseline": Path(baseline_trajectory).expanduser().resolve(),
        "candidate": Path(candidate_trajectory).expanduser().resolve(),
    }
    metrics = {}
    for role, directory in trajectory_dirs.items():
        if not directory.is_dir() or not (directory / "COMPLETE").is_file():
            raise ValueError(f"{role} trajectory is incomplete")
        metric = _load_json(
            directory / variant.metrics_filename,
            f"{role} trajectory metrics",
        )
        if not (
            metric.get("qualification_stage") == variant.qualification_stage
            and metric.get("classification") == "TRAJECTORY_COMPLETE"
            and metric.get("measurement_role") == role
            and metric.get("completed_steps") == STAGE_O43_TRAJECTORY_STEPS
            and metric.get("saved_steps") == [0, STAGE_O43_TRAJECTORY_STEPS]
            and metric.get("production_default_changed") is False
        ):
            raise ValueError(f"{role} trajectory contract differs")
        metrics[role] = metric
    input_identities = {
        (
            value["production_metadata_sha256"],
            value["production_initial_q_sha256"],
        )
        for value in metrics.values()
    }
    if len(input_identities) != 1:
        raise ValueError("Stage O.4.3 trajectory inputs differ")
    arrays = [
        _array_comparison(
            trajectory_dirs["baseline"] / f"{field}_{step}.npy",
            trajectory_dirs["candidate"] / f"{field}_{step}.npy",
            field=field,
            step=step,
        )
        for step in (0, STAGE_O43_TRAJECTORY_STEPS)
        for field in _FIELDS
    ]
    numerical_maximum = max(float(value["relative_l2"]) for value in arrays)
    q0_identical = next(
        value["byte_identical"]
        for value in arrays
        if value["field"] == "Q" and value["step"] == 0
    )

    profile_groups = {
        ("r128", "baseline"): baseline_r128_profiles,
        ("r128", "candidate"): candidate_r128_profiles,
        ("r320", "baseline"): baseline_r320_profiles,
        ("r320", "candidate"): candidate_r320_profiles,
    }
    profiles: dict[tuple[str, str], list[dict[str, object]]] = {}
    expected_shapes = {
        "r128": STAGE_O43_DIAGNOSTIC_SHAPE,
        "r320": STAGE_O43_DECISION_SHAPE,
    }
    for (scale, role), raw_paths in profile_groups.items():
        paths = tuple(Path(path).expanduser().resolve() for path in raw_paths)
        if len(paths) != STAGE_O43_PROFILE_TRIALS:
            raise ValueError("Stage O.4.3 requires three profiles per role/scale")
        loaded = [_load_json(path, f"{scale} {role} profile") for path in paths]
        for profile in loaded:
            try:
                selected_output_policy = (
                    variant.baseline_output_policy
                    if role == "baseline"
                    else variant.candidate_output_policy
                )
                selected_batch_policy = (
                    variant.baseline_batch_policy
                    if role == "baseline"
                    else variant.candidate_batch_policy
                )
                selected_producer_layout = (
                    variant.baseline_producer_output_layout
                    if role == "baseline"
                    else variant.candidate_producer_output_layout
                )
                producer_outputs = profile["producer_outputs"]
                valid = (
                    profile["qualification_stage"]
                    == variant.qualification_stage
                    and profile["classification"] == "PROFILE_COMPLETE"
                    and profile["measurement_role"] == role
                    and tuple(profile["configuration"]["shape"])
                    == expected_shapes[scale]
                    and profile["configuration"]["warmup_steps"] == 10
                    and profile["configuration"]["profile_steps"] == 20
                    and profile["finite"] is True
                    and profile["production_default_changed"] is False
                    and expected_gpu_name.lower()
                    in profile["environment"]["device_name"].lower()
                    and profile["environment"]["cuda_matmul_allow_tf32"]
                    is False
                    and profile["algebraic_output_publication_policy"]["mode"]
                    == selected_output_policy.mode.value
                    and profile["projected_batch_assembly_policy"]["mode"]
                    == selected_batch_policy.mode.value
                    and isinstance(producer_outputs, Mapping)
                    and set(producer_outputs)
                    == {"molecular_field", "nematic_stress"}
                    and all(
                        value["producer_output_layout"]
                        == selected_producer_layout
                        for value in producer_outputs.values()
                    )
                )
                if (
                    valid
                    and role == "candidate"
                    and variant.structural_contract
                    == "producer_owned_packing"
                ):
                    valid = all(
                        isinstance(value.get("producer_packing"), Mapping)
                        for value in producer_outputs.values()
                    )
            except (KeyError, TypeError):
                valid = False
            if not valid:
                raise ValueError(f"{scale} {role} profile contract differs")
        profiles[(scale, role)] = loaded

    profile_inputs = {
        (
            profile["production_metadata_sha256"],
            profile["production_initial_q_sha256"],
        )
        for loaded in profiles.values()
        for profile in loaded
    }
    if len(profile_inputs) != 2:
        raise ValueError("Stage O.4.3 requires one input per resolution")
    trajectory_identity = next(iter(input_identities))
    r128_profile_identities = {
        (
            profile["production_metadata_sha256"],
            profile["production_initial_q_sha256"],
        )
        for role in _ROLES
        for profile in profiles[("r128", role)]
    }
    if r128_profile_identities != {trajectory_identity}:
        raise ValueError("trajectory and R128 profile inputs differ")

    scales = {}
    for scale in expected_shapes:
        baseline = profiles[(scale, "baseline")]
        candidate = profiles[(scale, "candidate")]
        baseline_times = [
            float(value["throughput"]["mean_timestep_seconds"])
            for value in baseline
        ]
        candidate_times = [
            float(value["throughput"]["mean_timestep_seconds"])
            for value in candidate
        ]
        paired = [
            candidate_value / baseline_value
            for baseline_value, candidate_value in zip(
                baseline_times,
                candidate_times,
                strict=True,
            )
        ]
        baseline_peak = max(
            int(value["memory"]["peak_allocated_bytes"])
            for value in baseline
        )
        candidate_peak = max(
            int(value["memory"]["peak_allocated_bytes"])
            for value in candidate
        )
        scales[scale] = {
            "baseline_mean_timestep_seconds": _mean(baseline_times),
            "candidate_mean_timestep_seconds": _mean(candidate_times),
            "mean_timestep_ratio_candidate_over_baseline": (
                _mean(candidate_times) / _mean(baseline_times)
            ),
            "paired_timestep_ratios": paired,
            "paired_candidate_faster_count": sum(value < 1.0 for value in paired),
            "baseline_peak_allocated_bytes": baseline_peak,
            "candidate_peak_allocated_bytes": candidate_peak,
            "peak_allocated_ratio_candidate_over_baseline": (
                candidate_peak / baseline_peak
            ),
        }

    baseline_assemblies = [
        value["batch_assembly_diagnostics"]
        for value in profiles[("r320", "baseline")]
    ]
    candidate_assemblies = [
        value["batch_assembly_diagnostics"]
        for value in profiles[("r320", "candidate")]
    ]
    if variant.structural_contract == "scheduler_view_adoption":
        baseline_assembly = baseline_assemblies[0]
        structural_gate = bool(
            baseline_assembly["policy"]["mode"] == "copy_cat"
            and baseline_assembly["contiguous_view_batches"] == 0
            and all(
                value["policy"]["mode"] == "contiguous_storage_view"
                and value["contiguous_view_batches"] > 0
                and value["copy_cat_batches"]
                < baseline_assembly["copy_cat_batches"]
                and value["retained_tensor_references"] == 0
                for value in candidate_assemblies
            )
        )
    elif variant.structural_contract == "producer_owned_packing":
        producer_contract = all(
            all(
                output["producer_output_layout"] == "boundary_packed"
                and output["producer_packing"]["ownership"] == "producer"
                and output["producer_packing"]["packing_site"]
                == "inside_pointwise_kernel"
                and output["producer_packing"]["pointwise_execution"]
                == "compile"
                and output["producer_packing"]["post_kernel_stack"] is False
                and output["producer_packing"]["post_kernel_cat"] is False
                and output["producer_packing"]["cross_generation_reuse"]
                is False
                for output in profile["producer_outputs"].values()
            )
            for profile in profiles[("r320", "candidate")]
        )
        structural_gate = bool(
            producer_contract
            and all(
                value["policy"]["mode"] == "contiguous_storage_view"
                and value["retained_tensor_references"] == 0
                for value in (*baseline_assemblies, *candidate_assemblies)
            )
            and all(
                candidate["contiguous_view_batches"]
                > baseline["contiguous_view_batches"]
                and candidate["copy_cat_batches"]
                < baseline["copy_cat_batches"]
                for baseline, candidate in zip(
                    baseline_assemblies,
                    candidate_assemblies,
                    strict=True,
                )
            )
        )
    else:
        raise RuntimeError("unknown Stage O.4.3 structural contract")
    numerical_gate = bool(
        q0_identical
        and numerical_maximum <= STAGE_O43_RELATIVE_L2_TOLERANCE
    )
    decision = scales["r320"]
    performance_gate = bool(
        decision["mean_timestep_ratio_candidate_over_baseline"]
        <= STAGE_O43_MAXIMUM_MEAN_TIMESTEP_RATIO
        and max(decision["paired_timestep_ratios"])
        <= STAGE_O43_MAXIMUM_PAIRED_TIMESTEP_RATIO
        and decision["paired_candidate_faster_count"] >= 2
    )
    memory_gate = bool(
        decision["peak_allocated_ratio_candidate_over_baseline"]
        <= STAGE_O43_MAXIMUM_MEMORY_RATIO
    )
    safety_gate = bool(
        decision["mean_timestep_ratio_candidate_over_baseline"] <= 1.03
        and decision["peak_allocated_ratio_candidate_over_baseline"] <= 1.10
    )
    if not (numerical_gate and structural_gate and safety_gate):
        classification = "C_rejected"
    elif performance_gate and memory_gate:
        classification = "A_recommended"
    else:
        classification = "B_neutral"
    return {
        "schema_version": 1,
        "qualification_stage": variant.qualification_stage,
        "classification": classification,
        "architecture_decision": variant.architecture_decision,
        "eligible_for_stage_o44_decision": classification == "A_recommended",
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "stage_o42_evidence": {
            "path": str(evidence_path),
            "sha256": expected_stage_o42_sha256,
        },
        "trajectory": {
            "steps": STAGE_O43_TRAJECTORY_STEPS,
            "relative_l2_tolerance": STAGE_O43_RELATIVE_L2_TOLERANCE,
            "maximum_relative_l2": numerical_maximum,
            "q0_byte_identical": q0_identical,
            "arrays": arrays,
        },
        "scales": scales,
        "gates": {
            "numerical_equivalence": numerical_gate,
            "zero_copy_batch_assembly_exercised": structural_gate,
            "r320_performance_improvement": performance_gate,
            "r320_memory_non_regression": memory_gate,
            "candidate_safety_non_regression": safety_gate,
        },
        "thresholds": {
            "maximum_mean_timestep_ratio": (
                STAGE_O43_MAXIMUM_MEAN_TIMESTEP_RATIO
            ),
            "maximum_paired_timestep_ratio": (
                STAGE_O43_MAXIMUM_PAIRED_TIMESTEP_RATIO
            ),
            "maximum_memory_ratio": STAGE_O43_MAXIMUM_MEMORY_RATIO,
        },
    }


def analyze_stage_o43_qualification(
    baseline_trajectory: str | Path,
    candidate_trajectory: str | Path,
    baseline_r128_profiles: Sequence[str | Path],
    candidate_r128_profiles: Sequence[str | Path],
    baseline_r320_profiles: Sequence[str | Path],
    candidate_r320_profiles: Sequence[str | Path],
    *,
    stage_o42_report: str | Path,
    expected_stage_o42_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Analyze the original packed-publication Stage O.4.3 candidate."""

    return _analyze_stage_o43_variant_qualification(
        baseline_trajectory,
        candidate_trajectory,
        baseline_r128_profiles,
        candidate_r128_profiles,
        baseline_r320_profiles,
        candidate_r320_profiles,
        stage_o42_report=stage_o42_report,
        expected_stage_o42_sha256=expected_stage_o42_sha256,
        expected_gpu_name=expected_gpu_name,
        variant=_PACKED_PUBLICATION_VARIANT,
    )


def _require_rejected_stage_o43_evidence(
    stage_o43_report: str | Path,
    expected_stage_o43_sha256: str,
) -> tuple[Path, dict[str, object]]:
    evidence_path = Path(stage_o43_report).expanduser().resolve()
    if _sha256(evidence_path) != expected_stage_o43_sha256:
        raise ValueError("Stage O.4.3 evidence SHA-256 differs")
    evidence = _load_json(evidence_path, "Stage O.4.3 evidence")
    try:
        valid = bool(
            evidence["qualification_stage"] == "O.4.3"
            and evidence["classification"] == "C_rejected"
            and evidence["architecture_decision"]
            == "packed_generation_storage_with_safe_views"
            and evidence["eligible_for_stage_o44_decision"] is False
            and evidence["eligible_for_production_promotion"] is False
            and evidence["production_default_changed"] is False
            and evidence["gates"]["numerical_equivalence"] is True
            and evidence["gates"]["zero_copy_batch_assembly_exercised"]
            is True
            and evidence["gates"]["r320_performance_improvement"] is False
            and evidence["gates"]["r320_memory_non_regression"] is False
            and evidence["gates"]["candidate_safety_non_regression"] is False
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Stage O.4.3 rejection evidence contract differs")
    return evidence_path, evidence


def analyze_stage_o431_qualification(
    baseline_trajectory: str | Path,
    candidate_trajectory: str | Path,
    baseline_r128_profiles: Sequence[str | Path],
    candidate_r128_profiles: Sequence[str | Path],
    baseline_r320_profiles: Sequence[str | Path],
    candidate_r320_profiles: Sequence[str | Path],
    *,
    stage_o42_report: str | Path,
    expected_stage_o42_sha256: str,
    stage_o43_report: str | Path,
    expected_stage_o43_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Analyze safe views over naturally shared storage without repacking."""

    rejected_path, _ = _require_rejected_stage_o43_evidence(
        stage_o43_report,
        expected_stage_o43_sha256,
    )
    report = _analyze_stage_o43_variant_qualification(
        baseline_trajectory,
        candidate_trajectory,
        baseline_r128_profiles,
        candidate_r128_profiles,
        baseline_r320_profiles,
        candidate_r320_profiles,
        stage_o42_report=stage_o42_report,
        expected_stage_o42_sha256=expected_stage_o42_sha256,
        expected_gpu_name=expected_gpu_name,
        variant=_OPPORTUNISTIC_VIEW_VARIANT,
    )
    report["stage_o43_rejection_evidence"] = {
        "path": str(rejected_path),
        "sha256": expected_stage_o43_sha256,
        "packed_publication_must_remain_rejected": True,
    }
    return report


def _require_neutral_stage_o431_evidence(
    stage_o431_report: str | Path,
    expected_stage_o431_sha256: str,
) -> tuple[Path, dict[str, object]]:
    evidence_path = Path(stage_o431_report).expanduser().resolve()
    if _sha256(evidence_path) != expected_stage_o431_sha256:
        raise ValueError("Stage O.4.3.1 evidence SHA-256 differs")
    evidence = _load_json(evidence_path, "Stage O.4.3.1 evidence")
    try:
        valid = bool(
            evidence["qualification_stage"] == "O.4.3.1"
            and evidence["classification"] == "B_neutral"
            and evidence["architecture_decision"]
            == "opportunistic_natural_storage_views_without_republication"
            and evidence["eligible_for_stage_o44_decision"] is False
            and evidence["eligible_for_production_promotion"] is False
            and evidence["production_default_changed"] is False
            and evidence["gates"]["numerical_equivalence"] is True
            and evidence["gates"]["zero_copy_batch_assembly_exercised"]
            is True
            and evidence["gates"]["r320_performance_improvement"] is False
            and evidence["gates"]["candidate_safety_non_regression"] is True
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Stage O.4.3.1 neutral evidence contract differs")
    return evidence_path, evidence


def analyze_stage_o433_qualification(
    baseline_trajectory: str | Path,
    candidate_trajectory: str | Path,
    baseline_r128_profiles: Sequence[str | Path],
    candidate_r128_profiles: Sequence[str | Path],
    baseline_r320_profiles: Sequence[str | Path],
    candidate_r320_profiles: Sequence[str | Path],
    *,
    stage_o42_report: str | Path,
    expected_stage_o42_sha256: str,
    stage_o43_report: str | Path,
    expected_stage_o43_sha256: str,
    stage_o431_report: str | Path,
    expected_stage_o431_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Analyze producer-owned H/stress packing against the safe-view parent."""

    rejected_path, _ = _require_rejected_stage_o43_evidence(
        stage_o43_report,
        expected_stage_o43_sha256,
    )
    neutral_path, _ = _require_neutral_stage_o431_evidence(
        stage_o431_report,
        expected_stage_o431_sha256,
    )
    report = _analyze_stage_o43_variant_qualification(
        baseline_trajectory,
        candidate_trajectory,
        baseline_r128_profiles,
        candidate_r128_profiles,
        baseline_r320_profiles,
        candidate_r320_profiles,
        stage_o42_report=stage_o42_report,
        expected_stage_o42_sha256=expected_stage_o42_sha256,
        expected_gpu_name=expected_gpu_name,
        variant=_PRODUCER_PACKED_VARIANT,
    )
    report["prior_storage_evidence"] = {
        "stage_o43": {
            "path": str(rejected_path),
            "sha256": expected_stage_o43_sha256,
            "scheduler_republication_remains_rejected": True,
        },
        "stage_o431": {
            "path": str(neutral_path),
            "sha256": expected_stage_o431_sha256,
            "natural_view_route_closed_as_performance_neutral": True,
        },
    }
    return report


def profile_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile Stage O.4.3 on H100")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = profile_stage_o43_h100_runtime(
        args.production_reference_dir,
        role=args.role,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def trajectory_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage O.4.3 trajectory")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--confirm-steps", type=int, default=100)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    report = run_stage_o43_h100_trajectory(
        args.production_reference_dir,
        args.output_dir,
        role=args.role,
        confirmed_steps=args.confirm_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage O.4.3")
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    for scale in ("r128", "r320"):
        for role in _ROLES:
            parser.add_argument(
                f"--{role}-{scale}-profile",
                type=Path,
                action="append",
                required=True,
            )
    parser.add_argument("--stage-o42-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o42-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_o43_qualification(
        args.baseline_trajectory,
        args.candidate_trajectory,
        args.baseline_r128_profile,
        args.candidate_r128_profile,
        args.baseline_r320_profile,
        args.candidate_r320_profile,
        stage_o42_report=args.stage_o42_report,
        expected_stage_o42_sha256=args.expected_stage_o42_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["classification"] == "A_recommended" else 1


def profile_o431_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile Stage O.4.3.1 on H100")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = profile_stage_o431_h100_runtime(
        args.production_reference_dir,
        role=args.role,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def trajectory_o431_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage O.4.3.1 trajectory")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--confirm-steps", type=int, default=100)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    report = run_stage_o431_h100_trajectory(
        args.production_reference_dir,
        args.output_dir,
        role=args.role,
        confirmed_steps=args.confirm_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def analysis_o431_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage O.4.3.1")
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    for scale in ("r128", "r320"):
        for role in _ROLES:
            parser.add_argument(
                f"--{role}-{scale}-profile",
                type=Path,
                action="append",
                required=True,
            )
    parser.add_argument("--stage-o42-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o42-sha256", required=True)
    parser.add_argument("--stage-o43-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o43-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_o431_qualification(
        args.baseline_trajectory,
        args.candidate_trajectory,
        args.baseline_r128_profile,
        args.candidate_r128_profile,
        args.baseline_r320_profile,
        args.candidate_r320_profile,
        stage_o42_report=args.stage_o42_report,
        expected_stage_o42_sha256=args.expected_stage_o42_sha256,
        stage_o43_report=args.stage_o43_report,
        expected_stage_o43_sha256=args.expected_stage_o43_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["classification"] == "A_recommended" else 1


def profile_o433_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile Stage O.4.3.3 on H100")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = profile_stage_o433_h100_runtime(
        args.production_reference_dir,
        role=args.role,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def trajectory_o433_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage O.4.3.3 trajectory")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--confirm-steps", type=int, default=100)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    report = run_stage_o433_h100_trajectory(
        args.production_reference_dir,
        args.output_dir,
        role=args.role,
        confirmed_steps=args.confirm_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def analysis_o433_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage O.4.3.3")
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    for scale in ("r128", "r320"):
        for role in _ROLES:
            parser.add_argument(
                f"--{role}-{scale}-profile",
                type=Path,
                action="append",
                required=True,
            )
    parser.add_argument("--stage-o42-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o42-sha256", required=True)
    parser.add_argument("--stage-o43-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o43-sha256", required=True)
    parser.add_argument("--stage-o431-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o431-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_o433_qualification(
        args.baseline_trajectory,
        args.candidate_trajectory,
        args.baseline_r128_profile,
        args.candidate_r128_profile,
        args.baseline_r320_profile,
        args.candidate_r320_profile,
        stage_o42_report=args.stage_o42_report,
        expected_stage_o42_sha256=args.expected_stage_o42_sha256,
        stage_o43_report=args.stage_o43_report,
        expected_stage_o43_sha256=args.expected_stage_o43_sha256,
        stage_o431_report=args.stage_o431_report,
        expected_stage_o431_sha256=args.expected_stage_o431_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["classification"] == "A_recommended" else 1


__all__ = [
    "STAGE_O43_DECISION_SHAPE",
    "STAGE_O43_DIAGNOSTIC_SHAPE",
    "STAGE_O43_PROFILE_TRIALS",
    "STAGE_O43_RELATIVE_L2_TOLERANCE",
    "STAGE_O43_TRAJECTORY_STEPS",
    "analyze_stage_o43_qualification",
    "analyze_stage_o431_qualification",
    "analyze_stage_o433_qualification",
    "analysis_main",
    "analysis_o431_main",
    "analysis_o433_main",
    "profile_main",
    "profile_o431_main",
    "profile_o433_main",
    "profile_stage_o43_h100_runtime",
    "profile_stage_o431_h100_runtime",
    "profile_stage_o433_h100_runtime",
    "run_stage_o43_h100_trajectory",
    "run_stage_o431_h100_trajectory",
    "run_stage_o433_h100_trajectory",
    "trajectory_main",
    "trajectory_o431_main",
    "trajectory_o433_main",
]
