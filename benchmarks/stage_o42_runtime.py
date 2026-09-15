"""Build and inspect the two Stage O.4.2 Plane runtime roles."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable

import torch

try:
    from .profile_beris_edwards_timestep import (
        ProfileConfig,
        RegionTimer,
        _build_solver,
    )
except ImportError:  # Direct execution through the adjacent CLI wrapper.
    from profile_beris_edwards_timestep import (
        ProfileConfig,
        RegionTimer,
        _build_solver,
    )
from pssolver.diagnostics import build_tensor_inventory, cuda_memory_snapshot
from pssolver.experimental.h100_shadow_qualification import (
    _cuda_identity,
    _require_positive_integer,
    _write_new_json,
)
from pssolver.experimental.plane_shadow_driver import (
    load_h100_production_plane_reference,
)
from pssolver.experimental.shadow_metadata import (
    plane_beris_edwards_production_signature,
)
from pssolver.experimental.stage_n4_qualification import (
    _build_runtime,
    _initial_values,
)
from pssolver.experimental.stage_n_diagnostics import (
    summarize_audited_operators,
)
from pssolver.experimental.stage_o42_diagnostics import STAGE_O42_SHAPE

_ROLES = ("legacy", "canary")


def _plain_diagnostic_snapshot(
    snapshot: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Copy a read-only runtime snapshot across the JSON report boundary."""

    return dict(snapshot) if snapshot is not None else None


def _require_role(role: str) -> str:
    if role not in _ROLES:
        raise ValueError("role must be 'legacy' or 'canary'")
    return role


def _inventory_phase(
    roots: Mapping[str, object],
    *,
    device: torch.device,
) -> dict[str, object]:
    torch.cuda.synchronize(device)
    allocator = cuda_memory_snapshot(device)
    inventory = build_tensor_inventory(roots, device=device)
    allocated = int(allocator["allocated_bytes"])
    unique = int(inventory["unique_storage_bytes"])
    gap = allocated - unique
    return {
        "allocator": allocator,
        "inventory": inventory,
        "allocator_minus_inventory_bytes": gap,
        "inventory_exceeds_allocator_bytes": max(0, -gap),
        "storage_accounting_consistent": gap >= 0,
        "inventory_fraction_of_allocated": unique / allocated if allocated else 0.0,
    }


def _operator_audit(
    step: Callable[[], None],
    *,
    steps: int,
    device: torch.device,
) -> dict[str, object]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        for _ in range(steps):
            step()
    torch.cuda.synchronize(device)
    return {
        **summarize_audited_operators(tuple(profiler.key_averages())),
        "steps": steps,
        "scope": "separate_from_residency_and_semantic_timing",
        "memory_fields_are_allocator_effects_not_total_memory_traffic": True,
    }


def _legacy_config(
    reference,
    *,
    warmup_steps: int,
    diagnostic_steps: int,
) -> ProfileConfig:
    signature = plane_beris_edwards_production_signature(reference.metadata)
    solver = signature["solver"]
    numerics = signature["numerics"]
    config = ProfileConfig(
        shape=tuple(solver["shape"]),
        lengths=tuple(solver["lengths"]),
        device="cuda",
        dtype=solver["real_dtype"],
        dt=float(solver["dt"]),
        dealias_rule=numerics["dealias_rule"],
        projected_transform_execution=numerics["projected_transform_execution"],
        warmup_steps=warmup_steps,
        profile_steps=diagnostic_steps,
        spectral_refresh_interval=numerics["spectral_refresh_interval_steps"],
        pressure_diagnostics=bool(
            reference.metadata["numerics"]["pressure_residual_diagnostics"]
        ),
        reuse_q_gradients=bool(
            reference.metadata["numerics"]["q_gradient_reuse"]["enabled"]
        ),
        molecular_field_linear_space=numerics["molecular_field_linear_space"],
        stress_divergence_sum_space=numerics["stress_divergence_sum_space"],
        pointwise_execution=numerics["pointwise_execution"],
        transform_execution_order=solver["transform_execution_order"],
        spectral_storage=solver["spectral_storage"],
        initial_q_path=str(reference.initial_q_path),
        timing_scope="all_regions",
    )
    if config.shape != STAGE_O42_SHAPE:
        raise ValueError("Stage O.4.2 requires an R320 production reference")
    return config


def _legacy_diagnostic(
    reference,
    *,
    warmup_steps: int,
    diagnostic_steps: int,
    operator_audit_steps: int,
    device: torch.device,
) -> dict[str, object]:
    timer = RegionTimer(device)
    config = _legacy_config(
        reference,
        warmup_steps=warmup_steps,
        diagnostic_steps=diagnostic_steps,
    )
    solver = _build_solver(config, timer)
    roots = {"legacy_solver": solver}
    phases = {"after_build": _inventory_phase(roots, device=device)}
    for _ in range(warmup_steps):
        solver.integrator.step()
    phases["after_warmup"] = _inventory_phase(roots, device=device)
    timer.reset()
    for _ in range(diagnostic_steps):
        solver.integrator.step()
    semantic_regions = timer.summarize()
    timer.enabled = False
    phases["after_timestep"] = _inventory_phase(roots, device=device)
    solver.refresh_static_fields()
    phases["after_observation"] = _inventory_phase(roots, device=device)
    for _ in range(2):
        solver.integrator.step()
    operator_audit = _operator_audit(
        solver.integrator.step,
        steps=operator_audit_steps,
        device=device,
    )
    fields = solver.model.fields
    finite = bool(torch.isfinite(fields.spatial).all().item()) and bool(
        torch.isfinite(fields.spectral).all().item()
    )
    return {
        "configuration": asdict(config),
        "semantic_regions": {
            "schema_version": 1,
            "source": "legacy_region_timer",
            "steps": diagnostic_steps,
            "regions": semantic_regions,
        },
        "residency_phases": phases,
        "operator_audit": operator_audit,
        "finite": finite,
    }


def _canary_diagnostic(
    reference,
    *,
    warmup_steps: int,
    diagnostic_steps: int,
    operator_audit_steps: int,
    expected_gpu_name: str,
    device: torch.device,
) -> dict[str, object]:
    runtime, comparison = _build_runtime(
        reference,
        mode="candidate",
        expected_gpu_name=expected_gpu_name,
        device=device,
        instrument=True,
    )
    roots = {"separated_runtime": runtime}
    phases = {"after_build": _inventory_phase(roots, device=device)}
    runtime.reset(_initial_values(reference, device))
    phases["after_reset"] = _inventory_phase(roots, device=device)
    runtime.solver.run(warmup_steps)
    phases["after_warmup"] = _inventory_phase(roots, device=device)
    recorder = runtime.performance_recorder
    if recorder is None:
        raise RuntimeError("canary runtime lacks semantic instrumentation")
    recorder.reset()
    runtime.solver.run(diagnostic_steps)
    semantic_regions = recorder.snapshot()
    phases["after_timestep"] = _inventory_phase(roots, device=device)
    runtime.synchronize_algebraic_for_observation()
    phases["after_observation"] = _inventory_phase(roots, device=device)
    runtime.solver.run(2)
    operator_audit = _operator_audit(
        lambda: runtime.solver.run(1),
        steps=operator_audit_steps,
        device=device,
    )
    fields = runtime.solver.fields
    finite = bool(torch.isfinite(fields.spatial).all().item()) and bool(
        torch.isfinite(fields.spectral).all().item()
    )
    signature = comparison.production_signature
    return {
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
        },
        "configuration_comparison": comparison.to_metadata(),
        "semantic_regions": semantic_regions,
        "residency_phases": phases,
        "operator_audit": operator_audit,
        "algebraic_representation_reuse": _plain_diagnostic_snapshot(
            runtime.algebraic_representation_reuse_diagnostics()
        ),
        "algebraic_physical_materialization": _plain_diagnostic_snapshot(
            runtime.algebraic_physical_materialization_diagnostics()
        ),
        "finite": finite,
    }


def profile_stage_o42_runtime(
    production_directory: str | Path,
    *,
    role: str,
    warmup_steps: int = 10,
    diagnostic_steps: int = 2,
    operator_audit_steps: int = 2,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Profile one runtime without mutating production inputs or defaults."""

    role = _require_role(role)
    _require_positive_integer(warmup_steps, "warmup_steps")
    _require_positive_integer(diagnostic_steps, "diagnostic_steps")
    _require_positive_integer(operator_audit_steps, "operator_audit_steps")
    reference = load_h100_production_plane_reference(
        production_directory,
        expected_gpu_name=expected_gpu_name,
    )
    device = torch.device("cuda")
    environment = _cuda_identity(
        expected_gpu_name=expected_gpu_name,
        device=device,
    )
    torch.manual_seed(20260914)
    torch.cuda.manual_seed_all(20260914)
    with torch.no_grad():
        result = (
            _legacy_diagnostic(
                reference,
                warmup_steps=warmup_steps,
                diagnostic_steps=diagnostic_steps,
                operator_audit_steps=operator_audit_steps,
                device=device,
            )
            if role == "legacy"
            else _canary_diagnostic(
                reference,
                warmup_steps=warmup_steps,
                diagnostic_steps=diagnostic_steps,
                operator_audit_steps=operator_audit_steps,
                expected_gpu_name=expected_gpu_name,
                device=device,
            )
        )
    if result["finite"] is not True:
        raise RuntimeError("Stage O.4.2 diagnostic ended with non-finite fields")
    for phase in result["residency_phases"].values():
        if phase["inventory"]["truncated"] is True:
            raise RuntimeError("Stage O.4.2 tensor inventory was truncated")
    return {
        "schema_version": 1,
        "qualification_stage": "O.4.2",
        "measurement_role": role,
        "classification": "DIAGNOSTIC_COMPLETE",
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "production_directory": str(reference.directory),
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "environment": environment,
        "diagnostic_contract": {
            "shape": list(STAGE_O42_SHAPE),
            "warmup_steps": warmup_steps,
            "diagnostic_steps": diagnostic_steps,
            "operator_audit_steps": operator_audit_steps,
            "operator_audit_settling_steps": 2,
            "global_gc_traversal": False,
            "runtime_roots_only": True,
            "overlapping_storage_address_ranges_coalesced": True,
            "storage_accounting_consistency_is_diagnostic": True,
            "operator_memory_is_allocator_effect_not_total_traffic": True,
        },
        **result,
    }


def profile_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Stage O.4.2 diagnostic")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--diagnostic-steps", type=int, default=2)
    parser.add_argument("--operator-audit-steps", type=int, default=2)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = profile_stage_o42_runtime(
        args.production_reference_dir,
        role=args.role,
        warmup_steps=args.warmup_steps,
        diagnostic_steps=args.diagnostic_steps,
        operator_audit_steps=args.operator_audit_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["profile_main", "profile_stage_o42_runtime"]
