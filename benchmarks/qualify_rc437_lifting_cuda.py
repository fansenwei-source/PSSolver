#!/usr/bin/env python3
"""Candidate-only CUDA correctness and restart gates for RC4.3.7."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path

import torch

from pssolver.boundaries import assign_boundaries, free_slip_velocity, neumann_pressure_compatibility
from pssolver.configuration.active_nematics_simulation_adapters import compose_plane_beris_edwards_simulation
from pssolver.configuration.package_construction import plan_package_runtime_construction
from pssolver.configuration.plane_beris_edwards import create_plane_beris_edwards_run_spec
from pssolver.configuration.plane_beris_edwards_components import decompose_plane_beris_edwards_run_spec
from pssolver.models.active_nematics import Q_COMPONENTS, strong_planar_q
from pssolver.operators import localized_quintic_wall_transition
from pssolver.runtime.package_construction import PackageRuntimeConstructionInput, build_package_simulation_runtime
from pssolver.runtime.plane_beris_edwards import PlaneRuntimeBuildRequest, plane_physical_component
from pssolver.workflows.plane_checkpoint import capture_plane_checkpoint, load_plane_checkpoint, restore_plane_checkpoint, write_plane_checkpoint


SCHEMA = "pssolver.rc4_3_7.lifting_cuda_correctness.v1"
PASS = "PASS_RC4_3_7_INTERNAL_LIFTING_CUDA_CORRECTNESS_AND_RESTART"


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("correctness JSON round-trip differs")
    finally:
        if temporary.exists():
            temporary.unlink()


def _initial(shape: tuple[int, int, int]) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(437)
    return {
        name: 0.01 * torch.randn(shape, generator=generator, dtype=torch.float64)
        for name in Q_COMPONENTS
    }


def _build(
    *,
    device: str,
    apply_fresh: bool,
    pointwise_execution: str = "compile",
):
    shape = (8, 8, 8)
    spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=Path.cwd() / ".rc437_no_output",
        device=device,
        dtype="float64",
        pointwise_execution=pointwise_execution,
        runtime_path="legacy_production",
        nx=shape[0], ny=shape[1], nz=shape[2],
        lx=100.0, ly=100.0, height=20.0,
        steps=4, save_start_step=0, save_interval=4, diagnostic_interval=4,
        defect_min_separation=10.0, defect_core_radius=1.5,
        twist_modes=(1, 2, 3),
    )
    base = compose_plane_beris_edwards_simulation(decompose_plane_beris_edwards_run_spec(spec))
    normals = {(2, "lower"): (0.0, 0.0, -1.0), (2, "upper"): (0.0, 0.0, 1.0)}
    simulation = replace(
        base,
        boundaries=assign_boundaries(
            model=base.equation_system,
            geometry=base.geometry,
            policies={
                "Q": strong_planar_q(
                    scalar_order=0.6,
                    face_directors={(2, "lower"): (1.0, 0.0, 0.0), (2, "upper"): (0.0, 1.0, 0.0)},
                    face_normals=normals,
                ),
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
    )
    request = PlaneRuntimeBuildRequest(
        spec,
        {"configuration": spec.identity_metadata(), "runtime_selection": spec.runtime_selection_metadata()},
        _initial(shape),
        device,
        simulation,
        fresh_initial_conditioning=localized_quintic_wall_transition(2.0),
        apply_fresh_initial_conditioning=apply_fresh,
    )
    adapter = build_package_simulation_runtime(
        PackageRuntimeConstructionInput(plan_package_runtime_construction(simulation), request)
    )
    return spec, adapter


def _wall_residual(adapter) -> float:
    runtime = adapter.solver.model.static_lifting_runtime
    plan = runtime.operator.plan
    axis = plan.wall_normal_axis
    spacing = plan.domain_lengths[axis] / plan.domain_shape[axis]
    maximum = 0.0
    for index, name in enumerate(runtime.operator.component_order):
        component = plan.for_component(name)
        slope = (component.upper_value.value - component.lower_value.value) / plan.domain_lengths[axis]
        lower = runtime.operator.stacked_lift[index].select(axis, 0) - 0.5 * spacing * slope
        upper = runtime.operator.stacked_lift[index].select(axis, plan.domain_shape[axis] - 1) + 0.5 * spacing * slope
        maximum = max(
            maximum,
            float(torch.max(torch.abs(lower - component.lower_value.value)).item()),
            float(torch.max(torch.abs(upper - component.upper_value.value)).item()),
        )
    return maximum


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = torch.linalg.vector_norm((left - right).reshape(-1))
    scale = torch.linalg.vector_norm(right.reshape(-1))
    return float((delta / max(float(scale.item()), 1.0e-300)).item())


def run(checkpoint_dir: Path) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one allocated CUDA device is required")
    if str(torch.empty((), device="cuda").device) != "cuda:0":
        raise RuntimeError("allocated CUDA identity differs")
    _cpu_spec, cpu = _build(
        device="cpu",
        apply_fresh=True,
        pointwise_execution="eager",
    )
    cuda_spec, continuous = _build(device="cuda", apply_fresh=True)
    _, raw_cuda = _build(device="cuda", apply_fresh=False)
    _, segment = _build(device="cuda", apply_fresh=True)
    _, resumed = _build(device="cuda", apply_fresh=False)
    cpu_remainder = cpu.fields.spatial.detach().cpu()
    cuda_remainder = continuous.fields.spatial.detach().cpu()
    raw_remainder = raw_cuda.fields.spatial.detach().cpu()
    centers = (torch.arange(8, dtype=torch.float64) + 0.5) * 20.0 / 8
    interior = torch.minimum(centers, 20.0 - centers) >= 2.0
    cross_device_relative_l2 = _relative_l2(cuda_remainder, cpu_remainder)
    interior_equal = torch.equal(cuda_remainder[..., interior], raw_remainder[..., interior])
    transition_changed = bool(torch.any(cuda_remainder[..., ~interior] != raw_remainder[..., ~interior]).item())

    with torch.no_grad():
        continuous.advance(4)
        segment.advance(2)
        checkpoint = capture_plane_checkpoint(segment, run_spec=cuda_spec)
        persisted = load_plane_checkpoint(write_plane_checkpoint(checkpoint_dir, checkpoint))
        restored_steps = restore_plane_checkpoint(resumed, persisted, run_spec=cuda_spec)
        resumed.advance(2)
        torch.cuda.synchronize()
    names = (*Q_COMPONENTS, "ux", "uy", "uz", "p")
    field_equal = {name: bool(torch.equal(continuous.fields[name], resumed.fields[name])) for name in names}
    physical_equal = {
        name: bool(torch.equal(plane_physical_component(continuous, name), plane_physical_component(resumed, name)))
        for name in Q_COMPONENTS
    }
    provenance = persisted.fresh_construction_provenance
    restart = persisted.lifting_restart
    finite = all(bool(torch.isfinite(resumed.fields[name]).all().item()) for name in names)
    pointwise = continuous.solver.model.nlmodel.pointwise_kernels.metadata()
    passed = all((
        cross_device_relative_l2 <= 1.0e-15,
        interior_equal,
        transition_changed,
        _wall_residual(continuous) <= 1.0e-12,
        restored_steps == 2,
        all(field_equal.values()),
        all(physical_equal.values()),
        finite,
        provenance is not None,
        provenance["conditioning"]["effective_policy"] == "localized_quintic_wall_transition_v1",
        provenance["conditioning_applied"] is True,
        "fresh_initial_remainder_conditioning" not in restart["lifting"],
        pointwise["requested"] == pointwise["effective"] == "compile",
        pointwise["fallback_allowed"] is False,
    ))
    return {
        "schema": SCHEMA,
        "classification": PASS if passed else "FAIL_RC4_3_7_INTERNAL_LIFTING_CUDA_CORRECTNESS_AND_RESTART",
        "passed": passed,
        "environment": {
            "device": "cuda:0",
            "device_name": torch.cuda.get_device_name(0),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "pssolver_import": str(Path(__import__("pssolver").__file__).resolve()),
        },
        "fresh_conditioning": {
            "policy": "localized_quintic_wall_transition_v1",
            "transition_width": 2.0,
            "cross_device_relative_l2": cross_device_relative_l2,
            "cross_device_relative_l2_max": 1.0e-15,
            "interior_byte_identical_to_unconditioned": interior_equal,
            "transition_region_changed": transition_changed,
            "wall_residual_linf": _wall_residual(continuous),
            "wall_residual_linf_max": 1.0e-12,
        },
        "restart": {
            "segment_steps": 2,
            "final_steps": 4,
            "restored_steps": restored_steps,
            "fields_byte_identical": field_equal,
            "physical_q_byte_identical": physical_equal,
            "all_byte_identical": all((*field_equal.values(), *physical_equal.values())),
            "finite": finite,
            "fresh_construction_provenance": dict(provenance),
            "lifting_restart": dict(restart),
        },
        "pointwise_compile": pointwise,
        "qualification_boundary": {
            "internal_lifting_infrastructure_only": True,
            "public_lifted_plane_reopened": False,
            "localized_policy_publicly_qualified": False,
            "production_default_changed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.checkpoint_dir.expanduser().resolve())
    _atomic_json(args.output, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
