#!/usr/bin/env python3
"""Correct-layer CUDA lifting and restart gates for RC4.3.7 recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

if __package__:
    from benchmarks.qualify_rc437_lifting_cuda import (
        PASS,
        _atomic_json,
        _build,
        _initial,
        _relative_l2,
        _wall_residual,
    )
else:
    from qualify_rc437_lifting_cuda import (
        PASS,
        _atomic_json,
        _build,
        _initial,
        _relative_l2,
        _wall_residual,
    )

from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.plane_beris_edwards import plane_physical_component
from pssolver.workflows.plane_checkpoint import (
    capture_plane_checkpoint,
    load_plane_checkpoint,
    restore_plane_checkpoint,
    write_plane_checkpoint,
)


SCHEMA = "pssolver.rc4_3_7.lifting_cuda_correctness.v2"
MEASUREMENT_LAYER = "direct_conditioning_operator_output_before_spectral_projection"


def _direct_conditioning_evidence(
    runtime: Any,
    physical_values: dict[str, torch.Tensor],
) -> tuple[dict[str, object], torch.Tensor]:
    """Measure locality where the conditioning policy is actually defined."""

    conditioned = runtime.extract_initial_remainders(
        physical_values,
        apply_fresh_conditioning=True,
    )
    unconditioned = runtime.extract_initial_remainders(
        physical_values,
        apply_fresh_conditioning=False,
    )
    plan = runtime.plan
    axis = plan.wall_normal_axis
    count = plan.domain_shape[axis]
    wall_length = plan.domain_lengths[axis]
    width = runtime.fresh_initial_conditioning.transition_width
    if width is None:
        raise RuntimeError("localized conditioning width is missing")
    centers = (
        torch.arange(
            count,
            dtype=runtime.operator.dtype,
            device=runtime.operator.device,
        )
        + 0.5
    ) * (wall_length / count)
    distance = torch.minimum(centers, wall_length - centers)
    interior = distance >= width
    transition = ~interior
    components: dict[str, object] = {}
    for name in runtime.component_order:
        direct = conditioned[name]
        raw = unconditioned[name]
        components[name] = {
            "interior_byte_identical_to_unconditioned": bool(
                torch.equal(
                    direct.index_select(axis, torch.nonzero(interior).flatten()),
                    raw.index_select(axis, torch.nonzero(interior).flatten()),
                )
            ),
            "transition_region_changed": bool(
                torch.any(
                    direct.index_select(axis, torch.nonzero(transition).flatten())
                    != raw.index_select(axis, torch.nonzero(transition).flatten())
                ).item()
            ),
        }
    stacked = torch.stack(
        tuple(conditioned[name] for name in runtime.component_order),
        dim=0,
    )
    return {
        "measurement_layer": MEASUREMENT_LAYER,
        "interior_cell_count": int(torch.count_nonzero(interior).item()),
        "transition_cell_count": int(torch.count_nonzero(transition).item()),
        "components": components,
        "interior_byte_identical_to_unconditioned": all(
            value["interior_byte_identical_to_unconditioned"]
            for value in components.values()
        ),
        "transition_region_changed": any(
            value["transition_region_changed"]
            for value in components.values()
        ),
    }, stacked


def _post_projection_diagnostic(
    conditioned: Any,
    unconditioned: Any,
) -> dict[str, object]:
    """Describe, but never gate on, globally projected adapter fields."""

    centers = (
        torch.arange(8, dtype=torch.float64, device=conditioned.fields.spatial.device)
        + 0.5
    ) * 20.0 / 8
    interior = torch.minimum(centers, 20.0 - centers) >= 2.0
    components: dict[str, object] = {}
    for name in Q_COMPONENTS:
        left = conditioned.fields[name].detach().index_select(
            -1,
            torch.nonzero(interior).flatten(),
        )
        right = unconditioned.fields[name].detach().index_select(
            -1,
            torch.nonzero(interior).flatten(),
        )
        delta = torch.abs(left - right)
        components[name] = {
            "interior_byte_identical_to_unconditioned": bool(
                torch.equal(left, right)
            ),
            "interior_max_abs_difference": float(delta.max().item()),
            "interior_different_element_count": int(
                torch.count_nonzero(left != right).item()
            ),
        }
    return {
        "measurement_layer": (
            "adapter_fields_spatial_after_global_truncated_spectral_projection"
        ),
        "qualification_gate": False,
        "interpretation": (
            "A wall-local construction change may be redistributed globally by "
            "the truncated spectral projection; byte identity is not required here."
        ),
        "components": components,
        "all_interior_byte_identical_to_unconditioned": all(
            value["interior_byte_identical_to_unconditioned"]
            for value in components.values()
        ),
    }


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

    physical_values = _initial((8, 8, 8))
    cpu_direct, cpu_conditioned = _direct_conditioning_evidence(
        cpu.solver.model.static_lifting_runtime,
        physical_values,
    )
    cuda_direct, cuda_conditioned = _direct_conditioning_evidence(
        continuous.solver.model.static_lifting_runtime,
        physical_values,
    )
    cross_device_relative_l2 = _relative_l2(
        cuda_conditioned.detach().cpu(),
        cpu_conditioned.detach().cpu(),
    )
    post_projection = _post_projection_diagnostic(continuous, raw_cuda)

    with torch.no_grad():
        continuous.advance(4)
        segment.advance(2)
        checkpoint = capture_plane_checkpoint(segment, run_spec=cuda_spec)
        persisted = load_plane_checkpoint(
            write_plane_checkpoint(checkpoint_dir, checkpoint)
        )
        restored_steps = restore_plane_checkpoint(
            resumed,
            persisted,
            run_spec=cuda_spec,
        )
        resumed.advance(2)
        torch.cuda.synchronize()

    names = (*Q_COMPONENTS, "ux", "uy", "uz", "p")
    field_equal = {
        name: bool(torch.equal(continuous.fields[name], resumed.fields[name]))
        for name in names
    }
    physical_equal = {
        name: bool(
            torch.equal(
                plane_physical_component(continuous, name),
                plane_physical_component(resumed, name),
            )
        )
        for name in Q_COMPONENTS
    }
    provenance = persisted.fresh_construction_provenance
    restart = persisted.lifting_restart
    finite = all(
        bool(torch.isfinite(resumed.fields[name]).all().item()) for name in names
    )
    pointwise = continuous.solver.model.nlmodel.pointwise_kernels.metadata()
    passed = all(
        (
            cpu_direct["interior_byte_identical_to_unconditioned"],
            cuda_direct["interior_byte_identical_to_unconditioned"],
            cuda_direct["transition_region_changed"],
            cross_device_relative_l2 <= 1.0e-15,
            _wall_residual(continuous) <= 1.0e-12,
            post_projection["qualification_gate"] is False,
            restored_steps == 2,
            all(field_equal.values()),
            all(physical_equal.values()),
            finite,
            provenance is not None,
            provenance["conditioning"]["effective_policy"]
            == "localized_quintic_wall_transition_v1",
            provenance["conditioning_applied"] is True,
            "fresh_initial_remainder_conditioning" not in restart["lifting"],
            pointwise["requested"] == pointwise["effective"] == "compile",
            pointwise["fallback_allowed"] is False,
        )
    )
    return {
        "schema": SCHEMA,
        "classification": (
            PASS
            if passed
            else "FAIL_RC4_3_7_INTERNAL_LIFTING_CUDA_CORRECTNESS_AND_RESTART"
        ),
        "passed": passed,
        "environment": {
            "device": "cuda:0",
            "device_name": torch.cuda.get_device_name(0),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "pssolver_import": str(
                Path(__import__("pssolver").__file__).resolve()
            ),
        },
        "fresh_conditioning": {
            "policy": "localized_quintic_wall_transition_v1",
            "transition_width": 2.0,
            "measurement_layer": MEASUREMENT_LAYER,
            "cpu_reference": cpu_direct,
            "cuda": cuda_direct,
            "cross_device_relative_l2": cross_device_relative_l2,
            "cross_device_relative_l2_max": 1.0e-15,
            "interior_byte_identical_to_unconditioned": cuda_direct[
                "interior_byte_identical_to_unconditioned"
            ],
            "transition_region_changed": cuda_direct[
                "transition_region_changed"
            ],
            "wall_residual_linf": _wall_residual(continuous),
            "wall_residual_linf_max": 1.0e-12,
            "post_projection_diagnostic": post_projection,
        },
        "restart": {
            "segment_steps": 2,
            "final_steps": 4,
            "restored_steps": restored_steps,
            "fields_byte_identical": field_equal,
            "physical_q_byte_identical": physical_equal,
            "all_byte_identical": all(
                (*field_equal.values(), *physical_equal.values())
            ),
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
