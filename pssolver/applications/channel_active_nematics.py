"""Supported package application for the rectangular active-nematic Channel.

Importing this module is side-effect free.  The legacy runtime remains the
implicit default; the compiled runtime is selected only by an immutable run
specification and construction never falls back between paths.
"""

from __future__ import annotations

from collections.abc import Iterable
import platform
from pathlib import Path

import numpy as np
import torch

from pssolver.channel import build_active_nematic_channel
from pssolver.configuration.channel_active_nematics import ChannelActiveNematicRunSpec
from pssolver.configuration.channel_active_nematics_declarations import ChannelRuntimePath
from pssolver.models.active_nematics import Q_COMPONENTS, Q_convention_metadata, create_initial_condition, positive_equilibrium_S
from pssolver.run_metadata import prepare_new_run_directory
from pssolver.runtime.channel_active_nematics import (
    ChannelRuntimeBuildRequest,
    build_channel_active_nematic_runtime,
)
from pssolver.runtime.channel_application_bridge import (
    build_package_compiled_channel_runtime,
)
from pssolver.workflows.channel_active_nematics import ChannelActiveNematicsWorkflow, ChannelWorkflowResult


def _resolve_device(requested: str) -> torch.device:
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def _initial_q(run_spec: ChannelActiveNematicRunSpec) -> dict[str, torch.Tensor]:
    initial = run_spec.components.initial_condition
    if initial.mode != "generated":
        raise ValueError(
            "the P7.5 package workflow accepts generated initial state or "
            "a versioned workflow checkpoint; legacy snapshot input remains "
            "owned by Channel.py"
        )
    return create_initial_condition(
        "aligned_x_smooth_noise",
        shape=run_spec.shape,
        boundary_conditions=run_spec.boundaries.q,
        seed=initial.seed,
        S_initial=initial.initial_s,
        noise_theta=initial.noise_theta,
        noise_phi=initial.noise_phi,
        sigma_x=initial.smoothing_sigma[0],
        sigma_y=initial.smoothing_sigma[1],
        sigma_z=initial.smoothing_sigma[2],
        dtype=torch.float32,
    )


def _metadata(run_spec: ChannelActiveNematicRunSpec, device: torch.device) -> dict[str, object]:
    components = run_spec.components
    material = components.material
    return {
        "schema_version": 1,
        "application": "pssolver.applications.channel_active_nematics",
        "script": "Channel.py compatibility contract",
        "configuration": run_spec.identity_metadata(),
        "runtime_selection": run_spec.runtime_selection_metadata(),
        "runtime_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "resolved_device": str(device),
        },
        "solver": {
            "shape": list(run_spec.shape), "lengths": list(run_spec.lengths),
            "dt": run_spec.dt, "steps": run_spec.steps,
            "save_interval": run_spec.save_interval, "real_dtype": run_spec.dtype,
        },
        "model": {
            "name": "active_nematics",
            "variant": "legacy_channel_active_force_stokes",
            "Q_convention": Q_convention_metadata(),
            "parameters": {
                **material.to_metadata(),
                "S_initial": run_spec.initial_s,
                "S_bulk": positive_equilibrium_S(material.ldg_a, material.ldg_b, material.ldg_c),
            },
        },
        "boundary_conditions": run_spec.boundaries.to_metadata(),
        "numerics": components.numerics.to_metadata(),
        "initial_condition": components.initial_condition.to_metadata(),
    }


def run_channel_active_nematics(
    run_spec: ChannelActiveNematicRunSpec,
    *,
    progress: Iterable[int] | None = None,
) -> ChannelWorkflowResult:
    """Run one validated Channel request through the package-owned workflow."""

    if not isinstance(run_spec, ChannelActiveNematicRunSpec):
        raise TypeError("run_spec must be ChannelActiveNematicRunSpec")
    if run_spec.dtype != "float32":
        raise ValueError("P7.5 preserves the float32 Channel oracle")
    if run_spec.flow_alignment != 1.0:
        raise ValueError("P7.5 preserves the Channel flow_alignment=1 oracle")
    if run_spec.batch_size != 1:
        raise ValueError("P7.5 production output requires batch_size=1")
    device = _resolve_device(run_spec.device)
    initial_q = _initial_q(run_spec)
    metadata = _metadata(run_spec, device)
    request = ChannelRuntimeBuildRequest(run_spec, metadata, initial_q, device)
    material = run_spec.components.material
    pressure = run_spec.components.pressure_solver

    def legacy_builder():
        solver = build_active_nematic_channel(
            run_spec.shape, run_spec.lengths, run_spec.dt, initial_q,
            device=device, batchsize=run_spec.batch_size, rho=material.rho,
            elastic_constant=material.elastic_constant, beta=material.beta,
            friction=material.friction, viscosity=material.viscosity,
            pressure_rel_tol=pressure.relative_tolerance,
            pressure_max_iter=pressure.max_iterations,
            pressure_fixed_iterations=pressure.fixed_iterations,
        )
        solver.parameters["alpha"] = torch.full(
            (run_spec.batch_size, *run_spec.shape), material.activity,
            dtype=solver.dtype, device=solver.device,
        )
        return solver

    def compiled_builder():
        return build_package_compiled_channel_runtime(request)

    adapter = build_channel_active_nematic_runtime(
        request,
        legacy_builder=legacy_builder if run_spec.runtime_path is ChannelRuntimePath.LEGACY_CHANNEL else None,
        compiled_builder=compiled_builder if run_spec.runtime_path is ChannelRuntimePath.COMPILED_CHANNEL_V2 else None,
    )
    metadata["runtime_selection"] = {**run_spec.runtime_selection_metadata(), **adapter.to_metadata()}
    output = (
        run_spec.generated_output_directory
        if run_spec.initialization_mode == "generated"
        else run_spec.snapshot_output_directory
    )
    output = prepare_new_run_directory(output)
    return ChannelActiveNematicsWorkflow(adapter, run_spec, output, metadata).run(progress=progress)


__all__ = ["run_channel_active_nematics"]
