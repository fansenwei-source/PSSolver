"""Package-owned builders for the two qualified Channel runtimes."""

from __future__ import annotations

from pssolver.configuration.channel_active_nematics_declarations import (
    ChannelRuntimePath,
)
from pssolver.runtime.channel_active_nematics import ChannelRuntimeBuildRequest


def build_package_legacy_channel_runtime(
    request: ChannelRuntimeBuildRequest,
) -> object:
    """Construct the legacy Channel solver from one typed request."""

    if not isinstance(request, ChannelRuntimeBuildRequest):
        raise TypeError("request must be a ChannelRuntimeBuildRequest")
    if request.run_spec.runtime_path is not ChannelRuntimePath.LEGACY_CHANNEL:
        raise ValueError("legacy Channel builder requires legacy_channel")
    # Keep both the legacy implementation and Torch materialization lazy.
    import torch

    from pssolver.channel import build_active_nematic_channel

    run_spec = request.run_spec
    material = run_spec.components.material
    pressure = run_spec.components.pressure_solver
    solver = build_active_nematic_channel(
        run_spec.shape,
        run_spec.lengths,
        run_spec.dt,
        request.initial_q,
        device=request.device,
        batchsize=run_spec.batch_size,
        rho=material.rho,
        elastic_constant=material.elastic_constant,
        beta=material.beta,
        friction=material.friction,
        viscosity=material.viscosity,
        pressure_rel_tol=pressure.relative_tolerance,
        pressure_max_iter=pressure.max_iterations,
        pressure_fixed_iterations=pressure.fixed_iterations,
    )
    solver.parameters["alpha"] = torch.full(
        (run_spec.batch_size, *run_spec.shape),
        material.activity,
        dtype=solver.dtype,
        device=solver.device,
    )
    return solver


def build_package_compiled_channel_runtime(
    request: ChannelRuntimeBuildRequest,
) -> object:
    if not isinstance(request, ChannelRuntimeBuildRequest):
        raise TypeError("request must be a ChannelRuntimeBuildRequest")
    if (
        request.run_spec.runtime_path
        is not ChannelRuntimePath.COMPILED_CHANNEL_V2
    ):
        raise ValueError(
            "compiled Channel builder requires compiled_channel_v2"
        )
    # The experimental implementation remains lazy and is selected only after
    # the immutable request has chosen its qualified compiled path.
    from pssolver.experimental.channel_compiled_v2 import (
        build_channel_compiled_v2_runtime,
    )

    return build_channel_compiled_v2_runtime(
        request.run_spec,
        initial_q=request.initial_q,
        device=request.device,
    )


__all__ = [
    "build_package_compiled_channel_runtime",
    "build_package_legacy_channel_runtime",
]
