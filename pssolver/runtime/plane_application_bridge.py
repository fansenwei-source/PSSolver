"""Package-owned bridge for the qualified compiled Plane construction."""

from __future__ import annotations

from pssolver.configuration import PlaneRuntimePath

from .plane_beris_edwards import PlaneRuntimeBuildRequest


def build_package_compiled_plane_runtime(
    request: PlaneRuntimeBuildRequest,
) -> object:
    """Construct compiled Plane from one validated typed request."""

    if not isinstance(request, PlaneRuntimeBuildRequest):
        raise TypeError("request must be a PlaneRuntimeBuildRequest")
    if request.run_spec.runtime_path is not PlaneRuntimePath.COMPILED_V2:
        raise ValueError("compiled Plane builder requires compiled_v2")
    # Lazy by design: importing package construction for a legacy run must not
    # import the compiled workflow implementation.
    from pssolver.workflows.plane_compiled_v2 import (
        build_plane_compiled_v2_runtime,
    )

    return build_plane_compiled_v2_runtime(
        request.run_spec,
        device=request.device,
        initial_values=request.initial_values,
    )


__all__ = ["build_package_compiled_plane_runtime"]
