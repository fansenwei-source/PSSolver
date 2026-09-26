"""P7.7.5 package-owned construction for qualified simulation runtimes.

Applications provide one immutable plan and an existing typed build request.
This module owns all builder selection and returns the existing runtime adapter
directly.  Selection and validation happen once before runtime allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation_binding import bind_simulation_runtime
from pssolver.planning.construction import RuntimeConstructionKind
from pssolver.planning.package_construction import (
    PackageRuntimeConstructionPlan,
)

from .channel_active_nematics import ChannelRuntimeBuildRequest
from .channel_beris_edwards import ChannelBerisEdwardsRuntimeBuildRequest
from .channel_application_bridge import (
    build_package_compiled_channel_runtime,
    build_package_legacy_channel_runtime,
)
from .plane_application_bridge import build_package_compiled_plane_runtime
from .plane_beris_edwards import PlaneRuntimeBuildRequest
from .periodic_beris_edwards import PeriodicRuntimeBuildRequest
from .simulation_construction import (
    RuntimeAdapter,
    build_bound_simulation_runtime,
)


PackageBuildRequest = (
    PlaneRuntimeBuildRequest
    | ChannelRuntimeBuildRequest
    | PeriodicRuntimeBuildRequest
    | ChannelBerisEdwardsRuntimeBuildRequest
)


@dataclass(frozen=True, slots=True)
class PackageRuntimeConstructionInput:
    """Typed construction input with no application-supplied callable."""

    plan: PackageRuntimeConstructionPlan
    request: PackageBuildRequest

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PackageRuntimeConstructionPlan):
            raise TypeError("plan must be a PackageRuntimeConstructionPlan")
        if not isinstance(
            self.request,
            (
                PlaneRuntimeBuildRequest,
                ChannelRuntimeBuildRequest,
                PeriodicRuntimeBuildRequest,
                ChannelBerisEdwardsRuntimeBuildRequest,
            ),
        ):
            raise TypeError(
                "request must be a PlaneRuntimeBuildRequest or "
                "ChannelRuntimeBuildRequest, PeriodicRuntimeBuildRequest, "
                "or ChannelBerisEdwardsRuntimeBuildRequest"
            )

    def to_metadata(self) -> dict[str, object]:
        run_spec = self.request.run_spec
        return {
            "schema_version": 1,
            "plan_sha256": self.plan.canonical_sha256(),
            "request_type": (
                f"{type(self.request).__module__}."
                f"{type(self.request).__qualname__}"
            ),
            "runtime_path": (
                run_spec.runtime_path.value
                if hasattr(run_spec.runtime_path, "value")
                else str(run_spec.runtime_path)
            ),
            "configuration_sha256": run_spec.canonical_sha256(),
            "device": str(self.request.device),
            "application_builder_supplied": False,
        }


def _request_simulation(request: PackageBuildRequest):
    if isinstance(request, PlaneRuntimeBuildRequest):
        return compose_plane_beris_edwards_simulation(
            decompose_plane_beris_edwards_run_spec(request.run_spec)
        )
    if isinstance(request, ChannelRuntimeBuildRequest):
        return compose_channel_active_nematics_simulation(
            request.run_spec.components
        )
    if isinstance(request, ChannelBerisEdwardsRuntimeBuildRequest):
        return request.run_spec.simulation
    return request.run_spec.simulation


def build_package_simulation_runtime(
    construction: PackageRuntimeConstructionInput,
) -> RuntimeAdapter:
    """Build exactly one normalized package-owned runtime without fallback."""

    if not isinstance(construction, PackageRuntimeConstructionInput):
        raise TypeError(
            "construction must be a PackageRuntimeConstructionInput"
        )
    request = construction.request
    simulation = _request_simulation(request)
    expected_plan = plan_package_runtime_construction(simulation)
    if construction.plan != expected_plan:
        raise ValueError(
            "package construction plan does not match the build request"
        )
    binding = bind_simulation_runtime(simulation)
    kind = construction.plan.kind

    if kind is RuntimeConstructionKind.PLANE_LEGACY_PRODUCTION:
        return build_bound_simulation_runtime(binding, request)
    if kind is RuntimeConstructionKind.PLANE_COMPILED_V2:
        if not isinstance(request, PlaneRuntimeBuildRequest):
            raise TypeError("compiled Plane construction requires Plane request")
        return build_bound_simulation_runtime(
            binding,
            request,
            compiled_builder=lambda: build_package_compiled_plane_runtime(
                request
            ),
        )
    if kind is RuntimeConstructionKind.PERIODIC_COMPLETE_STRESS:
        if not isinstance(request, PeriodicRuntimeBuildRequest):
            raise TypeError(
                "periodic construction requires PeriodicRuntimeBuildRequest"
            )
        return build_bound_simulation_runtime(binding, request)
    if kind is RuntimeConstructionKind.CHANNEL_COMPLETE_STRESS:
        if not isinstance(request, ChannelBerisEdwardsRuntimeBuildRequest):
            raise TypeError(
                "complete-stress Channel construction requires Channel request"
            )
        return build_bound_simulation_runtime(binding, request)
    if kind is RuntimeConstructionKind.CHANNEL_LEGACY:
        if not isinstance(request, ChannelRuntimeBuildRequest):
            raise TypeError("legacy Channel construction requires Channel request")
        return build_bound_simulation_runtime(
            binding,
            request,
            legacy_builder=lambda: build_package_legacy_channel_runtime(
                request
            ),
        )
    if kind is RuntimeConstructionKind.CHANNEL_COMPILED_V2:
        if not isinstance(request, ChannelRuntimeBuildRequest):
            raise TypeError(
                "compiled Channel construction requires Channel request"
            )
        return build_bound_simulation_runtime(
            binding,
            request,
            compiled_builder=lambda: build_package_compiled_channel_runtime(
                request
            ),
        )
    raise AssertionError("unreachable package construction kind")


__all__ = [
    "PackageRuntimeConstructionInput",
    "build_package_simulation_runtime",
]
