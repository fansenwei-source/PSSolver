"""P7.7.5 package-owned construction for qualified simulation runtimes.

Applications provide one immutable plan and an existing typed build request.
This module owns all builder selection and returns the existing runtime adapter
directly.  Selection and validation happen once before runtime allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

from pssolver.configuration.application_registry import RUNTIME_BY_KIND
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.simulation_binding import bind_simulation_runtime
from pssolver.planning.package_construction import (
    PackageRuntimeConstructionPlan,
)

from .channel_beris_edwards import ChannelBerisEdwardsRuntimeBuildRequest
from .periodic_beris_edwards import PeriodicRuntimeBuildRequest
from .plane_application_bridge import (
    build_package_compiled_plane_runtime as build_package_compiled_plane_runtime,
)
from .plane_beris_edwards import PlaneRuntimeBuildRequest
from .simulation_construction import (
    RuntimeAdapter,
    build_bound_simulation_runtime,
)


PackageBuildRequest = (
    PlaneRuntimeBuildRequest | PeriodicRuntimeBuildRequest | ChannelBerisEdwardsRuntimeBuildRequest
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
                PeriodicRuntimeBuildRequest,
                ChannelBerisEdwardsRuntimeBuildRequest,
            ),
        ):
            raise TypeError(
                "request must be a PlaneRuntimeBuildRequest or "
                "PeriodicRuntimeBuildRequest, "
                "or ChannelBerisEdwardsRuntimeBuildRequest"
            )

    def to_metadata(self) -> dict[str, object]:
        run_spec = self.request.run_spec
        metadata = {
            "schema_version": 1,
            "plan_sha256": self.plan.canonical_sha256(),
            "request_type": (f"{type(self.request).__module__}.{type(self.request).__qualname__}"),
            "runtime_path": (
                run_spec.runtime_path.value
                if hasattr(run_spec.runtime_path, "value")
                else str(run_spec.runtime_path)
            ),
            "configuration_sha256": run_spec.canonical_sha256(),
            "device": str(self.request.device),
            "application_builder_supplied": False,
        }
        if isinstance(self.request, PlaneRuntimeBuildRequest):
            conditioning = self.request.fresh_initial_conditioning_metadata()
            if conditioning is not None:
                metadata["fresh_initial_conditioning"] = conditioning
                metadata["fresh_initial_conditioning_applied"] = (
                    self.request.apply_fresh_initial_conditioning
                )
        return metadata


def _request_simulation(request: PackageBuildRequest):
    if isinstance(request, PlaneRuntimeBuildRequest):
        return request.simulation
    if isinstance(request, ChannelBerisEdwardsRuntimeBuildRequest):
        return request.run_spec.simulation
    return request.run_spec.simulation


def build_package_simulation_runtime(
    construction: PackageRuntimeConstructionInput,
) -> RuntimeAdapter:
    """Build exactly one normalized package-owned runtime without fallback."""

    if not isinstance(construction, PackageRuntimeConstructionInput):
        raise TypeError("construction must be a PackageRuntimeConstructionInput")
    request = construction.request
    simulation = _request_simulation(request)
    expected_plan = plan_package_runtime_construction(simulation)
    if construction.plan != expected_plan:
        raise ValueError("package construction plan does not match the build request")
    binding = bind_simulation_runtime(simulation)
    kind = construction.plan.kind

    registration = RUNTIME_BY_KIND.get(kind)
    if registration is None:
        raise AssertionError("unreachable package construction kind")
    # Plane legacy historically delegates this check to the bound
    # constructor; retain that ordering rather than adding a guard.
    if registration.package_request_error is not None:
        request_type = globals()[registration.application.request_type.rsplit(".", 1)[1]]
        if not isinstance(request, request_type):
            raise TypeError(registration.package_request_error)
    if registration.package_builder is None:
        return build_bound_simulation_runtime(binding, request)
    builder = globals()[registration.package_builder.rsplit(".", 1)[1]]
    return build_bound_simulation_runtime(
        binding,
        request,
        **{registration.builder_parameter: lambda: builder(request)},
    )


__all__ = [
    "PackageRuntimeConstructionInput",
    "build_package_simulation_runtime",
]
