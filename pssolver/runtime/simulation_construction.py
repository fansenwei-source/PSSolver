"""Opt-in P7.7.4 connection from generic bindings to qualified runtimes.

The connector validates the complete declaration and binding before invoking
an existing Plane or Channel runtime factory.  It returns that factory's
adapter directly: no numerical wrapper, tensor copy, registry lookup, or
timestep dispatch is introduced.
"""

from __future__ import annotations

from collections.abc import Callable

from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation_binding import bind_simulation_runtime
from pssolver.planning.construction import (
    BuilderProvision,
    RuntimeConstructionBinding,
    RuntimeConstructionKind,
)

from .channel_active_nematics import (
    ChannelRuntimeAdapterProtocol,
    ChannelRuntimeBuildRequest,
    build_channel_active_nematic_runtime,
)
from .plane_beris_edwards import (
    PlaneRuntimeAdapterProtocol,
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)
from .periodic_beris_edwards import (
    PeriodicRuntimeAdapterProtocol,
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)


RuntimeAdapter = (
    PlaneRuntimeAdapterProtocol
    | ChannelRuntimeAdapterProtocol
    | PeriodicRuntimeAdapterProtocol
)
RuntimeBuilder = Callable[[], object]


def _request_simulation(request: object):
    if isinstance(request, PlaneRuntimeBuildRequest):
        return compose_plane_beris_edwards_simulation(
            decompose_plane_beris_edwards_run_spec(request.run_spec)
        )
    if isinstance(request, ChannelRuntimeBuildRequest):
        return compose_channel_active_nematics_simulation(
            request.run_spec.components
        )
    if isinstance(request, PeriodicRuntimeBuildRequest):
        return request.run_spec.simulation
    raise TypeError(
        "request must be a PlaneRuntimeBuildRequest or "
        "ChannelRuntimeBuildRequest or PeriodicRuntimeBuildRequest"
    )


def _require_builder_contract(
    binding: RuntimeConstructionBinding,
    *,
    legacy_builder: RuntimeBuilder | None,
    compiled_builder: RuntimeBuilder | None,
) -> None:
    for value, description in (
        (legacy_builder, "legacy_builder"),
        (compiled_builder, "compiled_builder"),
    ):
        if value is not None and not callable(value):
            raise TypeError(f"{description} must be callable")
    selected = binding.builder_parameter
    provided = {
        "legacy_builder": legacy_builder,
        "compiled_builder": compiled_builder,
    }
    unexpected = tuple(
        name
        for name, value in provided.items()
        if value is not None and name != selected
    )
    if unexpected:
        raise ValueError(
            f"unselected runtime builders are forbidden: {unexpected!r}"
        )
    if (
        binding.builder_provision is BuilderProvision.CALLER
        and provided[selected] is None
    ):
        raise RuntimeError(
            f"{binding.kind.value} requires caller-provided {selected}; "
            "runtime fallback is forbidden"
        )


def build_bound_simulation_runtime(
    binding: RuntimeConstructionBinding,
    request: (
        PlaneRuntimeBuildRequest
        | ChannelRuntimeBuildRequest
        | PeriodicRuntimeBuildRequest
    ),
    *,
    legacy_builder: RuntimeBuilder | None = None,
    compiled_builder: RuntimeBuilder | None = None,
) -> RuntimeAdapter:
    """Validate and delegate one opt-in generic construction request."""

    if not isinstance(binding, RuntimeConstructionBinding):
        raise TypeError("binding must be a RuntimeConstructionBinding")
    simulation = _request_simulation(request)
    expected = bind_simulation_runtime(simulation)
    if binding != expected:
        raise ValueError(
            "runtime construction binding does not match the build request"
        )
    _require_builder_contract(
        binding,
        legacy_builder=legacy_builder,
        compiled_builder=compiled_builder,
    )

    if binding.kind in {
        RuntimeConstructionKind.PLANE_LEGACY_PRODUCTION,
        RuntimeConstructionKind.PLANE_COMPILED_V2,
    }:
        if not isinstance(request, PlaneRuntimeBuildRequest):
            raise TypeError("Plane construction requires PlaneRuntimeBuildRequest")
        return build_plane_beris_edwards_runtime(
            request,
            legacy_builder=legacy_builder,
            compiled_builder=compiled_builder,
        )
    if binding.kind in {
        RuntimeConstructionKind.CHANNEL_LEGACY,
        RuntimeConstructionKind.CHANNEL_COMPILED_V2,
    }:
        if not isinstance(request, ChannelRuntimeBuildRequest):
            raise TypeError(
                "Channel construction requires ChannelRuntimeBuildRequest"
            )
        return build_channel_active_nematic_runtime(
            request,
            legacy_builder=legacy_builder,
            compiled_builder=compiled_builder,
        )
    if binding.kind is RuntimeConstructionKind.PERIODIC_COMPLETE_STRESS:
        if not isinstance(request, PeriodicRuntimeBuildRequest):
            raise TypeError(
                "periodic construction requires PeriodicRuntimeBuildRequest"
            )
        return build_periodic_beris_edwards_runtime(request)
    raise AssertionError("unreachable runtime construction kind")


__all__ = ["build_bound_simulation_runtime"]
