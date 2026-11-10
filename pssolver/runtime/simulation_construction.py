"""Opt-in P7.7.4 connection from generic bindings to qualified runtimes.

The connector validates the complete declaration and binding before invoking
an existing Plane or Channel runtime factory.  It returns that factory's
adapter directly: a registry selects construction once; no numerical wrapper, tensor copy, or
timestep dispatch is introduced.
"""

from __future__ import annotations

from collections.abc import Callable
from pssolver.configuration.application_registry import RUNTIME_BY_KIND
from pssolver.configuration.simulation_binding import bind_simulation_runtime
from pssolver.planning.construction import (
    BuilderProvision,
    RuntimeConstructionBinding,
)
from .channel_beris_edwards import ChannelBerisEdwardsRuntimeAdapterProtocol, ChannelBerisEdwardsRuntimeBuildRequest
from .plane_beris_edwards import PlaneRuntimeAdapterProtocol, PlaneRuntimeBuildRequest
from .channel_beris_edwards import (
    build_channel_beris_edwards_runtime as build_channel_beris_edwards_runtime,
)
from .plane_beris_edwards import (
    build_plane_beris_edwards_runtime as build_plane_beris_edwards_runtime,
)


RuntimeAdapter = (
    PlaneRuntimeAdapterProtocol
    | ChannelBerisEdwardsRuntimeAdapterProtocol
)
RuntimeBuilder = Callable[[], object]


def _request_simulation(request: object):
    if isinstance(request, PlaneRuntimeBuildRequest):
        return request.simulation
    if isinstance(request, ChannelBerisEdwardsRuntimeBuildRequest):
        return request.run_spec.simulation
    raise TypeError(
        "request must be a PlaneRuntimeBuildRequest or "
        "PeriodicRuntimeBuildRequest, or "
        "ChannelBerisEdwardsRuntimeBuildRequest"
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

        | ChannelBerisEdwardsRuntimeBuildRequest
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

    registration = RUNTIME_BY_KIND.get(binding.kind)
    if registration is None:
        raise AssertionError("unreachable runtime construction kind")
    application = registration.application
    request_type = globals()[application.request_type.rsplit(".", 1)[1]]
    if not isinstance(request, request_type):
        raise TypeError(application.runtime_request_error)
    factory = globals()[application.runtime_factory.rsplit(".", 1)[1]]
    if application.runtime_accepts_builders:
        return factory(request, legacy_builder=legacy_builder,
                       compiled_builder=compiled_builder)
    return factory(request)


__all__ = ["build_bound_simulation_runtime"]
