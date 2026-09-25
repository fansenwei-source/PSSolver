"""Tensor-free P7.7.5 normalization of runtime construction ownership."""

from __future__ import annotations

from pssolver.planning.construction import (
    RuntimeConstructionBinding,
    RuntimeConstructionKind,
)
from pssolver.planning.package_construction import (
    PackageRuntimeConstructionPlan,
)

from .simulation import SimulationSpec
from .simulation_binding import bind_simulation_runtime


_PACKAGE_FACTORY = (
    "pssolver.runtime.package_construction.build_package_simulation_runtime"
)
_IMPLEMENTATION_BUILDERS = {
    RuntimeConstructionKind.PERIODIC_COMPLETE_STRESS: (
        "pssolver.runtime.periodic_beris_edwards."
        "build_periodic_beris_edwards_runtime"
    ),
    RuntimeConstructionKind.PLANE_LEGACY_PRODUCTION: (
        "pssolver.runtime.plane_legacy.build_legacy_plane_runtime"
    ),
    RuntimeConstructionKind.PLANE_COMPILED_V2: (
        "pssolver.runtime.plane_application_bridge."
        "build_package_compiled_plane_runtime"
    ),
    RuntimeConstructionKind.CHANNEL_LEGACY: (
        "pssolver.runtime.channel_application_bridge."
        "build_package_legacy_channel_runtime"
    ),
    RuntimeConstructionKind.CHANNEL_COMPILED_V2: (
        "pssolver.runtime.channel_application_bridge."
        "build_package_compiled_channel_runtime"
    ),
}


def normalize_package_construction(
    binding: RuntimeConstructionBinding,
) -> PackageRuntimeConstructionPlan:
    """Replace caller builder ownership with an exact package builder name."""

    if not isinstance(binding, RuntimeConstructionBinding):
        raise TypeError("binding must be a RuntimeConstructionBinding")
    try:
        implementation_builder = _IMPLEMENTATION_BUILDERS[binding.kind]
    except KeyError as exc:
        raise ValueError("runtime binding kind has no package builder") from exc
    return PackageRuntimeConstructionPlan(
        source_binding_sha256=binding.canonical_sha256(),
        source_simulation_sha256=binding.source_simulation_sha256,
        lowering_plan_sha256=binding.lowering_plan_sha256,
        kind=binding.kind,
        request_type=binding.request_type,
        package_factory=_PACKAGE_FACTORY,
        implementation_builder=implementation_builder,
        adapter_protocol=binding.adapter_protocol,
    )


def plan_package_runtime_construction(
    simulation: SimulationSpec,
) -> PackageRuntimeConstructionPlan:
    """Lower, bind, and normalize one supported simulation declaration."""

    if not isinstance(simulation, SimulationSpec):
        raise TypeError("simulation must be a SimulationSpec")
    binding = bind_simulation_runtime(simulation)
    return normalize_package_construction(binding)


__all__ = [
    "normalize_package_construction",
    "plan_package_runtime_construction",
]
