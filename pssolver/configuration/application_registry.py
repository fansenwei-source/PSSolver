"""Private, tensor-free registry of the qualified application dispatch tables.

S-18/R1 first proved these selections in the inactive shadow stage.
Compiler discovery, lowering rejection metadata, and binding construction have
different historical orders; preserve each explicitly instead of sorting them.
Descriptors contain strings only: importing this module never resolves a
compiler, allocates a runtime, or changes the public capability surface.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ApplicationRegistration:
    equation_variant: str
    geometry_name: str
    application: str
    compiler: str
    compiler_accepts_result_type: bool
    lowering_kind: str
    topology: tuple[str, ...]
    capabilities: tuple[str, ...]
    runtime_paths: tuple[str, ...]
    request_type: str
    runtime_factory: str
    adapter_protocol: str
    runtime_accepts_builders: bool
    runtime_request_error: str

    @property
    def key(self) -> tuple[str, str]:
        return self.equation_variant, self.geometry_name

    def compiler_metadata(self) -> dict[str, str]:
        # Field order is part of the existing public metadata representation.
        return {
            "equation_variant": self.equation_variant,
            "geometry_name": self.geometry_name,
            "application": self.application,
            "adapter": self.compiler,
        }


@dataclass(frozen=True, slots=True)
class RuntimeRegistration:
    application: ApplicationRegistration
    runtime_path: str
    kind: str
    builder_provision: str
    builder_parameter: str | None
    package_builder: str | None
    package_request_error: str | None

    @property
    def key(self) -> tuple[str, str, str]:
        return (*self.application.key, self.runtime_path)

    def binding_fields(self) -> tuple[object, ...]:
        return (
            self.kind,
            self.application.request_type,
            self.application.runtime_factory,
            self.application.adapter_protocol,
            self.builder_provision,
            self.builder_parameter,
        )


_COMPLETE_CAPABILITIES = (
    "beris_edwards_q_evolution",
    "beris_edwards_q_gradient",
    "beris_edwards_velocity_gradient",
    "beris_edwards_molecular_field",
    "beris_edwards_stress",
    "beris_edwards_force",
    "incompressible_stokes",
)

_PLANE = ApplicationRegistration(
    "complete_stress_beris_edwards", "plane_slab",
    "plane_complete_stress_beris_edwards",
    "pssolver.configuration.public_simulation_runner."
    "_compile_plane_public_simulation",
    False, "plane", ("periodic", "periodic", "bounded"),
    _COMPLETE_CAPABILITIES, ("legacy_production", "compiled_v2"),
    "pssolver.runtime.plane_beris_edwards.PlaneRuntimeBuildRequest",
    "pssolver.runtime.plane_beris_edwards.build_plane_beris_edwards_runtime",
    "pssolver.runtime.plane_beris_edwards.PlaneRuntimeAdapterProtocol",
    True, "Plane construction requires PlaneRuntimeBuildRequest",
)
_PERIODIC = ApplicationRegistration(
    "complete_stress_beris_edwards", "periodic_box",
    "periodic_complete_stress_beris_edwards",
    "pssolver.configuration.legacy_periodic."
    "compile_periodic_public_simulation",
    True, "periodic", ("periodic", "periodic", "periodic"),
    _COMPLETE_CAPABILITIES, ("periodic_spectral",),
    "pssolver.configuration.legacy_periodic.PeriodicTranslation",
    "pssolver.engine.engine.Engine",
    "pssolver.engine.engine.Engine",
    False, "periodic construction requires PeriodicRuntimeBuildRequest",
)
_CHANNEL_COMPLETE = ApplicationRegistration(
    "complete_stress_beris_edwards", "rectangular_channel",
    "channel_complete_stress_beris_edwards",
    "pssolver.configuration.public_channel_beris_edwards_compiler."
    "compile_channel_beris_edwards_public_simulation",
    True, "channel_complete", ("periodic", "bounded", "bounded"),
    _COMPLETE_CAPABILITIES, ("channel_complete_stress",),
    "pssolver.runtime.channel_beris_edwards.ChannelBerisEdwardsRuntimeBuildRequest",
    "pssolver.runtime.channel_beris_edwards.build_channel_beris_edwards_runtime",
    "pssolver.runtime.channel_beris_edwards.ChannelBerisEdwardsRuntimeAdapterProtocol",
    False, "complete-stress Channel construction requires ChannelBerisEdwardsRuntimeBuildRequest",
)

# Public compiler/capability order, not lowering error order.
APPLICATIONS = (_PLANE, _PERIODIC, _CHANNEL_COMPLETE)
APPLICATION_BY_PAIR = MappingProxyType({item.key: item for item in APPLICATIONS})
LOWERING_REJECTION_PAIRS = tuple(item.key for item in (_PLANE, _PERIODIC, _CHANNEL_COMPLETE))

# Preserve the local binding table's historical insertion order.
RUNTIMES = (
    RuntimeRegistration(
        _PERIODIC, "periodic_spectral", "periodic_complete_stress",
        "package", None, None,
        "periodic construction requires PeriodicRuntimeBuildRequest",
    ),
    RuntimeRegistration(
        _CHANNEL_COMPLETE, "channel_complete_stress", "channel_complete_stress",
        "package", None, None,
        "complete-stress Channel construction requires Channel request",
    ),
    RuntimeRegistration(
        _PLANE, "legacy_production", "plane_legacy_production",
        "package", None, None,
        None,
    ),
    RuntimeRegistration(
        _PLANE, "compiled_v2", "plane_compiled_v2", "caller", "compiled_builder",
        "pssolver.runtime.plane_application_bridge.build_package_compiled_plane_runtime",
        "compiled Plane construction requires Plane request",
    ),
)
RUNTIME_BY_KEY = MappingProxyType({item.key: item for item in RUNTIMES})
RUNTIME_BY_KIND = MappingProxyType({item.kind: item for item in RUNTIMES})

# This is not a new public registry/extension API.
__all__ = []
