"""Tensor-free products of simulation capability lowering.

The objects in this module describe what an execution builder would have to
construct.  They contain no tensors, backend objects, callables, registries,
or solver instances.  P7.7.3 deliberately keeps these plans disconnected from
the qualified Plane and Channel runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math

from pssolver.core.boundary import BoundaryKind
from pssolver.core.fields import FieldRole
from pssolver.core.geometry import AxisTopology

from .plan import TransformKind


SIMULATION_LOWERING_PLAN_SCHEMA_VERSION = 1


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{description} must be a Python identifier")
    return value


def _qualified_name(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    if any(not part.isidentifier() for part in value.split(".")):
        raise ValueError(
            f"{description} must be a dotted Python identifier"
        )
    return value


def _string_tuple(values: object, description: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{description} must be an iterable, not a string")
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{description} must be iterable") from exc
    if any(
        not isinstance(value, str) or not value.isidentifier()
        for value in normalized
    ):
        raise ValueError(
            f"{description} must contain Python identifiers"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{description} must be unique")
    return normalized


def _canonical_json(value: object, description: str) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{description} must be finite and JSON-compatible"
        ) from exc


class BasisProvenance(str, Enum):
    """Reason that a component occupies a resolved basis space."""

    PHYSICAL_BOUNDARY = "physical_boundary"
    ALGEBRAIC_COMPATIBILITY = "algebraic_compatibility"
    DERIVATIVE_PARITY = "derivative_parity"
    CONSTITUTIVE_PARITY = "constitutive_parity"
    PROJECTED_OUTPUT = "projected_output"


class DerivativeMultiplier(str, Enum):
    """Exact modal action of one first derivative."""

    FOURIER_IK = "fourier_i_k"
    COSINE_TO_NEGATIVE_SINE = "cosine_to_negative_sine"
    SINE_TO_POSITIVE_COSINE = "sine_to_positive_cosine"


@dataclass(frozen=True, slots=True)
class AxisBasisRequirement:
    """Resolved scalar basis on one tensor-product axis.

    ``modal_kind`` labels periodic/even/odd modal parity.  It is a physical
    boundary kind only when the enclosing component provenance says so.
    Derived gradients and constitutive intermediates must not reinterpret the
    label as a separately prescribed wall law.
    """

    axis: int
    axis_name: str
    topology: AxisTopology
    modal_kind: BoundaryKind
    transform_kind: TransformKind
    physical_size: int
    spectral_size: int
    hermitian_packed: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.axis, int)
            or isinstance(self.axis, bool)
            or self.axis < 0
        ):
            raise ValueError("basis axis must be a non-negative integer")
        _identifier(self.axis_name, "basis axis name")
        if not isinstance(self.topology, AxisTopology):
            raise TypeError("basis topology must be an AxisTopology")
        if not isinstance(self.modal_kind, BoundaryKind):
            raise TypeError("basis modal kind must be a BoundaryKind")
        if not isinstance(self.transform_kind, TransformKind):
            raise TypeError("basis transform kind must be a TransformKind")
        for name in ("physical_size", "spectral_size"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.hermitian_packed, bool):
            raise TypeError("hermitian_packed must be a bool")
        periodic = self.topology is AxisTopology.PERIODIC
        if periodic != (self.modal_kind is BoundaryKind.PERIODIC):
            raise ValueError("basis modal kind conflicts with topology")
        if periodic != (self.transform_kind is TransformKind.FFT):
            raise ValueError("basis transform kind conflicts with topology")
        expected_size = (
            self.physical_size // 2 + 1
            if self.hermitian_packed
            else self.physical_size
        )
        if self.spectral_size != expected_size:
            raise ValueError("basis spectral size conflicts with packing")
        if self.hermitian_packed and not periodic:
            raise ValueError("Hermitian packing requires a periodic axis")

    def to_metadata(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "axis_name": self.axis_name,
            "topology": self.topology.value,
            "modal_kind": self.modal_kind.value,
            "transform_kind": self.transform_kind.value,
            "physical_size": self.physical_size,
            "spectral_size": self.spectral_size,
            "hermitian_packed": self.hermitian_packed,
        }


@dataclass(frozen=True, slots=True)
class ComponentBasisRequirement:
    """Unique resolved basis for one logical scalar component."""

    field_name: str
    component: str
    role: FieldRole
    axes: tuple[AxisBasisRequirement, ...]
    provenance: BasisProvenance
    source_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.field_name, "basis field name")
        _identifier(self.component, "basis component")
        if not isinstance(self.role, FieldRole):
            raise TypeError("basis role must be a FieldRole")
        axes = tuple(self.axes)
        if not axes or any(
            not isinstance(axis, AxisBasisRequirement) for axis in axes
        ):
            raise TypeError(
                "basis axes must contain AxisBasisRequirement objects"
            )
        if tuple(axis.axis for axis in axes) != tuple(range(len(axes))):
            raise ValueError("basis axes must be contiguous and ordered")
        if not isinstance(self.provenance, BasisProvenance):
            raise TypeError("basis provenance must be a BasisProvenance")
        object.__setattr__(
            self,
            "source_components",
            _string_tuple(self.source_components, "basis source components"),
        )
        object.__setattr__(self, "axes", axes)

    @property
    def transform_kinds(self) -> tuple[TransformKind, ...]:
        return tuple(axis.transform_kind for axis in self.axes)

    def to_metadata(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "component": self.component,
            "role": self.role.value,
            "axes": [axis.to_metadata() for axis in self.axes],
            "provenance": self.provenance.value,
            "source_components": list(self.source_components),
        }


@dataclass(frozen=True, slots=True)
class DerivativeRequirement:
    """One source-to-output first-derivative basis transition."""

    capability: str
    source_component: str
    output_component: str
    axis: int
    multiplier: DerivativeMultiplier
    source_transform_kinds: tuple[TransformKind, ...]
    output_transform_kinds: tuple[TransformKind, ...]

    def __post_init__(self) -> None:
        _identifier(self.capability, "derivative capability")
        _identifier(self.source_component, "derivative source component")
        _identifier(self.output_component, "derivative output component")
        if (
            not isinstance(self.axis, int)
            or isinstance(self.axis, bool)
            or self.axis < 0
        ):
            raise ValueError("derivative axis must be non-negative")
        if not isinstance(self.multiplier, DerivativeMultiplier):
            raise TypeError(
                "derivative multiplier must be a DerivativeMultiplier"
            )
        source = tuple(self.source_transform_kinds)
        output = tuple(self.output_transform_kinds)
        if not source or len(source) != len(output) or self.axis >= len(source):
            raise ValueError("derivative basis dimensions are inconsistent")
        if any(not isinstance(value, TransformKind) for value in (*source, *output)):
            raise TypeError("derivative bases must contain TransformKind values")
        for index, (before, after) in enumerate(zip(source, output)):
            if index != self.axis and before is not after:
                raise ValueError(
                    "a derivative can change basis only on its own axis"
                )
        expected = {
            TransformKind.FFT: (
                TransformKind.FFT,
                DerivativeMultiplier.FOURIER_IK,
            ),
            TransformKind.DCT: (
                TransformKind.DST,
                DerivativeMultiplier.COSINE_TO_NEGATIVE_SINE,
            ),
            TransformKind.DST: (
                TransformKind.DCT,
                DerivativeMultiplier.SINE_TO_POSITIVE_COSINE,
            ),
        }[source[self.axis]]
        if (output[self.axis], self.multiplier) != expected:
            raise ValueError("derivative multiplier and parity transition disagree")
        object.__setattr__(self, "source_transform_kinds", source)
        object.__setattr__(self, "output_transform_kinds", output)

    def to_metadata(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "source_component": self.source_component,
            "output_component": self.output_component,
            "axis": self.axis,
            "multiplier": self.multiplier.value,
            "source_transform_kinds": [
                value.value for value in self.source_transform_kinds
            ],
            "output_transform_kinds": [
                value.value for value in self.output_transform_kinds
            ],
        }


@dataclass(frozen=True, slots=True)
class CapabilityImplementationRequirement:
    """Construction-time binding of a scientific capability to an algorithm."""

    capability: str
    implementation_name: str
    strategy: str
    input_components: tuple[str, ...]
    output_components: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.capability, "capability")
        _identifier(self.implementation_name, "implementation name")
        _identifier(self.strategy, "implementation strategy")
        object.__setattr__(
            self,
            "input_components",
            _string_tuple(self.input_components, "capability inputs"),
        )
        object.__setattr__(
            self,
            "output_components",
            _string_tuple(self.output_components, "capability outputs"),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "implementation_name": self.implementation_name,
            "strategy": self.strategy,
            "input_components": list(self.input_components),
            "output_components": list(self.output_components),
        }


@dataclass(frozen=True, slots=True)
class NullspaceRequirement:
    """Pressure gauge and any uniform velocity-mode modeling choice."""

    pressure_component: str
    pressure_gauge: str
    tangential_velocity_components: tuple[str, ...]
    tangential_policy: str
    friction: float
    uniform_mode_action: str

    def __post_init__(self) -> None:
        _identifier(self.pressure_component, "pressure component")
        _identifier(self.pressure_gauge, "pressure gauge")
        object.__setattr__(
            self,
            "tangential_velocity_components",
            _string_tuple(
                self.tangential_velocity_components,
                "tangential velocity components",
            ),
        )
        _identifier(self.tangential_policy, "tangential policy")
        if (
            not isinstance(self.friction, (int, float))
            or isinstance(self.friction, bool)
            or not math.isfinite(float(self.friction))
            or float(self.friction) < 0.0
        ):
            raise ValueError("friction must be finite and non-negative")
        object.__setattr__(self, "friction", float(self.friction))
        _identifier(self.uniform_mode_action, "uniform mode action")

    def to_metadata(self) -> dict[str, object]:
        return {
            "pressure_component": self.pressure_component,
            "pressure_gauge": self.pressure_gauge,
            "tangential_velocity_components": list(
                self.tangential_velocity_components
            ),
            "tangential_policy": self.tangential_policy,
            "friction": self.friction,
            "uniform_mode_action": self.uniform_mode_action,
        }


@dataclass(frozen=True, slots=True)
class GeometrySolverRequirement:
    """Exact geometry-specific algebraic solver required by a plan."""

    capability: str
    family: str
    implementation: str
    force_components: tuple[str, ...]
    velocity_components: tuple[str, ...]
    pressure_component: str
    force_projection_components: tuple[str, ...]
    options_json: str

    def __post_init__(self) -> None:
        _identifier(self.capability, "solver capability")
        _identifier(self.family, "solver family")
        _qualified_name(self.implementation, "solver implementation")
        forces = _string_tuple(self.force_components, "solver forces")
        velocities = _string_tuple(
            self.velocity_components,
            "solver velocities",
        )
        projections = _string_tuple(
            self.force_projection_components,
            "force projection components",
        )
        if len(forces) != 3 or len(velocities) != 3 or len(projections) != 3:
            raise ValueError("the Stokes solver requires three vector components")
        _identifier(self.pressure_component, "solver pressure component")
        try:
            options = json.loads(self.options_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("solver options_json must be valid JSON") from exc
        canonical = _canonical_json(options, "solver options")
        if canonical != self.options_json:
            raise ValueError("solver options_json must use canonical JSON")
        object.__setattr__(self, "force_components", forces)
        object.__setattr__(self, "velocity_components", velocities)
        object.__setattr__(self, "force_projection_components", projections)

    def to_metadata(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "family": self.family,
            "implementation": self.implementation,
            "force_components": list(self.force_components),
            "velocity_components": list(self.velocity_components),
            "pressure_component": self.pressure_component,
            "force_projection_components": list(
                self.force_projection_components
            ),
            "options": json.loads(self.options_json),
        }


@dataclass(frozen=True, slots=True)
class SimulationLoweringPlan:
    """Complete tensor-free lowering of one supported simulation request."""

    source_simulation_sha256: str
    equation_variant: str
    geometry_name: str
    component_bases: tuple[ComponentBasisRequirement, ...]
    derivatives: tuple[DerivativeRequirement, ...]
    capabilities: tuple[CapabilityImplementationRequirement, ...]
    solver: GeometrySolverRequirement
    nullspace: NullspaceRequirement
    numerics_json: str
    time_integration_json: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_simulation_sha256, str)
            or len(self.source_simulation_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.source_simulation_sha256
            )
        ):
            raise ValueError("source simulation SHA-256 is invalid")
        _identifier(self.equation_variant, "equation variant")
        _identifier(self.geometry_name, "geometry name")
        component_bases = tuple(self.component_bases)
        derivatives = tuple(self.derivatives)
        capabilities = tuple(self.capabilities)
        if not component_bases or any(
            not isinstance(value, ComponentBasisRequirement)
            for value in component_bases
        ):
            raise TypeError("component_bases contain an invalid value")
        if any(
            not isinstance(value, DerivativeRequirement)
            for value in derivatives
        ):
            raise TypeError("derivatives contain an invalid value")
        if not capabilities or any(
            not isinstance(value, CapabilityImplementationRequirement)
            for value in capabilities
        ):
            raise TypeError("capabilities contain an invalid value")
        components = tuple(value.component for value in component_bases)
        if len(set(components)) != len(components):
            raise ValueError("component basis requirements must be unique")
        capability_names = tuple(value.capability for value in capabilities)
        if len(set(capability_names)) != len(capability_names):
            raise ValueError("capability requirements must be unique")
        derivative_outputs = tuple(value.output_component for value in derivatives)
        if len(set(derivative_outputs)) != len(derivative_outputs):
            raise ValueError("derivative output requirements must be unique")
        if not isinstance(self.solver, GeometrySolverRequirement):
            raise TypeError("solver must be a GeometrySolverRequirement")
        if not isinstance(self.nullspace, NullspaceRequirement):
            raise TypeError("nullspace must be a NullspaceRequirement")
        for value, description in (
            (self.numerics_json, "numerics"),
            (self.time_integration_json, "time integration"),
        ):
            try:
                decoded = json.loads(value)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{description}_json must be valid JSON") from exc
            if value != _canonical_json(decoded, description):
                raise ValueError(f"{description}_json must use canonical JSON")
        object.__setattr__(
            self,
            "component_bases",
            tuple(sorted(component_bases, key=lambda value: value.component)),
        )
        object.__setattr__(
            self,
            "derivatives",
            tuple(sorted(derivatives, key=lambda value: value.output_component)),
        )
        object.__setattr__(
            self,
            "capabilities",
            tuple(sorted(capabilities, key=lambda value: value.capability)),
        )

    def basis_for(self, component: str) -> ComponentBasisRequirement:
        for value in self.component_bases:
            if value.component == component:
                return value
        raise KeyError(component)

    def derivative_for(self, component: str) -> DerivativeRequirement:
        for value in self.derivatives:
            if value.output_component == component:
                return value
        raise KeyError(component)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": SIMULATION_LOWERING_PLAN_SCHEMA_VERSION,
            "source_simulation_sha256": self.source_simulation_sha256,
            "equation_variant": self.equation_variant,
            "geometry_name": self.geometry_name,
            "component_bases": [
                value.to_metadata() for value in self.component_bases
            ],
            "derivatives": [
                value.to_metadata() for value in self.derivatives
            ],
            "capabilities": [
                value.to_metadata() for value in self.capabilities
            ],
            "solver": self.solver.to_metadata(),
            "nullspace": self.nullspace.to_metadata(),
            "numerics": json.loads(self.numerics_json),
            "time_integration": json.loads(self.time_integration_json),
        }

    def canonical_sha256(self) -> str:
        payload = _canonical_json(self.to_metadata(), "lowering plan")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AxisBasisRequirement",
    "BasisProvenance",
    "CapabilityImplementationRequirement",
    "ComponentBasisRequirement",
    "DerivativeMultiplier",
    "DerivativeRequirement",
    "GeometrySolverRequirement",
    "NullspaceRequirement",
    "SIMULATION_LOWERING_PLAN_SCHEMA_VERSION",
    "SimulationLoweringPlan",
]
