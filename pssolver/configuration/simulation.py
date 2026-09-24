"""Tensor-free composition of scientific and operational run declarations.

P7.7.2 deliberately stops before capability lowering or runtime construction.
The objects here compose already validated declarations, separate identity
classes, and reject incompatible field/face/topology combinations without
allocating tensors or selecting FFT, DCT, DST, lifting, tau, or solver code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType

from pssolver.core.boundary import (
    BoundaryAssignment,
    BoundaryKind,
    BoundarySemantic,
)
from pssolver.core.fields import FieldRole
from pssolver.core.geometry import AxisTopology, GeometrySpec
from pssolver.core.integrators import IntegratorSpec
from pssolver.core.numerics import NumericsConfig
from pssolver.systems.equations import EquationSystemSpec


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{description} must be a Python identifier")
    return value


def _json_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    if any(
        not isinstance(name, str) or not name.isidentifier()
        for name in value
    ):
        raise ValueError(f"{description} keys must be Python identifiers")
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{description} must be finite and JSON-compatible"
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _mapping_metadata(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(
        json.dumps(
            dict(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class InitialConditionSource(str, Enum):
    """Origin of an initial-condition realization."""

    GENERATED = "generated"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class InitialConditionSpec:
    """Initial-condition family, source, and realization parameters."""

    family: str
    source: InitialConditionSource
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _identifier(self.family, "initial-condition family"),
        )
        if not isinstance(self.source, InitialConditionSource):
            raise TypeError("initial-condition source is invalid")
        object.__setattr__(
            self,
            "parameters",
            _json_mapping(
                self.parameters,
                "initial-condition parameters",
            ),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "family": self.family,
            "source": self.source.value,
            "parameters": _mapping_metadata(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class TimeIntegrationSpec:
    """Integrator identity plus non-integrator refresh policy."""

    integrator: IntegratorSpec
    refresh: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.integrator, IntegratorSpec):
            raise TypeError("integrator must be an IntegratorSpec")
        object.__setattr__(
            self,
            "refresh",
            _json_mapping(self.refresh, "time-integration refresh"),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "integrator": self.integrator.to_metadata(),
            "refresh": _mapping_metadata(self.refresh),
        }


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Backend/runtime selector and implementation policy."""

    backend: str
    runtime_path: str
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _identifier(self.backend, "execution backend"),
        )
        object.__setattr__(
            self,
            "runtime_path",
            _identifier(self.runtime_path, "runtime path"),
        )
        object.__setattr__(
            self,
            "options",
            _json_mapping(self.options, "execution options"),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "runtime_path": self.runtime_path,
            "options": _mapping_metadata(self.options),
        }


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """Finite run duration plus observation/output/restart policy."""

    steps: int
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.steps, int)
            or isinstance(self.steps, bool)
            or self.steps <= 0
        ):
            raise ValueError("workflow steps must be a positive integer")
        object.__setattr__(
            self,
            "options",
            _json_mapping(self.options, "workflow options"),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "options": _mapping_metadata(self.options),
        }


@dataclass(frozen=True, slots=True)
class InvocationSpec:
    """Non-scientific invocation and provenance inputs."""

    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "options",
            _json_mapping(self.options, "invocation options"),
        )

    def to_metadata(self) -> dict[str, object]:
        return _mapping_metadata(self.options)


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    """Validated, tensor-free composition of one simulation declaration.

    The declaration is not executable.  It contains no tensors, plans,
    transform objects, solver instances, workspaces, or runtime state.
    """

    equation_system: EquationSystemSpec
    geometry: GeometrySpec
    boundaries: BoundaryAssignment
    numerics: NumericsConfig
    time_integration: TimeIntegrationSpec
    discretization_parameters: Mapping[str, object]
    initial_condition: InitialConditionSpec
    execution: ExecutionSpec
    workflow: WorkflowSpec
    invocation: InvocationSpec
    compatibility_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        declarations = (
            ("equation_system", self.equation_system, EquationSystemSpec),
            ("geometry", self.geometry, GeometrySpec),
            ("boundaries", self.boundaries, BoundaryAssignment),
            ("numerics", self.numerics, NumericsConfig),
            (
                "time_integration",
                self.time_integration,
                TimeIntegrationSpec,
            ),
            (
                "initial_condition",
                self.initial_condition,
                InitialConditionSpec,
            ),
            ("execution", self.execution, ExecutionSpec),
            ("workflow", self.workflow, WorkflowSpec),
            ("invocation", self.invocation, InvocationSpec),
        )
        for name, value, value_type in declarations:
            if not isinstance(value, value_type):
                raise TypeError(f"{name} must be a {value_type.__name__}")
        object.__setattr__(
            self,
            "discretization_parameters",
            _json_mapping(
                self.discretization_parameters,
                "discretization parameters",
            ),
        )
        object.__setattr__(
            self,
            "compatibility_metadata",
            _json_mapping(
                self.compatibility_metadata,
                "compatibility metadata",
            ),
        )
        self._validate_boundary_composition()
        self._validate_initial_condition()

    def _validate_boundary_composition(self) -> None:
        if self.boundaries.ndim != self.geometry.domain.ndim:
            raise ValueError(
                "boundary assignment and geometry dimensions must agree"
            )
        roles = {
            component: field_spec.role
            for field_spec in self.equation_system.fields
            for component in field_spec.components
        }
        required = {
            component
            for component, role in roles.items()
            if role in {FieldRole.EVOLVED, FieldRole.ALGEBRAIC}
        }
        observed = set(self.boundaries.component_names)
        if observed != required:
            missing = tuple(sorted(required - observed))
            extra = tuple(sorted(observed - required))
            raise ValueError(
                "boundary assignments must cover exactly evolved and "
                f"algebraic components; missing={missing!r}, extra={extra!r}"
            )

        for component in self.boundaries.components:
            role = roles[component.component]
            if (
                role is FieldRole.EVOLVED
                and component.semantic is not BoundarySemantic.PHYSICAL
            ):
                raise ValueError(
                    f"evolved component '{component.component}' requires a "
                    "physical boundary semantic"
                )
            for face in component.faces:
                topology = self.geometry.axis_topologies[face.axis]
                kind = face.condition.kind
                if topology is AxisTopology.PERIODIC:
                    if kind is not BoundaryKind.PERIODIC:
                        raise ValueError(
                            f"periodic axis {face.axis} requires periodic "
                            f"conditions for '{component.component}'"
                        )
                elif kind is BoundaryKind.PERIODIC:
                    raise ValueError(
                        f"bounded axis {face.axis} cannot use a periodic "
                        f"condition for '{component.component}'"
                    )

    def _validate_initial_condition(self) -> None:
        initial = self.initial_condition
        if (
            initial.source is InitialConditionSource.GENERATED
            and initial.family
            not in self.equation_system.initial_condition_families
        ):
            raise ValueError(
                f"generated initial-condition family '{initial.family}' is "
                "not declared by the equation system"
            )

    def scientific_identity_metadata(self) -> dict[str, object]:
        domain = self.geometry.domain
        boundary_metadata = self.boundaries.to_metadata()
        return {
            "equation_system": self.equation_system.to_metadata(),
            "geometry": {
                "name": self.geometry.name,
                "lengths": list(domain.lengths),
                "axis_names": list(domain.axis_names),
                "axis_topologies": [
                    value.value for value in self.geometry.axis_topologies
                ],
            },
            "boundaries": {
                "ndim": boundary_metadata["ndim"],
                "components": boundary_metadata["components"],
            },
        }

    def discretization_identity_metadata(self) -> dict[str, object]:
        return {
            "grid": {
                "shape": list(self.geometry.domain.shape),
                "grid_placement": self.geometry.domain.grid_placement.value,
            },
            "numerics": self.numerics.to_metadata(),
            "time_integration": self.time_integration.to_metadata(),
            "parameters": _mapping_metadata(self.discretization_parameters),
        }

    def execution_identity_metadata(self) -> dict[str, object]:
        return self.execution.to_metadata()

    def run_identity_metadata(self) -> dict[str, object]:
        return {
            "initial_condition": self.initial_condition.to_metadata(),
            "workflow": self.workflow.to_metadata(),
            "invocation": self.invocation.to_metadata(),
        }

    def identity_metadata(self) -> dict[str, object]:
        identities = {
            "scientific": self.scientific_identity_metadata(),
            "discretization": self.discretization_identity_metadata(),
            "execution": self.execution_identity_metadata(),
            "run": self.run_identity_metadata(),
        }
        return {
            name: {"sha256": _sha256(value), "metadata": value}
            for name, value in identities.items()
        }

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "equation_system": self.equation_system.to_metadata(),
            "geometry": self.geometry.to_metadata(),
            "boundaries": self.boundaries.to_metadata(),
            "numerics": self.numerics.to_metadata(),
            "time_integration": self.time_integration.to_metadata(),
            "discretization_parameters": _mapping_metadata(
                self.discretization_parameters
            ),
            "initial_condition": self.initial_condition.to_metadata(),
            "execution": self.execution.to_metadata(),
            "workflow": self.workflow.to_metadata(),
            "invocation": self.invocation.to_metadata(),
            "identities": self.identity_metadata(),
            "compatibility_metadata": _mapping_metadata(
                self.compatibility_metadata
            ),
        }

    def canonical_sha256(self) -> str:
        return _sha256(self.to_metadata())


__all__ = [
    "ExecutionSpec",
    "InitialConditionSource",
    "InitialConditionSpec",
    "InvocationSpec",
    "SimulationSpec",
    "TimeIntegrationSpec",
    "WorkflowSpec",
]
