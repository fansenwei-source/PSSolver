"""Algebraic-field lifecycle, dependency-plan, and solver contracts.

Algebraic outputs are instantaneous functions of the current evolved state and
of earlier outputs in a frozen acyclic graph.  They are synchronized
immediately before evaluating the explicit RHS; individual field declarations
decide whether an output is stored or transient.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from pssolver.systems.algebraic import (
    AlgebraicSystemSpec,
    AlgebraicUpdatePhase,
)

from .contracts import (
    ExecutableModelProtocol,
    ModelExecutionContext,
)


@dataclass(frozen=True, slots=True)
class AlgebraicDependencyEdge:
    """One component-labelled dependency between algebraic systems."""

    producer: str
    consumer: str
    component: str

    def __post_init__(self) -> None:
        for value, description in (
            (self.producer, "producer"),
            (self.consumer, "consumer"),
            (self.component, "component"),
        ):
            if not isinstance(value, str) or not value.isidentifier():
                raise ValueError(f"{description} must be a Python identifier")
        if self.producer == self.consumer:
            raise ValueError("an algebraic dependency edge cannot be a self-edge")

    def to_metadata(self) -> dict[str, str]:
        return {
            "producer": self.producer,
            "consumer": self.consumer,
            "component": self.component,
        }


@dataclass(frozen=True, slots=True)
class AlgebraicExecutionPlan:
    """Frozen deterministic DAG for one pre-RHS algebraic evaluation."""

    declared_systems: tuple[AlgebraicSystemSpec, ...]
    execution_order: tuple[str, ...]
    dependency_edges: tuple[AlgebraicDependencyEdge, ...]
    initial_components: tuple[str, ...]
    output_components: tuple[str, ...]
    transient_components: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(
            isinstance(value, str)
            for value in (
                self.declared_systems,
                self.execution_order,
                self.dependency_edges,
                self.initial_components,
                self.output_components,
                self.transient_components,
            )
        ):
            raise TypeError("algebraic execution-plan collections cannot be strings")
        systems = tuple(self.declared_systems)
        order = tuple(self.execution_order)
        edges = tuple(self.dependency_edges)
        initial = tuple(self.initial_components)
        outputs = tuple(self.output_components)
        transients = tuple(self.transient_components)
        if not systems or not all(
            isinstance(system, AlgebraicSystemSpec) for system in systems
        ):
            raise TypeError(
                "declared_systems must contain AlgebraicSystemSpec objects"
            )
        names = tuple(system.name for system in systems)
        if len(set(names)) != len(names):
            raise ValueError("algebraic system names must be unique")
        if set(order) != set(names) or len(order) != len(names):
            raise ValueError("execution_order must contain every system once")
        if not all(isinstance(edge, AlgebraicDependencyEdge) for edge in edges):
            raise TypeError("dependency_edges have an invalid type")
        for values, description in (
            (initial, "initial_components"),
            (outputs, "output_components"),
            (transients, "transient_components"),
        ):
            if len(set(values)) != len(values) or any(
                not isinstance(name, str) or not name.isidentifier()
                for name in values
            ):
                raise ValueError(f"{description} must contain unique identifiers")
        if set(initial) & set(outputs):
            raise ValueError("initial and algebraic output components must differ")
        if not set(transients) <= set(outputs):
            raise ValueError("transient components must be algebraic outputs")
        owners: dict[str, str] = {}
        for system in systems:
            for component in system.output_components:
                if component in owners:
                    raise ValueError(
                        f"algebraic component {component!r} has multiple owners"
                    )
                owners[component] = system.name
        if set(owners) != set(outputs):
            raise ValueError(
                "execution-plan outputs do not match system ownership"
            )
        expected_edges = {
            (owners[component], system.name, component)
            for system in systems
            for component in system.dependencies
            if component in owners
        }
        if any(
            component not in set(initial) | set(owners)
            for system in systems
            for component in system.dependencies
        ):
            raise ValueError("execution-plan dependency has no source")
        actual_edges = {
            (edge.producer, edge.consumer, edge.component) for edge in edges
        }
        if actual_edges != expected_edges or len(edges) != len(expected_edges):
            raise ValueError("dependency_edges do not match system dependencies")
        position = {name: index for index, name in enumerate(order)}
        if any(
            position[producer] >= position[consumer]
            for producer, consumer, _ in expected_edges
        ):
            raise ValueError("execution_order is not topological")
        object.__setattr__(self, "declared_systems", systems)
        object.__setattr__(self, "execution_order", order)
        object.__setattr__(self, "dependency_edges", edges)
        object.__setattr__(self, "initial_components", initial)
        object.__setattr__(self, "output_components", outputs)
        object.__setattr__(self, "transient_components", transients)

    @property
    def ordered_systems(self) -> tuple[AlgebraicSystemSpec, ...]:
        by_name = {system.name: system for system in self.declared_systems}
        return tuple(by_name[name] for name in self.execution_order)

    def to_metadata(self) -> dict[str, object]:
        return {
            "declaration_order": [
                system.name for system in self.declared_systems
            ],
            "execution_order": list(self.execution_order),
            "dependency_edges": [
                edge.to_metadata() for edge in self.dependency_edges
            ],
            "initial_components": list(self.initial_components),
            "output_components": list(self.output_components),
            "transient_components": list(self.transient_components),
        }


def build_algebraic_execution_plan(
    systems: tuple[AlgebraicSystemSpec, ...],
    *,
    initial_components: tuple[str, ...],
    output_components: tuple[str, ...],
    transient_components: tuple[str, ...] = (),
) -> AlgebraicExecutionPlan:
    """Validate and topologically freeze an algebraic dependency graph."""

    if any(
        isinstance(value, str)
        for value in (
            systems,
            initial_components,
            output_components,
            transient_components,
        )
    ):
        raise TypeError("algebraic plan inputs cannot be strings")
    try:
        systems = tuple(systems)
        initial_components = tuple(initial_components)
        output_components = tuple(output_components)
        transient_components = tuple(transient_components)
    except TypeError as exc:
        raise TypeError("algebraic plan inputs must be iterable") from exc
    if not systems or not all(
        isinstance(system, AlgebraicSystemSpec) for system in systems
    ):
        raise TypeError("systems must contain AlgebraicSystemSpec objects")
    names = tuple(system.name for system in systems)
    if len(set(names)) != len(names):
        raise ValueError("algebraic system names must be unique")
    for values, description in (
        (initial_components, "initial_components"),
        (output_components, "output_components"),
        (transient_components, "transient_components"),
    ):
        if len(set(values)) != len(values) or any(
            not isinstance(name, str) or not name.isidentifier()
            for name in values
        ):
            raise ValueError(f"{description} must contain unique identifiers")
    if set(initial_components) & set(output_components):
        raise ValueError("initial and algebraic output components must differ")
    if not set(transient_components) <= set(output_components):
        raise ValueError("transient components must be algebraic outputs")

    owners: dict[str, str] = {}
    for system in systems:
        for component in system.output_components:
            if component in owners:
                raise ValueError(
                    f"algebraic component {component!r} has multiple owners"
                )
            owners[component] = system.name
    if set(owners) != set(output_components):
        missing = tuple(sorted(set(output_components) - set(owners)))
        unexpected = tuple(sorted(set(owners) - set(output_components)))
        raise ValueError(
            "algebraic system outputs do not match declared algebraic and "
            f"transient fields; missing={missing!r}, unexpected={unexpected!r}"
        )

    index = {name: position for position, name in enumerate(names)}
    prerequisites = {name: set() for name in names}
    consumers = {name: set() for name in names}
    edges = []
    available_initial = set(initial_components)
    for system in systems:
        for component in system.dependencies:
            if component in available_initial:
                continue
            try:
                producer = owners[component]
            except KeyError as exc:
                raise ValueError(
                    "algebraic dependencies must be evolved or algebraic "
                    f"outputs; {component!r} required by {system.name!r} "
                    "has no producer"
                ) from exc
            if producer == system.name:
                raise ValueError(
                    f"algebraic system {system.name!r} depends on its own "
                    f"output {component!r}"
                )
            prerequisites[system.name].add(producer)
            consumers[producer].add(system.name)
            edges.append(
                AlgebraicDependencyEdge(producer, system.name, component)
            )

    ready = [name for name in names if not prerequisites[name]]
    order = []
    while ready:
        ready.sort(key=index.__getitem__)
        current = ready.pop(0)
        order.append(current)
        for consumer in sorted(consumers[current], key=index.__getitem__):
            prerequisites[consumer].remove(current)
            if not prerequisites[consumer]:
                ready.append(consumer)
    if len(order) != len(names):
        cyclic = tuple(
            name for name in names if prerequisites[name]
        )
        raise ValueError(
            "algebraic dependency graph contains a cycle; "
            f"systems={cyclic!r}"
        )
    edges.sort(
        key=lambda edge: (
            index[edge.consumer],
            index[edge.producer],
            edge.component,
        )
    )
    return AlgebraicExecutionPlan(
        declared_systems=systems,
        execution_order=tuple(order),
        dependency_edges=tuple(edges),
        initial_components=initial_components,
        output_components=output_components,
        transient_components=transient_components,
    )


@runtime_checkable
class AlgebraicExecutableModelProtocol(ExecutableModelProtocol, Protocol):
    """Executable model that declares instantaneous algebraic systems."""

    def algebraic_system_specs(self) -> tuple[AlgebraicSystemSpec, ...]:
        """Return all algebraic systems in deterministic declaration order."""

        ...


@runtime_checkable
class AlgebraicSolverContext(ModelExecutionContext, Protocol):
    """Numerical operations available to a geometry-specific solver factory.

    This context is never passed to the physical model.
    """

    @property
    def geometry_name(self) -> str:
        """Stable geometry identifier used for exact dispatch."""

        ...

    @property
    def spectral_dtype(self) -> torch.dtype:
        """Native spectral tensor dtype."""

        ...

    def forward_projected(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Transform and project a physical tensor for one component."""

        ...

    def inverse_projected(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Invert native spectral coefficients for one component."""

        ...

    def boundary_conditions(self, component_name: str) -> tuple[str, ...]:
        """Return the resolved numerical boundary signature."""

        ...


@runtime_checkable
class AlgebraicSolverProtocol(Protocol):
    """Resolved geometry-specific algebraic solver."""

    @property
    def capability(self) -> str:
        ...

    @property
    def implementation_name(self) -> str:
        ...

    @property
    def output_components(self) -> tuple[str, ...]:
        ...

    def solve_spectral(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        """Return native spectral values for all requested outputs."""

        ...


@runtime_checkable
class AlgebraicPhysicalDependenciesProtocol(Protocol):
    """Optional physical-input declaration for a spectral algebraic solve.

    A representation-aware solver may consume some dependencies directly in
    spectral space.  Declaring the remaining physical inputs lets an adapter
    form a bounded physical computation island without exposing transform
    choices to the model specification.
    """

    @property
    def physical_dependencies(self) -> tuple[str, ...]:
        """Return the solver dependencies that must exist in physical space."""

        ...


@runtime_checkable
class InspectableAlgebraicSolverProtocol(Protocol):
    """Optional diagnostics and warm-start state for algebraic solvers.

    Diagnostics are observations of a solve, not stored PDE fields.  Restart
    state is likewise implementation state and must never be confused with
    the evolved or algebraic physical state declared by a model.
    """

    def observability_metadata(self) -> Mapping[str, object]:
        """Return static, JSON-compatible observability capabilities."""

        ...

    def diagnostic_snapshot(self) -> Mapping[str, object]:
        """Return diagnostics from the most recent algebraic solve."""

        ...

    def capture_restart_state(self) -> Mapping[str, torch.Tensor]:
        """Return detached implementation state needed for a warm restart."""

        ...

    def restore_restart_state(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        """Restore a previously captured warm-start state."""

        ...


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class AlgebraicSystemRestartState:
    """One solver's restart state bound to its dispatch identity."""

    system_name: str
    capability: str
    implementation_name: str
    provenance_sha256: str
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        for value, description in (
            (self.system_name, "system_name"),
            (self.capability, "capability"),
            (self.implementation_name, "implementation_name"),
        ):
            if not isinstance(value, str) or not value.isidentifier():
                raise ValueError(f"{description} must be a Python identifier")
        if (
            not isinstance(self.provenance_sha256, str)
            or len(self.provenance_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.provenance_sha256
            )
        ):
            raise ValueError("provenance_sha256 must be a lowercase SHA-256")
        if not isinstance(self.tensors, Mapping):
            raise TypeError("restart tensors must be a mapping")
        tensors: dict[str, torch.Tensor] = {}
        for name, value in self.tensors.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    "restart tensor names must be Python identifiers"
                )
            if not isinstance(value, torch.Tensor):
                raise TypeError("restart state values must be tensors")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("restart state tensors must be finite")
            tensors[name] = value.detach().clone()
        object.__setattr__(self, "tensors", MappingProxyType(tensors))

    def to_metadata(self) -> dict[str, object]:
        return {
            "system_name": self.system_name,
            "capability": self.capability,
            "implementation_name": self.implementation_name,
            "provenance_sha256": self.provenance_sha256,
            "tensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device_at_capture": str(value.device),
                    "sha256": _tensor_sha256(value),
                }
                for name, value in sorted(self.tensors.items())
            },
        }


@dataclass(frozen=True, slots=True)
class AlgebraicRuntimeRestartState:
    """Auditable warm-start state for all resolved algebraic systems."""

    format_version: int
    systems: tuple[AlgebraicSystemRestartState, ...]

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("unsupported algebraic restart format version")
        try:
            systems = tuple(self.systems)
        except TypeError as exc:
            raise TypeError("restart systems must be iterable") from exc
        if not all(
            isinstance(system, AlgebraicSystemRestartState)
            for system in systems
        ):
            raise TypeError(
                "restart systems must contain AlgebraicSystemRestartState"
            )
        names = tuple(system.system_name for system in systems)
        if len(set(names)) != len(names):
            raise ValueError("restart system names must be unique")
        object.__setattr__(self, "systems", systems)

    def to_metadata(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "systems": [system.to_metadata() for system in self.systems],
        }
