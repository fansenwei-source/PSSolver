"""Algebraic-field lifecycle and solver contracts.

Algebraic fields are instantaneous constraints of the current evolved state.
Stage F supports one explicit lifecycle: solve every algebraic system after the
evolved state is available and immediately before evaluating the explicit RHS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from .contracts import ExecutableModelProtocol, ModelExecutionContext


class AlgebraicUpdatePhase(str, Enum):
    """Point in a timestep at which an algebraic system is synchronized."""

    PRE_EXPLICIT_RHS = "pre_explicit_rhs"


@dataclass(frozen=True, slots=True)
class AlgebraicSystemSpec:
    """Declarative request for one coupled algebraic solve."""

    name: str
    capability: str
    output_components: tuple[str, ...]
    dependencies: tuple[str, ...]
    parameters: Mapping[str, object] = field(default_factory=dict)
    update_phase: AlgebraicUpdatePhase = (
        AlgebraicUpdatePhase.PRE_EXPLICIT_RHS
    )
    _parameters_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for value, description in (
            (self.name, "algebraic system name"),
            (self.capability, "algebraic capability"),
        ):
            if not isinstance(value, str) or not value.isidentifier():
                raise ValueError(f"{description} must be a Python identifier")
        if isinstance(self.output_components, str) or isinstance(
            self.dependencies,
            str,
        ):
            raise TypeError(
                "output_components and dependencies must be iterables, "
                "not strings"
            )
        try:
            outputs = tuple(self.output_components)
            dependencies = tuple(self.dependencies)
        except TypeError as exc:
            raise TypeError(
                "output_components and dependencies must be iterable"
            ) from exc
        if not outputs or any(
            not isinstance(name, str) or not name.isidentifier()
            for name in outputs
        ):
            raise ValueError(
                "output_components must contain Python identifiers"
            )
        if not dependencies or any(
            not isinstance(name, str) or not name.isidentifier()
            for name in dependencies
        ):
            raise ValueError("dependencies must contain Python identifiers")
        if len(set(outputs)) != len(outputs):
            raise ValueError("algebraic output components must be unique")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("algebraic dependencies must be unique")
        overlap = tuple(sorted(set(outputs) & set(dependencies)))
        if overlap:
            raise ValueError(
                "algebraic outputs cannot depend on themselves in Stage F; "
                f"overlap={overlap!r}"
            )
        if not isinstance(self.update_phase, AlgebraicUpdatePhase):
            raise TypeError("update_phase must be an AlgebraicUpdatePhase")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("algebraic parameters must be a mapping")
        if any(
            not isinstance(name, str) or not name.isidentifier()
            for name in self.parameters
        ):
            raise ValueError(
                "algebraic parameter names must be Python identifiers"
            )
        try:
            parameters_json = json.dumps(
                dict(self.parameters),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "algebraic parameters must be finite and JSON-compatible"
            ) from exc
        normalized_parameters = json.loads(parameters_json)
        object.__setattr__(self, "output_components", outputs)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(normalized_parameters),
        )
        object.__setattr__(self, "_parameters_json", parameters_json)

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "capability": self.capability,
            "output_components": list(self.output_components),
            "dependencies": list(self.dependencies),
            "parameters": json.loads(self._parameters_json),
            "update_phase": self.update_phase.value,
        }


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
