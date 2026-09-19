"""Tensor-free declarations for coupled algebraic subsystem requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
from types import MappingProxyType


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
                "algebraic outputs cannot depend on themselves; "
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
