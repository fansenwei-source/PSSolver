"""Exact geometry/capability dispatch for algebraic solver factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pssolver.core import GeometrySpec

from .algebraic import (
    AlgebraicSolverContext,
    AlgebraicSolverProtocol,
    AlgebraicSystemSpec,
)
from .contracts import (
    ExecutableModelProtocol,
    ExplicitRHSExecutorProtocol,
    ModelExecutionContext,
)


AlgebraicSolverFactory = Callable[
    [AlgebraicSolverContext, AlgebraicSystemSpec],
    AlgebraicSolverProtocol,
]
ExplicitRHSExecutorFactory = Callable[
    [ModelExecutionContext, ExecutableModelProtocol],
    ExplicitRHSExecutorProtocol,
]


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{description} must be a Python identifier")
    return value


@dataclass(frozen=True, slots=True)
class AlgebraicSolverRegistration:
    """One exact geometry/capability implementation registration."""

    geometry_type: type[GeometrySpec]
    geometry_name: str
    capability: str
    implementation_name: str
    factory: AlgebraicSolverFactory

    def __init__(
        self,
        geometry_type: type[GeometrySpec],
        geometry_name: str,
        capability: str,
        implementation_name: str,
        factory: AlgebraicSolverFactory,
    ) -> None:
        if not isinstance(geometry_type, type) or not issubclass(
            geometry_type,
            GeometrySpec,
        ):
            raise TypeError("geometry_type must be a GeometrySpec subclass")
        object.__setattr__(self, "geometry_type", geometry_type)
        object.__setattr__(
            self,
            "geometry_name",
            _identifier(geometry_name, "geometry_name"),
        )
        object.__setattr__(
            self,
            "capability",
            _identifier(capability, "capability"),
        )
        object.__setattr__(
            self,
            "implementation_name",
            _identifier(implementation_name, "implementation_name"),
        )
        if not callable(factory):
            raise TypeError("factory must be callable")
        object.__setattr__(self, "factory", factory)

    def to_metadata(self) -> dict[str, str]:
        return {
            "geometry_type": (
                f"{self.geometry_type.__module__}."
                f"{self.geometry_type.__qualname__}"
            ),
            "geometry_name": self.geometry_name,
            "capability": self.capability,
            "implementation_name": self.implementation_name,
        }


@dataclass(frozen=True, slots=True)
class ExplicitRHSExecutorRegistration:
    """One exact geometry/model explicit-RHS implementation."""

    geometry_type: type[GeometrySpec]
    geometry_name: str
    model_type: type
    implementation_name: str
    factory: ExplicitRHSExecutorFactory

    def __init__(
        self,
        geometry_type: type[GeometrySpec],
        geometry_name: str,
        model_type: type,
        implementation_name: str,
        factory: ExplicitRHSExecutorFactory,
    ) -> None:
        if not isinstance(geometry_type, type) or not issubclass(
            geometry_type,
            GeometrySpec,
        ):
            raise TypeError("geometry_type must be a GeometrySpec subclass")
        if not isinstance(model_type, type):
            raise TypeError("model_type must be a type")
        object.__setattr__(self, "geometry_type", geometry_type)
        object.__setattr__(
            self,
            "geometry_name",
            _identifier(geometry_name, "geometry_name"),
        )
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(
            self,
            "implementation_name",
            _identifier(implementation_name, "implementation_name"),
        )
        if not callable(factory):
            raise TypeError("factory must be callable")
        object.__setattr__(self, "factory", factory)

    def to_metadata(self) -> dict[str, str]:
        return {
            "geometry_type": (
                f"{self.geometry_type.__module__}."
                f"{self.geometry_type.__qualname__}"
            ),
            "geometry_name": self.geometry_name,
            "model_type": (
                f"{self.model_type.__module__}.{self.model_type.__qualname__}"
            ),
            "implementation_name": self.implementation_name,
        }


class GeometrySolverRegistry:
    """Mutable construction registry with exact lookup and no fallback."""

    def __init__(self) -> None:
        self._registrations: dict[
            tuple[type[GeometrySpec], str],
            AlgebraicSolverRegistration,
        ] = {}
        self._explicit_rhs_registrations: dict[
            tuple[type[GeometrySpec], type],
            ExplicitRHSExecutorRegistration,
        ] = {}

    def register(
        self,
        *,
        geometry_type: type[GeometrySpec],
        geometry_name: str,
        capability: str,
        implementation_name: str,
        factory: AlgebraicSolverFactory,
    ) -> AlgebraicSolverRegistration:
        registration = AlgebraicSolverRegistration(
            geometry_type,
            geometry_name,
            capability,
            implementation_name,
            factory,
        )
        key = (registration.geometry_type, registration.capability)
        if key in self._registrations:
            raise ValueError(
                "duplicate algebraic solver registration for "
                f"geometry_type={key[0].__name__!r}, "
                f"capability={key[1]!r}"
            )
        self._registrations[key] = registration
        return registration

    def resolve(
        self,
        geometry: GeometrySpec,
        system: AlgebraicSystemSpec,
    ) -> AlgebraicSolverRegistration:
        if not isinstance(geometry, GeometrySpec):
            raise TypeError("geometry must be a GeometrySpec")
        if not isinstance(system, AlgebraicSystemSpec):
            raise TypeError("system must be an AlgebraicSystemSpec")
        key = (type(geometry), system.capability)
        try:
            registration = self._registrations[key]
        except KeyError as exc:
            raise LookupError(
                "no exact algebraic solver registration for "
                f"geometry_type={type(geometry).__name__!r}, "
                f"capability={system.capability!r}; implicit geometry "
                "fallback is disabled"
            ) from exc
        if registration.geometry_name != geometry.name:
            raise ValueError(
                "registered geometry name does not match the geometry "
                f"instance: expected {registration.geometry_name!r}, "
                f"got {geometry.name!r}"
            )
        return registration

    def register_explicit_rhs(
        self,
        *,
        geometry_type: type[GeometrySpec],
        geometry_name: str,
        model_type: type,
        implementation_name: str,
        factory: ExplicitRHSExecutorFactory,
    ) -> ExplicitRHSExecutorRegistration:
        """Register one exact geometry/model RHS executor without fallback."""

        registration = ExplicitRHSExecutorRegistration(
            geometry_type,
            geometry_name,
            model_type,
            implementation_name,
            factory,
        )
        key = (registration.geometry_type, registration.model_type)
        if key in self._explicit_rhs_registrations:
            raise ValueError(
                "duplicate explicit RHS executor registration for "
                f"geometry_type={key[0].__name__!r}, "
                f"model_type={key[1].__name__!r}"
            )
        self._explicit_rhs_registrations[key] = registration
        return registration

    def find_explicit_rhs(
        self,
        geometry: GeometrySpec,
        model: ExecutableModelProtocol,
    ) -> ExplicitRHSExecutorRegistration | None:
        """Return an exact optional RHS registration.

        Absence means that the generic adapter must call the physical model's
        ``explicit_rhs`` method.  Registrations for base geometry or model
        classes are deliberately not inherited.
        """

        if not isinstance(geometry, GeometrySpec):
            raise TypeError("geometry must be a GeometrySpec")
        if not isinstance(model, ExecutableModelProtocol):
            raise TypeError("model must implement ExecutableModelProtocol")
        registration = self._explicit_rhs_registrations.get(
            (type(geometry), type(model))
        )
        if registration is None:
            return None
        if registration.geometry_name != geometry.name:
            raise ValueError(
                "registered geometry name does not match the geometry "
                f"instance: expected {registration.geometry_name!r}, "
                f"got {geometry.name!r}"
            )
        return registration

    def to_metadata(self) -> list[dict[str, str]]:
        return [
            self._registrations[key].to_metadata()
            for key in sorted(
                self._registrations,
                key=lambda item: (
                    item[0].__module__,
                    item[0].__qualname__,
                    item[1],
                ),
            )
        ]

    def explicit_rhs_metadata(self) -> list[dict[str, str]]:
        """Return deterministic metadata for optional RHS registrations."""

        return [
            self._explicit_rhs_registrations[key].to_metadata()
            for key in sorted(
                self._explicit_rhs_registrations,
                key=lambda item: (
                    item[0].__module__,
                    item[0].__qualname__,
                    item[1].__module__,
                    item[1].__qualname__,
                ),
            )
        ]
