"""Immutable discovery records for the public declaration surface.

Declaration presence and qualified execution are intentionally separate.
The catalog reports both without importing a runtime or probing hardware.
"""

from __future__ import annotations

from dataclasses import dataclass


_VALID_DECLARATION_KINDS = frozenset(
    {"model", "geometry", "boundary_policy"}
)


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{description} must be a Python identifier")
    return value


@dataclass(frozen=True, slots=True)
class DeclarationCapability:
    """One public constructor and its already-qualified applications."""

    kind: str
    key: str
    constructor: str
    qualified_applications: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _VALID_DECLARATION_KINDS:
            raise ValueError("unsupported declaration-capability kind")
        _identifier(self.key, "capability key")
        if not isinstance(self.constructor, str) or "." not in self.constructor:
            raise ValueError("constructor must be a fully qualified import path")
        applications = tuple(self.qualified_applications)
        for application in applications:
            _identifier(application, "qualified application")
        if len(set(applications)) != len(applications):
            raise ValueError("qualified applications must be unique")
        object.__setattr__(self, "qualified_applications", applications)

    @property
    def executable(self) -> bool:
        """Whether this declaration participates in a qualified application."""

        return bool(self.qualified_applications)

    def to_metadata(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "key": self.key,
            "constructor": self.constructor,
            "declarable": True,
            "executable": self.executable,
            "qualified_applications": list(self.qualified_applications),
        }


@dataclass(frozen=True, slots=True)
class QualifiedCombination:
    """One model/geometry pair with a qualified compiler adapter."""

    equation_variant: str
    geometry_name: str
    application: str
    adapter: str
    runtime_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.equation_variant, "equation variant")
        _identifier(self.geometry_name, "geometry name")
        _identifier(self.application, "application")
        if not isinstance(self.adapter, str) or "." not in self.adapter:
            raise ValueError("adapter must be a fully qualified import path")
        runtime_paths = tuple(self.runtime_paths)
        if not runtime_paths:
            raise ValueError("a qualified combination requires a runtime path")
        for runtime_path in runtime_paths:
            _identifier(runtime_path, "runtime path")
        if len(set(runtime_paths)) != len(runtime_paths):
            raise ValueError("runtime paths must be unique")
        object.__setattr__(self, "runtime_paths", runtime_paths)

    def to_metadata(self) -> dict[str, object]:
        return {
            "equation_variant": self.equation_variant,
            "geometry_name": self.geometry_name,
            "application": self.application,
            "adapter": self.adapter,
            "runtime_paths": list(self.runtime_paths),
            "qualified": True,
        }


@dataclass(frozen=True, slots=True)
class PublicCapabilityCatalog:
    """Complete immutable P8.1 view of public and executable capabilities."""

    models: tuple[DeclarationCapability, ...]
    geometries: tuple[DeclarationCapability, ...]
    boundary_policies: tuple[DeclarationCapability, ...]
    qualified_combinations: tuple[QualifiedCombination, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "models": [item.to_metadata() for item in self.models],
            "geometries": [item.to_metadata() for item in self.geometries],
            "boundary_policies": [
                item.to_metadata() for item in self.boundary_policies
            ],
            "qualified_combinations": [
                item.to_metadata() for item in self.qualified_combinations
            ],
        }


_PLANE_APPLICATION = "plane_complete_stress_beris_edwards"
_CHANNEL_APPLICATION = "channel_legacy_active_force_active_nematics"
_PERIODIC_APPLICATION = "periodic_complete_stress_beris_edwards"
_CHANNEL_COMPLETE_APPLICATION = "channel_complete_stress_beris_edwards"

_MODELS = (
    DeclarationCapability(
        kind="model",
        key="complete_stress_beris_edwards",
        constructor=(
            "pssolver.models.active_nematics.CompleteStressBerisEdwards"
        ),
        qualified_applications=(
            _PLANE_APPLICATION,
            _PERIODIC_APPLICATION,
            _CHANNEL_COMPLETE_APPLICATION,
        ),
    ),
    DeclarationCapability(
        kind="model",
        key="legacy_active_force_active_nematics",
        constructor=(
            "pssolver.models.active_nematics."
            "LegacyActiveForceActiveNematics"
        ),
        qualified_applications=(_CHANNEL_APPLICATION,),
    ),
)

_GEOMETRIES = (
    DeclarationCapability(
        kind="geometry",
        key="periodic_box",
        constructor="pssolver.geometries.PeriodicBox",
        qualified_applications=(_PERIODIC_APPLICATION,),
    ),
    DeclarationCapability(
        kind="geometry",
        key="plane_slab",
        constructor="pssolver.geometries.PlaneSlab",
        qualified_applications=(_PLANE_APPLICATION,),
    ),
    DeclarationCapability(
        kind="geometry",
        key="rectangular_channel",
        constructor="pssolver.geometries.RectangularChannel",
        qualified_applications=(
            _CHANNEL_APPLICATION,
            _CHANNEL_COMPLETE_APPLICATION,
        ),
    ),
)

_BOUNDARY_POLICIES = (
    DeclarationCapability(
        kind="boundary_policy",
        key="neumann_q",
        constructor="pssolver.boundaries.neumann_q",
        qualified_applications=(
            _PLANE_APPLICATION,
            _CHANNEL_APPLICATION,
            _PERIODIC_APPLICATION,
            _CHANNEL_COMPLETE_APPLICATION,
        ),
    ),
    DeclarationCapability(
        kind="boundary_policy",
        key="free_slip_velocity",
        constructor="pssolver.boundaries.free_slip_velocity",
        qualified_applications=(_PLANE_APPLICATION, _PERIODIC_APPLICATION),
    ),
    DeclarationCapability(
        kind="boundary_policy",
        key="no_slip_velocity",
        constructor="pssolver.boundaries.no_slip_velocity",
        qualified_applications=(
            _CHANNEL_APPLICATION,
            _CHANNEL_COMPLETE_APPLICATION,
        ),
    ),
    DeclarationCapability(
        kind="boundary_policy",
        key="neumann_pressure_compatibility",
        constructor="pssolver.boundaries.neumann_pressure_compatibility",
        qualified_applications=(
            _PLANE_APPLICATION,
            _CHANNEL_APPLICATION,
            _PERIODIC_APPLICATION,
            _CHANNEL_COMPLETE_APPLICATION,
        ),
    ),
)

_RUNTIME_PATHS = {
    _PLANE_APPLICATION: ("legacy_production", "compiled_v2"),
    _CHANNEL_APPLICATION: ("legacy_channel", "compiled_channel_v2"),
    _PERIODIC_APPLICATION: ("periodic_spectral",),
    _CHANNEL_COMPLETE_APPLICATION: ("channel_complete_stress",),
}


def available_models() -> tuple[DeclarationCapability, ...]:
    """Return every ergonomic public model declaration."""

    return _MODELS


def available_geometries() -> tuple[DeclarationCapability, ...]:
    """Return every ergonomic public geometry declaration."""

    return _GEOMETRIES


def available_boundary_policies() -> tuple[DeclarationCapability, ...]:
    """Return public field-level boundary declarations.

    Periodic continuation is geometry-induced and is therefore not listed as
    a wall policy.
    """

    return _BOUNDARY_POLICIES


def available_combinations() -> tuple[QualifiedCombination, ...]:
    """Return exactly the model/geometry pairs registered for execution."""

    from pssolver.configuration.public_simulation_runner import (
        public_compiler_capabilities,
    )

    combinations = []
    for registration in public_compiler_capabilities():
        application = registration["application"]
        try:
            runtime_paths = _RUNTIME_PATHS[application]
        except KeyError as exc:
            raise RuntimeError(
                "a public compiler registration lacks catalog runtime paths"
            ) from exc
        combinations.append(
            QualifiedCombination(
                equation_variant=registration["equation_variant"],
                geometry_name=registration["geometry_name"],
                application=application,
                adapter=registration["adapter"],
                runtime_paths=runtime_paths,
            )
        )
    return tuple(combinations)


def capability_catalog() -> PublicCapabilityCatalog:
    """Return the immutable declaration and qualification catalog."""

    return PublicCapabilityCatalog(
        models=available_models(),
        geometries=available_geometries(),
        boundary_policies=available_boundary_policies(),
        qualified_combinations=available_combinations(),
    )


__all__ = [
    "DeclarationCapability",
    "PublicCapabilityCatalog",
    "QualifiedCombination",
    "available_boundary_policies",
    "available_combinations",
    "available_geometries",
    "available_models",
    "capability_catalog",
]
