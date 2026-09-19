"""Dependency-neutral shared declarations for the Plane run facade.

The supported import and pickle identity of these declarations predates their
physical extraction from :mod:`plane_beris_edwards`.  The explicit legacy
``__module__`` assignments preserve that nominal contract while the old
module re-exports the exact canonical objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pssolver.adapters.legacy_boundaries import boundary_set_to_legacy
from pssolver.core import (
    BoundaryCondition,
    BoundarySet,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    PeriodicBC,
)


DEFAULT_PLANE_RUNTIME_PATH = "legacy_production"
DEFAULT_ZERO_MODE_POLICY = "zero_mean"
DEFAULT_FRICTION_MODE_FRIC = 0.1
DEFAULT_SPECTRAL_REFRESH_TIME = 0.2
_LEGACY_DECLARATION_MODULE = (
    "pssolver.configuration.plane_beris_edwards"
)


class PlaneRuntimePath(str, Enum):
    """Mutually exclusive Plane Beris--Edwards runtime implementations."""

    LEGACY_PRODUCTION = "legacy_production"
    SEPARATED_CANARY = "separated_canary"


PlaneRuntimePath.__module__ = _LEGACY_DECLARATION_MODULE


def _periodic_periodic(condition: BoundaryCondition) -> BoundarySet:
    return BoundarySet((PeriodicBC(), PeriodicBC(), condition))


@dataclass(frozen=True, slots=True)
class PlaneFreeSlipBoundaryConditions:
    """Physical BC declaration for the free-slip/free-Q Plane model."""

    q: BoundarySet = field(
        default_factory=lambda: _periodic_periodic(HomogeneousNeumannBC())
    )
    tangential_velocity: BoundarySet = field(
        default_factory=lambda: _periodic_periodic(HomogeneousNeumannBC())
    )
    normal_velocity: BoundarySet = field(
        default_factory=lambda: _periodic_periodic(HomogeneousDirichletBC())
    )
    pressure_modal: BoundarySet = field(
        default_factory=lambda: _periodic_periodic(HomogeneousNeumannBC())
    )
    distortion_odd_z: BoundarySet = field(
        default_factory=lambda: _periodic_periodic(HomogeneousDirichletBC())
    )

    def __post_init__(self) -> None:
        values = (
            self.q,
            self.tangential_velocity,
            self.normal_velocity,
            self.pressure_modal,
            self.distortion_odd_z,
        )
        if not all(isinstance(value, BoundarySet) for value in values):
            raise TypeError("Plane boundary entries must be BoundarySet objects")
        if any(value.ndim != 3 for value in values):
            raise ValueError("Plane boundary entries must be three-dimensional")

    def to_legacy(self) -> dict[str, tuple[str, ...]]:
        """Translate the physical declaration at the legacy adapter edge."""

        return {
            "q": boundary_set_to_legacy(self.q),
            "tangential_velocity": boundary_set_to_legacy(
                self.tangential_velocity
            ),
            "normal_velocity": boundary_set_to_legacy(self.normal_velocity),
            "pressure_modal": boundary_set_to_legacy(self.pressure_modal),
            "distortion_odd_z": boundary_set_to_legacy(
                self.distortion_odd_z
            ),
        }

    def to_metadata(self) -> dict[str, object]:
        return {
            "q": self.q.to_metadata(),
            "tangential_velocity": self.tangential_velocity.to_metadata(),
            "normal_velocity": self.normal_velocity.to_metadata(),
            "pressure_modal": self.pressure_modal.to_metadata(),
            "distortion_odd_z": self.distortion_odd_z.to_metadata(),
            "wall_normal_axis": 2,
        }


PlaneFreeSlipBoundaryConditions.__module__ = _LEGACY_DECLARATION_MODULE
PLANE_FREE_SLIP_BOUNDARIES = PlaneFreeSlipBoundaryConditions()


@dataclass(frozen=True, slots=True)
class SpectralRefreshSpec:
    """Resolved periodic dynamic-spectrum refresh policy."""

    mode: str
    requested_interval_time: float | None
    requested_interval_steps: int | None
    effective_interval_steps: int | None
    effective_interval_time: float | None

    def __post_init__(self) -> None:
        if self.mode not in {"physical_time", "steps", "disabled"}:
            raise ValueError("invalid spectral refresh mode")

    def to_metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "requested_interval_time": self.requested_interval_time,
            "requested_interval_steps": self.requested_interval_steps,
            "effective_interval_steps": self.effective_interval_steps,
            "effective_interval_time": self.effective_interval_time,
        }


SpectralRefreshSpec.__module__ = _LEGACY_DECLARATION_MODULE


__all__ = [
    "DEFAULT_FRICTION_MODE_FRIC",
    "DEFAULT_PLANE_RUNTIME_PATH",
    "DEFAULT_SPECTRAL_REFRESH_TIME",
    "DEFAULT_ZERO_MODE_POLICY",
    "PLANE_FREE_SLIP_BOUNDARIES",
    "PlaneFreeSlipBoundaryConditions",
    "PlaneRuntimePath",
    "SpectralRefreshSpec",
]
