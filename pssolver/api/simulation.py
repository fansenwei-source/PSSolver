"""Public, immutable, tensor-free composition of a simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType

from pssolver.boundaries.homogeneous import (
    BoundaryPolicy,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.core._canonical import canonical_sha256
from pssolver.geometries.public import PeriodicBox, PlaneSlab, RectangularChannel

from .declarations import (
    GeneratedInitialCondition,
    Output,
    SemiImplicitEuler,
    SnapshotInitialCondition,
    SpectralNumerics,
    TorchSpectralExecution,
)
from .models import CompleteStressBerisEdwards, ShendrukPlaneBerisEdwards


@dataclass(frozen=True, slots=True)
class Simulation:
    """Compose one complete simulation request without allocating fields."""

    model: CompleteStressBerisEdwards | ShendrukPlaneBerisEdwards
    geometry: PeriodicBox | PlaneSlab | RectangularChannel
    boundaries: Mapping[str, BoundaryPolicy]
    numerics: SpectralNumerics
    time: SemiImplicitEuler
    initial_condition: GeneratedInitialCondition | SnapshotInitialCondition
    execution: TorchSpectralExecution
    output: Output

    def __post_init__(self) -> None:
        fields: dict[str, type | tuple[type, ...]] = {
            "model": (CompleteStressBerisEdwards, ShendrukPlaneBerisEdwards),
            "geometry": (PeriodicBox, PlaneSlab, RectangularChannel),
            "numerics": SpectralNumerics,
            "time": SemiImplicitEuler,
            "initial_condition": (GeneratedInitialCondition, SnapshotInitialCondition),
            "execution": TorchSpectralExecution,
            "output": Output,
        }
        for name, allowed in fields.items():
            if not isinstance(getattr(self, name), allowed):
                raise TypeError(f"unsupported Simulation {name} declaration")
        if not isinstance(self.boundaries, Mapping) or not all(
            isinstance(field, str) and isinstance(value, BoundaryPolicy)
            for field, value in self.boundaries.items()
        ):
            raise TypeError("boundaries must map field names to boundary policies")
        object.__setattr__(self, "boundaries", MappingProxyType(dict(self.boundaries)))

    def to_metadata(self) -> dict[str, object]:
        """Return the JSON-compatible public declaration."""
        return {
            "model": {"kind": type(self.model).__name__, **asdict(self.model)},
            "geometry": self.geometry.to_metadata(),
            "boundaries": {
                field: policy.to_metadata() if hasattr(policy, "to_metadata") else asdict(policy)
                for field, policy in self.boundaries.items()
            },
            "numerics": self.numerics.to_metadata(),
            "time": self.time.to_metadata(),
            "initial_condition": self.initial_condition.to_metadata(),
            "execution": self.execution.to_metadata(),
            "output": self.output.to_metadata(),
        }

    def canonical_sha256(self) -> str:
        """Hash the declaration, including non-Plan execution and output choices."""
        return canonical_sha256(self.to_metadata())

    def identity_metadata(self) -> dict[str, object]:
        """Return declaration metadata and the identity of its assembled Plan."""
        from .runner import compile_simulation

        return {
            "declaration": self.to_metadata(),
            "plan_identity_sha256": compile_simulation(self).plan_identity_sha256,
        }


def _free_slip_plane_declaration(shape, lengths):
    """Return the fixed geometry and wall policies used by the Plane command."""
    return PlaneSlab(shape=shape, lengths=lengths), {
        "Q": neumann_q(),
        "velocity": free_slip_velocity(),
        "pressure": neumann_pressure_compatibility(),
    }


__all__ = ["Simulation"]
