"""Tensor-free declarations for the qualified two-component modal block."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _finite(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{description} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class TwoComponentModalOperatorSpec:
    """Fixed component order, diffusion, and constant reaction coupling."""

    component_order: tuple[str, str]
    diffusion: tuple[float, float]
    coupling: tuple[tuple[float, float], tuple[float, float]]
    geometry_identity: str = "periodic_1d"
    basis_signature: tuple[str, ...] = ("periodic",)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.component_order, tuple)
            or len(self.component_order) != 2
            or len(set(self.component_order)) != 2
            or any(
                not isinstance(name, str) or not name.isidentifier()
                for name in self.component_order
            )
        ):
            raise ValueError(
                "component_order must contain exactly two unique identifiers"
            )
        if not isinstance(self.diffusion, tuple) or len(self.diffusion) != 2:
            raise ValueError("diffusion must contain exactly two values")
        diffusion = tuple(
            _finite(value, "diffusion coefficient")
            for value in self.diffusion
        )
        if any(value < 0.0 for value in diffusion):
            raise ValueError("diffusion coefficients must be non-negative")
        object.__setattr__(self, "diffusion", diffusion)

        if (
            not isinstance(self.coupling, tuple)
            or len(self.coupling) != 2
            or any(not isinstance(row, tuple) or len(row) != 2 for row in self.coupling)
        ):
            raise ValueError("coupling must have exact shape (2, 2)")
        coupling = tuple(
            tuple(_finite(value, "coupling coefficient") for value in row)
            for row in self.coupling
        )
        object.__setattr__(self, "coupling", coupling)

        if self.geometry_identity != "periodic_1d":
            raise ValueError("Phase 4 supports only geometry_identity='periodic_1d'")
        if self.basis_signature != ("periodic",):
            raise ValueError(
                "Phase 4 supports only basis_signature=('periodic',)"
            )

    @property
    def block_size(self) -> int:
        return 2

    def coefficient_matrix(
        self,
        laplacian_eigenvalue: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the real linear matrix at one Laplacian eigenvalue."""

        eigenvalue = _finite(
            laplacian_eigenvalue,
            "laplacian_eigenvalue",
        )
        return (
            (
                self.coupling[0][0] + self.diffusion[0] * eigenvalue,
                self.coupling[0][1],
            ),
            (
                self.coupling[1][0],
                self.coupling[1][1] + self.diffusion[1] * eigenvalue,
            ),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "identity": "two_component_modal_operator",
            "block_size": self.block_size,
            "component_order": list(self.component_order),
            "diffusion": list(self.diffusion),
            "coupling": [list(row) for row in self.coupling],
            "geometry_identity": self.geometry_identity,
            "basis_signature": list(self.basis_signature),
            "physical_coefficient_policy": "constant",
            "mode_coupling": False,
        }


__all__ = ["TwoComponentModalOperatorSpec"]
