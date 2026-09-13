"""Explicit numerical-policy contracts used during future plan assembly."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Precision(str, Enum):
    """Supported real arithmetic precision."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class DealiasRule(str, Enum):
    """Nonlinear spectral projection rule."""

    NONE = "none"
    TWO_THIRDS = "two_thirds"
    CUBIC_HALF = "cubic_half"


class TransformExecutionOrder(str, Enum):
    """Tensor-product transform execution order."""

    LEGACY = "legacy"
    REAL_FIRST = "real_first"


class ProjectedTransformExecution(str, Enum):
    """Work performed by transforms adjacent to spectral projection."""

    FULL = "full"
    TRUNCATED = "truncated"


class SpectralStorage(str, Enum):
    """Native Fourier coefficient storage."""

    FULL_COMPLEX = "full_complex"
    HERMITIAN_HALF = "hermitian_half"


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    """Explicit, geometry-independent spectral execution choices.

    Stage A deliberately provides no implicit production preset.  A later
    geometry policy will construct qualified Plane, Channel, or reference
    configurations without turning a Plane assumption into a global default.
    """

    precision: Precision
    dealias_rule: DealiasRule
    transform_execution_order: TransformExecutionOrder
    projected_transform_execution: ProjectedTransformExecution
    spectral_storage: SpectralStorage
    hermitian_axis: int | None = None

    def __post_init__(self) -> None:
        enum_fields = (
            ("precision", self.precision, Precision),
            ("dealias_rule", self.dealias_rule, DealiasRule),
            (
                "transform_execution_order",
                self.transform_execution_order,
                TransformExecutionOrder,
            ),
            (
                "projected_transform_execution",
                self.projected_transform_execution,
                ProjectedTransformExecution,
            ),
            ("spectral_storage", self.spectral_storage, SpectralStorage),
        )
        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}")

        if (
            self.projected_transform_execution
            is ProjectedTransformExecution.TRUNCATED
            and self.dealias_rule is DealiasRule.NONE
        ):
            raise ValueError("truncated projected transforms require dealiasing")

        if self.spectral_storage is SpectralStorage.HERMITIAN_HALF:
            if self.transform_execution_order is not TransformExecutionOrder.REAL_FIRST:
                raise ValueError("Hermitian storage requires real-first transforms")
            if (
                not isinstance(self.hermitian_axis, int)
                or isinstance(self.hermitian_axis, bool)
                or self.hermitian_axis < 0
            ):
                raise ValueError(
                    "Hermitian storage requires a non-negative integer axis"
                )
        elif self.hermitian_axis is not None:
            raise ValueError(
                "hermitian_axis must be None for full-complex storage"
            )

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible numerical configuration."""

        return {
            "precision": self.precision.value,
            "dealias_rule": self.dealias_rule.value,
            "transform_execution_order": self.transform_execution_order.value,
            "projected_transform_execution": (
                self.projected_transform_execution.value
            ),
            "spectral_storage": self.spectral_storage.value,
            "hermitian_axis": self.hermitian_axis,
        }
