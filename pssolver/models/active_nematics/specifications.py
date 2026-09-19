"""Tensor-free active-nematic model requests.

The immutable values in this module describe model-owned inputs only.  They
do not select a geometry, resolve a benchmark preset, allocate arrays, or
choose a numerical backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real


def _finite_real(value: object, description: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{description} must be a finite real number")
    return float(value)


def _positive_real(value: object, description: str) -> float:
    normalized = _finite_real(value, description)
    if normalized <= 0.0:
        raise ValueError(f"{description} must be positive")
    return normalized


def _integer(value: object, description: str, *, positive: bool) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        qualifier = "a positive integer" if positive else "an integer"
        raise ValueError(f"{description} must be {qualifier}")
    normalized = int(value)
    if positive and normalized <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return normalized


@dataclass(frozen=True, slots=True)
class BerisEdwardsMaterialRequest:
    """Geometry-neutral material inputs for Beris--Edwards dynamics."""

    ldg_a: float
    ldg_b: float
    ldg_c: float
    gamma: float
    flow_alignment: float
    beta: float

    def __post_init__(self) -> None:
        for name in (
            "ldg_a",
            "ldg_b",
            "ldg_c",
            "flow_alignment",
            "beta",
        ):
            object.__setattr__(
                self,
                name,
                _finite_real(getattr(self, name), name),
            )
        object.__setattr__(self, "gamma", _positive_real(self.gamma, "gamma"))

    def to_metadata(self) -> dict[str, float]:
        """Return the request as a fresh JSON-compatible mapping."""

        return {
            "ldg_a": self.ldg_a,
            "ldg_b": self.ldg_b,
            "ldg_c": self.ldg_c,
            "gamma": self.gamma,
            "flow_alignment": self.flow_alignment,
            "beta": self.beta,
        }


@dataclass(frozen=True, slots=True)
class ExtrudedDefectGasInitialConditionSpec:
    """Model-owned request for the extruded periodic defect-gas initial Q."""

    seed: int
    num_defect_pairs: int
    defect_min_separation: float
    defect_core_radius: float
    background_angle: float
    twist_amplitude: float
    twist_modes: tuple[int, ...]
    initial_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _integer(self.seed, "seed", positive=False))
        object.__setattr__(
            self,
            "num_defect_pairs",
            _integer(
                self.num_defect_pairs,
                "num_defect_pairs",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "defect_min_separation",
            _positive_real(
                self.defect_min_separation,
                "defect_min_separation",
            ),
        )
        object.__setattr__(
            self,
            "defect_core_radius",
            _positive_real(self.defect_core_radius, "defect_core_radius"),
        )
        object.__setattr__(
            self,
            "background_angle",
            _finite_real(self.background_angle, "background_angle"),
        )
        twist_amplitude = _finite_real(
            self.twist_amplitude,
            "twist_amplitude",
        )
        if twist_amplitude < 0.0:
            raise ValueError("twist_amplitude must be non-negative")
        object.__setattr__(self, "twist_amplitude", twist_amplitude)

        if isinstance(self.twist_modes, (str, bytes)):
            raise ValueError("twist_modes must be an iterable of integers")
        try:
            requested_modes = tuple(self.twist_modes)
        except TypeError as exc:
            raise ValueError(
                "twist_modes must be an iterable of integers"
            ) from exc
        if not requested_modes:
            raise ValueError("twist_modes must not be empty")
        modes = tuple(
            _integer(mode, "twist mode", positive=True)
            for mode in requested_modes
        )
        if len(set(modes)) != len(modes):
            raise ValueError("twist_modes must be unique")
        object.__setattr__(self, "twist_modes", modes)
        object.__setattr__(
            self,
            "initial_s",
            _positive_real(self.initial_s, "initial_s"),
        )

    def to_metadata(self) -> dict[str, object]:
        """Return the request as a fresh JSON-compatible mapping."""

        return {
            "seed": self.seed,
            "num_defect_pairs": self.num_defect_pairs,
            "defect_min_separation": self.defect_min_separation,
            "defect_core_radius": self.defect_core_radius,
            "background_angle": self.background_angle,
            "twist_amplitude": self.twist_amplitude,
            "twist_modes": list(self.twist_modes),
            "initial_s": self.initial_s,
        }


__all__ = [
    "BerisEdwardsMaterialRequest",
    "ExtrudedDefectGasInitialConditionSpec",
]
