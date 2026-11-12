"""Tensor-free model requests for the unified public simulation interface."""

import math
from dataclasses import dataclass
from numbers import Real


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    normalized = float(value)
    if positive and normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True, slots=True, kw_only=True)
class CompleteStressBerisEdwards:
    """Declare complete-stress Beris–Edwards material and Stokes coefficients."""

    ldg_a: float
    ldg_b: float
    ldg_c: float
    ldg_l1: float
    gamma: float
    flow_alignment: float
    activity: float
    beta: float
    viscosity: float
    friction: float = 0.0
    tangential_zero_mode_policy: str = "zero_mean"

    def __post_init__(self) -> None:
        policy = self.tangential_zero_mode_policy
        if policy not in ("zero_mean", "friction"):
            raise ValueError("tangential_zero_mode_policy must be 'zero_mean' or 'friction'")
        viscosity = _finite(self.viscosity, "viscosity", positive=True)
        friction = _finite(self.friction, "friction")
        if friction < 0:
            raise ValueError("friction must be non-negative")
        if policy == "zero_mean" and friction != 0:
            raise ValueError("zero_mean tangential policy requires friction == 0")
        if policy == "friction" and friction <= 0:
            raise ValueError("friction tangential policy requires friction > 0")
        object.__setattr__(self, "viscosity", viscosity)
        object.__setattr__(self, "friction", friction)
        for name in ("ldg_a", "ldg_b", "ldg_c", "flow_alignment", "beta"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in ("gamma", "ldg_l1"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))
        object.__setattr__(self, "activity", _finite(self.activity, "activity_amplitude"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ShendrukPlaneBerisEdwards:
    """Declare the Plane benchmark's activity-number parameterization."""

    activity_number: float
    parameterization: str = "paper-window"
    frank_k: float = 0.01
    coefficient_min: float = 0.01
    coefficient_max: float = 0.05
    ldg_a: float = 0.0
    ldg_b: float = -0.3
    ldg_c: float = 0.3
    gamma: float = 2.94
    flow_alignment: float = 0.3
    eta: float = 2.0 / 3.0
    beta: float = -1.0
    zero_mode_policy: str = "zero_mean"
    friction_mode_fric: float = 0.1

    def __post_init__(self) -> None:
        if self.parameterization not in {"paper-window", "fixed-k"}:
            raise ValueError("parameterization must be 'paper-window' or 'fixed-k'")
        if self.zero_mode_policy not in {"zero_mean", "friction"}:
            raise ValueError("zero_mode_policy must be 'zero_mean' or 'friction'")
        for name in ("ldg_a", "ldg_b", "ldg_c", "flow_alignment", "beta"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in (
            "activity_number",
            "frank_k",
            "coefficient_min",
            "coefficient_max",
            "gamma",
            "eta",
            "friction_mode_fric",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))
        if self.coefficient_max <= self.coefficient_min:
            raise ValueError("coefficient bounds must be positive and strictly increasing")


__all__ = ["CompleteStressBerisEdwards", "ShendrukPlaneBerisEdwards"]
