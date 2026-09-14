"""Pure Shendruk-inspired Plane parameter resolution.

The preset translates the dimensionless activity number into the coefficients
used by the current quasistatic Beris--Edwards--Stokes model.  It does not
construct fields, transforms, solvers, or runtime tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from pssolver.models.active_nematics.q_tensor import positive_equilibrium_S


@dataclass(frozen=True, slots=True)
class ShendrukPlanePreset:
    """Resolved coefficients for one Shendruk-inspired Plane run."""

    activity_number: float
    height: float
    parameterization: str
    activity_ratio: float
    frank_k: float
    zeta: float
    ldg_a: float
    ldg_b: float
    ldg_c: float
    ldg_l1: float
    rotational_viscosity: float
    equilibrium_s: float
    equilibrium_q_amplitude: float

    @property
    def ldg_a_over_gamma(self) -> float:
        return self.ldg_a / self.rotational_viscosity

    @property
    def ldg_b_over_gamma(self) -> float:
        return self.ldg_b / self.rotational_viscosity

    @property
    def ldg_c_over_gamma(self) -> float:
        return self.ldg_c / self.rotational_viscosity

    @property
    def ldg_l1_over_gamma(self) -> float:
        return self.ldg_l1 / self.rotational_viscosity

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible record of inputs and resolved values."""

        return {
            "activity_number": self.activity_number,
            "height": self.height,
            "parameterization": self.parameterization,
            "activity_ratio_zeta_over_k": self.activity_ratio,
            "frank_k": self.frank_k,
            "zeta": self.zeta,
            "ldg_a": self.ldg_a,
            "ldg_b": self.ldg_b,
            "ldg_c": self.ldg_c,
            "ldg_l1": self.ldg_l1,
            "rotational_viscosity": self.rotational_viscosity,
            "equilibrium_s": self.equilibrium_s,
            "equilibrium_q_amplitude": self.equilibrium_q_amplitude,
            "ldg_a_over_gamma": self.ldg_a_over_gamma,
            "ldg_b_over_gamma": self.ldg_b_over_gamma,
            "ldg_c_over_gamma": self.ldg_c_over_gamma,
            "ldg_l1_over_gamma": self.ldg_l1_over_gamma,
        }


def resolve_shendruk_plane_preset(
    *,
    activity_number: float,
    height: float,
    parameterization: str,
    frank_k: float,
    coefficient_min: float,
    coefficient_max: float,
    ldg_a: float,
    ldg_b: float,
    ldg_c: float,
    gamma: float,
) -> ShendrukPlanePreset:
    """Resolve the current benchmark mapping without runtime side effects."""

    if parameterization not in {"paper-window", "fixed-k"}:
        raise ValueError(
            "parameterization must be 'paper-window' or 'fixed-k'"
        )
    numeric = (
        activity_number,
        height,
        frank_k,
        coefficient_min,
        coefficient_max,
        ldg_a,
        ldg_b,
        ldg_c,
        gamma,
    )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in numeric
    ):
        raise ValueError("Shendruk preset coefficients must be finite numbers")
    if activity_number <= 0.0 or height <= 0.0 or frank_k <= 0.0:
        raise ValueError("activity number, height, and Frank K must be positive")
    if coefficient_min <= 0.0 or coefficient_max <= coefficient_min:
        raise ValueError(
            "coefficient bounds must be positive and strictly increasing"
        )
    if gamma <= 0.0:
        raise ValueError("rotational viscosity gamma must be positive")

    activity_ratio = (activity_number / height) ** 2
    if parameterization == "paper-window":
        ratio_min = coefficient_min / coefficient_max
        ratio_max = coefficient_max / coefficient_min
        if not ratio_min <= activity_ratio <= ratio_max:
            activity_min = height * ratio_min**0.5
            activity_max = height * ratio_max**0.5
            raise ValueError(
                f"A={activity_number} is outside the paper-window interval "
                f"[{activity_min:.6g}, {activity_max:.6g}] for H={height}"
            )
        if activity_ratio <= 1.0:
            resolved_zeta = coefficient_min
            resolved_frank_k = resolved_zeta / activity_ratio
        else:
            resolved_frank_k = coefficient_min
            resolved_zeta = resolved_frank_k * activity_ratio
    else:
        resolved_frank_k = frank_k
        resolved_zeta = resolved_frank_k * activity_ratio

    equilibrium_s = positive_equilibrium_S(ldg_a, ldg_b, ldg_c)
    equilibrium_q_amplitude = 1.5 * equilibrium_s
    ldg_l1 = resolved_frank_k / (2.0 * equilibrium_q_amplitude**2)
    return ShendrukPlanePreset(
        activity_number=float(activity_number),
        height=float(height),
        parameterization=parameterization,
        activity_ratio=activity_ratio,
        frank_k=resolved_frank_k,
        zeta=resolved_zeta,
        ldg_a=float(ldg_a),
        ldg_b=float(ldg_b),
        ldg_c=float(ldg_c),
        ldg_l1=ldg_l1,
        rotational_viscosity=float(gamma),
        equilibrium_s=equilibrium_s,
        equilibrium_q_amplitude=equilibrium_q_amplitude,
    )


__all__ = [
    "ShendrukPlanePreset",
    "resolve_shendruk_plane_preset",
]
