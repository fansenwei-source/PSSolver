"""Yingyou Ma's principal-plane and ideal-initial-loop conventions.

The principal frame is obtained from the orientational second moment
``q_box = <n n>_box``.  Its eigenvectors, ordered by decreasing eigenvalue,
are N, M, and L.  The N-M principal plane therefore has normal L.

For the thesis's ideal initial loops, the additional model assumptions are
``k || nu`` and ``Omega || L``.  The complete pi director rotation along k is
a one-dimensional skyrmion, whose mode mapping is

    k || N: pure splay,  k || M: pure bend,  k || L: pure twist.

This skyrmion mapping must not be confused with the small-perturbation mapping,
for which k || N is bend and k || M is splay.
"""

from __future__ import annotations

import numpy as np


CONVENTION_NAME = "Yingyou-Ma-ideal-initial-loop"
CONVENTION_SOURCE = "3D Dry Uniaxial Active Nematics in Bulk, pp. 68-92"
IDEAL_LOOP_MODE_AXES = {
    "pure-splay": "N",
    "pure-bend": "M",
    "pure-twist": "L",
}


def _unit(vector, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a three-vector, got shape {vector.shape}.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Cannot normalize {name}: norm={norm!r}.")
    return vector / norm


def validate_nml_frame(axis_n, axis_m, axis_l, *, atol: float = 1e-6) -> np.ndarray:
    """Return a validated right-handed matrix with N, M, and L as columns."""

    axes = np.column_stack(
        (
            _unit(axis_n, name="N"),
            _unit(axis_m, name="M"),
            _unit(axis_l, name="L"),
        )
    )
    if not np.allclose(axes.T @ axes, np.eye(3), atol=atol):
        raise ValueError("N, M, and L must form an orthonormal frame.")
    if np.linalg.det(axes) < 1.0 - atol:
        raise ValueError("N, M, and L must form a right-handed frame with L = N x M.")
    return axes


def unsigned_axis_angle_deg(vector, axis) -> float:
    """Return the nematic unsigned angle between two vectors in degrees."""

    cosine = float(
        np.clip(abs(np.dot(_unit(vector, name="vector"), _unit(axis, name="axis"))), 0.0, 1.0)
    )
    return float(np.degrees(np.arccos(cosine)))


def ideal_loop_axis_angles(k, axes_nml) -> dict[str, float]:
    """Return k's angles to the pure-splay, pure-bend, and pure-twist axes."""

    axes = np.asarray(axes_nml, dtype=float)
    if axes.shape != (3, 3):
        raise ValueError(f"axes_nml must have shape (3, 3), got {axes.shape}.")
    validate_nml_frame(axes[:, 0], axes[:, 1], axes[:, 2])
    return {
        "pure-splay": unsigned_axis_angle_deg(k, axes[:, 0]),
        "pure-bend": unsigned_axis_angle_deg(k, axes[:, 1]),
        "pure-twist": unsigned_axis_angle_deg(k, axes[:, 2]),
    }


def classify_ideal_loop(k, axes_nml) -> tuple[str, dict[str, float]]:
    """Classify an ideal loop by the nearest Yingyou k axis."""

    angles = ideal_loop_axis_angles(k, axes_nml)
    return min(angles, key=angles.get), angles


def convention_metadata(axis_n, axis_m, axis_l, k, nu, omega) -> dict[str, object]:
    """Build explicit, JSON-ready metadata for one ideal initial loop."""

    axes = validate_nml_frame(axis_n, axis_m, axis_l)
    k = _unit(k, name="k")
    nu = _unit(nu, name="nu")
    omega = _unit(omega, name="Omega")
    loop_type, angles = classify_ideal_loop(k, axes)
    k_nml = axes.T @ k
    return {
        "name": CONVENTION_NAME,
        "source": CONVENTION_SOURCE,
        "orientation_second_moment": "q_box = <n n>_box",
        "eigenvalue_order": "lambda_N >= lambda_M >= lambda_L",
        "principal_plane": "N-M",
        "principal_plane_normal": "L",
        "loop_plane_normal": "nu",
        "rotation_axis": "Omega",
        "ideal_loop_assumptions": ["k parallel nu", "Omega parallel L"],
        "ideal_skyrmion_mode_axes": dict(IDEAL_LOOP_MODE_AXES),
        "small_perturbation_warning": (
            "Do not use the small-perturbation mapping: there k||N is bend and k||M is splay."
        ),
        "N": axes[:, 0].tolist(),
        "M": axes[:, 1].tolist(),
        "L": axes[:, 2].tolist(),
        "k": k.tolist(),
        "nu": nu.tolist(),
        "Omega": omega.tolist(),
        "k_NML": k_nml.tolist(),
        "classified_type": loop_type,
        "axis_angles_deg": angles,
        "angle_k_nu_deg": unsigned_axis_angle_deg(k, nu),
        "gamma_angle_nu_Omega_deg": unsigned_axis_angle_deg(nu, omega),
    }


__all__ = [
    "CONVENTION_NAME",
    "CONVENTION_SOURCE",
    "IDEAL_LOOP_MODE_AXES",
    "classify_ideal_loop",
    "convention_metadata",
    "ideal_loop_axis_angles",
    "unsigned_axis_angle_deg",
    "validate_nml_frame",
]
