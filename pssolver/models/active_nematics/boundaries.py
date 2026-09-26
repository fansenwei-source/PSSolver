"""Active-nematic conveniences for canonical prescribed-Q wall data.

The functions in this module translate physical Q language into the generic
field-level :class:`StaticPrescribedDirichletPolicy`.  They do not select a
geometry, transform, lifting extension, runtime, or solver.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

import numpy as np
import torch

from pssolver.boundaries import (
    StaticPrescribedDirichletPolicy,
    prescribed_dirichlet,
)
from pssolver.core.boundary import BoundarySide

from .fields import Q_COMPONENTS
from .q_tensor import Q_components, uniaxial_Q


OrientedFace = tuple[int, BoundarySide | str]


def _side(value: BoundarySide | str) -> BoundarySide:
    if isinstance(value, BoundarySide):
        return value
    try:
        return BoundarySide(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Q face side must be 'lower' or 'upper'") from exc


def _face_key(value: object) -> tuple[int, BoundarySide]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("Q face keys must be (axis, side) tuples")
    axis, side = value
    if (
        not isinstance(axis, int)
        or isinstance(axis, bool)
        or axis < 0
    ):
        raise ValueError("Q face axis must be a non-negative integer")
    return axis, _side(side)


def _faces(value: object, description: str) -> dict[tuple[int, BoundarySide], Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    if not value:
        raise ValueError(f"{description} must not be empty")
    normalized: dict[tuple[int, BoundarySide], Any] = {}
    for key, item in value.items():
        normalized_key = _face_key(key)
        if normalized_key in normalized:
            raise ValueError(f"{description} contains a duplicate oriented face")
        normalized[normalized_key] = item
    return normalized


def _constant_component(value: object, component: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"prescribed {component} must be spatially constant"
            )
        scalar = float(value.detach().cpu().item())
    else:
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(
                f"prescribed {component} must be spatially constant"
            )
        scalar = float(array.reshape(()))
    if not math.isfinite(scalar):
        raise ValueError(f"prescribed {component} must be finite")
    return scalar


def _constant_q_components(value: object) -> dict[str, float]:
    components = Q_components(value)
    return {
        component: _constant_component(components[component], component)
        for component in Q_COMPONENTS
    }


def _scalar_order(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("scalar_order must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError("scalar_order must be finite and non-negative")
    return normalized


def _unit_vector(value: object, description: str) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy().astype(float, copy=False)
    else:
        array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{description} must have shape (3,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{description} must be finite")
    norm = float(np.linalg.norm(array))
    if not math.isclose(norm, 1.0, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError(f"{description} must be unit length")
    return array


def prescribed_q(
    face_values: Mapping[OrientedFace, object],
) -> StaticPrescribedDirichletPolicy:
    """Prescribe one constant symmetric-traceless Q tensor on each face.

    Each value may be a full ``(3, 3)`` tensor, the canonical five-component
    sequence, or a mapping keyed by ``Qxx, Qxy, Qxz, Qyy, Qyz``.
    """

    faces = _faces(face_values, "prescribed Q face values")
    generic = {}
    for (axis, side), value in faces.items():
        for component, scalar in _constant_q_components(value).items():
            generic[(component, axis, side)] = scalar
    return prescribed_dirichlet("Q", generic)


def strong_homeotropic_q(
    *,
    scalar_order: Real,
    face_normals: Mapping[OrientedFace, object],
) -> StaticPrescribedDirichletPolicy:
    """Return strong homeotropic Q data using oriented unit face normals."""

    order = _scalar_order(scalar_order)
    normals = _faces(face_normals, "homeotropic face normals")
    tensors = {
        face: uniaxial_Q(
            _unit_vector(normal, "homeotropic face normal"),
            order,
        )
        for face, normal in normals.items()
    }
    return prescribed_q(tensors)


def strong_planar_q(
    *,
    scalar_order: Real,
    face_directors: Mapping[OrientedFace, object],
    face_normals: Mapping[OrientedFace, object],
) -> StaticPrescribedDirichletPolicy:
    """Return strong planar Q data from explicit in-plane unit directors."""

    order = _scalar_order(scalar_order)
    directors = _faces(face_directors, "planar face directors")
    normals = _faces(face_normals, "planar face normals")
    if set(directors) != set(normals):
        raise ValueError(
            "planar directors and normals must cover the same oriented faces"
        )
    tensors = {}
    for face in directors:
        director = _unit_vector(directors[face], "planar face director")
        normal = _unit_vector(normals[face], "planar face normal")
        if not math.isclose(
            float(np.dot(director, normal)),
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise ValueError("planar face director must be tangent to its face")
        tensors[face] = uniaxial_Q(director, order)
    return prescribed_q(tensors)


__all__ = [
    "OrientedFace",
    "prescribed_q",
    "strong_homeotropic_q",
    "strong_planar_q",
]
