"""Active-nematic conveniences for canonical prescribed-Q wall data.

The functions in this module translate physical Q language into the generic
field-level :class:`StaticPrescribedDirichletPolicy`.  They do not select a
geometry, transform, lifting extension, runtime, or solver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np
import torch

from pssolver.boundaries import (
    RobinFaceLaw,
    StaticPrescribedDirichletPolicy,
    StaticRobinBoundaryPolicy,
    prescribed_dirichlet,
)
from pssolver.core.boundary import BoundarySide

from .fields import Q_COMPONENTS
from .q_tensor import Q_components, Q_convention_metadata, uniaxial_Q


OrientedFace = tuple[int, BoundarySide | str]

QUADRATIC_FINITE_Q_SURFACE_LAW_ID = (
    "quadratic_one_constant_q_anchoring_v1"
)
Q_COMPONENT_METRIC = (
    (2.0, 0.0, 0.0, 1.0, 0.0),
    (0.0, 2.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 2.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 2.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 2.0),
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _positive(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{description} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{description} must be positive and finite")
    return normalized


def _nonnegative(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{description} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{description} must be finite and non-negative")
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


def _full_from_component_tuple(
    values: tuple[float, ...],
) -> np.ndarray:
    qxx, qxy, qxz, qyy, qyz = values
    return np.asarray(
        (
            (qxx, qxy, qxz),
            (qxy, qyy, qyz),
            (qxz, qyz, -(qxx + qyy)),
        ),
        dtype=float,
    )


def _component_tuple(value: object) -> tuple[float, ...]:
    components = _constant_q_components(value)
    return tuple(components[name] for name in Q_COMPONENTS)


def _q_coordinate_basis() -> tuple[np.ndarray, ...]:
    basis = []
    for index in range(len(Q_COMPONENTS)):
        values = [0.0] * len(Q_COMPONENTS)
        values[index] = 1.0
        basis.append(_full_from_component_tuple(tuple(values)))
    return tuple(basis)


def q_component_metric() -> tuple[tuple[float, ...], ...]:
    """Return the Gram metric induced by the full tensor contraction."""

    basis = _q_coordinate_basis()
    derived = tuple(
        tuple(float(np.sum(left * right)) for right in basis)
        for left in basis
    )
    if derived != Q_COMPONENT_METRIC:
        raise RuntimeError("canonical Q component metric is inconsistent")
    return derived


@dataclass(frozen=True, slots=True)
class FiniteQAnchoringFace:
    """One constant quadratic surface energy on an oriented face."""

    axis: int
    side: BoundarySide
    wall_strength: float
    target_components: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.axis, int)
            or isinstance(self.axis, bool)
            or self.axis < 0
        ):
            raise ValueError("finite-Q face axis must be non-negative")
        if not isinstance(self.side, BoundarySide):
            raise TypeError("finite-Q face side must be a BoundarySide")
        object.__setattr__(
            self,
            "wall_strength",
            _nonnegative(self.wall_strength, "wall_strength"),
        )
        values = tuple(float(value) for value in self.target_components)
        if len(values) != len(Q_COMPONENTS) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("target Q must have five finite components")
        object.__setattr__(self, "target_components", values)

    @property
    def key(self) -> tuple[int, BoundarySide]:
        return self.axis, self.side

    @property
    def target_full(self) -> np.ndarray:
        return _full_from_component_tuple(self.target_components)

    def to_metadata(self) -> dict[str, object]:
        target = {
            name: value
            for name, value in zip(Q_COMPONENTS, self.target_components)
        }
        return {
            "axis": self.axis,
            "side": self.side.value,
            "wall_strength": self.wall_strength,
            "target_components": target,
            "target_full": self.target_full.tolist(),
            "target_sha256": _canonical_sha256(target),
        }


@dataclass(frozen=True, slots=True)
class QuadraticFiniteQAnchoring:
    """Model specialization for one-constant quadratic finite anchoring."""

    k_q: float
    faces: tuple[FiniteQAnchoringFace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_q", _positive(self.k_q, "k_q"))
        try:
            faces = tuple(self.faces)
        except TypeError as exc:
            raise TypeError("finite-Q faces must be iterable") from exc
        if not faces or not all(
            isinstance(value, FiniteQAnchoringFace) for value in faces
        ):
            raise ValueError(
                "finite-Q anchoring requires FiniteQAnchoringFace values"
            )
        keys = tuple(value.key for value in faces)
        if len(set(keys)) != len(keys):
            raise ValueError("finite-Q anchoring faces must be unique")
        q_component_metric()
        object.__setattr__(
            self,
            "faces",
            tuple(
                sorted(
                    faces,
                    key=lambda value: (value.axis, value.side.value),
                )
            ),
        )

    def to_robin_policy(self) -> StaticRobinBoundaryPolicy:
        laws = []
        for face in self.faces:
            for component, target in zip(
                Q_COMPONENTS,
                face.target_components,
            ):
                laws.append(
                    RobinFaceLaw(
                        component,
                        face.axis,
                        face.side,
                        (
                            face.wall_strength,
                            self.k_q,
                            face.wall_strength * target,
                        ),
                    )
                )
        return StaticRobinBoundaryPolicy("Q", tuple(laws))

    def to_metadata(self) -> dict[str, object]:
        faces = []
        for value in self.faces:
            metadata = value.to_metadata()
            metadata["extrapolation_length"] = (
                None
                if value.wall_strength == 0.0
                else self.k_q / value.wall_strength
            )
            faces.append(metadata)
        return {
            "schema_version": 1,
            "surface_law_id": QUADRATIC_FINITE_Q_SURFACE_LAW_ID,
            "bulk_energy": "(K_Q/2)*integral[(partial_k Q_ij)^2]",
            "surface_energy": "(W/2)*integral[(Q_ij-Qstar_ij)^2]",
            "natural_boundary_law": (
                "K_Q*(n_dot_grad_Q_ij)+W*(Q_ij-Qstar_ij)=0"
            ),
            "robin_mapping": {
                "alpha": "W",
                "beta": "K_Q",
                "gamma": "W*Qstar_component",
            },
            "k_q": self.k_q,
            "q_convention": Q_convention_metadata(),
            "component_order": list(Q_COMPONENTS),
            "component_metric": [list(row) for row in q_component_metric()],
            "metric_cancellation": (
                "same_invertible_Gram_matrix_in_bulk_and_surface_variations"
            ),
            "faces": faces,
            "w_zero_limit": "homogeneous_neumann",
            "strong_dirichlet_is_distinct_policy": True,
            "generic_robin_policy_sha256": (
                self.to_robin_policy().canonical_sha256()
            ),
        }

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_metadata())


def quadratic_finite_q_anchoring(
    *,
    k_q: Real,
    wall_strengths: Mapping[OrientedFace, Real],
    target_q: Mapping[OrientedFace, object],
) -> QuadraticFiniteQAnchoring:
    """Create finite-Q anchoring and its field-neutral Robin policy."""

    strengths = _faces(wall_strengths, "finite-Q wall strengths")
    targets = _faces(target_q, "finite-Q target tensors")
    if set(strengths) != set(targets):
        raise ValueError(
            "finite-Q strengths and targets must cover the same faces"
        )
    return QuadraticFiniteQAnchoring(
        k_q=_positive(k_q, "k_q"),
        faces=tuple(
            FiniteQAnchoringFace(
                axis=axis,
                side=side,
                wall_strength=_nonnegative(
                    strengths[(axis, side)],
                    "wall_strength",
                ),
                target_components=_component_tuple(targets[(axis, side)]),
            )
            for axis, side in sorted(
                strengths,
                key=lambda value: (value[0], value[1].value),
            )
        ),
    )


def finite_homeotropic_q_anchoring(
    *,
    k_q: Real,
    wall_strengths: Mapping[OrientedFace, Real],
    scalar_order: Real,
    face_normals: Mapping[OrientedFace, object],
) -> QuadraticFiniteQAnchoring:
    """Create quadratic anchoring toward normal-aligned uniaxial Q."""

    order = _scalar_order(scalar_order)
    normals = _faces(face_normals, "homeotropic face normals")
    return quadratic_finite_q_anchoring(
        k_q=k_q,
        wall_strengths=wall_strengths,
        target_q={
            face: uniaxial_Q(
                _unit_vector(normal, "homeotropic face normal"),
                order,
            )
            for face, normal in normals.items()
        },
    )


def finite_planar_q_anchoring(
    *,
    k_q: Real,
    wall_strengths: Mapping[OrientedFace, Real],
    scalar_order: Real,
    face_directors: Mapping[OrientedFace, object],
    face_normals: Mapping[OrientedFace, object],
) -> QuadraticFiniteQAnchoring:
    """Create quadratic anchoring toward explicit in-plane uniaxial Q."""

    order = _scalar_order(scalar_order)
    directors = _faces(face_directors, "planar face directors")
    normals = _faces(face_normals, "planar face normals")
    if set(directors) != set(normals):
        raise ValueError(
            "planar directors and normals must cover the same oriented faces"
        )
    targets = {}
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
        targets[face] = uniaxial_Q(director, order)
    return quadratic_finite_q_anchoring(
        k_q=k_q,
        wall_strengths=wall_strengths,
        target_q=targets,
    )


def finite_q_variational_residual(
    *,
    q: object,
    normal_derivative_q: object,
    target_q: object,
    k_q: Real,
    wall_strength: Real,
) -> dict[str, np.ndarray]:
    """Return full-tensor and five-coordinate surface variations."""

    elastic = _positive(k_q, "k_q")
    strength = _nonnegative(wall_strength, "wall_strength")
    q_values = np.asarray(_component_tuple(q), dtype=float)
    derivative_values = np.asarray(
        _component_tuple(normal_derivative_q),
        dtype=float,
    )
    target_values = np.asarray(_component_tuple(target_q), dtype=float)
    component_law = (
        elastic * derivative_values
        + strength * (q_values - target_values)
    )
    metric = np.asarray(q_component_metric(), dtype=float)
    coordinate_variation = metric @ component_law
    full_tensor_residual = _full_from_component_tuple(
        tuple(float(value) for value in component_law)
    )
    projected_full_variation = np.asarray(
        [
            float(np.sum(basis * full_tensor_residual))
            for basis in _q_coordinate_basis()
        ]
    )
    if not np.allclose(
        coordinate_variation,
        projected_full_variation,
        rtol=2.0e-15,
        atol=2.0e-15,
    ):
        raise RuntimeError("five-component Q metric cancellation failed")
    return {
        "component_law": component_law,
        "coordinate_variation": coordinate_variation,
        "projected_full_variation": projected_full_variation,
        "full_tensor_residual": full_tensor_residual,
    }


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
    "FiniteQAnchoringFace",
    "OrientedFace",
    "Q_COMPONENT_METRIC",
    "QUADRATIC_FINITE_Q_SURFACE_LAW_ID",
    "QuadraticFiniteQAnchoring",
    "finite_homeotropic_q_anchoring",
    "finite_planar_q_anchoring",
    "finite_q_variational_residual",
    "prescribed_q",
    "q_component_metric",
    "quadratic_finite_q_anchoring",
    "strong_homeotropic_q",
    "strong_planar_q",
]
