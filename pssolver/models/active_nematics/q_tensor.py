"""Canonical three-dimensional active-nematic Q-tensor operations.

The model uses exactly one convention,

    Q = (3 S / 2) (n n - I / 3),

where ``n`` is a unit director and ``S = lambda_max(Q)`` for a prolate
uniaxial tensor.  Functions accept NumPy arrays or PyTorch tensors and retain
the input backend.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import torch

from .fields import Q_COMPONENTS


Q_CONVENTION_ID = "de_gennes_S_lambda_max_v1"
Q_CONVENTION_DEFINITION = "Q=(3S/2)(nn-I/3)"


def _torch_float_dtype(value: torch.Tensor) -> torch.dtype:
    if value.is_floating_point():
        return value.dtype
    return torch.get_default_dtype()


def _validate_director_numpy(director: np.ndarray) -> None:
    if director.ndim < 1 or director.shape[-1] != 3:
        raise ValueError(
            "director must have final dimension 3, got "
            f"shape {director.shape}"
        )
    if not np.all(np.isfinite(director)):
        raise ValueError("director must contain only finite values")
    norm = np.linalg.norm(director, axis=-1)
    if not np.allclose(norm, 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("director must be unit length")


def _validate_director_torch(director: torch.Tensor) -> None:
    if director.ndim < 1 or director.shape[-1] != 3:
        raise ValueError(
            "director must have final dimension 3, got "
            f"shape {tuple(director.shape)}"
        )
    if not bool(torch.isfinite(director).all()):
        raise ValueError("director must contain only finite values")
    norm = torch.linalg.vector_norm(director, dim=-1)
    if not torch.allclose(
        norm,
        torch.ones_like(norm),
        rtol=1e-5,
        atol=1e-7,
    ):
        raise ValueError("director must be unit length")


def uniaxial_Q(
    director: np.ndarray | torch.Tensor,
    S: float | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Construct canonical prolate uniaxial Q tensors.

    ``director`` has shape ``(..., 3)`` and must be unit length. ``S`` may be
    a non-negative scalar or an array broadcastable to the director's leading
    dimensions.  The result has shape ``(..., 3, 3)``.
    """
    if torch.is_tensor(director) or torch.is_tensor(S):
        if torch.is_tensor(director):
            dtype = _torch_float_dtype(director)
            director_tensor = director.to(dtype=dtype)
        else:
            assert torch.is_tensor(S)
            dtype = _torch_float_dtype(S)
            director_tensor = torch.as_tensor(
                director,
                dtype=dtype,
                device=S.device,
            )
        S_tensor = torch.as_tensor(
            S,
            dtype=director_tensor.dtype,
            device=director_tensor.device,
        )
        _validate_director_torch(director_tensor)
        if not bool(torch.isfinite(S_tensor).all()) or bool((S_tensor < 0).any()):
            raise ValueError("S must contain only non-negative finite values")
        try:
            leading_shape = torch.broadcast_shapes(
                director_tensor.shape[:-1],
                S_tensor.shape,
            )
        except RuntimeError as error:
            raise ValueError(
                f"S shape {tuple(S_tensor.shape)} is not broadcastable with "
                f"director shape {tuple(director_tensor.shape)}"
            ) from error
        director_tensor = torch.broadcast_to(
            director_tensor,
            (*leading_shape, 3),
        )
        S_tensor = torch.broadcast_to(S_tensor, leading_shape)
        dyad = director_tensor.unsqueeze(-1) * director_tensor.unsqueeze(-2)
        identity = torch.eye(
            3,
            dtype=director_tensor.dtype,
            device=director_tensor.device,
        )
        return 1.5 * S_tensor[..., None, None] * (dyad - identity / 3.0)

    director_array = np.asarray(director)
    if not np.issubdtype(director_array.dtype, np.floating):
        director_array = director_array.astype(float)
    S_array = np.asarray(S, dtype=director_array.dtype)
    _validate_director_numpy(director_array)
    if not np.all(np.isfinite(S_array)) or np.any(S_array < 0):
        raise ValueError("S must contain only non-negative finite values")
    try:
        leading_shape = np.broadcast_shapes(director_array.shape[:-1], S_array.shape)
    except ValueError as error:
        raise ValueError(
            f"S shape {S_array.shape} is not broadcastable with director "
            f"shape {director_array.shape}"
        ) from error
    director_array = np.broadcast_to(director_array, (*leading_shape, 3))
    S_array = np.broadcast_to(S_array, leading_shape)
    dyad = director_array[..., :, None] * director_array[..., None, :]
    return 1.5 * S_array[..., None, None] * (
        dyad - np.eye(3, dtype=director_array.dtype) / 3.0
    )


def _stack_mapping_components(
    Q: Mapping[str, Any],
) -> np.ndarray | torch.Tensor:
    names = set(Q)
    required = set(Q_COMPONENTS)
    missing = required - names
    extra = names - required
    if missing or extra:
        raise ValueError(
            "Q mapping must contain exactly the five independent components; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    values = [Q[name] for name in Q_COMPONENTS]
    tensor_values = [value for value in values if torch.is_tensor(value)]
    if tensor_values:
        reference = tensor_values[0]
        if reference.is_complex():
            raise ValueError("Q must be real-valued")
        dtype = _torch_float_dtype(reference)
        converted = [
            torch.as_tensor(value, dtype=dtype, device=reference.device)
            for value in values
        ]
        try:
            converted = list(torch.broadcast_tensors(*converted))
        except RuntimeError as error:
            raise ValueError("Q component shapes are not broadcastable") from error
        return torch.stack(converted, dim=-1)

    arrays = [np.asarray(value) for value in values]
    if any(np.iscomplexobj(value) for value in arrays):
        raise ValueError("Q must be real-valued")
    try:
        arrays = list(np.broadcast_arrays(*arrays))
    except ValueError as error:
        raise ValueError("Q component shapes are not broadcastable") from error
    return np.stack(arrays, axis=-1)


def _validate_full_Q_structure(Q: np.ndarray | torch.Tensor) -> None:
    """Require an explicit full Q tensor to be finite, symmetric, and traceless."""
    if torch.is_tensor(Q):
        if not bool(torch.isfinite(Q).all()):
            raise ValueError("Q must contain only finite values")
        if not torch.allclose(
            Q,
            Q.transpose(-2, -1),
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError("full Q tensors must be symmetric")
        trace = Q.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        trace_tolerance = (
            8.0
            * torch.finfo(Q.dtype).eps
            * Q.abs().amax(dim=(-2, -1))
            + 1.0e-12
        )
        if bool((trace.abs() > trace_tolerance).any()):
            raise ValueError("full Q tensors must be traceless")
        return

    if not np.all(np.isfinite(Q)):
        raise ValueError("Q must contain only finite values")
    if not np.allclose(
        Q,
        np.swapaxes(Q, -2, -1),
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("full Q tensors must be symmetric")
    trace = np.trace(Q, axis1=-2, axis2=-1)
    trace_tolerance = (
        8.0 * np.finfo(Q.dtype).eps * np.max(np.abs(Q), axis=(-2, -1))
        + 1.0e-12
    )
    if np.any(np.abs(trace) > trace_tolerance):
        raise ValueError("full Q tensors must be traceless")


def _full_Q(
    Q: Mapping[str, Any] | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    values = _stack_mapping_components(Q) if isinstance(Q, Mapping) else Q

    if torch.is_tensor(values):
        if values.is_complex():
            raise ValueError("Q must be real-valued")
        if not values.is_floating_point():
            values = values.to(dtype=torch.get_default_dtype())
        if values.ndim >= 2 and tuple(values.shape[-2:]) == (3, 3):
            _validate_full_Q_structure(values)
            return values
        if values.ndim < 1 or values.shape[-1] != len(Q_COMPONENTS):
            raise ValueError(
                "Q must have shape (..., 3, 3), shape (..., 5), or be a "
                "five-component mapping"
            )
        Qxx, Qxy, Qxz, Qyy, Qyz = values.unbind(dim=-1)
        Qzz = -(Qxx + Qyy)
        return torch.stack(
            (
                torch.stack((Qxx, Qxy, Qxz), dim=-1),
                torch.stack((Qxy, Qyy, Qyz), dim=-1),
                torch.stack((Qxz, Qyz, Qzz), dim=-1),
            ),
            dim=-2,
        )

    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError("Q must be real-valued")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(float)
    if array.ndim >= 2 and array.shape[-2:] == (3, 3):
        _validate_full_Q_structure(array)
        return array
    if array.ndim < 1 or array.shape[-1] != len(Q_COMPONENTS):
        raise ValueError(
            "Q must have shape (..., 3, 3), shape (..., 5), or be a "
            "five-component mapping"
        )
    Qxx, Qxy, Qxz, Qyy, Qyz = np.moveaxis(array, -1, 0)
    Qzz = -(Qxx + Qyy)
    return np.stack(
        (
            np.stack((Qxx, Qxy, Qxz), axis=-1),
            np.stack((Qxy, Qyy, Qyz), axis=-1),
            np.stack((Qxz, Qyz, Qzz), axis=-1),
        ),
        axis=-2,
    )


def Q_components(
    Q: Mapping[str, Any] | np.ndarray | torch.Tensor,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Return the five independent components of a symmetric traceless Q."""
    full = _full_Q(Q)
    return {
        "Qxx": full[..., 0, 0],
        "Qxy": full[..., 0, 1],
        "Qxz": full[..., 0, 2],
        "Qyy": full[..., 1, 1],
        "Qyz": full[..., 1, 2],
    }


def _validate_finite_Q(Q: np.ndarray | torch.Tensor) -> None:
    if torch.is_tensor(Q):
        finite = bool(torch.isfinite(Q).all())
    else:
        finite = bool(np.all(np.isfinite(Q)))
    if not finite:
        raise ValueError("Q must contain only finite values")


def S_from_Q(
    Q: Mapping[str, Any] | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Return the principal order parameter ``S = lambda_max(Q)``."""
    full = _full_Q(Q)
    _validate_finite_Q(full)
    if torch.is_tensor(full):
        return torch.linalg.eigvalsh(full)[..., -1]
    return np.linalg.eigvalsh(full)[..., -1]


def Q_magnitude(
    Q: Mapping[str, Any] | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Return ``sqrt((2/3) Q:Q)`` without assuming uniaxiality."""
    full = _full_Q(Q)
    _validate_finite_Q(full)
    contraction = (2.0 / 3.0) * (full * full).sum(axis=(-2, -1))
    if torch.is_tensor(full):
        return torch.sqrt(torch.clamp_min(contraction, 0.0))
    return np.sqrt(np.maximum(contraction, 0.0))


def positive_equilibrium_S(A: float, B: float, C: float) -> float:
    """Return the positive root of ``3 C S^2 + B S + 2 A = 0``."""
    coefficients = (float(A), float(B), float(C))
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("A, B, and C must be finite")
    A_value, B_value, C_value = coefficients
    if C_value <= 0.0:
        raise ValueError(f"C must be positive, got {C_value}")
    discriminant = B_value**2 - 24.0 * A_value * C_value
    if discriminant < 0.0:
        raise ValueError(
            "bulk coefficients do not have a real uniaxial equilibrium: "
            f"discriminant={discriminant}"
        )
    square_root = math.sqrt(discriminant)
    roots = (
        (-B_value + square_root) / (6.0 * C_value),
        (-B_value - square_root) / (6.0 * C_value),
    )
    positive_roots = [root for root in roots if root > 0.0]
    if not positive_roots:
        raise ValueError(f"bulk coefficients have no positive root: roots={roots}")
    return max(positive_roots)


def Q_convention_metadata() -> dict[str, Any]:
    """Return a fresh metadata record for the model's sole Q convention."""
    return {
        "id": Q_CONVENTION_ID,
        "definition": Q_CONVENTION_DEFINITION,
        "S_definition": "S=lambda_max(Q)",
        "uniaxial_eigenvalues": ["S", "-S/2", "-S/2"],
        "version": 1,
    }


__all__ = [
    "Q_components",
    "Q_convention_metadata",
    "Q_magnitude",
    "S_from_Q",
    "positive_equilibrium_S",
    "uniaxial_Q",
]
