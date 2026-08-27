"""Canonical Q-tensor operations at the Nematics3D boundary.

PSSolver defines ``S = lambda_max(Q)``. Some Nematics3D releases use
``S_n3d = 1.5 * lambda_max(Q)`` and their analytic director routine is
ill-conditioned for the repeated transverse eigenvalues of a uniaxial Q
tensor. Production analysis therefore enters through this adapter:

* scalar order, directors, and eigenframes use NumPy's symmetric eigensolver;
* canonical Q passed to Nematics3D's higher-level ``QFieldObject`` is scaled
  by ``2/3``, so its internal scalar order equals canonical ``S``.
"""

from __future__ import annotations

from importlib import import_module

import numpy as np


def _full_Q_numpy(Q: np.ndarray) -> np.ndarray:
    """Convert a compact five-component Q field to validated full matrix form."""
    values = np.asarray(Q)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Q must contain only finite values")
    if values.ndim >= 2 and values.shape[-2:] == (3, 3):
        scale = np.max(np.abs(values), axis=(-2, -1))
        tolerance = (
            8.0 * np.finfo(values.dtype).eps * scale + 1.0e-12
        )
        symmetry_error = np.max(
            np.abs(values - np.swapaxes(values, -2, -1)),
            axis=(-2, -1),
        )
        if not np.all(symmetry_error <= tolerance):
            raise ValueError("full Q tensors must be symmetric")
        trace_error = np.abs(np.trace(values, axis1=-2, axis2=-1))
        if not np.all(trace_error <= tolerance):
            raise ValueError("full Q tensors must be traceless")
        return values
    if values.ndim < 1 or values.shape[-1] != 5:
        raise ValueError(
            "Q must have shape (..., 5) or (..., 3, 3), "
            f"got {values.shape}"
        )
    Qxx, Qxy, Qxz, Qyy, Qyz = np.moveaxis(values, -1, 0)
    Qzz = -(Qxx + Qyy)
    return np.stack(
        (
            np.stack((Qxx, Qxy, Qxz), axis=-1),
            np.stack((Qxy, Qyy, Qyz), axis=-1),
            np.stack((Qxz, Qyz, Qzz), axis=-1),
        ),
        axis=-2,
    )


def S_and_director_from_Q(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical ``S = lambda_max(Q)`` and the dominant director."""
    eigenvalues, eigenvectors = np.linalg.eigh(_full_Q_numpy(Q))
    return eigenvalues[..., -1], eigenvectors[..., :, -1]


def director_from_Q(Q: np.ndarray) -> np.ndarray:
    """Return the dominant nematic director for a five- or nine-component Q."""
    _, director = S_and_director_from_Q(Q)
    return director


def eigenframe_from_Q(Q: np.ndarray) -> np.ndarray:
    """Return a descending-eigenvalue eigenvector frame.

    Eigenvectors occupy columns of the final two axes. At an exactly degenerate
    tensor (including ``Q=0``), the returned orthonormal frame is a finite but
    mathematically arbitrary representative of the eigenspace.
    """
    _, eigenframe = np.linalg.eigh(_full_Q_numpy(Q))
    return np.flip(eigenframe, axis=-1)


def q_field_object_from_Q(
    Q: np.ndarray,
    *,
    box_periodic_flag=False,
    name: str = "Q",
    **kwargs,
):
    """Build a Nematics3D QFieldObject from canonical PSSolver Q.

    The canonical tensor is scaled by ``2/3`` for Nematics3D's internal Q
    convention. The object's scalar order and director fields are then
    replaced by the robust symmetric-eigensolver result before optional defect
    detection, avoiding the installed analytic diagonalizer's uniaxial
    degeneracy bug while retaining the complete (possibly biaxial) Q field.
    """
    values = np.asarray(Q)
    canonical_S, canonical_director = S_and_director_from_Q(values)
    try:
        n3d = import_module("nematics3d")
        InputQ = import_module(
            "nematics3d.classes.q_field_object"
        ).InputQ
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Nematics3D is required to construct a QFieldObject"
        ) from exc

    is_detect_defects = bool(kwargs.pop("is_detect_defects", True))
    is_classify_lines = bool(kwargs.pop("is_classify_lines", True))
    # The installed analytic diagonalizer can emit a divide warning before
    # these provisional values are replaced below. Defect detection is still
    # disabled during construction, so suppress only that NumPy arithmetic
    # warning inside the known-buggy call.
    with np.errstate(invalid="ignore", divide="ignore"):
        q_object = n3d.QFieldObject(
            inputValue=InputQ(
                Q=(2.0 / 3.0) * values,
                box_periodic_flag=box_periodic_flag,
            ),
            name=name,
            is_detect_defects=False,
            is_classify_lines=False,
            **kwargs,
        )
    object.__setattr__(q_object, "_raw_S", canonical_S)
    object.__setattr__(q_object, "_raw_n", canonical_director)

    if is_detect_defects:
        q_object.act_defect_detect()
        if is_classify_lines:
            q_object.act_lines_classify()
    return q_object
