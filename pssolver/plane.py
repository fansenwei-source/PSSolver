"""Numerical defaults qualified specifically for the Plane geometry.

These defaults must not be reused implicitly by geometries whose periodic
axes, boundary-condition layout, or real-valued storage assumptions differ.
"""

from .geometries.plane_numerics import (
    DEFAULT_PLANE_SPECTRAL_STORAGE,
    PLANE_HERMITIAN_AXIS,
)


__all__ = [
    "DEFAULT_PLANE_SPECTRAL_STORAGE",
    "PLANE_HERMITIAN_AXIS",
]
