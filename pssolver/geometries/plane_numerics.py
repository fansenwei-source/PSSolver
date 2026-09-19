"""Tensor-free numerical policy qualified for the production Plane slab.

This module selects a storage representation for the existing Plane topology;
it does not construct transforms, allocate tensors, or define a global
spectral default for other geometries.
"""

from pssolver.core.numerics import SpectralStorage


DEFAULT_PLANE_SPECTRAL_STORAGE = SpectralStorage.HERMITIAN_HALF.value
PLANE_HERMITIAN_AXIS = 1


__all__ = [
    "DEFAULT_PLANE_SPECTRAL_STORAGE",
    "PLANE_HERMITIAN_AXIS",
]
