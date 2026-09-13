"""Small scalar models used to qualify the new execution architecture."""

from .scalar import (
    AllenCahnModel,
    DiffusionHelmholtzCouplingModel,
    ScalarDiffusionModel,
)
from .stokes import BodyForceStokesCanaryModel

__all__ = [
    "AllenCahnModel",
    "BodyForceStokesCanaryModel",
    "DiffusionHelmholtzCouplingModel",
    "ScalarDiffusionModel",
]
