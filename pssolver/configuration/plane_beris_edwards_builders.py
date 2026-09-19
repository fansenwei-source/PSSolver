"""Narrow pure builders for derived Plane run-configuration views.

Each function owns exactly one historical ``PlaneBerisEdwardsRunSpec`` view.
They intentionally do not construct the aggregate component graph, cache
values, or validate fields unrelated to the requested view.
"""

from __future__ import annotations

from pssolver.core import (
    DealiasRule,
    DomainSpec,
    NumericsConfig,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.geometries import PlaneSlab
from pssolver.presets import ShendrukPlanePreset, resolve_shendruk_plane_preset


def build_plane_beris_edwards_domain(
    *,
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    height: float,
) -> DomainSpec:
    """Build only the historical Plane domain view."""

    return DomainSpec((nx, ny, nz), (lx, ly, height))


def build_plane_beris_edwards_geometry(domain: DomainSpec) -> PlaneSlab:
    """Build only the historical Plane slab view."""

    return PlaneSlab(domain, wall_normal_axis=2)


def build_plane_beris_edwards_numerics(
    *,
    dtype: str,
    dealias_rule: str,
    transform_execution_order: str,
    projected_transform_execution: str,
    spectral_storage: str,
    hermitian_axis: int,
) -> NumericsConfig:
    """Build only the historical Plane numerics view."""

    return NumericsConfig(
        precision=Precision(dtype),
        dealias_rule=DealiasRule(dealias_rule),
        transform_execution_order=TransformExecutionOrder(
            transform_execution_order
        ),
        projected_transform_execution=ProjectedTransformExecution(
            projected_transform_execution
        ),
        spectral_storage=SpectralStorage(spectral_storage),
        hermitian_axis=(
            hermitian_axis
            if spectral_storage == "hermitian_half"
            else None
        ),
    )


def build_plane_beris_edwards_shendruk_preset(
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
    """Build only the historical resolved Shendruk preset view."""

    return resolve_shendruk_plane_preset(
        activity_number=activity_number,
        height=height,
        parameterization=parameterization,
        frank_k=frank_k,
        coefficient_min=coefficient_min,
        coefficient_max=coefficient_max,
        ldg_a=ldg_a,
        ldg_b=ldg_b,
        ldg_c=ldg_c,
        gamma=gamma,
    )


__all__ = [
    "build_plane_beris_edwards_domain",
    "build_plane_beris_edwards_geometry",
    "build_plane_beris_edwards_numerics",
    "build_plane_beris_edwards_shendruk_preset",
]
