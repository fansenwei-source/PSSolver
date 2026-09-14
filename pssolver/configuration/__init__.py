"""Immutable run specifications at the CLI-to-runtime boundary."""

from .plane_beris_edwards import (
    DEFAULT_FRICTION_MODE_FRIC,
    DEFAULT_SPECTRAL_REFRESH_TIME,
    DEFAULT_ZERO_MODE_POLICY,
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
    PLANE_FREE_SLIP_BOUNDARIES,
    PLANE_RUN_SPEC_SCHEMA_VERSION,
    PlaneBerisEdwardsRunSpec,
    PlaneFreeSlipBoundaryConditions,
    SpectralRefreshSpec,
    create_plane_beris_edwards_run_spec,
    parse_plane_beris_edwards_run_spec,
)

__all__ = [
    "DEFAULT_FRICTION_MODE_FRIC",
    "DEFAULT_SPECTRAL_REFRESH_TIME",
    "DEFAULT_ZERO_MODE_POLICY",
    "PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES",
    "PLANE_FREE_SLIP_BOUNDARIES",
    "PLANE_RUN_SPEC_SCHEMA_VERSION",
    "PlaneBerisEdwardsRunSpec",
    "PlaneFreeSlipBoundaryConditions",
    "SpectralRefreshSpec",
    "create_plane_beris_edwards_run_spec",
    "parse_plane_beris_edwards_run_spec",
]
