"""Compatibility facade for the v0.1 transform import surface.

Canonical implementations live in ``backends``, ``operators``, and
``linear_solvers``.  The aliases here preserve historical imports and legacy
pickle global paths without introducing wrappers or a second implementation.
"""

from .backends.bounded import (
    BoundedAxisPlanKey,
    DenseBoundedAxisExecutionPlan,
    build_dense_orthonormal_matrix,
)
from .backends.tensor_product import (
    DEFAULT_PERIODIC_TRANSFORM_EXECUTION,
    DEFAULT_SPECTRAL_STORAGE,
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    PERIODIC_TRANSFORM_EXECUTION_MODES,
    SPECTRAL_STORAGE_MODES,
    TensorProductTransformBackend,
    TransformMetadata,
)
from .linear_solvers.stokes.plane_free_slip import FreeSlipModalStokesSolver
from .operators.projection import (
    DEALIAS_RULE_FRACTIONS,
    DEFAULT_DEALIAS_RULE,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
    PROJECTED_TRANSFORM_EXECUTION_MODES,
    BasisAwareSpectralProjector,
)
from .operators.tensor_divergence import (
    projected_common_basis_stress_divergence,
    projected_distortion_stress_divergence,
)


__all__ = (
    "DEFAULT_DEALIAS_RULE",
    "DEFAULT_TRANSFORM_EXECUTION_ORDER",
    "DEFAULT_PROJECTED_TRANSFORM_EXECUTION",
    "PROJECTED_TRANSFORM_EXECUTION_MODES",
    "DEFAULT_SPECTRAL_STORAGE",
    "SPECTRAL_STORAGE_MODES",
    "DEFAULT_PERIODIC_TRANSFORM_EXECUTION",
    "PERIODIC_TRANSFORM_EXECUTION_MODES",
    "DEALIAS_RULE_FRACTIONS",
    "TransformMetadata",
    "TensorProductTransformBackend",
    "BasisAwareSpectralProjector",
    "projected_common_basis_stress_divergence",
    "projected_distortion_stress_divergence",
    "FreeSlipModalStokesSolver",
    "BoundedAxisPlanKey",
    "DenseBoundedAxisExecutionPlan",
    "build_dense_orthonormal_matrix",
)
