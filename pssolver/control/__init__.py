"""Discrete-adjoint optimal-control tools for PSSolver."""

from .active_force import active_force_divergence
from .controls import (
    TemporalMaskControl,
    partition_mask_along_axis,
    partition_mask_rbf,
    smooth_box_mask,
)
from .dal import DALResult, DiscreteAdjointLoop
from .functional import FunctionalSemiImplicitStep
from .grid_transfer import conservative_block_average, periodic_translate_axis
from .loop_diagnostics import (
    core_centerline_geometry,
    defect_plaquette_counts,
    loop_core_metrics,
    periodic_profile_shift,
    principal_director_and_S,
    soft_core_center,
    soft_core_density,
    x_disturbance_profile,
)
from .objectives import (
    CoreAwareAlignedShapeObjective,
    CoreAwareComovingQTrajectoryObjective,
    CoreAwareMomentObjective,
    CoreTranslationNoMassObjective,
    CoreTranslationObjective,
    FreePathShapePreservingObjective,
    LOSS_FUNCTIONS,
    QTrackingObjective,
    available_loss_functions,
    build_loss_function,
    load_q_target,
)

__all__ = [
    "DALResult",
    "DiscreteAdjointLoop",
    "FunctionalSemiImplicitStep",
    "CoreAwareAlignedShapeObjective",
    "CoreAwareComovingQTrajectoryObjective",
    "CoreAwareMomentObjective",
    "CoreTranslationNoMassObjective",
    "CoreTranslationObjective",
    "FreePathShapePreservingObjective",
    "LOSS_FUNCTIONS",
    "QTrackingObjective",
    "TemporalMaskControl",
    "active_force_divergence",
    "available_loss_functions",
    "build_loss_function",
    "conservative_block_average",
    "core_centerline_geometry",
    "defect_plaquette_counts",
    "load_q_target",
    "loop_core_metrics",
    "periodic_profile_shift",
    "periodic_translate_axis",
    "partition_mask_along_axis",
    "partition_mask_rbf",
    "principal_director_and_S",
    "soft_core_center",
    "soft_core_density",
    "smooth_box_mask",
    "x_disturbance_profile",
]
