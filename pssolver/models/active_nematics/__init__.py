"""Public building blocks for the active-nematic PDE model."""

from .fields import Q_COMPONENTS
from .initial_conditions import (
    aligned_x_smooth_noise,
    analytic_periodic_defect_gas_2d,
    available_initial_conditions,
    create_initial_condition,
    extruded_2d_twist,
    neumann_twist_profile,
    sample_periodic_neutral_defects_2d,
)
from .nematics3d_adapter import (
    S_and_director_from_Q,
    director_from_Q,
    eigenframe_from_Q,
    q_field_object_from_Q,
)
from .q_tensor import (
    Q_components,
    Q_convention_metadata,
    Q_magnitude,
    S_from_Q,
    positive_equilibrium_S,
    uniaxial_Q,
)

__all__ = [
    "Q_COMPONENTS",
    "Q_components",
    "Q_convention_metadata",
    "Q_magnitude",
    "S_from_Q",
    "S_and_director_from_Q",
    "director_from_Q",
    "eigenframe_from_Q",
    "q_field_object_from_Q",
    "aligned_x_smooth_noise",
    "analytic_periodic_defect_gas_2d",
    "available_initial_conditions",
    "create_initial_condition",
    "extruded_2d_twist",
    "neumann_twist_profile",
    "positive_equilibrium_S",
    "sample_periodic_neutral_defects_2d",
    "uniaxial_Q",
]
