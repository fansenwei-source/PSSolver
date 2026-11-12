"""Public tensor-free active-nematic model constructors.

The constructors in this module return canonical equation-system
declarations.  They do not attach a geometry or boundary condition, select a
runtime, allocate a tensor, or execute a timestep.
"""

from __future__ import annotations

from pssolver.functional.legacy_declarations.active_nematics_equation_systems import (
    COMPLETE_STRESS_FORCE_COMPONENTS,
    CompleteStressBerisEdwardsEquationRequest,
)
from pssolver.functional.legacy_declarations.active_nematics_specifications import (
    BerisEdwardsMaterialRequest,
)
from pssolver.functional.legacy_declarations.systems_equations import EquationSystemSpec
from pssolver.functional.legacy_declarations.systems_stokes import (
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)


def CompleteStressBerisEdwards(
    *,
    ldg_a: float,
    ldg_b: float,
    ldg_c: float,
    ldg_l1: float,
    gamma: float,
    flow_alignment: float,
    activity: float,
    beta: float,
    viscosity: float,
    friction: float = 0.0,
    tangential_zero_mode_policy: str | TangentialZeroModePolicy = (
        TangentialZeroModePolicy.ZERO_MEAN
    ),
) -> EquationSystemSpec:
    """Declare the qualified complete-stress Beris--Edwards equations.

    The default zero-wave-number treatment remains the explicit zero-mean
    velocity policy with zero friction.  The alternative ``friction`` policy
    retains the uniform velocity mode and requires a strictly positive drag
    coefficient.  Pressure always uses the zero-mean gauge.
    """

    try:
        zero_mode_policy = TangentialZeroModePolicy(
            tangential_zero_mode_policy
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tangential_zero_mode_policy must be 'zero_mean' or 'friction'"
        ) from exc
    if zero_mode_policy is TangentialZeroModePolicy.NOT_APPLICABLE:
        raise ValueError(
            "complete-stress periodic/Plane flow requires zero_mean or "
            "friction tangential_zero_mode_policy"
        )

    stokes = IncompressibleStokesSystemSpec(
        name="flow",
        force_components=COMPLETE_STRESS_FORCE_COMPONENTS,
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=viscosity,
        friction=friction,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=zero_mode_policy,
    )
    request = CompleteStressBerisEdwardsEquationRequest(
        material=BerisEdwardsMaterialRequest(
            ldg_a=ldg_a,
            ldg_b=ldg_b,
            ldg_c=ldg_c,
            gamma=gamma,
            flow_alignment=flow_alignment,
            beta=beta,
        ),
        ldg_l1=ldg_l1,
        activity_amplitude=activity,
        stokes_system=stokes,
    )
    return request.to_equation_system_spec()


__all__ = [
    "CompleteStressBerisEdwards",
]
