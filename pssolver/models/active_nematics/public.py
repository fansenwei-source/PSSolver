"""Public tensor-free active-nematic model constructors.

The constructors in this module return canonical equation-system
declarations.  They do not attach a geometry or boundary condition, select a
runtime, allocate a tensor, or execute a timestep.
"""

from __future__ import annotations

from .equation_systems import (
    ACTIVE_FORCE_COMPONENTS,
    COMPLETE_STRESS_FORCE_COMPONENTS,
    CompleteStressBerisEdwardsEquationRequest,
    EquationSystemSpec,
    IncompressibleStokesSystemSpec,
    LegacyActiveForceEquationRequest,
    PressureGauge,
    TangentialZeroModePolicy,
)
from .specifications import BerisEdwardsMaterialRequest


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
) -> EquationSystemSpec:
    """Declare the qualified complete-stress Beris--Edwards equations.

    The zero-wave-number treatment is the explicit zero-mean tangential-flow
    policy with zero friction.  A future public constructor may expose the
    already-supported friction policy, but this convenience surface does not
    silently select it.
    """

    stokes = IncompressibleStokesSystemSpec(
        name="flow",
        force_components=COMPLETE_STRESS_FORCE_COMPONENTS,
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=viscosity,
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


def LegacyActiveForceActiveNematics(
    *,
    rho: float,
    elastic_constant: float,
    activity: float,
    beta: float,
    flow_alignment: float,
    friction: float,
    viscosity: float,
) -> EquationSystemSpec:
    """Declare the already-qualified Channel active-force-only equations.

    This is an ergonomic facade over the canonical rho-parameterized request
    used by the existing Channel application.  Every physical coefficient is
    explicit; the constructor does not attach a geometry, boundary policy,
    pressure algorithm, numerical method, or runtime.
    """

    stokes = IncompressibleStokesSystemSpec(
        name="channel_stokes",
        force_components=ACTIVE_FORCE_COMPONENTS,
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=viscosity,
        friction=friction,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=(
            TangentialZeroModePolicy.NOT_APPLICABLE
        ),
    )
    request = LegacyActiveForceEquationRequest(
        rho=rho,
        elastic_constant=elastic_constant,
        activity=activity,
        beta=beta,
        flow_alignment=flow_alignment,
        stokes_system=stokes,
    )
    return request.to_equation_system_spec()


__all__ = [
    "CompleteStressBerisEdwards",
    "LegacyActiveForceActiveNematics",
]
