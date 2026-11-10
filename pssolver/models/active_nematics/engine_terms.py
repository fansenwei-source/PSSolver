"""Periodic model factories; numerical composition moved from the v0.4 runtime."""

import math

from pssolver.linear_solvers.stokes import PeriodicModalStokesSolver
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsPointwiseKernels,
    BerisEdwardsQGradientCache,
    BerisEdwardsQNonlinearModel,
)


PERIODIC_BOUNDARIES = ("periodic", "periodic", "periodic")


class _BerisEdwardsPeriodicStokes(
    PeriodicModalStokesSolver,
    BerisEdwardsFreeSlipStokes,
):
    """Runtime composition of complete nematic force and periodic Stokes."""

    def __init__(
        self,
        solver,
        spectral_projector,
        *,
        beta_value,
        friction,
        viscosity,
        ldg_a,
        ldg_b,
        ldg_c,
        ldg_l1,
        flow_alignment,
        molecular_field_linear_space,
        stress_divergence_sum_space,
        cache_force_diagnostics,
        cache_pressure_diagnostics,
        q_gradient_cache,
        pointwise_kernels,
        zero_mode_policy,
    ):
        numeric_values = (
            beta_value,
            friction,
            viscosity,
            ldg_a,
            ldg_b,
            ldg_c,
            ldg_l1,
            flow_alignment,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("Stokes and nematic coefficients must be finite")
        if ldg_l1 <= 0:
            raise ValueError("the one-constant L1 coefficient must be positive")
        if molecular_field_linear_space not in {"physical", "spectral"}:
            raise ValueError("invalid molecular-field linear space")
        if stress_divergence_sum_space not in {"physical", "spectral"}:
            raise ValueError("invalid stress-divergence sum space")

        PeriodicModalStokesSolver.__init__(
            self,
            solver.transform_backend,
            boundary_conditions=PERIODIC_BOUNDARIES,
            friction=friction,
            viscosity=viscosity,
            zero_mode_policy=zero_mode_policy,
            pressure_diagnostics=cache_pressure_diagnostics,
        )
        self.beta = float(beta_value)
        self.ldg_a = float(ldg_a)
        self.ldg_b = float(ldg_b)
        self.ldg_c = float(ldg_c)
        self.ldg_l1 = float(ldg_l1)
        self.flow_alignment = float(flow_alignment)
        self.molecular_field_linear_space = molecular_field_linear_space
        self.stress_divergence_sum_space = stress_divergence_sum_space
        self.cache_force_diagnostics = bool(cache_force_diagnostics)
        self.pointwise_kernels = pointwise_kernels
        self.q_gradient_cache = q_gradient_cache
        self.spectral_projector = spectral_projector
        self.q_boundary_conditions = PERIODIC_BOUNDARIES
        self.distortion_odd_boundary_conditions = PERIODIC_BOUNDARIES
        self.last_tangential_force_mean = None
        self.last_total_tangential_force_mean = None
        self.last_active_tangential_force_mean = None
        self.last_passive_tangential_force_mean = None
        self.last_removed_tangential_force_mean = None
        self.last_projected_normal_force = None
        self.last_uniform_velocity_mode_action = (
            "remove_all_uniform_velocity_and_force"
            if zero_mode_policy == "zero_mean"
            else "retain_all_uniform_velocity_resolved_by_friction"
        )


def periodic_explicit_term(projector, plan):
    """Materialize the explicit term and its shared pointwise/cache objects."""
    cache = BerisEdwardsQGradientCache() if plan.q_gradient_reuse else None
    kernels = BerisEdwardsPointwiseKernels(plan.pointwise_execution)
    return BerisEdwardsQNonlinearModel(
        projector,
        PERIODIC_BOUNDARIES,
        **plan.arguments,
        q_gradient_cache=cache,
        pointwise_kernels=kernels,
    )


def periodic_stokes_term(host, projector, plan, explicit):
    """Use the unchanged Periodic composition, not its stateful forward."""
    return _BerisEdwardsPeriodicStokes(
        host,
        projector,
        **plan.arguments,
        cache_force_diagnostics=False,
        cache_pressure_diagnostics=plan.pressure_diagnostics,
        q_gradient_cache=explicit.q_gradient_cache,
        pointwise_kernels=explicit.pointwise_kernels,
    )
