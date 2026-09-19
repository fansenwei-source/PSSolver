"""Upper compatibility adapter for provisional Plane run components.

The dependency-neutral component graph does not import the supported flat
facade.  This module preserves the P2.1/P2.2 direct import paths, performs the
concrete facade type check, and forwards all flat values exactly once.
"""

from __future__ import annotations

from .plane_beris_edwards import (
    PLANE_HERMITIAN_AXIS,
    PlaneBerisEdwardsRunSpec,
)
from .plane_beris_edwards_component_graph import (
    BerisEdwardsMaterialRequest,
    ExtrudedDefectGasInitialConditionSpec,
    IncompressibleStokesSystemSpec,
    NumericsConfig,
    Path,
    PlaneBerisEdwardsExecutionSpec,
    PlaneBerisEdwardsPhysicsSpec,
    PlaneBerisEdwardsRunComponents,
    PlaneInvocationSpec,
    PlaneSlab,
    PlaneTimeSteppingSpec,
    PlaneWorkflowSpec,
    ShendrukPlaneParameterRequest,
    ShendrukPlanePreset,
    build_plane_beris_edwards_run_components as _build_run_components,
)
from .plane_beris_edwards_declarations import (
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneFreeSlipBoundaryConditions,
    PlaneRuntimePath,
    SpectralRefreshSpec,
)

# The non-component names imported from the lower graph are intentional
# compatibility bindings.  Component classes retain this module as their
# nominal ``__module__``, so no-argument ``typing.get_type_hints`` resolves
# their postponed annotations through this namespace exactly as before.

# These classes retain their provisional P2.1 nominal path.  Rebinding the
# exact module-name object also preserves protocol-4 memoization when several
# component classes occur in one pickle.
for _component_type in (
    PlaneBerisEdwardsPhysicsSpec,
    PlaneTimeSteppingSpec,
    PlaneBerisEdwardsExecutionSpec,
    PlaneWorkflowSpec,
    PlaneInvocationSpec,
    PlaneBerisEdwardsRunComponents,
):
    _component_type.__module__ = __name__


def decompose_plane_beris_edwards_run_spec(
    legacy_spec: PlaneBerisEdwardsRunSpec,
) -> PlaneBerisEdwardsRunComponents:
    """Purely decompose one qualified flat facade into provisional parts."""

    if not isinstance(legacy_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError(
            "legacy_spec must be a PlaneBerisEdwardsRunSpec"
        )

    return _build_run_components(
        activity_number=legacy_spec.activity_number,
        output_dir=legacy_spec.output_dir,
        height=legacy_spec.height,
        parameterization=legacy_spec.parameterization,
        frank_k=legacy_spec.frank_k,
        coefficient_min=legacy_spec.coefficient_min,
        coefficient_max=legacy_spec.coefficient_max,
        lx=legacy_spec.lx,
        ly=legacy_spec.ly,
        nx=legacy_spec.nx,
        ny=legacy_spec.ny,
        nz=legacy_spec.nz,
        dt=legacy_spec.dt,
        steps=legacy_spec.steps,
        save_start_step=legacy_spec.save_start_step,
        save_interval=legacy_spec.save_interval,
        diagnostic_interval=legacy_spec.diagnostic_interval,
        seed=legacy_spec.seed,
        num_defect_pairs=legacy_spec.num_defect_pairs,
        defect_min_separation=legacy_spec.defect_min_separation,
        defect_core_radius=legacy_spec.defect_core_radius,
        background_angle=legacy_spec.background_angle,
        twist_amplitude=legacy_spec.twist_amplitude,
        twist_modes=legacy_spec.twist_modes,
        ldg_a=legacy_spec.ldg_a,
        ldg_b=legacy_spec.ldg_b,
        ldg_c=legacy_spec.ldg_c,
        gamma=legacy_spec.gamma,
        flow_alignment=legacy_spec.flow_alignment,
        eta=legacy_spec.eta,
        zero_mode_policy=legacy_spec.zero_mode_policy,
        friction_mode_fric=legacy_spec.friction_mode_fric,
        dealias_rule=legacy_spec.dealias_rule,
        projected_transform_execution=(
            legacy_spec.projected_transform_execution
        ),
        beta=legacy_spec.beta,
        initial_s=legacy_spec.initial_s,
        device=legacy_spec.device,
        dtype=legacy_spec.dtype,
        molecular_field_linear_space=(
            legacy_spec.molecular_field_linear_space
        ),
        stress_divergence_sum_space=(
            legacy_spec.stress_divergence_sum_space
        ),
        pointwise_execution=legacy_spec.pointwise_execution,
        transform_execution_order=(
            legacy_spec.transform_execution_order
        ),
        spectral_storage=legacy_spec.spectral_storage,
        tf32=legacy_spec.tf32,
        spectral_refresh=legacy_spec.spectral_refresh,
        diagnostics=legacy_spec.diagnostics,
        disable_q_gradient_reuse=legacy_spec.disable_q_gradient_reuse,
        save_hydrodynamics=legacy_spec.save_hydrodynamics,
        validation_config_sha256=legacy_spec.validation_config_sha256,
        dry_run=legacy_spec.dry_run,
        checkpoint_interval=legacy_spec.checkpoint_interval,
        restart_from=legacy_spec.restart_from,
        runtime_path=legacy_spec.runtime_path,
        boundaries=legacy_spec.boundaries,
        hermitian_axis=PLANE_HERMITIAN_AXIS,
    )


__all__ = [
    "decompose_plane_beris_edwards_run_spec",
    "PlaneBerisEdwardsPhysicsSpec",
    "PlaneBerisEdwardsRunComponents",
    "PlaneBerisEdwardsExecutionSpec",
    "PlaneInvocationSpec",
    "PlaneTimeSteppingSpec",
    "PlaneWorkflowSpec",
]
