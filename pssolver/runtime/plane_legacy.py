"""Package-owned legacy production assembly for Plane Beris--Edwards runs.

This module is the Stage R boundary between the stable Plane runtime factory
and the legacy spectral implementation.  It intentionally preserves the
validated numerical algorithm while removing solver construction from the
top-level command-line script.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from pssolver.configuration import (
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneBerisEdwardsRunSpec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsPointwiseKernels,
    BerisEdwardsQGradientCache,
    BerisEdwardsQNonlinearModel,
    beris_edwards_linear_operator,
)
from pssolver.solver import SpectralSolver
from pssolver.operators.projection import BasisAwareSpectralProjector


_LEGACY_BOUNDARIES = PLANE_FREE_SLIP_BOUNDARIES.to_legacy()
Q_BOUNDARIES = _LEGACY_BOUNDARIES["q"]
TANGENTIAL_VELOCITY_BOUNDARIES = _LEGACY_BOUNDARIES[
    "tangential_velocity"
]
NORMAL_VELOCITY_BOUNDARIES = _LEGACY_BOUNDARIES["normal_velocity"]
PRESSURE_MODAL_BOUNDARIES = _LEGACY_BOUNDARIES["pressure_modal"]


class DealiasedSemiImplicitEulerIntegrator(SemiImplicitEulerIntegrator):
    """Apply the projected-mode contract around each legacy IMEX update."""

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        self.spectral_projector = model.spectral_projector

    def _refresh_dynamic_spectra(self):
        self.spectral_projector.refresh_dynamic_fields(
            self.model.fields,
            sync_spatial=True,
        )

    def step(self, pre_update_callback=None):
        if self._static_fields_are_current:
            self._static_fields_are_current = False
        else:
            self.model.update_static_fields()

        if pre_update_callback is not None:
            pre_update_callback()

        nonlinear_hats = self.model.compute_nonlinear()
        dynamic_fields = self.model.fields.spectral[: self.dyn_count]
        dynamic_fields.add_(self.dt * nonlinear_hats)
        dynamic_fields.div_(self.denom)
        self.spectral_projector.project_dynamic_fields(
            self.model.fields,
            sync_spatial=False,
        )

        for group in self.dynamic_transform_groups:
            boundary_conditions = self.model.fields.get_boundary_conditions(
                group[0]
            )
            self.model.fields.store_spatial_group(
                group,
                self.spectral_projector.inverse_transform(
                    self.model.fields.select_spectral_group(group),
                    boundary_conditions,
                ),
            )

        self._advance_spectral_refresh_clock()


def _real_dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as exc:
        raise ValueError("Plane runtime dtype must be float32 or float64") from exc


def build_legacy_plane_runtime(
    run_spec: PlaneBerisEdwardsRunSpec,
    *,
    device: object,
    initial_values: Mapping[str, object],
) -> tuple[SpectralSolver, BasisAwareSpectralProjector]:
    """Construct the validated Plane production backend.

    Parameters
    ----------
    run_spec
        Fully resolved immutable Plane configuration.
    device
        Torch-compatible execution device selected by the application edge.
    initial_values
        Initial values for the five independent Q components.

    Returns
    -------
    solver, projector
        The legacy production solver and its projected-transform authority.

    Notes
    -----
    Stage R changes ownership only.  The equations, operation order,
    transforms, boundary conditions, zero-mode rule, and time integrator are
    the same as the previously in-script production assembly.
    """

    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PlaneBerisEdwardsRunSpec")
    if not isinstance(initial_values, Mapping):
        raise TypeError("initial_values must be a mapping")

    components = decompose_plane_beris_edwards_run_spec(run_spec)
    domain = components.geometry.domain
    numerics = components.numerics
    material = components.physics.material
    stokes_request = components.physics.stokes
    preset = components.preset
    time_stepping = components.time_stepping
    execution = components.execution
    workflow = components.workflow
    boundaries = components.effective_boundaries.to_legacy()
    q_boundaries = boundaries["q"]
    tangential_velocity_boundaries = boundaries["tangential_velocity"]
    normal_velocity_boundaries = boundaries["normal_velocity"]
    pressure_modal_boundaries = boundaries["pressure_modal"]

    real_dtype = _real_dtype(numerics.precision.value)
    pointwise_kernels = BerisEdwardsPointwiseKernels(
        execution.pointwise_execution
    )

    solver = SpectralSolver(
        shape=domain.shape,
        L=domain.lengths,
        dt=time_stepping.dt,
        device=device,
        batchsize=1,
        dtype=real_dtype,
        transform_execution_order=(
            numerics.transform_execution_order.value
        ),
        spectral_storage=numerics.spectral_storage.value,
        hermitian_axis=components.geometry.periodic_axes[-1],
    )
    spectral_projector = BasisAwareSpectralProjector(
        solver,
        rule=numerics.dealias_rule.value,
        transform_execution=(
            numerics.projected_transform_execution.value
        ),
    )
    solver.model.spectral_projector = spectral_projector
    solver.model.set_static_inverse_transform(
        spectral_projector.inverse_transform
    )
    solver.integrator_cl = DealiasedSemiImplicitEulerIntegrator
    q2_q = solver.get_q2(q_boundaries)
    q_linear_operator = beris_edwards_linear_operator(
        q2_q,
        ldg_a=material.ldg_a,
        ldg_l1=preset.ldg_l1,
        rotational_viscosity=preset.rotational_viscosity,
    )
    for name, initial_value in initial_values.items():
        solver.model.add_dynamic_field(
            name,
            init=initial_value,
            L_hat=q_linear_operator,
            boundary_conditions=q_boundaries,
        )
    solver.model.add_static_field(
        "ux", boundary_conditions=tangential_velocity_boundaries
    )
    solver.model.add_static_field(
        "uy", boundary_conditions=tangential_velocity_boundaries
    )
    solver.model.add_static_field(
        "uz", boundary_conditions=normal_velocity_boundaries
    )
    solver.model.add_static_field(
        "p", boundary_conditions=pressure_modal_boundaries
    )

    q_gradient_cache = (
        None
        if execution.disable_q_gradient_reuse
        else BerisEdwardsQGradientCache()
    )
    solver.model.set_nonlinear_model(
        BerisEdwardsQNonlinearModel(
            spectral_projector,
            q_boundaries,
            ldg_b=material.ldg_b,
            ldg_c=material.ldg_c,
            rotational_viscosity=preset.rotational_viscosity,
            flow_alignment=material.flow_alignment,
            q_gradient_cache=q_gradient_cache,
            pointwise_kernels=pointwise_kernels,
        )
    )
    solver.model.set_static_compute_model(
        BerisEdwardsFreeSlipStokes(
            solver,
            spectral_projector=spectral_projector,
            beta_value=material.beta,
            friction=stokes_request.friction,
            viscosity=stokes_request.viscosity,
            ldg_a=material.ldg_a,
            ldg_b=material.ldg_b,
            ldg_c=material.ldg_c,
            ldg_l1=preset.ldg_l1,
            flow_alignment=material.flow_alignment,
            molecular_field_linear_space=(
                execution.molecular_field_linear_space
            ),
            stress_divergence_sum_space=(
                execution.stress_divergence_sum_space
            ),
            cache_force_diagnostics=workflow.diagnostics,
            cache_pressure_diagnostics=workflow.diagnostics,
            q_gradient_cache=q_gradient_cache,
            pointwise_kernels=pointwise_kernels,
            zero_mode_policy=(
                stokes_request.tangential_zero_mode_policy.value
            ),
        )
    )
    alpha = torch.tensor(
        preset.zeta,
        device=device,
        dtype=real_dtype,
    )
    solver.model.parameters.new_param("alpha", alpha)
    solver.build()
    solver.integrator.set_spectral_refresh_interval(
        time_stepping.spectral_refresh.effective_interval_steps
    )
    spectral_projector.project_dynamic_fields(
        solver.model.fields,
        sync_spatial=True,
    )
    if spectral_projector.enabled:
        solver.integrator._static_fields_are_current = False
    return solver, spectral_projector


__all__ = [
    "DealiasedSemiImplicitEulerIntegrator",
    "NORMAL_VELOCITY_BOUNDARIES",
    "PRESSURE_MODAL_BOUNDARIES",
    "Q_BOUNDARIES",
    "TANGENTIAL_VELOCITY_BOUNDARIES",
    "build_legacy_plane_runtime",
]
