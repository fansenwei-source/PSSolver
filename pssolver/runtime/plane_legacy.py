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
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsPointwiseKernels,
    BerisEdwardsQGradientCache,
    BerisEdwardsQNonlinearModel,
    beris_edwards_linear_operator,
)
from pssolver.plane import PLANE_HERMITIAN_AXIS
from pssolver.solver import SpectralSolver
from pssolver.transforms import BasisAwareSpectralProjector


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
            self.model.fields.spatial[group] = (
                self.spectral_projector.inverse_transform(
                    self.model.fields.spectral[group],
                    boundary_conditions,
                )
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

    real_dtype = _real_dtype(run_spec.dtype)
    preset = run_spec.shendruk_preset
    friction = (
        0.0
        if run_spec.zero_mode_policy == "zero_mean"
        else float(run_spec.friction_mode_fric)
    )
    pointwise_kernels = BerisEdwardsPointwiseKernels(
        run_spec.pointwise_execution
    )

    solver = SpectralSolver(
        shape=(run_spec.nx, run_spec.ny, run_spec.nz),
        L=(run_spec.lx, run_spec.ly, run_spec.height),
        dt=run_spec.dt,
        device=device,
        batchsize=1,
        dtype=real_dtype,
        transform_execution_order=run_spec.transform_execution_order,
        spectral_storage=run_spec.spectral_storage,
        hermitian_axis=PLANE_HERMITIAN_AXIS,
    )
    spectral_projector = BasisAwareSpectralProjector(
        solver,
        rule=run_spec.dealias_rule,
        transform_execution=run_spec.projected_transform_execution,
    )
    solver.model.spectral_projector = spectral_projector
    solver.model.set_static_inverse_transform(
        spectral_projector.inverse_transform
    )
    solver.integrator_cl = DealiasedSemiImplicitEulerIntegrator
    q2_q = solver.get_q2(Q_BOUNDARIES)
    q_linear_operator = beris_edwards_linear_operator(
        q2_q,
        ldg_a=run_spec.ldg_a,
        ldg_l1=preset.ldg_l1,
        rotational_viscosity=preset.rotational_viscosity,
    )
    for name, initial_value in initial_values.items():
        solver.model.add_dynamic_field(
            name,
            init=initial_value,
            L_hat=q_linear_operator,
            boundary_conditions=Q_BOUNDARIES,
        )
    solver.model.add_static_field(
        "ux", boundary_conditions=TANGENTIAL_VELOCITY_BOUNDARIES
    )
    solver.model.add_static_field(
        "uy", boundary_conditions=TANGENTIAL_VELOCITY_BOUNDARIES
    )
    solver.model.add_static_field(
        "uz", boundary_conditions=NORMAL_VELOCITY_BOUNDARIES
    )
    solver.model.add_static_field(
        "p", boundary_conditions=PRESSURE_MODAL_BOUNDARIES
    )

    q_gradient_cache = (
        None
        if run_spec.disable_q_gradient_reuse
        else BerisEdwardsQGradientCache()
    )
    solver.model.set_nonlinear_model(
        BerisEdwardsQNonlinearModel(
            spectral_projector,
            Q_BOUNDARIES,
            ldg_b=run_spec.ldg_b,
            ldg_c=run_spec.ldg_c,
            rotational_viscosity=preset.rotational_viscosity,
            flow_alignment=run_spec.flow_alignment,
            q_gradient_cache=q_gradient_cache,
            pointwise_kernels=pointwise_kernels,
        )
    )
    solver.model.set_static_compute_model(
        BerisEdwardsFreeSlipStokes(
            solver,
            spectral_projector=spectral_projector,
            beta_value=run_spec.beta,
            friction=friction,
            viscosity=run_spec.eta,
            ldg_a=run_spec.ldg_a,
            ldg_b=run_spec.ldg_b,
            ldg_c=run_spec.ldg_c,
            ldg_l1=preset.ldg_l1,
            flow_alignment=run_spec.flow_alignment,
            molecular_field_linear_space=(
                run_spec.molecular_field_linear_space
            ),
            stress_divergence_sum_space=(
                run_spec.stress_divergence_sum_space
            ),
            cache_force_diagnostics=run_spec.diagnostics,
            cache_pressure_diagnostics=run_spec.diagnostics,
            q_gradient_cache=q_gradient_cache,
            pointwise_kernels=pointwise_kernels,
            zero_mode_policy=run_spec.zero_mode_policy,
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
        run_spec.spectral_refresh_interval_steps
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
