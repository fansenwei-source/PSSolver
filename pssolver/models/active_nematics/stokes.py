"""Importable complete-stress Beris--Edwards free-slip Stokes adapter."""

import math

import torch

from pssolver.transforms import (
    FreeSlipModalStokesSolver,
    projected_common_basis_stress_divergence,
    projected_distortion_stress_divergence,
)

from .beris_edwards import (
    BerisEdwardsQGradientCache,
    beris_edwards_algebraic_stress_components,
    beris_edwards_distortion_stress_components,
    beris_edwards_molecular_field_components,
)
from .fields import Q_COMPONENTS


PLANE_Q_BOUNDARY_CONDITIONS = ("periodic", "periodic", "neumann")
PLANE_TANGENTIAL_VELOCITY_BOUNDARY_CONDITIONS = (
    "periodic",
    "periodic",
    "neumann",
)
PLANE_NORMAL_VELOCITY_BOUNDARY_CONDITIONS = (
    "periodic",
    "periodic",
    "dirichlet",
)
PLANE_PRESSURE_BOUNDARY_CONDITIONS = ("periodic", "periodic", "neumann")
PLANE_DISTORTION_ODD_BOUNDARY_CONDITIONS = (
    "periodic",
    "periodic",
    "dirichlet",
)


class BerisEdwardsFreeSlipStokes(FreeSlipModalStokesSolver):
    """Complete one-constant nematic force coupled to modal free-slip flow.

    The adapter evaluates the raw molecular field, reactive plus active
    algebraic stress, and distortion stress before applying the row-wise
    divergence.  It then projects the complete force into the native velocity
    spaces and uses :class:`FreeSlipModalStokesSolver` for the saddle solve.

    ``zero_mean`` with zero friction removes the two uniform tangential force
    modes and fixes the plug-flow reference frame.  ``friction`` with positive
    drag retains and determines those modes.  This distinction is a modeling
    choice, not a pressure gauge.

    Disabling pressure diagnostics skips only residual measurements and their
    host synchronizations. The computed pressure and velocity are unchanged.
    """

    def __init__(
        self,
        solver,
        spectral_projector,
        beta_value=-1.0,
        friction=0.0,
        viscosity=2.0 / 3.0,
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=0.02,
        flow_alignment=0.3,
        cache_force_diagnostics=False,
        cache_pressure_diagnostics=True,
        q_gradient_cache=None,
        zero_mode_policy="zero_mean",
        q_boundary_conditions=PLANE_Q_BOUNDARY_CONDITIONS,
        tangential_velocity_boundary_conditions=(
            PLANE_TANGENTIAL_VELOCITY_BOUNDARY_CONDITIONS
        ),
        normal_velocity_boundary_conditions=(
            PLANE_NORMAL_VELOCITY_BOUNDARY_CONDITIONS
        ),
        pressure_boundary_conditions=PLANE_PRESSURE_BOUNDARY_CONDITIONS,
        distortion_odd_boundary_conditions=(
            PLANE_DISTORTION_ODD_BOUNDARY_CONDITIONS
        ),
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
            raise ValueError("Stokes and nematic coefficients must be finite.")
        if ldg_l1 <= 0:
            raise ValueError("The one-constant L1 coefficient must be positive.")

        q_bcs = tuple(q_boundary_conditions)
        tangential_bcs = tuple(tangential_velocity_boundary_conditions)
        normal_bcs = tuple(normal_velocity_boundary_conditions)
        pressure_bcs = tuple(pressure_boundary_conditions)
        distortion_odd_bcs = tuple(distortion_odd_boundary_conditions)
        if q_bcs != tangential_bcs or q_bcs != pressure_bcs:
            raise ValueError(
                "This adapter requires Q, tangential velocity, and pressure "
                "to share periodic/periodic/Neumann parity."
            )
        if distortion_odd_bcs != normal_bcs:
            raise ValueError(
                "Odd distortion stress and normal velocity must share "
                "periodic/periodic/Dirichlet parity."
            )

        super().__init__(
            solver.transform_backend,
            tangential_boundary_conditions=tangential_bcs,
            normal_boundary_conditions=normal_bcs,
            pressure_boundary_conditions=pressure_bcs,
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
        self.cache_force_diagnostics = bool(cache_force_diagnostics)
        if q_gradient_cache is not None and not isinstance(
            q_gradient_cache,
            BerisEdwardsQGradientCache,
        ):
            raise TypeError(
                "q_gradient_cache must be a BerisEdwardsQGradientCache or None."
            )
        self.q_gradient_cache = q_gradient_cache
        self.spectral_projector = spectral_projector
        self.q_boundary_conditions = q_bcs
        self.distortion_odd_boundary_conditions = distortion_odd_bcs
        self.last_tangential_force_mean = None
        self.last_total_tangential_force_mean = None
        self.last_active_tangential_force_mean = None
        self.last_passive_tangential_force_mean = None
        self.last_removed_tangential_force_mean = None
        self.last_projected_normal_force = None

    def _project_physical_tensor(self, fields, tensor, boundary_conditions):
        spectral = fields.transform_tensor(tensor, boundary_conditions)
        spectral = self.spectral_projector.project(
            spectral,
            boundary_conditions,
        )
        return fields.inverse_transform_tensor(
            spectral,
            boundary_conditions,
        )

    def compute_nematic_force(self, fields, alpha):
        """Return the projected complete force and active tangential part."""
        if self.q_gradient_cache is not None:
            self.q_gradient_cache.clear()
        q_components = tuple(fields[name] for name in Q_COMPONENTS)
        laplacian_components = tuple(
            fields.laplacian(name) for name in Q_COMPONENTS
        )

        # Stress uses raw H, not H/gamma.  Project H as one resolved field,
        # then project the complete reactive stress rather than Q:H alone; this
        # preserves the discrete reactive/alignment energy exchange.
        raw_h_components = beris_edwards_molecular_field_components(
            q_components,
            laplacian_components,
            ldg_a=self.ldg_a,
            ldg_b=self.ldg_b,
            ldg_c=self.ldg_c,
            ldg_l1=self.ldg_l1,
        )
        h_tensor = self._project_physical_tensor(
            fields,
            torch.stack(raw_h_components),
            self.q_boundary_conditions,
        )
        h_components = tuple(h_tensor[index] for index in range(5))
        active_prefactor = self.beta * alpha
        algebraic_stress = beris_edwards_algebraic_stress_components(
            q_components,
            h_components,
            flow_alignment=self.flow_alignment,
            active_prefactor=active_prefactor,
        )
        algebraic_force = projected_common_basis_stress_divergence(
            self.transform_backend,
            algebraic_stress,
            self.q_boundary_conditions,
            projector=self.spectral_projector,
        )
        del (
            algebraic_stress,
            h_components,
            h_tensor,
            laplacian_components,
            raw_h_components,
        )

        q_gradients = tuple(
            tuple(
                fields.gradient(name, axis=axis) for name in Q_COMPONENTS
            )
            for axis in range(3)
        )
        distortion_stress = beris_edwards_distortion_stress_components(
            q_gradients,
            ldg_l1=self.ldg_l1,
        )
        distortion_force = projected_distortion_stress_divergence(
            self.transform_backend,
            distortion_stress,
            self.q_boundary_conditions,
            self.distortion_odd_boundary_conditions,
            projector=self.spectral_projector,
        )
        total_force = algebraic_force + distortion_force
        del distortion_stress

        gradient_x, gradient_y, gradient_z = q_gradients
        active_tangential_force = active_prefactor * torch.stack(
            (
                gradient_x[0] + gradient_y[1] + gradient_z[2],
                gradient_x[1] + gradient_y[3] + gradient_z[4],
            )
        )
        if self.q_gradient_cache is not None:
            self.q_gradient_cache.stage(fields, q_gradients)
        return total_force, active_tangential_force

    def after_static_fields_updated(self, fields):
        """Publish staged Q gradients after static field synchronization."""
        if self.q_gradient_cache is not None:
            self.q_gradient_cache.publish(fields)

    # Preserve the private name used by older callers of the benchmark class.
    def _compute_nematic_force(self, fields, alpha):
        return self.compute_nematic_force(fields, alpha)

    def forward(self, fields, params):
        alpha = params["alpha"]
        force, active_tangential_force = self.compute_nematic_force(
            fields,
            alpha,
        )

        total_mean = force[:2].mean(dim=(-3, -2, -1)).detach()
        active_mean = active_tangential_force.mean(
            dim=(-3, -2, -1)
        ).detach()
        self.last_total_tangential_force_mean = total_mean
        self.last_tangential_force_mean = total_mean
        self.last_active_tangential_force_mean = active_mean
        self.last_passive_tangential_force_mean = total_mean - active_mean
        self.last_removed_tangential_force_mean = (
            total_mean
            if self.has_tangential_null_mode
            else torch.zeros_like(total_mean)
        )

        # Project the complete force before the coupled saddle solve.
        force_tangential_hat = self.spectral_projector.project(
            fields.transform_tensor(
                force[:2],
                self.tangential_boundary_conditions,
            ),
            self.tangential_boundary_conditions,
        )
        force_normal_hat = self.spectral_projector.project(
            fields.transform_tensor(
                force[2],
                self.normal_boundary_conditions,
            ),
            self.normal_boundary_conditions,
        )
        if self.cache_force_diagnostics:
            self.last_projected_normal_force = (
                fields.inverse_transform_tensor(
                    force_normal_hat,
                    self.normal_boundary_conditions,
                ).detach()
            )
        else:
            self.last_projected_normal_force = None
        del active_tangential_force, force

        fx_hat, fy_hat = force_tangential_hat
        ux_hat, uy_hat, uz_hat, pressure_hat = self.solve_force_hats(
            fx_hat,
            fy_hat,
            force_normal_hat,
        )
        return torch.stack((ux_hat, uy_hat, uz_hat, pressure_hat))
