"""Direct fully periodic incompressible Stokes--Brinkman solver."""

from __future__ import annotations

import math

import torch


class PeriodicModalStokesSolver(torch.nn.Module):
    """Solve the constant-coefficient periodic Stokes saddle system.

    The pressure gauge and uniform-velocity policy are deliberately separate.
    Pressure always uses the zero-mean gauge.  ``zero_mean`` removes the
    complete all-zero-wave-number velocity/force mode when friction is zero;
    ``friction`` retains that mode and resolves it with positive drag.
    """

    def __init__(
        self,
        backend,
        *,
        boundary_conditions=("periodic", "periodic", "periodic"),
        friction=0.0,
        viscosity=1.0,
        zero_mode_policy="zero_mean",
        pressure_diagnostics=True,
    ):
        # Call Module directly so this solver remains safe as the leading base
        # of the complete-stress periodic adapter's multiple-inheritance MRO.
        torch.nn.Module.__init__(self)
        if backend.dim != 3:
            raise ValueError("Periodic Stokes solve requires three dimensions.")
        if not math.isfinite(float(viscosity)) or viscosity <= 0:
            raise ValueError("Periodic Stokes solve requires finite viscosity > 0.")
        if not math.isfinite(float(friction)) or friction < 0:
            raise ValueError("Periodic Stokes solve requires finite friction >= 0.")
        if zero_mode_policy not in {"zero_mean", "friction"}:
            raise ValueError("zero_mode_policy must be 'zero_mean' or 'friction'.")
        if zero_mode_policy == "zero_mean" and friction != 0:
            raise ValueError("zero_mean mode requires friction == 0.")
        if zero_mode_policy == "friction" and friction <= 0:
            raise ValueError("friction mode requires friction > 0.")
        if not isinstance(pressure_diagnostics, bool):
            raise TypeError("pressure_diagnostics must be a bool.")

        bcs = tuple(boundary_conditions)
        if bcs != ("periodic", "periodic", "periodic"):
            raise ValueError("Periodic Stokes fields require periodic boundaries.")

        metadata = backend.get_metadata(bcs)
        wave_numbers = torch.meshgrid(*metadata.axis_modes, indexing="ij")
        k_components = tuple(
            value.to(dtype=backend.spectral_dtype) for value in wave_numbers
        )
        k2 = sum(value.square() for value in wave_numbers)
        uniform_mask = k2 == 0
        helmholtz = float(friction) + float(viscosity) * k2
        helmholtz_null_mask = helmholtz == 0
        helmholtz_safe = helmholtz.masked_fill(helmholtz_null_mask, 1.0)
        helmholtz_inverse = helmholtz_safe.reciprocal()
        if zero_mode_policy == "zero_mean":
            helmholtz_inverse.masked_fill_(helmholtz_null_mask, 0.0)
        pressure_k2_safe = k2.masked_fill(uniform_mask, 1.0)

        self.transform_backend = backend
        self.boundary_conditions = bcs
        # The common complete-stress adapter uses these names to transform the
        # force.  In a periodic box all components share the same FFT basis.
        self.tangential_boundary_conditions = bcs
        self.normal_boundary_conditions = bcs
        self.pressure_boundary_conditions = bcs
        self.friction = float(friction)
        self.viscosity = float(viscosity)
        self.zero_mode_policy = zero_mode_policy
        self.pressure_diagnostics = pressure_diagnostics
        self.register_buffer(
            "ik",
            torch.stack(tuple(1j * value for value in k_components)),
        )
        self.register_buffer("k2_safe", pressure_k2_safe)
        self.register_buffer("helmholtz_inverse", helmholtz_inverse)
        self.register_buffer("uniform_mask", uniform_mask)
        self.has_tangential_null_mode = bool(
            zero_mode_policy == "zero_mean" and uniform_mask.any().item()
        )
        self.last_pressure_hat = None
        self.last_pressure_iterations = 0
        self.last_pressure_residual = 0.0
        self.last_pressure_relative_residual = 0.0

    def _project_pressure_gauge(self, pressure_hat):
        extra = pressure_hat.ndim - self.uniform_mask.ndim
        mask = self.uniform_mask.reshape(
            *((1,) * extra), *self.uniform_mask.shape
        )
        return pressure_hat.masked_fill(mask, 0)

    def _ik_for(self, vector):
        extra = vector.ndim - self.ik.ndim
        return self.ik.reshape(
            self.ik.shape[0],
            *((1,) * extra),
            *self.ik.shape[1:],
        )

    def _scalar_for(self, coefficient, scalar):
        extra = scalar.ndim - coefficient.ndim
        return coefficient.reshape(*((1,) * extra), *coefficient.shape)

    def divergence_hat(self, ux_hat, uy_hat, uz_hat):
        velocity = torch.stack((ux_hat, uy_hat, uz_hat))
        return (self._ik_for(velocity) * velocity).sum(dim=0)

    def pressure_gradient_hats(self, pressure_hat):
        pressure_hat = self._project_pressure_gauge(pressure_hat)
        ik = self._ik_for(pressure_hat.unsqueeze(0))
        return tuple(ik[index] * pressure_hat for index in range(3))

    def solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        """Return divergence-free velocity and zero-mean pressure spectra."""

        force = torch.stack((fx_hat, fy_hat, fz_hat))
        if force.shape[-3:] != self.helmholtz_inverse.shape:
            raise ValueError("periodic force spectra have an incompatible shape")
        ik = self._ik_for(force)
        uniform_mask = self._scalar_for(self.uniform_mask, fx_hat)
        inverse = self._scalar_for(self.helmholtz_inverse, fx_hat)
        k2_safe = self._scalar_for(self.k2_safe, fx_hat)
        if self.zero_mode_policy == "zero_mean":
            force = force.masked_fill(uniform_mask.unsqueeze(0), 0)
        divergence_force = (ik * force).sum(dim=0)
        pressure_hat = self._project_pressure_gauge(
            -divergence_force / k2_safe
        )
        gradient = ik * pressure_hat.unsqueeze(0)
        velocity = (force - gradient) * inverse.unsqueeze(0)

        self.last_pressure_hat = pressure_hat.detach()
        self.last_pressure_iterations = 1
        if self.pressure_diagnostics:
            residual = self.divergence_hat(*velocity)
            residual_norm = torch.linalg.vector_norm(residual.reshape(-1)).item()
            force_norm = torch.linalg.vector_norm(force.reshape(-1)).item()
            self.last_pressure_residual = residual_norm
            self.last_pressure_relative_residual = (
                residual_norm / force_norm if force_norm else 0.0
            )
        else:
            self.last_pressure_residual = math.nan
            self.last_pressure_relative_residual = math.nan
        return (*tuple(velocity), pressure_hat)

    def _solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        return self.solve_force_hats(fx_hat, fy_hat, fz_hat)


__all__ = ["PeriodicModalStokesSolver"]
