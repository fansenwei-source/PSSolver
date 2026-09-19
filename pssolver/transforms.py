import math

import torch

from .backends.bounded import (
    BoundedAxisPlanKey,
    DenseBoundedAxisExecutionPlan,
    build_dense_orthonormal_matrix,
)
from .backends.tensor_product import (
    DEFAULT_PERIODIC_TRANSFORM_EXECUTION,
    DEFAULT_SPECTRAL_STORAGE,
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    PERIODIC_TRANSFORM_EXECUTION_MODES,
    SPECTRAL_STORAGE_MODES,
    TensorProductTransformBackend,
    TransformMetadata,
)
from .operators.projection import (
    DEALIAS_RULE_FRACTIONS,
    DEFAULT_DEALIAS_RULE,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
    PROJECTED_TRANSFORM_EXECUTION_MODES,
    BasisAwareSpectralProjector,
)
from .operators.tensor_divergence import (
    projected_common_basis_stress_divergence,
    projected_distortion_stress_divergence,
)


class FreeSlipModalStokesSolver(torch.nn.Module):
    """Mixed DCT/DST Stokes--Brinkman saddle solver for z-normal walls.

    Pressure residual diagnostics are enabled by default for backward
    compatibility. Production callers may disable them to avoid an additional
    residual operator and GPU-to-host scalar synchronizations; this does not
    change the pressure or velocity solution.
    """

    def __init__(
        self,
        backend,
        *,
        tangential_boundary_conditions,
        normal_boundary_conditions,
        pressure_boundary_conditions,
        friction=0.0,
        viscosity=1.0,
        zero_mode_policy="zero_mean",
        pressure_diagnostics=True,
    ):
        super().__init__()
        if backend.dim != 3:
            raise ValueError("Free-slip modal Stokes solve requires three dimensions.")
        if not math.isfinite(float(viscosity)) or viscosity <= 0:
            raise ValueError("Free-slip Stokes solve requires finite viscosity > 0.")
        if not math.isfinite(float(friction)) or friction < 0:
            raise ValueError("Free-slip Stokes solve requires finite friction >= 0.")
        if zero_mode_policy not in ("zero_mean", "friction"):
            raise ValueError(
                "zero_mode_policy must be 'zero_mean' or 'friction'."
            )
        if not isinstance(pressure_diagnostics, bool):
            raise TypeError("pressure_diagnostics must be a bool.")
        if zero_mode_policy == "zero_mean" and friction != 0:
            raise ValueError("zero_mean mode requires friction == 0.")
        if zero_mode_policy == "friction" and friction <= 0:
            raise ValueError("friction mode requires friction > 0.")

        tangential_bcs = tuple(tangential_boundary_conditions)
        normal_bcs = tuple(normal_boundary_conditions)
        pressure_bcs = tuple(pressure_boundary_conditions)
        if any(len(bcs) != 3 for bcs in (tangential_bcs, normal_bcs, pressure_bcs)):
            raise ValueError("Each Stokes field requires three boundary conditions.")
        if tangential_bcs[:2] != ("periodic", "periodic"):
            raise ValueError("Tangential velocity must be periodic in x and y.")
        if pressure_bcs[:2] != ("periodic", "periodic"):
            raise ValueError("Pressure must be periodic in x and y.")
        if normal_bcs[:2] != ("periodic", "periodic"):
            raise ValueError("Normal velocity must be periodic in x and y.")
        if tangential_bcs[2] != "neumann":
            raise ValueError("Tangential velocity requires Neumann z parity.")
        if pressure_bcs[2] != "neumann":
            raise ValueError("Pressure requires Neumann z parity.")
        if normal_bcs[2] != "dirichlet":
            raise ValueError("Normal velocity requires Dirichlet z parity.")

        self.transform_backend = backend
        self.tangential_boundary_conditions = tangential_bcs
        self.normal_boundary_conditions = normal_bcs
        self.pressure_boundary_conditions = pressure_bcs
        self.friction = float(friction)
        self.viscosity = float(viscosity)
        self.zero_mode_policy = zero_mode_policy
        self.pressure_diagnostics = pressure_diagnostics

        tangential_metadata = backend.get_metadata(tangential_bcs)
        normal_metadata = backend.get_metadata(normal_bcs)
        pressure_metadata = backend.get_metadata(pressure_bcs)
        spectral_dtype = backend.spectral_dtype
        real_dtype = backend.real_dtype
        device = backend.device

        qx_xy, qy_xy = torch.meshgrid(
            pressure_metadata.axis_modes[0],
            pressure_metadata.axis_modes[1],
            indexing="ij",
        )
        kz_tangential = tangential_metadata.axis_modes[2]
        kz_normal = normal_metadata.axis_modes[2]
        kz_pressure = pressure_metadata.axis_modes[2]
        nz = kz_normal.numel()

        # Row-vector coefficient convention. The terminal DST mode maps to
        # unavailable DCT mode m=N and is absent from this derivative pair.
        dz_dirichlet_to_neumann = torch.zeros(
            (nz, nz), device=device, dtype=real_dtype
        )
        if nz > 1:
            index = torch.arange(nz - 1, device=device)
            dz_dirichlet_to_neumann[index, index + 1] = kz_normal[:-1]
        dz_neumann_to_dirichlet = -dz_dirichlet_to_neumann.transpose(0, 1)

        kxy2 = qx_xy.square() + qy_xy.square()
        a_tangential = self.friction + self.viscosity * (
            kxy2.unsqueeze(-1)
            + kz_tangential.square().view(1, 1, -1)
        )
        a_normal = self.friction + self.viscosity * (
            kxy2.unsqueeze(-1) + kz_normal.square().view(1, 1, -1)
        )
        tangential_null_mask = a_tangential == 0
        a_tangential_safe = a_tangential.masked_fill(
            tangential_null_mask, 1.0
        )
        a_tangential_inv = 1.0 / a_tangential_safe
        a_tangential_inv.masked_fill_(tangential_null_mask, 0.0)
        a_normal_inv = 1.0 / a_normal

        # S = -D A^{-1} G. Pressure cosine m>=1 maps to sine slot m-1.
        schur_diag = kxy2.unsqueeze(-1) * a_tangential_inv
        if nz > 1:
            schur_diag[..., 1:] += (
                kz_pressure[1:].square().view(1, 1, -1)
                * a_normal_inv[..., :-1]
            )

        pressure_null_mask = torch.zeros(
            (1, *qx_xy.shape, nz), device=device, dtype=torch.bool
        )
        pressure_null_mask[:, 0, 0, 0] = True
        schur_diag_safe = schur_diag.unsqueeze(0).clone()
        schur_diag_safe.masked_fill_(pressure_null_mask, 1.0)
        unresolved = (schur_diag_safe == 0) & ~pressure_null_mask
        if unresolved.any():
            raise ValueError(
                "Free-slip pressure Schur complement contains a non-gauge "
                "null mode."
            )

        self.register_buffer(
            "ikx",
            (1j * qx_xy).view(1, *qx_xy.shape, 1).to(dtype=spectral_dtype),
        )
        self.register_buffer(
            "iky",
            (1j * qy_xy).view(1, *qy_xy.shape, 1).to(dtype=spectral_dtype),
        )
        self.register_buffer(
            "a_tangential_inv", a_tangential_inv.unsqueeze(0)
        )
        self.register_buffer("a_normal_inv", a_normal_inv.unsqueeze(0))
        self.register_buffer(
            "tangential_null_mask", tangential_null_mask.unsqueeze(0)
        )
        self.register_buffer(
            "dz_neumann_to_dirichlet",
            dz_neumann_to_dirichlet.to(dtype=spectral_dtype),
        )
        self.register_buffer(
            "dz_dirichlet_to_neumann",
            dz_dirichlet_to_neumann.to(dtype=spectral_dtype),
        )
        self.register_buffer("schur_diag_safe", schur_diag_safe)
        self.register_buffer("pressure_null_mask", pressure_null_mask)

        self.has_tangential_null_mode = bool(tangential_null_mask.any().item())
        self.last_pressure_hat = None
        self.last_pressure_iterations = 0
        self.last_pressure_residual = 0.0
        self.last_pressure_relative_residual = 0.0

    @staticmethod
    def _matmul_lastdim(tensor, matrix):
        return torch.matmul(tensor, matrix)

    def _project_pressure_gauge(self, pressure_hat):
        return pressure_hat.masked_fill(self.pressure_null_mask, 0)

    def _tangential_helmholtz_inverse(self, rhs_hat):
        return (rhs_hat * self.a_tangential_inv).masked_fill(
            self.tangential_null_mask, 0
        )

    def _normal_helmholtz_inverse(self, rhs_hat):
        return rhs_hat * self.a_normal_inv

    def _pressure_grad_z(self, pressure_hat):
        return self._matmul_lastdim(
            pressure_hat, self.dz_neumann_to_dirichlet
        )

    def _velocity_div_z(self, velocity_hat):
        return self._matmul_lastdim(
            velocity_hat, self.dz_dirichlet_to_neumann
        )

    def divergence_hat(self, ux_hat, uy_hat, uz_hat):
        """Return the native pressure-basis divergence coefficients."""
        return (
            self.ikx * ux_hat
            + self.iky * uy_hat
            + self._velocity_div_z(uz_hat)
        )

    def pressure_gradient_hats(self, pressure_hat):
        """Return x/y DCT and z DST coefficients of ``grad(p)``."""
        pressure_hat = self._project_pressure_gauge(pressure_hat)
        return (
            self.ikx * pressure_hat,
            self.iky * pressure_hat,
            self._pressure_grad_z(pressure_hat),
        )

    def _pressure_operator(self, pressure_hat):
        pressure_hat = self._project_pressure_gauge(pressure_hat)
        ux_hat = self._tangential_helmholtz_inverse(self.ikx * pressure_hat)
        uy_hat = self._tangential_helmholtz_inverse(self.iky * pressure_hat)
        uz_hat = self._normal_helmholtz_inverse(
            self._pressure_grad_z(pressure_hat)
        )
        return self._project_pressure_gauge(
            -self.divergence_hat(ux_hat, uy_hat, uz_hat)
        )

    def _solve_pressure(self, rhs_hat):
        rhs_hat = self._project_pressure_gauge(rhs_hat)
        if self.pressure_diagnostics:
            rhs_norm = torch.linalg.vector_norm(rhs_hat.reshape(-1)).item()
            if rhs_norm == 0.0:
                self.last_pressure_iterations = 0
                self.last_pressure_residual = 0.0
                self.last_pressure_relative_residual = 0.0
                return torch.zeros_like(rhs_hat)

        pressure_hat = self._project_pressure_gauge(
            rhs_hat / self.schur_diag_safe
        )
        self.last_pressure_iterations = 1
        if self.pressure_diagnostics:
            residual = rhs_hat - self._pressure_operator(pressure_hat)
            residual_norm = torch.linalg.vector_norm(residual.reshape(-1)).item()
            self.last_pressure_residual = residual_norm
            self.last_pressure_relative_residual = residual_norm / rhs_norm
        else:
            # The diagonal Schur solve itself is unchanged. Only host-synchronizing
            # residual measurements are omitted from the production hot path.
            self.last_pressure_residual = math.nan
            self.last_pressure_relative_residual = math.nan
        return pressure_hat

    def solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        """Solve the mixed-basis saddle system for native force spectra."""
        ux_hat_free = self._tangential_helmholtz_inverse(fx_hat)
        uy_hat_free = self._tangential_helmholtz_inverse(fy_hat)
        uz_hat_free = self._normal_helmholtz_inverse(fz_hat)

        pressure_rhs = self._project_pressure_gauge(
            -self.divergence_hat(ux_hat_free, uy_hat_free, uz_hat_free)
        )
        pressure_hat = self._solve_pressure(pressure_rhs)
        self.last_pressure_hat = pressure_hat.detach()

        ux_hat = ux_hat_free - self._tangential_helmholtz_inverse(
            self.ikx * pressure_hat
        )
        uy_hat = uy_hat_free - self._tangential_helmholtz_inverse(
            self.iky * pressure_hat
        )
        uz_hat = uz_hat_free - self._normal_helmholtz_inverse(
            self._pressure_grad_z(pressure_hat)
        )
        return ux_hat, uy_hat, uz_hat, pressure_hat

    def _solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        """Backward-compatible name used by the simulation script."""
        return self.solve_force_hats(fx_hat, fy_hat, fz_hat)
