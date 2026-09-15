import math
from dataclasses import dataclass

import torch


DEFAULT_DEALIAS_RULE = "cubic_half"
DEFAULT_TRANSFORM_EXECUTION_ORDER = "real_first"
DEALIAS_RULE_FRACTIONS = {
    "none": None,
    "two_thirds": 2.0 / 3.0,
    "cubic_half": 0.5,
}


def _real_dtype(dtype):
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    return torch.float32


def _complex_dtype(dtype):
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    return torch.complex64


@dataclass(frozen=True)
class TransformMetadata:
    boundary_conditions: tuple[str, ...]
    transform_kinds: tuple[str, ...]
    axis_modes: tuple[torch.Tensor, ...]
    q2: torch.Tensor
    q2_safe: torch.Tensor
    laplacian_eigs: torch.Tensor


class TensorProductTransformBackend:
    """Tensor-product spectral transforms on a shared cell-centered grid."""

    _execution_orders = ("legacy", "real_first")

    _transform_kind_map = {
        "periodic": "fft",
        "dirichlet": "dst",
        "neumann": "dct",
    }

    _gradient_bc_map = {
        "periodic": "periodic",
        "dirichlet": "neumann",
        "neumann": "dirichlet",
    }

    def __init__(
        self,
        shape,
        lengths,
        device="cuda",
        dtype=torch.float32,
        execution_order=DEFAULT_TRANSFORM_EXECUTION_ORDER,
    ):
        if execution_order not in self._execution_orders:
            raise ValueError(
                f"execution_order must be one of {self._execution_orders}, "
                f"got {execution_order!r}"
            )
        self.shape = tuple(shape)
        self.lengths = tuple(self._normalize_length(length) for length in lengths)
        self.device = device
        self.dtype = dtype
        self.real_dtype = _real_dtype(dtype)
        self.spectral_dtype = _complex_dtype(dtype)
        self.execution_order = execution_order
        self.dim = len(self.shape)

        self._matrix_cache = {}
        self._metadata_cache = {}

        self.axes = tuple(self._build_axis_grid(axis) for axis in range(self.dim))
        self.spatial_grids = torch.meshgrid(*self.axes, indexing="ij")

    def _normalize_length(self, length):
        if isinstance(length, torch.Tensor):
            return float(length.detach().cpu().item())
        return float(length)

    def _build_axis_grid(self, axis):
        n = self.shape[axis]
        length = self.lengths[axis]
        dx = length / n
        return (torch.arange(n, device=self.device, dtype=self.real_dtype) + 0.5) * dx

    def _broadcast_axis_values(self, values, axis):
        view_shape = [1] * self.dim
        view_shape[axis] = values.shape[0]
        return values.reshape(*view_shape)

    def _get_matrix(self, kind, size):
        key = (kind, size, str(self.device), self.real_dtype)
        if key in self._matrix_cache:
            return self._matrix_cache[key]

        n = torch.arange(size, device=self.device, dtype=self.real_dtype)
        k = n.unsqueeze(1)
        phase = math.pi * (n + 0.5) / size

        if kind == "dct":
            matrix = torch.cos(k * phase)
            matrix[0] *= math.sqrt(1.0 / size)
            if size > 1:
                matrix[1:] *= math.sqrt(2.0 / size)
        elif kind == "dst":
            matrix = math.sqrt(2.0 / size) * torch.sin((k + 1.0) * phase)
            if size > 0:
                matrix[-1] *= math.sqrt(0.5)
        else:
            raise ValueError(f"Unsupported transform kind '{kind}'.")

        self._matrix_cache[key] = matrix
        return matrix

    def _apply_axis_transform(self, tensor, kind, axis, inverse=False):
        if kind == "fft":
            if inverse:
                return torch.fft.ifft(tensor, dim=axis)
            return torch.fft.fft(tensor, dim=axis)

        matrix = self._get_matrix(kind, tensor.shape[axis])
        moved = tensor.movedim(axis, -1)
        matrix = matrix.to(device=moved.device, dtype=moved.dtype)
        if inverse:
            transformed = moved @ matrix
        else:
            transformed = moved @ matrix.transpose(-1, -2)
        return transformed.movedim(-1, axis)

    def _transform_kinds(self, boundary_conditions):
        return tuple(self._transform_kind_map[bc] for bc in boundary_conditions)

    def _ordered_axis_transforms(self, transform_kinds, inverse):
        indexed = list(enumerate(transform_kinds))
        if self.execution_order == "legacy":
            return list(reversed(indexed)) if inverse else indexed

        real_transforms = [item for item in indexed if item[1] != "fft"]
        periodic_transforms = [item for item in indexed if item[1] == "fft"]
        if inverse:
            return list(reversed(periodic_transforms)) + list(
                reversed(real_transforms)
            )
        return real_transforms + periodic_transforms

    def get_metadata(self, boundary_conditions):
        boundary_conditions = tuple(boundary_conditions)
        if boundary_conditions in self._metadata_cache:
            return self._metadata_cache[boundary_conditions]

        axis_modes = []
        q2 = None
        for axis, bc in enumerate(boundary_conditions):
            n = self.shape[axis]
            length = self.lengths[axis]

            if bc == "periodic":
                modes = torch.fft.fftfreq(
                    n,
                    d=length / n,
                    device=self.device,
                    dtype=self.real_dtype,
                ) * (2.0 * math.pi)
            elif bc == "dirichlet":
                modes = (
                    torch.arange(1, n + 1, device=self.device, dtype=self.real_dtype)
                    * (math.pi / length)
                )
            elif bc == "neumann":
                modes = (
                    torch.arange(n, device=self.device, dtype=self.real_dtype)
                    * (math.pi / length)
                )
            else:
                raise ValueError(f"Unsupported boundary condition '{bc}'.")

            axis_modes.append(modes)
            axis_q2 = self._broadcast_axis_values(modes.square(), axis)
            q2 = axis_q2 if q2 is None else q2 + axis_q2

        q2_safe = q2.clone()
        zero_mask = q2_safe == 0
        if zero_mask.any():
            q2_safe[zero_mask] = torch.finfo(self.real_dtype).eps

        metadata = TransformMetadata(
            boundary_conditions=boundary_conditions,
            transform_kinds=self._transform_kinds(boundary_conditions),
            axis_modes=tuple(axis_modes),
            q2=q2,
            q2_safe=q2_safe,
            laplacian_eigs=-q2,
        )
        self._metadata_cache[boundary_conditions] = metadata
        return metadata

    def get_q2(self, boundary_conditions, regularize=False):
        metadata = self.get_metadata(boundary_conditions)
        return metadata.q2_safe if regularize else metadata.q2

    def get_laplacian_eigs(self, boundary_conditions):
        return self.get_metadata(boundary_conditions).laplacian_eigs

    def get_gradient_boundary_conditions(self, boundary_conditions, axis):
        boundary_conditions = tuple(boundary_conditions)
        if axis < 0 or axis >= len(boundary_conditions):
            raise IndexError(f"Axis {axis} is out of range for boundary conditions.")

        derivative_bcs = list(boundary_conditions)
        derivative_bcs[axis] = self._gradient_bc_map[boundary_conditions[axis]]
        return tuple(derivative_bcs)

    def forward(self, tensor, boundary_conditions):
        output = tensor
        metadata = self.get_metadata(boundary_conditions)
        for local_axis, kind in self._ordered_axis_transforms(
            metadata.transform_kinds,
            inverse=False,
        ):
            axis = output.ndim - self.dim + local_axis
            output = self._apply_axis_transform(output, kind, axis, inverse=False)
        return output.to(self.spectral_dtype)

    def inverse(self, spectral, boundary_conditions):
        output = spectral
        metadata = self.get_metadata(boundary_conditions)
        for local_axis, kind in self._ordered_axis_transforms(
            metadata.transform_kinds,
            inverse=True,
        ):
            if (
                self.execution_order == "real_first"
                and kind != "fft"
                and output.is_complex()
            ):
                # Real basis matrices commute with taking the real part. For
                # physical Hermitian spectra the imaginary component is only
                # roundoff; for arbitrary spectra inverse() has always
                # discarded the same component after the real transforms.
                output = output.real
            axis = output.ndim - self.dim + local_axis
            output = self._apply_axis_transform(output, kind, axis, inverse=True)
        return output.real

    def laplacian_hat(self, spectral, boundary_conditions):
        laplacian_eigs = self.get_laplacian_eigs(boundary_conditions)
        return spectral * laplacian_eigs.to(device=spectral.device)

    def gradient_hat(self, spectral, boundary_conditions, axis):
        boundary_conditions = tuple(boundary_conditions)
        metadata = self.get_metadata(boundary_conditions)
        bc = boundary_conditions[axis]
        modes = metadata.axis_modes[axis].to(device=spectral.device)
        spectral_axis = spectral.ndim - self.dim + axis

        if bc == "periodic":
            factors = self._broadcast_axis_values(1j * modes, axis).to(dtype=spectral.dtype)
            return spectral * factors, boundary_conditions

        moved = spectral.movedim(spectral_axis, -1)
        derivative = torch.zeros_like(moved)

        if bc == "dirichlet":
            if moved.shape[-1] > 1:
                derivative[..., 1:] = moved[..., :-1] * modes[:-1]
        elif bc == "neumann":
            if moved.shape[-1] > 1:
                derivative[..., :-1] = -moved[..., 1:] * modes[1:]
        else:
            raise ValueError(f"Unsupported boundary condition '{bc}' for gradient.")

        derivative = derivative.movedim(-1, spectral_axis)
        derivative_bcs = self.get_gradient_boundary_conditions(boundary_conditions, axis)
        return derivative, derivative_bcs


class BasisAwareSpectralProjector:
    """Sharp tensor-product projector for the native FFT/DCT/DST bases.

    The cutoff is strict. For a fraction ``f`` the retained integer modes are
    ``abs(k) < f*N/2`` for FFT, ``m < f*N`` for DCT, and ``r < f*N`` for
    DST. Both filtering rules therefore remove the terminal DST mode, whose
    derivative would require the unavailable DCT mode ``m=N``.
    """

    def __init__(self, solver, rule=DEFAULT_DEALIAS_RULE):
        if rule not in DEALIAS_RULE_FRACTIONS:
            raise ValueError(
                f"Unknown dealias rule {rule!r}; expected one of "
                f"{tuple(DEALIAS_RULE_FRACTIONS)}."
            )
        self.rule = rule
        self.fraction = DEALIAS_RULE_FRACTIONS[rule]
        self.shape = tuple(solver.shape)
        self.device = solver.transform_backend.device
        self.real_dtype = solver.transform_backend.real_dtype
        self._mask_cache = {}
        self._axis_mask_cache = {}

    @property
    def enabled(self):
        return self.fraction is not None

    def _axis_mode_numbers(self, axis, boundary_condition):
        key = (axis, boundary_condition)
        if key in self._axis_mask_cache:
            return self._axis_mask_cache[key]

        n = self.shape[axis]
        if boundary_condition == "periodic":
            mode_numbers = torch.fft.fftfreq(
                n,
                d=1.0 / n,
                device=self.device,
                dtype=self.real_dtype,
            )
            nyquist_mode = n / 2.0
        elif boundary_condition == "neumann":
            mode_numbers = torch.arange(
                n,
                device=self.device,
                dtype=self.real_dtype,
            )
            nyquist_mode = float(n)
        elif boundary_condition == "dirichlet":
            mode_numbers = torch.arange(
                1,
                n + 1,
                device=self.device,
                dtype=self.real_dtype,
            )
            nyquist_mode = float(n)
        else:
            raise ValueError(
                f"Unsupported boundary condition {boundary_condition!r}."
            )

        keep = mode_numbers.abs() < self.fraction * nyquist_mode
        self._axis_mask_cache[key] = keep
        return keep

    def mask(self, boundary_conditions):
        if not self.enabled:
            raise RuntimeError("The spectral projector is disabled.")

        boundary_conditions = tuple(boundary_conditions)
        if len(boundary_conditions) != len(self.shape):
            raise ValueError(
                "Boundary-condition count must match the spectral dimension."
            )
        if boundary_conditions in self._mask_cache:
            return self._mask_cache[boundary_conditions]

        mask = torch.ones(self.shape, device=self.device, dtype=torch.bool)
        for axis, boundary_condition in enumerate(boundary_conditions):
            axis_keep = self._axis_mode_numbers(axis, boundary_condition)
            view_shape = [1] * len(self.shape)
            view_shape[axis] = self.shape[axis]
            mask &= axis_keep.reshape(view_shape)

        self._mask_cache[boundary_conditions] = mask
        return mask

    def project(self, spectral, boundary_conditions):
        """Project spectral data in its trailing spatial dimensions."""
        if not self.enabled:
            return spectral
        if tuple(spectral.shape[-len(self.shape):]) != self.shape:
            raise ValueError(
                f"Expected trailing spectral shape {self.shape}, got "
                f"{tuple(spectral.shape[-len(self.shape):])}."
            )
        mask = self.mask(boundary_conditions)
        return spectral * mask.to(dtype=spectral.dtype)

    def project_dynamic_fields(self, fields, *, sync_spatial):
        """Project dynamic transform groups and optionally sync real fields."""
        if not self.enabled:
            return
        groups = fields.group_indices_by_boundary_conditions(
            range(fields.dyn_count)
        )
        for group in groups:
            boundary_conditions = fields.get_boundary_conditions(group[0])
            fields.spectral[group] = self.project(
                fields.spectral[group],
                boundary_conditions,
            )
            if sync_spatial:
                fields.spatial[group] = fields.inverse_transform_group(group)

    def retained_axis_counts(self, boundary_conditions):
        if not self.enabled:
            return tuple(self.shape)
        return tuple(
            int(self._axis_mode_numbers(axis, bc).sum().item())
            for axis, bc in enumerate(boundary_conditions)
        )


def _project_if_enabled(projector, spectral, boundary_conditions):
    if projector is None:
        return spectral
    return projector.project(spectral, boundary_conditions)


def _inverse_spectral_gradient(
    backend,
    spectral,
    boundary_conditions,
    axis,
):
    gradient_hat, gradient_bcs = backend.gradient_hat(
        spectral,
        boundary_conditions,
        axis,
    )
    return backend.inverse(gradient_hat, gradient_bcs)


def projected_common_basis_stress_divergence(
    backend,
    stress_components,
    boundary_conditions,
    *,
    projector=None,
):
    """Return row-wise ``partial_j stress_ij`` for one shared basis.

    Components are the nine row-major entries
    ``(xx, xy, xz, yx, yy, yz, zx, zy, zz)``.
    """
    if len(stress_components) != 9:
        raise ValueError("A three-dimensional stress requires nine components.")
    boundary_conditions = tuple(boundary_conditions)
    stress_hat = backend.forward(
        torch.stack(tuple(stress_components)),
        boundary_conditions,
    )
    stress_hat = _project_if_enabled(
        projector,
        stress_hat,
        boundary_conditions,
    )
    derivative_x = _inverse_spectral_gradient(
        backend,
        stress_hat[[0, 3, 6]],
        boundary_conditions,
        axis=0,
    )
    derivative_y = _inverse_spectral_gradient(
        backend,
        stress_hat[[1, 4, 7]],
        boundary_conditions,
        axis=1,
    )
    derivative_z = _inverse_spectral_gradient(
        backend,
        stress_hat[[2, 5, 8]],
        boundary_conditions,
        axis=2,
    )
    return derivative_x + derivative_y + derivative_z


def projected_distortion_stress_divergence(
    backend,
    stress_components,
    even_boundary_conditions,
    odd_boundary_conditions,
    *,
    projector=None,
):
    """Differentiate row-major distortion stress with its z parity split."""
    if len(stress_components) != 9:
        raise ValueError("A three-dimensional stress requires nine components.")
    even_boundary_conditions = tuple(even_boundary_conditions)
    odd_boundary_conditions = tuple(odd_boundary_conditions)
    even_components = torch.stack(
        tuple(stress_components[index] for index in (0, 1, 3, 4, 8))
    )
    odd_components = torch.stack(
        tuple(stress_components[index] for index in (2, 5, 6, 7))
    )
    even_hat = _project_if_enabled(
        projector,
        backend.forward(even_components, even_boundary_conditions),
        even_boundary_conditions,
    )
    odd_hat = _project_if_enabled(
        projector,
        backend.forward(odd_components, odd_boundary_conditions),
        odd_boundary_conditions,
    )

    even_x = _inverse_spectral_gradient(
        backend, even_hat[[0, 2]], even_boundary_conditions, axis=0
    )
    even_y = _inverse_spectral_gradient(
        backend, even_hat[[1, 3]], even_boundary_conditions, axis=1
    )
    even_z = _inverse_spectral_gradient(
        backend, even_hat[4], even_boundary_conditions, axis=2
    )
    odd_x = _inverse_spectral_gradient(
        backend, odd_hat[2], odd_boundary_conditions, axis=0
    )
    odd_y = _inverse_spectral_gradient(
        backend, odd_hat[3], odd_boundary_conditions, axis=1
    )
    odd_z = _inverse_spectral_gradient(
        backend, odd_hat[[0, 1]], odd_boundary_conditions, axis=2
    )
    return torch.stack(
        (
            even_x[0] + even_y[0] + odd_z[0],
            even_x[1] + even_y[1] + odd_z[1],
            odd_x + odd_y + even_z,
        )
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
