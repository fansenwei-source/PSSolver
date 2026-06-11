import math
from dataclasses import dataclass

import torch


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

    def __init__(self, shape, lengths, device="cuda", dtype=torch.float32):
        self.shape = tuple(shape)
        self.lengths = tuple(self._normalize_length(length) for length in lengths)
        self.device = device
        self.dtype = dtype
        self.real_dtype = _real_dtype(dtype)
        self.spectral_dtype = _complex_dtype(dtype)
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
                modes = torch.fft.fftfreq(n, d=length / n).to(
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
        for local_axis, kind in enumerate(metadata.transform_kinds):
            axis = output.ndim - self.dim + local_axis
            output = self._apply_axis_transform(output, kind, axis, inverse=False)
        return output.to(self.spectral_dtype)

    def inverse(self, spectral, boundary_conditions):
        output = spectral
        metadata = self.get_metadata(boundary_conditions)
        for local_axis, kind in reversed(list(enumerate(metadata.transform_kinds))):
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
