import math
from dataclasses import dataclass

import torch

from .bounded import (
    BoundedAxisPlanKey,
    DenseBoundedAxisExecutionPlan,
    build_dense_orthonormal_matrix,
)
from pssolver.core.numerics import (
    DEFAULT_SPECTRAL_STORAGE,
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    SPECTRAL_STORAGE_MODES,
)


DEFAULT_PERIODIC_TRANSFORM_EXECUTION = "multidim"
PERIODIC_TRANSFORM_EXECUTION_MODES = ("axiswise", "multidim")


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
        spectral_storage=DEFAULT_SPECTRAL_STORAGE,
        hermitian_axis=None,
        periodic_transform_execution=DEFAULT_PERIODIC_TRANSFORM_EXECUTION,
    ):
        if execution_order not in self._execution_orders:
            raise ValueError(
                f"execution_order must be one of {self._execution_orders}, "
                f"got {execution_order!r}"
            )
        if spectral_storage not in SPECTRAL_STORAGE_MODES:
            raise ValueError(
                f"spectral_storage must be one of {SPECTRAL_STORAGE_MODES}, "
                f"got {spectral_storage!r}"
            )
        if periodic_transform_execution not in PERIODIC_TRANSFORM_EXECUTION_MODES:
            raise ValueError(
                "periodic_transform_execution must be one of "
                f"{PERIODIC_TRANSFORM_EXECUTION_MODES}, got "
                f"{periodic_transform_execution!r}"
            )
        if spectral_storage == "hermitian_half":
            if not isinstance(hermitian_axis, int) or isinstance(
                hermitian_axis,
                bool,
            ):
                raise TypeError("hermitian_axis must be an integer")
            if hermitian_axis < 0 or hermitian_axis >= len(shape):
                raise ValueError(
                    "hermitian_axis is outside the transform dimension"
                )
            if execution_order != "real_first":
                raise ValueError(
                    "hermitian_half storage requires real_first execution"
                )
        self.shape = tuple(shape)
        self.lengths = tuple(self._normalize_length(length) for length in lengths)
        self.device = device
        self.dtype = dtype
        self.real_dtype = _real_dtype(dtype)
        self.spectral_dtype = _complex_dtype(dtype)
        self.execution_order = execution_order
        self.spectral_storage = spectral_storage
        self.hermitian_axis = hermitian_axis
        self.periodic_transform_execution = periodic_transform_execution
        self.dim = len(self.shape)
        spectral_shape = list(self.shape)
        if self.spectral_storage == "hermitian_half":
            spectral_shape[self.hermitian_axis] = (
                self.shape[self.hermitian_axis] // 2 + 1
            )
        self.spectral_shape = tuple(spectral_shape)

        self._matrix_cache = {}
        self._bounded_axis_plan_cache = {}
        self._metadata_cache = {}
        self._periodic_gradient_factor_cache = {}

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

        matrix = build_dense_orthonormal_matrix(
            kind,
            size,
            device=self.device,
            dtype=self.real_dtype,
        )

        self._matrix_cache[key] = matrix
        return matrix

    def _get_bounded_axis_execution_plan(
        self,
        tensor,
        kind,
        physical_size,
        retained_count,
    ):
        value_type = "complex" if tensor.is_complex() else "real"
        # Device and real dtype are immutable backend properties.  Keep the
        # hot-path lookup compact and construct the validated, fully explicit
        # public plan key only on a cache miss.
        cache_key = (kind, physical_size, retained_count, value_type)
        cached = self._bounded_axis_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        full_matrix = self._get_matrix(kind, physical_size)
        key = BoundedAxisPlanKey(
            kind=kind,
            physical_size=physical_size,
            retained_count=retained_count,
            device=full_matrix.device,
            real_dtype=self.real_dtype,
            value_type=value_type,
        )
        matrix = full_matrix[:retained_count]
        plan = DenseBoundedAxisExecutionPlan(key=key, matrix=matrix)
        self._bounded_axis_plan_cache[cache_key] = plan
        return plan

    def _apply_axis_transform(self, tensor, kind, axis, inverse=False):
        if kind == "fft":
            if inverse:
                return torch.fft.ifft(tensor, dim=axis)
            return torch.fft.fft(tensor, dim=axis)

        size = tensor.shape[axis]
        plan = self._get_bounded_axis_execution_plan(
            tensor,
            kind,
            size,
            size,
        )
        moved = tensor.movedim(axis, -1)
        if inverse:
            transformed = plan.inverse_last_axis(moved)
        else:
            transformed = plan.forward_last_axis(moved)
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

    def _periodic_axes(self, transform_kinds, tensor_ndim):
        return tuple(
            tensor_ndim - self.dim + local_axis
            for local_axis, kind in enumerate(transform_kinds)
            if kind == "fft"
        )

    def _uses_multidim_periodic_transform(self, transform_kinds):
        """Return whether periodic axes can be executed as one FFT call.

        Real-first transforms already place every bounded transform before the
        periodic transforms (and after them during inverse execution), so
        grouping the periodic axes preserves that contract. An all-periodic
        legacy transform is also safe to group. Mixed legacy transforms retain
        their historical per-axis order and fall back to ``axiswise``.
        """
        if self.periodic_transform_execution != "multidim":
            return False
        periodic_count = sum(kind == "fft" for kind in transform_kinds)
        if periodic_count < 2:
            return False
        return self.execution_order == "real_first" or periodic_count == self.dim

    def periodic_transform_execution_metadata(self, boundary_conditions):
        """Describe the requested and effective periodic transform policy."""
        transform_kinds = self.get_metadata(boundary_conditions).transform_kinds
        effective = (
            "multidim"
            if self._uses_multidim_periodic_transform(transform_kinds)
            else "axiswise"
        )
        fallback_reason = None
        if (
            self.periodic_transform_execution == "multidim"
            and effective == "axiswise"
        ):
            periodic_count = sum(kind == "fft" for kind in transform_kinds)
            if periodic_count < 2:
                fallback_reason = "fewer_than_two_periodic_axes"
            else:
                fallback_reason = "mixed_legacy_execution_order"
        return {
            "requested": self.periodic_transform_execution,
            "effective": effective,
            "fallback_reason": fallback_reason,
        }

    def get_metadata(self, boundary_conditions):
        boundary_conditions = tuple(boundary_conditions)
        if self.spectral_storage == "hermitian_half":
            self._validate_hermitian_boundary_conditions(boundary_conditions)
        if boundary_conditions in self._metadata_cache:
            return self._metadata_cache[boundary_conditions]

        axis_modes = []
        q2 = None
        for axis, bc in enumerate(boundary_conditions):
            n = self.shape[axis]
            length = self.lengths[axis]

            if bc == "periodic":
                frequency = (
                    torch.fft.rfftfreq
                    if self.spectral_storage == "hermitian_half"
                    and axis == self.hermitian_axis
                    else torch.fft.fftfreq
                )
                modes = frequency(
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

    def _normalize_retained_axis_counts(
        self,
        retained_axis_counts,
        transform_kinds,
    ):
        try:
            counts = tuple(retained_axis_counts)
        except TypeError as exc:
            raise TypeError(
                "retained_axis_counts must be an iterable of integers."
            ) from exc
        if len(counts) != self.dim:
            raise ValueError(
                "retained_axis_counts must match the transform dimension."
            )
        for axis, (count, kind) in enumerate(zip(counts, transform_kinds)):
            size = (
                self.spectral_shape[axis]
                if kind == "fft"
                else self.shape[axis]
            )
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(
                    "retained_axis_counts must contain integers."
                )
            if count < 0 or count > size:
                raise ValueError(
                    f"Retained count {count} is invalid for axis {axis} "
                    f"with size {size}."
                )
            if kind == "fft" and count != size:
                raise ValueError(
                    "Periodic FFT axes cannot use contiguous retained-mode "
                    "truncation."
                )
        return counts

    def _apply_retained_real_axis_transform(
        self,
        tensor,
        kind,
        local_axis,
        retained_count,
        *,
        inverse,
    ):
        axis = tensor.ndim - self.dim + local_axis
        plan = self._get_bounded_axis_execution_plan(
            tensor,
            kind,
            self.shape[local_axis],
            retained_count,
        )
        moved = tensor.movedim(axis, -1)
        if inverse:
            transformed = plan.inverse_last_axis(moved)
        else:
            transformed = plan.forward_last_axis(moved)
        return transformed.movedim(-1, axis)

    def forward(
        self,
        tensor,
        boundary_conditions,
        *,
        retained_axis_counts=None,
    ):
        metadata = self.get_metadata(boundary_conditions)
        counts = (
            None
            if retained_axis_counts is None
            else self._normalize_retained_axis_counts(
                retained_axis_counts,
                metadata.transform_kinds,
            )
        )
        if self.spectral_storage == "hermitian_half":
            return self._forward_hermitian(
                tensor,
                boundary_conditions,
                retained_axis_counts=counts,
            )

        output = tensor
        if self._uses_multidim_periodic_transform(metadata.transform_kinds):
            for local_axis, kind in self._ordered_axis_transforms(
                metadata.transform_kinds,
                inverse=False,
            ):
                if kind == "fft":
                    continue
                output = self._apply_retained_real_axis_transform(
                    output,
                    kind,
                    local_axis,
                    self.shape[local_axis]
                    if counts is None
                    else counts[local_axis],
                    inverse=False,
                )
            periodic_axes = self._periodic_axes(
                metadata.transform_kinds,
                output.ndim,
            )
            return torch.fft.fftn(output, dim=periodic_axes).to(
                self.spectral_dtype
            )

        for local_axis, kind in self._ordered_axis_transforms(
            metadata.transform_kinds,
            inverse=False,
        ):
            axis = output.ndim - self.dim + local_axis
            if counts is None or kind == "fft":
                output = self._apply_axis_transform(
                    output,
                    kind,
                    axis,
                    inverse=False,
                )
            else:
                output = self._apply_retained_real_axis_transform(
                    output,
                    kind,
                    local_axis,
                    counts[local_axis],
                    inverse=False,
                )
        return output.to(self.spectral_dtype)

    def inverse(
        self,
        spectral,
        boundary_conditions,
        *,
        retained_axis_counts=None,
    ):
        metadata = self.get_metadata(boundary_conditions)
        counts = (
            None
            if retained_axis_counts is None
            else self._normalize_retained_axis_counts(
                retained_axis_counts,
                metadata.transform_kinds,
            )
        )
        output = spectral
        if counts is not None:
            trailing_shape = tuple(spectral.shape[-self.dim :])
            if trailing_shape != self.spectral_shape:
                raise ValueError(
                    "Expected trailing spectral shape "
                    f"{self.spectral_shape}, got "
                    f"{trailing_shape}."
                )
            slices = [slice(None)] * spectral.ndim
            for local_axis, kind in enumerate(metadata.transform_kinds):
                if kind != "fft":
                    axis = spectral.ndim - self.dim + local_axis
                    slices[axis] = slice(0, counts[local_axis])
            output = spectral[tuple(slices)]

        if self.spectral_storage == "hermitian_half":
            return self._inverse_hermitian(
                output,
                boundary_conditions,
                retained_axis_counts=counts,
            )

        if self._uses_multidim_periodic_transform(metadata.transform_kinds):
            periodic_axes = self._periodic_axes(
                metadata.transform_kinds,
                output.ndim,
            )
            output = torch.fft.ifftn(output, dim=periodic_axes)
            for local_axis, kind in self._ordered_axis_transforms(
                metadata.transform_kinds,
                inverse=True,
            ):
                if kind == "fft":
                    continue
                if output.is_complex():
                    output = output.real
                output = self._apply_retained_real_axis_transform(
                    output,
                    kind,
                    local_axis,
                    self.shape[local_axis]
                    if counts is None
                    else counts[local_axis],
                    inverse=True,
                )
            return output.real

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
            if counts is None or kind == "fft":
                output = self._apply_axis_transform(
                    output,
                    kind,
                    axis,
                    inverse=True,
                )
            else:
                output = self._apply_retained_real_axis_transform(
                    output,
                    kind,
                    local_axis,
                    counts[local_axis],
                    inverse=True,
                )
        return output.real

    def _validate_hermitian_boundary_conditions(self, boundary_conditions):
        boundary_conditions = tuple(boundary_conditions)
        if len(boundary_conditions) != self.dim:
            raise ValueError(
                "Boundary-condition count must match the transform dimension."
            )
        if boundary_conditions[self.hermitian_axis] != "periodic":
            raise ValueError(
                "hermitian_half storage requires a periodic boundary "
                f"condition on axis {self.hermitian_axis}"
            )
        return boundary_conditions

    def _hermitian_periodic_axes(self, transform_kinds, tensor_ndim):
        periodic_axes = [
            axis for axis, kind in enumerate(transform_kinds) if kind == "fft"
        ]
        periodic_axes.remove(self.hermitian_axis)
        periodic_axes.append(self.hermitian_axis)
        return tuple(tensor_ndim - self.dim + axis for axis in periodic_axes)

    def _forward_hermitian(
        self,
        tensor,
        boundary_conditions,
        *,
        retained_axis_counts=None,
    ):
        boundary_conditions = self._validate_hermitian_boundary_conditions(
            boundary_conditions
        )
        metadata = self.get_metadata(boundary_conditions)
        output = tensor
        for local_axis, kind in enumerate(metadata.transform_kinds):
            if kind == "fft":
                continue
            axis = output.ndim - self.dim + local_axis
            if retained_axis_counts is None:
                output = self._apply_axis_transform(
                    output,
                    kind,
                    axis,
                    inverse=False,
                )
            else:
                output = self._apply_retained_real_axis_transform(
                    output,
                    kind,
                    local_axis,
                    retained_axis_counts[local_axis],
                    inverse=False,
                )
        periodic_axes = self._hermitian_periodic_axes(
            metadata.transform_kinds,
            output.ndim,
        )
        output = torch.fft.rfftn(output, dim=periodic_axes)
        return output.to(self.spectral_dtype)

    def _inverse_hermitian(
        self,
        spectral,
        boundary_conditions,
        *,
        retained_axis_counts=None,
    ):
        boundary_conditions = self._validate_hermitian_boundary_conditions(
            boundary_conditions
        )
        trailing_shape = tuple(spectral.shape[-self.dim :])
        expected_shape = (
            self.spectral_shape
            if retained_axis_counts is None
            else tuple(retained_axis_counts)
        )
        if trailing_shape != expected_shape:
            raise ValueError(
                f"Expected trailing spectral shape {expected_shape}, "
                f"got {trailing_shape}."
            )
        metadata = self.get_metadata(boundary_conditions)
        periodic_axes = self._hermitian_periodic_axes(
            metadata.transform_kinds,
            spectral.ndim,
        )
        local_periodic_axes = tuple(
            axis - (spectral.ndim - self.dim) for axis in periodic_axes
        )
        physical_sizes = tuple(
            self.shape[axis] for axis in local_periodic_axes
        )
        output = torch.fft.irfftn(
            spectral,
            s=physical_sizes,
            dim=periodic_axes,
        )
        for local_axis, kind in reversed(
            tuple(enumerate(metadata.transform_kinds))
        ):
            if kind == "fft":
                continue
            axis = output.ndim - self.dim + local_axis
            if retained_axis_counts is None:
                output = self._apply_axis_transform(
                    output,
                    kind,
                    axis,
                    inverse=True,
                )
            else:
                output = self._apply_retained_real_axis_transform(
                    output,
                    kind,
                    local_axis,
                    retained_axis_counts[local_axis],
                    inverse=True,
                )
        return output.real

    def laplacian_hat(self, spectral, boundary_conditions):
        laplacian_eigs = self.get_laplacian_eigs(boundary_conditions)
        return spectral * laplacian_eigs.to(device=spectral.device)

    def _periodic_gradient_factors(
        self,
        spectral,
        boundary_conditions,
        axis,
    ):
        key = (
            tuple(boundary_conditions),
            axis,
            spectral.device,
            spectral.dtype,
        )
        factors = self._periodic_gradient_factor_cache.get(key)
        if factors is not None:
            return factors

        metadata = self.get_metadata(boundary_conditions)
        modes = metadata.axis_modes[axis]
        if (
            self.spectral_storage == "hermitian_half"
            and self.shape[axis] % 2 == 0
        ):
            # A first derivative of an even-grid Nyquist mode has no
            # real-valued collocation-grid representation. Cache the exact
            # zeroed multiplier instead of cloning and mutating the mode
            # vector on every gradient evaluation.
            modes = modes.clone()
            modes[self.shape[axis] // 2] = 0
        factors = self._broadcast_axis_values(1j * modes, axis).to(
            device=spectral.device,
            dtype=spectral.dtype,
        )
        self._periodic_gradient_factor_cache[key] = factors
        return factors

    def gradient_hat(self, spectral, boundary_conditions, axis):
        boundary_conditions = tuple(boundary_conditions)
        bc = boundary_conditions[axis]
        spectral_axis = spectral.ndim - self.dim + axis

        if bc == "periodic":
            factors = self._periodic_gradient_factors(
                spectral,
                boundary_conditions,
                axis,
            )
            return spectral * factors, boundary_conditions

        metadata = self.get_metadata(boundary_conditions)
        modes = metadata.axis_modes[axis].to(device=spectral.device)
        moved = spectral.movedim(spectral_axis, -1)
        derivative = torch.zeros_like(moved)
        preserve_autograd = torch.is_grad_enabled() and moved.requires_grad

        if bc == "dirichlet":
            if moved.shape[-1] > 1:
                if preserve_autograd:
                    derivative[..., 1:] = moved[..., :-1] * modes[:-1]
                else:
                    # The derivative buffer is owned by this call. Write the
                    # shifted product directly into it to avoid materializing
                    # a full temporary followed by a strided copy kernel.
                    torch.mul(
                        moved[..., :-1],
                        modes[:-1],
                        out=derivative[..., 1:],
                    )
        elif bc == "neumann":
            if moved.shape[-1] > 1:
                if preserve_autograd:
                    derivative[..., :-1] = -moved[..., 1:] * modes[1:]
                else:
                    torch.neg(
                        moved[..., 1:],
                        out=derivative[..., :-1],
                    )
                    derivative[..., :-1].mul_(modes[1:])
        else:
            raise ValueError(f"Unsupported boundary condition '{bc}' for gradient.")

        derivative = derivative.movedim(-1, spectral_axis)
        derivative_bcs = self.get_gradient_boundary_conditions(
            boundary_conditions,
            axis,
        )
        return derivative, derivative_bcs
