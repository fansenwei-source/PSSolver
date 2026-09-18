import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .backends.bounded import (
    BoundedAxisPlanKey,
    DenseBoundedAxisExecutionPlan,
    build_dense_orthonormal_matrix,
)


DEFAULT_DEALIAS_RULE = "cubic_half"
DEFAULT_TRANSFORM_EXECUTION_ORDER = "real_first"
DEFAULT_PROJECTED_TRANSFORM_EXECUTION = "truncated"
PROJECTED_TRANSFORM_EXECUTION_MODES = ("full", "truncated")
DEFAULT_SPECTRAL_STORAGE = "full_complex"
SPECTRAL_STORAGE_MODES = ("full_complex", "hermitian_half")
DEFAULT_PERIODIC_TRANSFORM_EXECUTION = "multidim"
PERIODIC_TRANSFORM_EXECUTION_MODES = ("axiswise", "multidim")
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


class BasisAwareSpectralProjector:
    """Sharp tensor-product projector for the native FFT/DCT/DST bases.

    The cutoff is strict. For a fraction ``f`` the retained integer modes are
    ``abs(k) < f*N/2`` for FFT, ``m < f*N`` for DCT, and ``r < f*N`` for
    DST. Both filtering rules therefore remove the terminal DST mode, whose
    derivative would require the unavailable DCT mode ``m=N``.
    """

    def __init__(
        self,
        solver,
        rule=DEFAULT_DEALIAS_RULE,
        transform_execution=None,
    ):
        if rule not in DEALIAS_RULE_FRACTIONS:
            raise ValueError(
                f"Unknown dealias rule {rule!r}; expected one of "
                f"{tuple(DEALIAS_RULE_FRACTIONS)}."
            )
        if transform_execution is None:
            # With projection disabled there are no discarded modes to skip;
            # preserve the historical full transform automatically. An
            # explicit truncated+none request remains an error below.
            transform_execution = (
                "full"
                if rule == "none"
                else DEFAULT_PROJECTED_TRANSFORM_EXECUTION
            )
        if transform_execution not in PROJECTED_TRANSFORM_EXECUTION_MODES:
            raise ValueError(
                "transform_execution must be 'full' or 'truncated'."
            )
        if transform_execution == "truncated" and rule == "none":
            raise ValueError(
                "truncated projected transforms require enabled dealiasing."
            )
        self.rule = rule
        self.fraction = DEALIAS_RULE_FRACTIONS[rule]
        self.transform_execution = transform_execution
        self.physical_shape = tuple(solver.shape)
        self.transform_backend = solver.transform_backend
        self.shape = self.transform_backend.spectral_shape
        self.device = self.transform_backend.device
        self.real_dtype = self.transform_backend.real_dtype
        self._mask_cache = {}
        self._axis_mask_cache = {}
        self._retained_counts_cache = {}
        self._reduced_periodic_mask_cache = {}

    @property
    def enabled(self):
        return self.fraction is not None

    def _axis_mode_numbers(self, axis, boundary_condition):
        key = (axis, boundary_condition)
        if key in self._axis_mask_cache:
            return self._axis_mask_cache[key]

        n = self.physical_shape[axis]
        if boundary_condition == "periodic":
            frequency = (
                torch.fft.rfftfreq
                if self.transform_backend.spectral_storage == "hermitian_half"
                and axis == self.transform_backend.hermitian_axis
                else torch.fft.fftfreq
            )
            mode_numbers = frequency(
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
            view_shape[axis] = axis_keep.shape[0]
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
        # Let PyTorch promote the Boolean mask inside the multiplication
        # kernel.  Materializing a full complex-valued copy of the mask adds
        # one allocation and one device copy at every projection while giving
        # exactly the same finite 0/1 multiplication.
        return spectral * mask

    def project_(self, spectral, boundary_conditions):
        """Project owned spectral storage in place without materializing a copy."""
        if not self.enabled:
            return spectral
        if tuple(spectral.shape[-len(self.shape) :]) != self.shape:
            raise ValueError(
                f"Expected trailing spectral shape {self.shape}, got "
                f"{tuple(spectral.shape[-len(self.shape):])}."
            )
        spectral.mul_(self.mask(boundary_conditions))
        return spectral

    def _retained_counts(self, boundary_conditions):
        boundary_conditions = tuple(boundary_conditions)
        if boundary_conditions not in self._retained_counts_cache:
            self._retained_counts_cache[boundary_conditions] = tuple(
                (
                    self.shape[axis]
                    if bc == "periodic"
                    else int(self._axis_mode_numbers(axis, bc).sum().item())
                )
                for axis, bc in enumerate(boundary_conditions)
            )
        return self._retained_counts_cache[boundary_conditions]

    def _reduced_periodic_mask(self, boundary_conditions):
        boundary_conditions = tuple(boundary_conditions)
        if boundary_conditions in self._reduced_periodic_mask_cache:
            return self._reduced_periodic_mask_cache[boundary_conditions]
        counts = self._retained_counts(boundary_conditions)
        mask = torch.ones(counts, device=self.device, dtype=torch.bool)
        for axis, bc in enumerate(boundary_conditions):
            if bc != "periodic":
                continue
            axis_keep = self._axis_mode_numbers(axis, bc)
            view_shape = [1] * len(self.shape)
            view_shape[axis] = self.shape[axis]
            mask &= axis_keep.reshape(view_shape)
        self._reduced_periodic_mask_cache[boundary_conditions] = mask
        return mask

    def forward_transform(self, tensor, boundary_conditions):
        """Transform and project, optionally avoiding discarded real modes."""
        boundary_conditions = tuple(boundary_conditions)
        if self.transform_execution == "full":
            return self.project(
                self.transform_backend.forward(tensor, boundary_conditions),
                boundary_conditions,
            )

        counts = self._retained_counts(boundary_conditions)
        if any(count == 0 for count in counts):
            return torch.zeros(
                (*tensor.shape[: -len(self.shape)], *self.shape),
                device=tensor.device,
                dtype=self.transform_backend.spectral_dtype,
            )
        reduced = self.transform_backend.forward(
            tensor,
            boundary_conditions,
            retained_axis_counts=counts,
        )
        reduced.mul_(self._reduced_periodic_mask(boundary_conditions))
        padding = []
        for full_size, retained_count in reversed(
            tuple(zip(self.shape, counts))
        ):
            padding.extend((0, full_size - retained_count))
        return functional.pad(reduced, tuple(padding))

    def inverse_transform(self, spectral, boundary_conditions):
        """Invert coefficients whose high modes have already been projected."""
        boundary_conditions = tuple(boundary_conditions)
        if self.transform_execution == "full":
            return self.transform_backend.inverse(
                spectral,
                boundary_conditions,
            )
        counts = self._retained_counts(boundary_conditions)
        if any(count == 0 for count in counts):
            return torch.zeros(
                (*spectral.shape[: -len(self.shape)], *self.physical_shape),
                device=spectral.device,
                dtype=self.real_dtype,
            )
        return self.transform_backend.inverse(
            spectral,
            boundary_conditions,
            retained_axis_counts=counts,
        )

    def execution_metadata(self):
        """Return JSON-compatible provenance for projected transforms."""
        enabled = self.transform_execution == "truncated"
        spectral_storage = self.transform_backend.spectral_storage
        return {
            "requested": self.transform_execution,
            "effective": self.transform_execution,
            "fallback_allowed": False,
            "fallback_reason": None,
            "truncated_real_basis_axes": enabled,
            "spectral_storage": spectral_storage,
            "backend_storage_shape_preserved": True,
            "full_spectral_storage_preserved": (
                spectral_storage == "full_complex"
            ),
        }

    def computed_axis_sizes(self, boundary_conditions):
        """Return transform extents actually evaluated along each axis."""
        if self.transform_execution == "full":
            return self.shape
        return self._retained_counts(boundary_conditions)

    def project_dynamic_fields(self, fields, *, sync_spatial):
        """Project dynamic transform groups and optionally sync real fields."""
        if not self.enabled:
            return
        groups = fields.group_indices_by_boundary_conditions(
            range(fields.dyn_count)
        )
        for group in groups:
            boundary_conditions = fields.get_boundary_conditions(group[0])
            spectral = fields.select_spectral_group(group)
            self.project_(spectral, boundary_conditions)
            if (
                fields.transform_group_indexing_metadata(group)["effective"]
                == "advanced"
            ):
                fields.store_spectral_group(group, spectral)
            if sync_spatial:
                fields.store_spatial_group(
                    group,
                    self.inverse_transform(spectral, boundary_conditions),
                )

    def refresh_dynamic_fields(self, fields, *, sync_spatial):
        """Rebuild projected dynamic spectra directly from spatial fields."""
        groups = fields.group_indices_by_boundary_conditions(
            range(fields.dyn_count)
        )
        for group in groups:
            boundary_conditions = fields.get_boundary_conditions(group[0])
            spectral = self.forward_transform(
                fields.select_spatial_group(group),
                boundary_conditions,
            )
            fields.store_spectral_group(group, spectral)
            if sync_spatial:
                fields.store_spatial_group(
                    group,
                    self.inverse_transform(spectral, boundary_conditions),
                )

    def retained_axis_counts(self, boundary_conditions):
        if not self.enabled:
            return tuple(self.shape)
        return tuple(
            int(self._axis_mode_numbers(axis, bc).sum().item())
            for axis, bc in enumerate(boundary_conditions)
        )


def _forward_projected(
    backend,
    projector,
    tensor,
    boundary_conditions,
):
    if projector is None:
        return backend.forward(tensor, boundary_conditions)
    return projector.forward_transform(tensor, boundary_conditions)


def _inverse_projected(
    backend,
    projector,
    spectral,
    boundary_conditions,
):
    if projector is None:
        return backend.inverse(spectral, boundary_conditions)
    return projector.inverse_transform(spectral, boundary_conditions)


def _inverse_spectral_gradient(
    backend,
    spectral,
    boundary_conditions,
    axis,
    *,
    projector=None,
):
    gradient_hat, gradient_bcs = backend.gradient_hat(
        spectral,
        boundary_conditions,
        axis,
    )
    return _inverse_projected(
        backend,
        projector,
        gradient_hat,
        gradient_bcs,
    )


def _validate_divergence_sum_space(sum_space):
    if sum_space not in {"physical", "spectral"}:
        raise ValueError(
            "sum_space must be 'physical' or 'spectral'."
        )


def _spectral_gradient(backend, spectral, boundary_conditions, axis):
    return backend.gradient_hat(
        spectral,
        boundary_conditions,
        axis,
    )


def projected_common_basis_stress_divergence(
    backend,
    stress_components,
    boundary_conditions,
    *,
    projector=None,
    sum_space="physical",
):
    """Return row-wise ``partial_j stress_ij`` for one shared basis.

    Components are the nine row-major entries
    ``(xx, xy, xz, yx, yy, yz, zx, zy, zz)``.
    ``sum_space='spectral'`` combines the x/y derivative coefficients, which
    share a basis, before inversion.  The z derivative retains its parity-
    changed basis and is combined in physical space.
    """
    if len(stress_components) != 9:
        raise ValueError("A three-dimensional stress requires nine components.")
    _validate_divergence_sum_space(sum_space)
    boundary_conditions = tuple(boundary_conditions)
    stress_hat = _forward_projected(
        backend,
        projector,
        torch.stack(tuple(stress_components)),
        boundary_conditions,
    )
    stress_matrix = stress_hat.unflatten(0, (3, 3))
    if sum_space == "physical":
        derivative_x = _inverse_spectral_gradient(
            backend,
            stress_matrix[:, 0],
            boundary_conditions,
            axis=0,
            projector=projector,
        )
        derivative_y = _inverse_spectral_gradient(
            backend,
            stress_matrix[:, 1],
            boundary_conditions,
            axis=1,
            projector=projector,
        )
        derivative_z = _inverse_spectral_gradient(
            backend,
            stress_matrix[:, 2],
            boundary_conditions,
            axis=2,
            projector=projector,
        )
        return derivative_x + derivative_y + derivative_z

    derivative_x_hat, derivative_x_bcs = _spectral_gradient(
        backend,
        stress_matrix[:, 0],
        boundary_conditions,
        axis=0,
    )
    derivative_y_hat, derivative_y_bcs = _spectral_gradient(
        backend,
        stress_matrix[:, 1],
        boundary_conditions,
        axis=1,
    )
    if derivative_x_bcs != derivative_y_bcs:
        raise RuntimeError(
            "The x/y stress derivatives must share one spectral basis."
        )
    derivative_xy = _inverse_projected(
        backend,
        projector,
        derivative_x_hat + derivative_y_hat,
        derivative_x_bcs,
    )
    derivative_z = _inverse_spectral_gradient(
        backend,
        stress_matrix[:, 2],
        boundary_conditions,
        axis=2,
        projector=projector,
    )
    return derivative_xy + derivative_z


def projected_distortion_stress_divergence(
    backend,
    stress_components,
    even_boundary_conditions,
    odd_boundary_conditions,
    *,
    projector=None,
    sum_space="physical",
):
    """Differentiate row-major distortion stress with its z parity split.

    ``sum_space='spectral'`` assembles the two tangential components in their
    common Neumann basis and the normal component in its Dirichlet basis before
    inversion.  This is the same linear divergence with fewer transforms.
    """
    if len(stress_components) != 9:
        raise ValueError("A three-dimensional stress requires nine components.")
    _validate_divergence_sum_space(sum_space)
    even_boundary_conditions = tuple(even_boundary_conditions)
    odd_boundary_conditions = tuple(odd_boundary_conditions)
    even_components = torch.stack(
        tuple(stress_components[index] for index in (0, 1, 3, 4, 8))
    )
    odd_components = torch.stack(
        tuple(stress_components[index] for index in (2, 5, 6, 7))
    )
    even_hat = _forward_projected(
        backend,
        projector,
        even_components,
        even_boundary_conditions,
    )
    odd_hat = _forward_projected(
        backend,
        projector,
        odd_components,
        odd_boundary_conditions,
    )
    even_matrix = even_hat[:4].unflatten(0, (2, 2))
    odd_tangential = odd_hat[:2]

    if sum_space == "physical":
        even_x = _inverse_spectral_gradient(
            backend,
            even_matrix[:, 0],
            even_boundary_conditions,
            axis=0,
            projector=projector,
        )
        even_y = _inverse_spectral_gradient(
            backend,
            even_matrix[:, 1],
            even_boundary_conditions,
            axis=1,
            projector=projector,
        )
        even_z = _inverse_spectral_gradient(
            backend,
            even_hat[4],
            even_boundary_conditions,
            axis=2,
            projector=projector,
        )
        odd_x = _inverse_spectral_gradient(
            backend,
            odd_hat[2],
            odd_boundary_conditions,
            axis=0,
            projector=projector,
        )
        odd_y = _inverse_spectral_gradient(
            backend,
            odd_hat[3],
            odd_boundary_conditions,
            axis=1,
            projector=projector,
        )
        odd_z = _inverse_spectral_gradient(
            backend,
            odd_tangential,
            odd_boundary_conditions,
            axis=2,
            projector=projector,
        )
        return torch.stack(
            (
                even_x[0] + even_y[0] + odd_z[0],
                even_x[1] + even_y[1] + odd_z[1],
                odd_x + odd_y + even_z,
            )
        )

    even_x_hat, even_x_bcs = _spectral_gradient(
        backend, even_matrix[:, 0], even_boundary_conditions, axis=0
    )
    even_y_hat, even_y_bcs = _spectral_gradient(
        backend, even_matrix[:, 1], even_boundary_conditions, axis=1
    )
    odd_z_hat, odd_z_bcs = _spectral_gradient(
        backend, odd_tangential, odd_boundary_conditions, axis=2
    )
    if not (even_x_bcs == even_y_bcs == odd_z_bcs):
        raise RuntimeError(
            "Tangential distortion-force terms must share one spectral basis."
        )
    tangential = _inverse_projected(
        backend,
        projector,
        even_x_hat + even_y_hat + odd_z_hat,
        even_x_bcs,
    )

    odd_x_hat, odd_x_bcs = _spectral_gradient(
        backend, odd_hat[2], odd_boundary_conditions, axis=0
    )
    odd_y_hat, odd_y_bcs = _spectral_gradient(
        backend, odd_hat[3], odd_boundary_conditions, axis=1
    )
    even_z_hat, even_z_bcs = _spectral_gradient(
        backend, even_hat[4], even_boundary_conditions, axis=2
    )
    if not (odd_x_bcs == odd_y_bcs == even_z_bcs):
        raise RuntimeError(
            "Normal distortion-force terms must share one spectral basis."
        )
    normal = _inverse_projected(
        backend,
        projector,
        odd_x_hat + odd_y_hat + even_z_hat,
        odd_x_bcs,
    )
    return torch.stack((tangential[0], tangential[1], normal))


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
