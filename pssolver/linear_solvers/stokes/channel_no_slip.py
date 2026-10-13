"""No-slip rectangular-Channel modal Stokes/Brinkman solver.

The solver consumes force coefficients in periodic/Dirichlet/Dirichlet
velocity bases and returns velocity coefficients in the same bases plus a
zero-mean pressure in periodic/Neumann/Neumann space.  It deliberately keeps
the legacy Channel operation order while separating the linear solve from the
active-nematic force model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch


CHANNEL_VELOCITY_BOUNDARY_CONDITIONS = (
    "periodic",
    "dirichlet",
    "dirichlet",
)
CHANNEL_PRESSURE_BOUNDARY_CONDITIONS = (
    "periodic",
    "neumann",
    "neumann",
)

CHANNEL_PRESSURE_RHS_NORM_DEFINITION = "active_mode_root_mean_square.v1"
CHANNEL_PRESSURE_NORMALIZED_RESIDUAL_DEFINITION = (
    "active_rms(projected_residual)/active_rms(projected_rhs)"
)
CHANNEL_PRESSURE_WORKING_VECTOR_SCALING = (
    "largest_power_of_two_not_greater_than_active_component_amax"
)
CHANNEL_PRESSURE_BREAKDOWN_DEFINITION = (
    "abs(mean_active(conj(left)*right))"
    "<=eps*active_rms(left)*active_rms(right)"
)


class ChannelPressureConvergenceError(RuntimeError):
    """Fail-closed production Channel pressure-solve error."""

    code = "channel_pressure_convergence_error"

    def __init__(self, message: str, *, details: dict[str, object]) -> None:
        super().__init__(message)
        self.details = dict(details)
        self.diagnostics = dict(details)


@dataclass(frozen=True, slots=True)
class ChannelPressureMagnitude:
    """JSON-safe active-mode RMS, including unrepresentable magnitudes."""

    is_zero: bool
    float64_value: float | None
    representable_as_positive_float64: bool
    log2_value: float | None
    binary_mantissa: float | None
    binary_exponent: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.is_zero, bool):
            raise TypeError("pressure magnitude is_zero must be bool")
        if not isinstance(self.representable_as_positive_float64, bool):
            raise TypeError(
                "pressure magnitude representability flag must be bool"
            )
        if self.is_zero:
            if (
                self.float64_value != 0.0
                or self.representable_as_positive_float64
                or self.log2_value is not None
                or self.binary_mantissa is not None
                or self.binary_exponent is not None
            ):
                raise ValueError("zero pressure magnitude has invalid metadata")
            return
        if (
            not isinstance(self.log2_value, float)
            or not math.isfinite(self.log2_value)
            or not isinstance(self.binary_mantissa, float)
            or not math.isfinite(self.binary_mantissa)
            or not 0.5 <= self.binary_mantissa < 1.0
            or not isinstance(self.binary_exponent, int)
            or isinstance(self.binary_exponent, bool)
        ):
            raise ValueError("nonzero pressure magnitude has invalid binary metadata")
        expected_log2 = (
            math.log2(self.binary_mantissa) + self.binary_exponent
        )
        if not math.isclose(
            self.log2_value,
            expected_log2,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("pressure magnitude binary metadata is inconsistent")
        if self.representable_as_positive_float64:
            if (
                not isinstance(self.float64_value, float)
                or not math.isfinite(self.float64_value)
                or self.float64_value <= 0.0
            ):
                raise ValueError(
                    "representable pressure magnitude needs a positive float64"
                )
        elif self.float64_value is not None:
            raise ValueError(
                "unrepresentable pressure magnitude float64 value must be null"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "is_zero": self.is_zero,
            "float64_value": self.float64_value,
            "representable_as_positive_float64": (
                self.representable_as_positive_float64
            ),
            "log2_value": self.log2_value,
            "binary_mantissa": self.binary_mantissa,
            "binary_exponent": self.binary_exponent,
        }

    def shifted(self, binary_exponent: int) -> ChannelPressureMagnitude:
        """Return this magnitude multiplied by an exact power of two."""

        if self.is_zero:
            return self
        if not isinstance(binary_exponent, int) or isinstance(
            binary_exponent, bool
        ):
            raise TypeError("binary exponent shift must be an integer")
        exponent = int(self.binary_exponent) + binary_exponent
        return _channel_pressure_magnitude_from_binary(
            float(self.binary_mantissa),
            exponent,
        )


def _channel_pressure_zero_magnitude() -> ChannelPressureMagnitude:
    return ChannelPressureMagnitude(
        is_zero=True,
        float64_value=0.0,
        representable_as_positive_float64=False,
        log2_value=None,
        binary_mantissa=None,
        binary_exponent=None,
    )


def _channel_pressure_magnitude_from_binary(
    mantissa: float,
    exponent: int,
) -> ChannelPressureMagnitude:
    if not math.isfinite(mantissa) or not 0.5 <= mantissa < 1.0:
        raise ValueError("binary magnitude mantissa must be finite in [0.5, 1)")
    log2_value = math.log2(mantissa) + exponent
    try:
        value = math.ldexp(mantissa, exponent)
    except OverflowError:
        value = math.inf
    minimum_log2 = -1074.0
    maximum_log2 = math.log2(torch.finfo(torch.float64).max)
    representable = bool(
        minimum_log2 <= log2_value <= maximum_log2
        and value > 0.0
        and math.isfinite(value)
    )
    return ChannelPressureMagnitude(
        is_zero=False,
        float64_value=float(value) if representable else None,
        representable_as_positive_float64=representable,
        log2_value=float(log2_value),
        binary_mantissa=float(mantissa),
        binary_exponent=int(exponent),
    )


def _channel_pressure_project_active(value, inactive_mask):
    if not isinstance(value, torch.Tensor):
        raise TypeError("Channel pressure vector must be a torch.Tensor")
    if not value.is_complex():
        raise TypeError("Channel pressure vector must have a complex dtype")
    try:
        mask = inactive_mask.expand_as(value)
    except RuntimeError as exc:
        raise ValueError("Channel pressure mask does not match the vector") from exc
    return value.masked_fill(mask, 0), mask


def _channel_pressure_component_amax(value, inactive_mask):
    projected, mask = _channel_pressure_project_active(value, inactive_mask)
    component_amax = torch.maximum(
        projected.real.abs(),
        projected.imag.abs(),
    ).amax()
    component_value = float(component_amax.item())
    if not math.isfinite(component_value):
        raise ValueError("Channel pressure vector contains NaN or Inf")
    active_count = (~mask).sum(dtype=projected.real.dtype)
    return projected, component_value, active_count


def channel_pressure_active_rms(
    value: torch.Tensor,
    inactive_mask: torch.Tensor,
) -> ChannelPressureMagnitude:
    """Return the scale-safe RMS on the projected pressure subspace."""

    projected, component_amax, active_count = _channel_pressure_component_amax(
        value,
        inactive_mask,
    )
    if component_amax == 0.0:
        return _channel_pressure_zero_magnitude()
    _component_mantissa, component_exponent = math.frexp(component_amax)
    normalized = channel_pressure_scale_power_of_two(
        projected,
        -int(component_exponent - 1),
    )
    # The normalized components are bounded by two, so this reduction cannot
    # overflow or underflow even when the original coefficients are extreme.
    squared = normalized.real.square() + normalized.imag.square()
    rms_factor = math.sqrt(float((squared.sum() / active_count).item()))
    if not math.isfinite(rms_factor) or rms_factor <= 0.0:
        raise ValueError("Channel pressure subspace has no active coefficients")
    factor_mantissa, factor_exponent = math.frexp(rms_factor)
    exponent = component_exponent - 1 + factor_exponent
    return _channel_pressure_magnitude_from_binary(
        factor_mantissa,
        exponent,
    )


def _channel_pressure_working_magnitude(
    value: torch.Tensor,
    inactive_mask: torch.Tensor,
    *,
    finite_companions: tuple[torch.Tensor, ...] = (),
    active_count: int | None = None,
) -> tuple[ChannelPressureMagnitude, bool]:
    """Measure an already scaled PCG vector with one host synchronization.

    ``channel_pressure_working_vector`` first places the production solve in a
    safe quadratic range.  Subsequent residuals normally remain in that range,
    so their RMS and companion finite flags can be copied to the host together.
    If a finite nonzero vector nevertheless falls outside that range, fall
    back to the fully scale-safe public primitive.
    """

    if active_count is None:
        count = (~inactive_mask.expand_as(value)).sum(dtype=value.real.dtype)
    else:
        if active_count <= 0:
            raise ValueError("active pressure degree count must be positive")
        count = active_count
    rms = torch.linalg.vector_norm(value.reshape(-1)) / math.sqrt(float(count))
    metrics = [rms]
    metrics.extend(
        torch.isfinite(companion).all().to(dtype=value.real.dtype)
        for companion in finite_companions
    )
    observed = torch.stack(metrics).tolist()
    rms_value = float(observed[0])
    companions_finite = all(item == 1.0 for item in observed[1:])
    if not companions_finite:
        return _channel_pressure_zero_magnitude(), False
    if not math.isfinite(rms_value) or rms_value <= 0.0:
        # This branch is intentionally exceptional.  It preserves the public
        # underflow/overflow contract without charging the ordinary CUDA path
        # for a second scalar synchronization on every PCG iteration.
        try:
            return channel_pressure_active_rms(value, inactive_mask), True
        except ValueError:
            return _channel_pressure_zero_magnitude(), False
    mantissa, exponent = math.frexp(rms_value)
    return _channel_pressure_magnitude_from_binary(mantissa, exponent), True


def _channel_pressure_working_breakdown(
    left: torch.Tensor,
    right: torch.Tensor,
    inactive_mask: torch.Tensor,
    *,
    inner_product: torch.Tensor | None = None,
) -> tuple[bool, bool]:
    """Evaluate the relative breakdown gate with one host synchronization.

    The caller supplies vectors in the safe PCG working scale.  The returned
    booleans are ``(breakdown, finite)``.  Exceptional finite scales fall back
    to the fully scale-safe public implementation; non-finite vectors are
    reported separately so the caller can retain its fail-closed reason.
    """

    if inner_product is None:
        inner_product = torch.sum(torch.conj(left) * right)
    paired_square_sum = torch.sum(
        torch.complex(
            left.real.square() + left.imag.square(),
            right.real.square() + right.imag.square(),
        )
    )
    observed = torch.stack(
        (
            torch.abs(inner_product),
            paired_square_sum.real,
            paired_square_sum.imag,
        )
    ).tolist()
    inner_value, left_square_value, right_square_value = (
        float(item) for item in observed
    )
    if not all(
        math.isfinite(item) and item > 0.0
        for item in (left_square_value, right_square_value)
    ) or not math.isfinite(inner_value):
        try:
            return channel_pressure_relative_breakdown(
                left,
                right,
                inactive_mask,
            ), True
        except ValueError:
            return False, False
    epsilon = torch.finfo(left.real.dtype).eps
    threshold = epsilon * math.sqrt(left_square_value * right_square_value)
    return inner_value <= threshold, True


def _channel_pressure_working_residual_observation(
    residual: torch.Tensor,
    preconditioned: torch.Tensor,
    inactive_mask: torch.Tensor,
    *,
    inner_product: torch.Tensor,
    active_count: int,
    finite_companions: tuple[torch.Tensor, ...] = (),
) -> tuple[ChannelPressureMagnitude, bool, bool, bool]:
    """Observe residual magnitude and preconditioned breakdown in one sync."""

    if active_count <= 0:
        raise ValueError("active pressure degree count must be positive")
    residual_square_sum = torch.sum(
        residual.real.square() + residual.imag.square()
    )
    preconditioned_square_sum = torch.sum(
        preconditioned.real.square() + preconditioned.imag.square()
    )
    metrics = [
        torch.abs(inner_product),
        residual_square_sum,
        preconditioned_square_sum,
    ]
    metrics.extend(
        torch.isfinite(companion).all().to(dtype=residual.real.dtype)
        for companion in finite_companions
    )
    observed = torch.stack(metrics).tolist()
    inner_value, residual_square_value, preconditioned_square_value = (
        float(item) for item in observed[:3]
    )
    companions_finite = all(item == 1.0 for item in observed[3:])
    ordinary = (
        companions_finite
        and math.isfinite(inner_value)
        and math.isfinite(residual_square_value)
        and residual_square_value > 0.0
        and math.isfinite(preconditioned_square_value)
        and preconditioned_square_value > 0.0
    )
    if not ordinary:
        magnitude, residual_finite = _channel_pressure_working_magnitude(
            residual,
            inactive_mask,
            finite_companions=finite_companions,
            active_count=active_count,
        )
        breakdown, pair_finite = _channel_pressure_working_breakdown(
            residual,
            preconditioned,
            inactive_mask,
            inner_product=inner_product,
        )
        return magnitude, residual_finite, breakdown, pair_finite
    rms_value = math.sqrt(residual_square_value / active_count)
    mantissa, exponent = math.frexp(rms_value)
    magnitude = _channel_pressure_magnitude_from_binary(mantissa, exponent)
    threshold = torch.finfo(residual.real.dtype).eps * math.sqrt(
        residual_square_value * preconditioned_square_value
    )
    return magnitude, True, inner_value <= threshold, True


def channel_pressure_scale_power_of_two(
    value: torch.Tensor,
    binary_exponent: int,
) -> torch.Tensor:
    """Multiply a real or complex tensor by 2**exponent without scalar loss."""

    if not isinstance(binary_exponent, int) or isinstance(binary_exponent, bool):
        raise TypeError("binary exponent must be an integer")

    def scale_real(component: torch.Tensor) -> torch.Tensor:
        remaining = int(binary_exponent)
        result = component
        real_dtype = component.dtype
        maximum_step = math.frexp(torch.finfo(real_dtype).max)[1] - 1
        minimum_step = math.frexp(torch.finfo(real_dtype).tiny)[1] - 1
        while remaining:
            step = min(remaining, maximum_step) if remaining > 0 else max(
                remaining,
                minimum_step,
            )
            exponent = torch.tensor(
                step,
                device=component.device,
                dtype=torch.int32,
            )
            result = torch.ldexp(result, exponent)
            remaining -= step
        return result

    if value.is_complex():
        return torch.complex(scale_real(value.real), scale_real(value.imag))
    return scale_real(value)


def channel_pressure_working_vector(
    value: torch.Tensor,
    inactive_mask: torch.Tensor,
) -> tuple[torch.Tensor, int | None, ChannelPressureMagnitude]:
    """Project and scale a pressure vector into a safe quadratic range."""

    projected, component_amax, active_count = _channel_pressure_component_amax(
        value,
        inactive_mask,
    )
    if component_amax == 0.0:
        return projected, None, _channel_pressure_zero_magnitude()
    working_exponent = math.frexp(component_amax)[1] - 1
    working = channel_pressure_scale_power_of_two(
        projected,
        -working_exponent,
    )
    squared = working.real.square() + working.imag.square()
    rms_factor = math.sqrt(float((squared.sum() / active_count).item()))
    if not math.isfinite(rms_factor) or rms_factor <= 0.0:
        raise ValueError("Channel pressure subspace has no active coefficients")
    factor_mantissa, factor_exponent = math.frexp(rms_factor)
    magnitude = _channel_pressure_magnitude_from_binary(
        factor_mantissa,
        working_exponent + factor_exponent,
    )
    return working, int(working_exponent), magnitude


def channel_pressure_magnitude_ratio(
    numerator: ChannelPressureMagnitude,
    denominator: ChannelPressureMagnitude,
) -> float:
    if denominator.is_zero:
        if numerator.is_zero:
            return 0.0
        raise ValueError("nonzero magnitude cannot be normalized by zero")
    if numerator.is_zero:
        return 0.0
    difference = float(numerator.log2_value) - float(denominator.log2_value)
    try:
        # ``math.exp2`` is unavailable on supported Python 3.10.  ``math.pow``
        # preserves the same base-two ratio and overflow contract across all
        # supported Python versions without introducing a version branch.
        value = math.pow(2.0, difference)
    except OverflowError as exc:
        raise ValueError("normalized pressure magnitude is not finite") from exc
    if not math.isfinite(value):
        raise ValueError("normalized pressure magnitude is not finite")
    return float(value)


def channel_pressure_relative_breakdown(
    left: torch.Tensor,
    right: torch.Tensor,
    inactive_mask: torch.Tensor,
) -> bool:
    """Evaluate the frozen scale-relative active-mode inner-product gate."""

    left_projected, mask = _channel_pressure_project_active(left, inactive_mask)
    right_projected, _right_mask = _channel_pressure_project_active(
        right,
        inactive_mask,
    )
    maxima = torch.stack(
        (
            torch.maximum(
                left_projected.real.abs(),
                left_projected.imag.abs(),
            ).amax(),
            torch.maximum(
                right_projected.real.abs(),
                right_projected.imag.abs(),
            ).amax(),
        )
    )
    left_maximum, right_maximum = (
        float(value) for value in maxima.tolist()
    )
    if not all(math.isfinite(value) for value in (left_maximum, right_maximum)):
        raise ValueError("Channel pressure vector contains NaN or Inf")
    if left_maximum == 0.0 or right_maximum == 0.0:
        return True
    left_exponent = math.frexp(left_maximum)[1] - 1
    right_exponent = math.frexp(right_maximum)[1] - 1
    left_work = channel_pressure_scale_power_of_two(
        left_projected,
        -left_exponent,
    )
    right_work = channel_pressure_scale_power_of_two(
        right_projected,
        -right_exponent,
    )
    active = ~mask
    left_active = left_work.masked_select(active)
    right_active = right_work.masked_select(active)
    mean_inner = torch.mean(torch.conj(left_active) * right_active)
    left_rms = torch.sqrt(
        torch.mean(left_active.real.square() + left_active.imag.square())
    )
    right_rms = torch.sqrt(
        torch.mean(right_active.real.square() + right_active.imag.square())
    )
    epsilon = torch.finfo(left.real.dtype).eps
    return bool(
        (
            torch.abs(mean_inner)
            <= epsilon * left_rms * right_rms
        ).item()
    )


def _finite_coefficient(
    value: object,
    description: str,
    *,
    positive: bool,
) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
        or (not positive and float(value) < 0.0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{description} must be finite and {qualifier}")
    return float(value)


def _positive_integer(value: object, description: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) <= 0
    ):
        raise ValueError(f"{description} must be a positive integer")
    return int(value)


class ChannelNoSlipModalStokesSolver(torch.nn.Module):
    """Matrix-free mixed FFT/DST/DCT Channel Stokes/Brinkman solve.

    ``pressure_fixed_iterations`` selects exact fixed work.  In that mode
    ``pressure_max_iterations`` is inactive and relative tolerance is
    diagnostic-only; only an exact zero residual may terminate early.
    """

    def __init__(
        self,
        transform_backend,
        *,
        friction: float = 0.0,
        viscosity: float = 1.0,
        pressure_relative_tolerance: float = 1.0e-6,
        pressure_max_iterations: int = 80,
        pressure_fixed_iterations: int | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(transform_backend, "get_metadata"):
            raise TypeError("transform_backend must provide transform metadata")
        if len(tuple(transform_backend.shape)) != 3:
            raise ValueError("Channel Stokes solver requires three dimensions")
        friction = _finite_coefficient(
            friction,
            "friction",
            positive=False,
        )
        viscosity = _finite_coefficient(
            viscosity,
            "viscosity",
            positive=True,
        )
        pressure_relative_tolerance = _finite_coefficient(
            pressure_relative_tolerance,
            "pressure_relative_tolerance",
            positive=True,
        )
        pressure_max_iterations = _positive_integer(
            pressure_max_iterations,
            "pressure_max_iterations",
        )
        if pressure_fixed_iterations is not None:
            pressure_fixed_iterations = _positive_integer(
                pressure_fixed_iterations,
                "pressure_fixed_iterations",
            )

        backend = transform_backend
        spectral_dtype = backend.spectral_dtype
        real_dtype = backend.real_dtype
        device = backend.device
        velocity_metadata = backend.get_metadata(
            CHANNEL_VELOCITY_BOUNDARY_CONDITIONS
        )
        pressure_metadata = backend.get_metadata(
            CHANNEL_PRESSURE_BOUNDARY_CONDITIONS
        )
        qx = velocity_metadata.axis_modes[0].clone()
        ky_d = velocity_metadata.axis_modes[1]
        kz_d = velocity_metadata.axis_modes[2]
        ky_n = pressure_metadata.axis_modes[1]
        kz_n = pressure_metadata.axis_modes[2]
        nx, ny, nz = qx.numel(), ky_d.numel(), kz_d.numel()
        nyquist_mask = torch.zeros(
            (1, nx, ny, nz),
            device=device,
            dtype=torch.bool,
        )
        periodic_size = int(backend.shape[0])
        if periodic_size % 2 == 0:
            nyquist_index = periodic_size // 2
            if nyquist_index < nx:
                qx[nyquist_index] = 0
                nyquist_mask[:, nyquist_index, :, :] = True

        wall_ops = {}
        for axis, size, k_dirichlet in (
            (1, ny, ky_d),
            (2, nz, kz_d),
        ):
            dst = backend._get_matrix("dst", size).to(
                device=device,
                dtype=real_dtype,
            )
            dct = backend._get_matrix("dct", size).to(
                device=device,
                dtype=real_dtype,
            )
            d_d_to_n = torch.zeros(
                (size, size),
                device=device,
                dtype=real_dtype,
            )
            if size > 1:
                index = torch.arange(size - 1, device=device)
                d_d_to_n[index, index + 1] = k_dirichlet[:-1]
            wall_ops[axis] = {
                "n_to_d": (dct @ dst.transpose(0, 1)).to(spectral_dtype),
                "d_to_n": (dst @ dct.transpose(0, 1)).to(spectral_dtype),
                "dn_to_dd": (-d_d_to_n.transpose(0, 1)).to(spectral_dtype),
                "dd_to_dn": d_d_to_n.to(spectral_dtype),
            }

        a_diag = friction + viscosity * (
            qx.square().view(nx, 1, 1)
            + ky_d.square().view(1, ny, 1)
            + kz_d.square().view(1, 1, nz)
        )
        pressure_q2 = (
            qx.square().view(nx, 1, 1)
            + ky_n.square().view(1, ny, 1)
            + kz_n.square().view(1, 1, nz)
        )
        schur_diag = pressure_q2 / (
            friction
            + viscosity
            * pressure_q2.clamp_min(torch.finfo(real_dtype).eps)
        )
        null_mask = torch.zeros(
            (1, nx, ny, nz),
            device=device,
            dtype=torch.bool,
        )
        null_mask[:, 0, 0, 0] = True
        schur_safe = schur_diag.unsqueeze(0).clone()
        schur_safe[null_mask | nyquist_mask] = 1.0

        self.register_buffer(
            "ikx",
            (1j * qx).view(1, nx, 1, 1).to(spectral_dtype),
        )
        self.register_buffer("a_inv", (1.0 / a_diag).unsqueeze(0))
        for axis_name, axis in (("y", 1), ("z", 2)):
            for operator_name, operator in wall_ops[axis].items():
                self.register_buffer(f"{operator_name}_{axis_name}", operator)
        self.register_buffer("schur_diag_safe", schur_safe)
        self.register_buffer("pressure_null_mask", null_mask)
        # The periodic Nyquist plane has no real first derivative on an even
        # grid.  It is derived from the transform shape and deliberately kept
        # out of state_dict to preserve the checkpoint schema.
        self.register_buffer(
            "nyquist_mask",
            nyquist_mask,
            persistent=False,
        )
        self._active_pressure_degrees_of_freedom = int(
            (~(null_mask | nyquist_mask)).sum().item()
        )
        self.pressure_rel_tol = pressure_relative_tolerance
        self.pressure_max_iter = pressure_max_iterations
        self.pressure_fixed_iterations = pressure_fixed_iterations
        self.pressure_guess = None
        self.last_pressure_iterations = 0
        self.last_pressure_residual = 0.0
        self.last_pressure_relative_residual = 0.0
        self.last_pressure_residual_magnitude = (
            _channel_pressure_zero_magnitude().to_metadata()
        )
        self.last_pressure_termination_reason = "not_run"
        self.last_pressure_acceptable = False
        self.last_pressure_working_scale_binary_exponent = None
        self._pressure_solve_sequence_id = 0
        self.last_pressure_diagnostics = self._pressure_diagnostics_metadata(
            rhs_norm=_channel_pressure_zero_magnitude(),
            initial_residual=_channel_pressure_zero_magnitude(),
            achieved_residual=_channel_pressure_zero_magnitude(),
            relative_residual=None,
            iterations=0,
            termination_reason="not_run",
            acceptable=False,
            working_scale_binary_exponent=None,
            solve_phase="not_run",
        )

    def pressure_convergence_policy(self) -> dict[str, object]:
        """Return the explicit precedence of the PCG stopping controls."""

        if self.pressure_fixed_iterations is None:
            return {
                "schema_version": 1,
                "convergence_mode": "relative_tolerance",
                "authoritative_iteration_limit": self.pressure_iteration_limit,
                "authoritative_iteration_limit_source": "max_iterations",
                "max_iterations_role": "authoritative_iteration_cap",
                "relative_tolerance_role": "required_postcondition",
                "initial_guess_policy": "persistent_production_warm_start",
                "rhs_norm_definition": CHANNEL_PRESSURE_RHS_NORM_DEFINITION,
                "working_vector_scaling": (
                    CHANNEL_PRESSURE_WORKING_VECTOR_SCALING
                ),
                "breakdown_definition": CHANNEL_PRESSURE_BREAKDOWN_DEFINITION,
            }
        return {
            "schema_version": 1,
            "convergence_mode": "fixed_work",
            "authoritative_iteration_limit": self.pressure_iteration_limit,
            "authoritative_iteration_limit_source": "fixed_iterations",
            "max_iterations_role": "inactive_compatibility_field",
            "relative_tolerance_role": "diagnostic_only_not_early_exit",
            "initial_guess_policy": "persistent_production_warm_start",
            "rhs_norm_definition": CHANNEL_PRESSURE_RHS_NORM_DEFINITION,
            "working_vector_scaling": CHANNEL_PRESSURE_WORKING_VECTOR_SCALING,
            "breakdown_definition": CHANNEL_PRESSURE_BREAKDOWN_DEFINITION,
        }

    @property
    def pressure_iteration_limit(self) -> int:
        """Return the currently authoritative fixed-work or tolerance cap."""

        if self.pressure_fixed_iterations is not None:
            return int(self.pressure_fixed_iterations)
        return int(self.pressure_max_iter)

    @property
    def pressure_inactive_mask(self) -> torch.Tensor:
        return self.pressure_null_mask | self.nyquist_mask

    def _legacy_magnitude_value(
        self,
        magnitude: ChannelPressureMagnitude,
    ) -> float:
        if magnitude.float64_value is not None:
            return float(magnitude.float64_value)
        if float(magnitude.log2_value) < -1074.0:
            return 0.0
        return float(torch.finfo(torch.float64).max)

    def _record_pressure_outcome(
        self,
        *,
        rhs_norm: ChannelPressureMagnitude,
        initial_residual: ChannelPressureMagnitude,
        iterations: int,
        residual: ChannelPressureMagnitude,
        relative_residual: float | None,
        termination_reason: str,
        acceptable: bool,
        working_scale_binary_exponent: int | None,
        solve_phase: str = "production_primal",
    ) -> None:
        self.last_pressure_iterations = int(iterations)
        self.last_pressure_residual = self._legacy_magnitude_value(residual)
        self.last_pressure_relative_residual = (
            None if relative_residual is None else float(relative_residual)
        )
        self.last_pressure_residual_magnitude = residual.to_metadata()
        self.last_pressure_termination_reason = termination_reason
        self.last_pressure_acceptable = bool(acceptable)
        self.last_pressure_working_scale_binary_exponent = (
            working_scale_binary_exponent
        )
        self.last_pressure_diagnostics = self._pressure_diagnostics_metadata(
            rhs_norm=rhs_norm,
            initial_residual=initial_residual,
            achieved_residual=residual,
            relative_residual=relative_residual,
            iterations=iterations,
            termination_reason=termination_reason,
            acceptable=acceptable,
            working_scale_binary_exponent=working_scale_binary_exponent,
            solve_phase=solve_phase,
        )

    def _pressure_diagnostics_metadata(
        self,
        *,
        rhs_norm: ChannelPressureMagnitude,
        initial_residual: ChannelPressureMagnitude,
        achieved_residual: ChannelPressureMagnitude,
        relative_residual: float | None,
        iterations: int,
        termination_reason: str,
        acceptable: bool,
        working_scale_binary_exponent: int | None,
        solve_phase: str,
    ) -> dict[str, object]:
        policy = self.pressure_convergence_policy()
        fixed_work = self.pressure_fixed_iterations is not None
        required_postcondition = (
            "zero_rhs_or_exact_zero_residual_or_exact_requested_fixed_work"
            if fixed_work
            else (
                "zero_rhs_or_exact_zero_residual_or_normalized_residual_not_"
                "above_requested_relative_tolerance"
            )
        )
        return {
            "schema_version": 2,
            "solve_sequence_id": self._pressure_solve_sequence_id,
            "source_primal_sequence_id": None,
            "solver_identity": (
                "persistent_warm_start_preconditioned_conjugate_gradient"
            ),
            "operator_identity": "channel_pressure_schur_primal",
            "solve_phase": solve_phase,
            "initial_guess_policy": "persistent_production_warm_start",
            "convergence_mode": policy["convergence_mode"],
            "requested_relative_tolerance": self.pressure_rel_tol,
            "requested_fixed_iterations": self.pressure_fixed_iterations,
            "requested_max_iterations": self.pressure_max_iter,
            "authoritative_iteration_limit": (
                policy["authoritative_iteration_limit"]
            ),
            "authoritative_iteration_limit_source": (
                policy["authoritative_iteration_limit_source"]
            ),
            "max_iterations_role": policy["max_iterations_role"],
            "relative_tolerance_role": policy["relative_tolerance_role"],
            "active_pressure_degrees_of_freedom": (
                self._active_pressure_degrees_of_freedom
            ),
            "rhs_norm_definition": CHANNEL_PRESSURE_RHS_NORM_DEFINITION,
            "rhs_norm": rhs_norm.to_metadata(),
            "initial_absolute_residual": initial_residual.to_metadata(),
            "achieved_absolute_residual": achieved_residual.to_metadata(),
            "normalized_residual_definition": (
                CHANNEL_PRESSURE_NORMALIZED_RESIDUAL_DEFINITION
            ),
            "achieved_normalized_residual": relative_residual,
            "working_vector_scaling": CHANNEL_PRESSURE_WORKING_VECTOR_SCALING,
            "working_scale_binary_exponent": (
                working_scale_binary_exponent
            ),
            "breakdown_definition": CHANNEL_PRESSURE_BREAKDOWN_DEFINITION,
            "achieved_iteration_count": int(iterations),
            "termination_reason": termination_reason,
            "required_postcondition": required_postcondition,
            "postcondition_satisfied": bool(acceptable),
            "acceptable": bool(acceptable),
        }

    def _raise_pressure_failure(
        self,
        *,
        rhs_norm: ChannelPressureMagnitude,
        initial_residual: ChannelPressureMagnitude,
        iterations: int,
        residual: ChannelPressureMagnitude,
        relative_residual: float | None,
        termination_reason: str,
        working_scale_binary_exponent: int | None,
    ) -> None:
        self._record_pressure_outcome(
            rhs_norm=rhs_norm,
            initial_residual=initial_residual,
            iterations=iterations,
            residual=residual,
            relative_residual=relative_residual,
            termination_reason=termination_reason,
            acceptable=False,
            working_scale_binary_exponent=(
                working_scale_binary_exponent
            ),
        )
        raise ChannelPressureConvergenceError(
            "production Channel pressure solve did not satisfy its contract",
            details=self.last_pressure_diagnostics,
        )

    def _apply_axis_matrix(self, tensor, matrix, axis):
        spectral_axis = tensor.ndim - 3 + axis
        moved = tensor.movedim(spectral_axis, -1)
        return torch.matmul(moved, matrix).movedim(-1, spectral_axis)

    def _project_pressure_gauge(self, pressure_hat):
        return pressure_hat.masked_fill(
            self.pressure_null_mask | self.nyquist_mask,
            0,
        )

    def _helmholtz_inverse(self, rhs_hat):
        return (rhs_hat * self.a_inv).masked_fill(self.nyquist_mask, 0)

    def _pressure_to_velocity(self, pressure_hat):
        out = self._apply_axis_matrix(pressure_hat, self.n_to_d_y, axis=1)
        return self._apply_axis_matrix(out, self.n_to_d_z, axis=2)

    def _velocity_to_pressure(self, velocity_hat):
        out = self._apply_axis_matrix(velocity_hat, self.d_to_n_y, axis=1)
        return self._apply_axis_matrix(out, self.d_to_n_z, axis=2)

    def _pressure_gradient(self, pressure_hat, axis):
        if axis == 0:
            return self.ikx * self._pressure_to_velocity(pressure_hat)
        if axis == 1:
            out = self._apply_axis_matrix(
                pressure_hat,
                self.dn_to_dd_y,
                axis=1,
            )
            return self._apply_axis_matrix(out, self.n_to_d_z, axis=2)
        if axis == 2:
            out = self._apply_axis_matrix(
                pressure_hat,
                self.n_to_d_y,
                axis=1,
            )
            return self._apply_axis_matrix(out, self.dn_to_dd_z, axis=2)
        raise IndexError(f"Unsupported pressure-gradient axis {axis}.")

    def _velocity_divergence_component(self, velocity_hat, axis):
        if axis == 0:
            return self._velocity_to_pressure(self.ikx * velocity_hat)
        if axis == 1:
            out = self._apply_axis_matrix(
                velocity_hat,
                self.dd_to_dn_y,
                axis=1,
            )
            return self._apply_axis_matrix(out, self.d_to_n_z, axis=2)
        if axis == 2:
            out = self._apply_axis_matrix(
                velocity_hat,
                self.d_to_n_y,
                axis=1,
            )
            return self._apply_axis_matrix(out, self.dd_to_dn_z, axis=2)
        raise IndexError(f"Unsupported velocity-divergence axis {axis}.")

    def pressure_gradient_hats(self, pressure_hat):
        """Return pressure-gradient coefficients in velocity bases."""

        return tuple(
            self._pressure_gradient(pressure_hat, axis)
            for axis in range(3)
        )

    def divergence_hat(self, ux_hat, uy_hat, uz_hat):
        """Return velocity divergence in the native pressure basis."""

        return sum(
            self._velocity_divergence_component(velocity_hat, axis)
            for axis, velocity_hat in enumerate((ux_hat, uy_hat, uz_hat))
        )

    def _pressure_operator(self, pressure_hat):
        pressure_hat = self._project_pressure_gauge(pressure_hat)
        velocity_hats = [
            self._helmholtz_inverse(
                self._pressure_gradient(pressure_hat, axis)
            )
            for axis in range(3)
        ]
        divergence_hat = self.divergence_hat(*velocity_hats)
        return self._project_pressure_gauge(-divergence_hat)

    def _solve_pressure(self, rhs_hat):
        self._pressure_solve_sequence_id += 1
        rhs_hat = self._project_pressure_gauge(rhs_hat)
        if not bool(torch.isfinite(rhs_hat).all().item()):
            unavailable = _channel_pressure_zero_magnitude()
            self._record_pressure_outcome(
                rhs_norm=unavailable,
                initial_residual=unavailable,
                iterations=0,
                residual=unavailable,
                relative_residual=None,
                termination_reason="nonfinite_input",
                acceptable=False,
                working_scale_binary_exponent=None,
            )
            raise ChannelPressureConvergenceError(
                "production Channel pressure input contains NaN or Inf",
                details=self.last_pressure_diagnostics,
            )
        working_rhs, working_exponent, rhs_magnitude = (
            channel_pressure_working_vector(
                rhs_hat,
                self.pressure_inactive_mask,
            )
        )
        zero = _channel_pressure_zero_magnitude()
        if rhs_magnitude.is_zero:
            pressure_hat = torch.zeros_like(rhs_hat)
            self.pressure_guess = pressure_hat.detach()
            self._record_pressure_outcome(
                rhs_norm=rhs_magnitude,
                initial_residual=zero,
                iterations=0,
                residual=zero,
                relative_residual=0.0,
                termination_reason="zero_rhs",
                acceptable=True,
                working_scale_binary_exponent=None,
            )
            return pressure_hat
        if self.pressure_guess is None or self.pressure_guess.shape != rhs_hat.shape:
            pressure_work = torch.zeros_like(working_rhs)
        else:
            pressure_work = self._project_pressure_gauge(
                channel_pressure_scale_power_of_two(
                    self.pressure_guess.to(
                        device=rhs_hat.device,
                        dtype=rhs_hat.dtype,
                    ),
                    -int(working_exponent),
                )
            )
        residual = self._project_pressure_gauge(
            working_rhs - self._pressure_operator(pressure_work)
        )
        active_count = (
            self._active_pressure_degrees_of_freedom
            * residual.numel()
            // self.pressure_inactive_mask.numel()
        )
        preconditioned = residual / self.schur_diag_safe
        rz_old = torch.sum(torch.conj(residual) * preconditioned).real
        (
            residual_work_magnitude,
            initial_state_finite,
            initial_breakdown,
            initial_pair_finite,
        ) = _channel_pressure_working_residual_observation(
            residual,
            preconditioned,
            self.pressure_inactive_mask,
            inner_product=rz_old,
            active_count=active_count,
            finite_companions=(pressure_work,),
        )
        if not initial_state_finite:
            self._raise_pressure_failure(
                rhs_norm=rhs_magnitude,
                initial_residual=rhs_magnitude,
                iterations=0,
                residual=rhs_magnitude,
                relative_residual=1.0,
                termination_reason="nonfinite_intermediate",
                working_scale_binary_exponent=working_exponent,
            )
        rhs_work_magnitude = rhs_magnitude.shifted(-int(working_exponent))
        residual_magnitude = residual_work_magnitude.shifted(
            int(working_exponent)
        )
        initial_residual = residual_magnitude
        relative_residual = channel_pressure_magnitude_ratio(
            residual_work_magnitude,
            rhs_work_magnitude,
        )
        if residual_work_magnitude.is_zero:
            pressure_hat = channel_pressure_scale_power_of_two(
                pressure_work,
                int(working_exponent),
            )
            self.pressure_guess = pressure_hat.detach()
            self._record_pressure_outcome(
                rhs_norm=rhs_magnitude,
                initial_residual=initial_residual,
                iterations=0,
                residual=zero,
                relative_residual=0.0,
                termination_reason="exact_zero_residual",
                acceptable=True,
                working_scale_binary_exponent=working_exponent,
            )
            return pressure_hat
        fixed_work = self.pressure_fixed_iterations is not None
        if not fixed_work and relative_residual <= self.pressure_rel_tol:
            pressure_hat = channel_pressure_scale_power_of_two(
                pressure_work,
                int(working_exponent),
            )
            self.pressure_guess = pressure_hat.detach()
            self._record_pressure_outcome(
                rhs_norm=rhs_magnitude,
                initial_residual=initial_residual,
                iterations=0,
                residual=residual_magnitude,
                relative_residual=relative_residual,
                termination_reason="relative_tolerance_met",
                acceptable=True,
                working_scale_binary_exponent=working_exponent,
            )
            return pressure_hat
        if not initial_pair_finite:
            self._raise_pressure_failure(
                rhs_norm=rhs_magnitude,
                initial_residual=initial_residual,
                iterations=0,
                residual=residual_magnitude,
                relative_residual=relative_residual,
                termination_reason="nonfinite_intermediate",
                working_scale_binary_exponent=working_exponent,
            )
        if initial_breakdown:
            self._raise_pressure_failure(
                rhs_norm=rhs_magnitude,
                initial_residual=initial_residual,
                iterations=0,
                residual=residual_magnitude,
                relative_residual=relative_residual,
                termination_reason="preconditioned_residual_breakdown",
                working_scale_binary_exponent=working_exponent,
            )
        direction = preconditioned.clone()
        iterations = 0
        iteration_limit = self.pressure_iteration_limit
        termination_reason = "iteration_limit_reached_without_postcondition"
        acceptable = False
        while iterations < iteration_limit:
            operator_direction = self._pressure_operator(direction)
            direction_operator_inner = torch.sum(
                torch.conj(direction) * operator_direction
            )
            operator_breakdown, operator_pair_finite = (
                _channel_pressure_working_breakdown(
                    direction,
                    operator_direction,
                    self.pressure_inactive_mask,
                    inner_product=direction_operator_inner,
                )
            )
            if not operator_pair_finite:
                termination_reason = "nonfinite_intermediate"
                break
            if operator_breakdown:
                termination_reason = "operator_breakdown"
                break
            denominator = direction_operator_inner.real
            step = rz_old / denominator
            pressure_work = self._project_pressure_gauge(
                pressure_work + step * direction
            )
            residual = self._project_pressure_gauge(
                residual - step * operator_direction
            )
            iterations += 1
            preconditioned = residual / self.schur_diag_safe
            rz_new = torch.sum(torch.conj(residual) * preconditioned).real
            (
                residual_work_magnitude,
                state_finite,
                preconditioned_breakdown,
                preconditioned_pair_finite,
            ) = _channel_pressure_working_residual_observation(
                residual,
                preconditioned,
                self.pressure_inactive_mask,
                inner_product=rz_new,
                active_count=active_count,
                finite_companions=(pressure_work,),
            )
            if not state_finite:
                termination_reason = "nonfinite_intermediate"
                break
            residual_magnitude = residual_work_magnitude.shifted(
                int(working_exponent)
            )
            relative_residual = channel_pressure_magnitude_ratio(
                residual_work_magnitude,
                rhs_work_magnitude,
            )
            if residual_work_magnitude.is_zero:
                termination_reason = "exact_zero_residual"
                acceptable = True
                break
            if not fixed_work and relative_residual <= self.pressure_rel_tol:
                termination_reason = "relative_tolerance_met"
                acceptable = True
                break
            if iterations == iteration_limit:
                if fixed_work:
                    termination_reason = "fixed_work_completed"
                    acceptable = True
                break
            if not preconditioned_pair_finite:
                termination_reason = "nonfinite_intermediate"
                break
            if preconditioned_breakdown:
                termination_reason = "preconditioned_residual_breakdown"
                break
            direction = preconditioned + (rz_new / rz_old) * direction
            rz_old = rz_new
        if not acceptable:
            self._raise_pressure_failure(
                rhs_norm=rhs_magnitude,
                initial_residual=initial_residual,
                iterations=iterations,
                residual=residual_magnitude,
                relative_residual=relative_residual,
                termination_reason=termination_reason,
                working_scale_binary_exponent=working_exponent,
            )
        pressure_hat = channel_pressure_scale_power_of_two(
            pressure_work,
            int(working_exponent),
        )
        if not bool(torch.isfinite(pressure_hat).all().item()):
            self._raise_pressure_failure(
                rhs_norm=rhs_magnitude,
                initial_residual=initial_residual,
                iterations=iterations,
                residual=residual_magnitude,
                relative_residual=relative_residual,
                termination_reason="nonfinite_intermediate",
                working_scale_binary_exponent=working_exponent,
            )
        self.pressure_guess = pressure_hat.detach()
        self._record_pressure_outcome(
            rhs_norm=rhs_magnitude,
            initial_residual=initial_residual,
            iterations=iterations,
            residual=residual_magnitude,
            relative_residual=relative_residual,
            termination_reason=termination_reason,
            acceptable=True,
            working_scale_binary_exponent=working_exponent,
        )
        return pressure_hat

    def solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        """Solve the Channel saddle system for native force spectra."""

        force_hats = tuple(
            force_hat.masked_fill(self.nyquist_mask, 0)
            for force_hat in (fx_hat, fy_hat, fz_hat)
        )
        free_velocity = [
            self._helmholtz_inverse(force_hat)
            for force_hat in force_hats
        ]
        provisional_divergence = self.divergence_hat(*free_velocity)
        pressure_hat = self._solve_pressure(
            self._project_pressure_gauge(-provisional_divergence)
        )
        velocity_hat = [
            free_velocity[axis]
            - self._helmholtz_inverse(
                self._pressure_gradient(pressure_hat, axis)
            )
            for axis in range(3)
        ]
        return (*velocity_hat, pressure_hat)


__all__ = [
    "CHANNEL_PRESSURE_BREAKDOWN_DEFINITION",
    "CHANNEL_PRESSURE_BOUNDARY_CONDITIONS",
    "CHANNEL_PRESSURE_NORMALIZED_RESIDUAL_DEFINITION",
    "CHANNEL_PRESSURE_RHS_NORM_DEFINITION",
    "CHANNEL_PRESSURE_WORKING_VECTOR_SCALING",
    "CHANNEL_VELOCITY_BOUNDARY_CONDITIONS",
    "ChannelPressureConvergenceError",
    "ChannelPressureMagnitude",
    "ChannelNoSlipModalStokesSolver",
    "channel_pressure_active_rms",
    "channel_pressure_magnitude_ratio",
    "channel_pressure_relative_breakdown",
    "channel_pressure_scale_power_of_two",
    "channel_pressure_working_vector",
]
