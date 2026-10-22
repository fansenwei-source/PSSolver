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

from pssolver.core.pressure import (
    channel_pressure_minimum_relative_tolerance_for_precision,
)
from pssolver.core.errors import SimulationExecutionError


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
    "active_rms(projected_pcg_recurrence_residual)/"
    "active_rms(projected_rhs)"
)
CHANNEL_PRESSURE_WORKING_VECTOR_SCALING = (
    "largest_power_of_two_not_greater_than_active_component_amax"
)
CHANNEL_PRESSURE_BREAKDOWN_DEFINITION = (
    "abs(mean_active(conj(left)*right))"
    "<=eps*active_rms(left)*active_rms(right)"
)
_CHANNEL_PRESSURE_BREAKDOWN_CERTIFICATE_EPSILON_MARGIN = 4096.0
CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION = (
    "fixed_work_roundoff_saturation"
)


def channel_pressure_minimum_relative_tolerance(
    dtype: torch.dtype | str,
) -> float:
    """Return the smallest supported tolerance-mode pressure target.

    The PCG recurrence residual can continue decreasing below the accuracy of
    the represented solution even though the residual of the returned state
    has reached its dtype-dependent roundoff floor.  Rejecting targets below a
    conservative supported floor excludes the empirically observed region in
    which those quantities decouple.  ``relative_tolerance_met`` is explicitly
    a postcondition on the projected PCG recurrence residual; it does not claim
    that a separately recomputed residual was evaluated on the hot path.

    Fixed-work mode does not use the tolerance as a stopping postcondition and
    therefore deliberately does not apply this floor.
    """

    # The independent rc5 acceptance matrix found recurrence/represented-state
    # residual disagreement at two and four epsilons.  Eight epsilons remains
    # the public floor; tolerance-mode solves independently verify the
    # residual of the represented return value before accepting it.
    real_dtype = _channel_pressure_real_dtype(dtype)
    precision = "float32" if real_dtype == torch.float32 else "float64"
    return channel_pressure_minimum_relative_tolerance_for_precision(precision)


def _channel_pressure_real_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Return the real component dtype without constructing a tensor view."""

    if isinstance(dtype, str):
        try:
            return {
                "float32": torch.float32,
                "float64": torch.float64,
                "complex64": torch.float32,
                "complex128": torch.float64,
            }[dtype]
        except KeyError as exc:
            raise TypeError(
                f"unsupported Channel pressure dtype: {dtype}"
            ) from exc
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    if dtype in {torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        return dtype
    raise TypeError(f"unsupported Channel pressure dtype: {dtype}")


class ChannelPressureConvergenceError(SimulationExecutionError):
    """Fail-closed production Channel pressure-solve error.

    The pressure solver does not own the enclosing workflow clock, so its
    stable execution-error context names the inner solve and uses step zero.
    Workflow-level failure metadata separately records the runtime's actual
    completed-step clock.
    """

    code = "channel_pressure_convergence_error"

    def __init__(self, message: str, *, details: dict[str, object]) -> None:
        super().__init__(
            message,
            application_family="channel_pressure",
            stage="pressure_solve",
            step=0,
            details=details,
        )
        # Preserve the pre-rc6 public mapping behavior of this established
        # exception while still inheriting the common execution-error API.
        self.details = dict(self.details)
        self.diagnostics = dict(self.details)

    def __reduce__(self):
        """Preserve the public exception contract across pickle protocols."""

        extra_state = {
            name: value
            for name, value in self.__dict__.items()
            if name
            not in {
                "application_family",
                "stage",
                "step",
                "details",
                "diagnostics",
            }
        }
        return (
            _restore_channel_pressure_convergence_error,
            (
                type(self),
                str(self),
                self.application_family,
                self.stage,
                self.step,
                dict(self.details),
                dict(self.diagnostics),
                extra_state,
            ),
        )


def _restore_channel_pressure_convergence_error(
    error_type,
    message: str,
    application_family: str,
    stage: str,
    step: int,
    details: dict[str, object],
    diagnostics: dict[str, object],
    extra_state: dict[str, object],
):
    """Rebuild Channel pressure errors without narrowing subclass identity."""

    error = error_type.__new__(error_type)
    SimulationExecutionError.__init__(
        error,
        message,
        application_family=application_family,
        stage=stage,
        step=step,
        details=details,
    )
    # Match ChannelPressureConvergenceError's established mutable mapping
    # projections and restore FunctionalConvergenceError.operation when the
    # dual-hierarchy subclass is pickled.
    error.details = dict(error.details)
    error.diagnostics = dict(diagnostics)
    for name, value in extra_state.items():
        setattr(error, name, value)
    return error


def channel_pressure_fixed_work_roundoff_saturated(
    relative_residual: float | None,
    dtype: torch.dtype | str,
) -> bool:
    """Return whether fixed-work PCG reached its represented precision floor."""

    return bool(
        relative_residual is not None
        and math.isfinite(float(relative_residual))
        and float(relative_residual)
        <= channel_pressure_minimum_relative_tolerance(dtype)
    )


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
    active_count = (~mask).sum(
        dtype=_channel_pressure_real_dtype(projected.dtype)
    )
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
    flattened = value.reshape(-1)
    square_sum = torch.vdot(flattened, flattened).real
    metrics = [square_sum]
    metrics.extend(
        torch.maximum(
            companion.real.abs(),
            companion.imag.abs(),
        ).amax()
        for companion in finite_companions
    )
    observed = torch.stack(metrics).tolist()
    square_sum_value = float(observed[0])
    companions_finite = all(math.isfinite(item) for item in observed[1:])
    if not companions_finite:
        return _channel_pressure_zero_magnitude(), False
    if not math.isfinite(square_sum_value) or square_sum_value <= 0.0:
        # This branch is intentionally exceptional.  It preserves the public
        # underflow/overflow contract without charging the ordinary CUDA path
        # for a second scalar synchronization on every PCG iteration.
        try:
            return channel_pressure_active_rms(value, inactive_mask), True
        except ValueError:
            return _channel_pressure_zero_magnitude(), False
    rms_value = math.sqrt(square_sum_value / float(count))
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
    left_flattened = left.reshape(-1)
    right_flattened = right.reshape(-1)
    left_square_sum = torch.vdot(left_flattened, left_flattened).real
    right_square_sum = torch.vdot(right_flattened, right_flattened).real
    observed = torch.stack(
        (
            torch.abs(inner_product),
            left_square_sum,
            right_square_sum,
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
    epsilon = torch.finfo(_channel_pressure_real_dtype(left.dtype)).eps
    threshold = epsilon * math.sqrt(left_square_value) * math.sqrt(
        right_square_value
    )
    if not math.isfinite(threshold):
        try:
            return channel_pressure_relative_breakdown(
                left,
                right,
                inactive_mask,
            ), True
        except ValueError:
            return False, False
    return inner_value <= threshold, True


def _channel_pressure_certified_inner_observation(
    inner_product: torch.Tensor,
) -> tuple[bool, bool]:
    """Validate an inner product after a constructor-time spectral proof.

    For a certified positive-definite pair the frozen relative breakdown
    inequality cannot hold for finite nonzero vectors.  The per-iteration
    check therefore only needs to reject a non-finite or non-positive inner
    product.  Callers without that certificate continue to use the full
    vector-norm definition above.
    """

    observed = complex(inner_product.item())
    finite = math.isfinite(observed.real) and math.isfinite(observed.imag)
    if not finite:
        return False, False
    return observed.real <= 0.0, True


def _channel_pressure_operator_breakdown(
    direction: torch.Tensor,
    operator_direction: torch.Tensor,
    inactive_mask: torch.Tensor,
    *,
    inner_product: torch.Tensor,
    operator_norm_upper_bound: float,
) -> tuple[bool, bool]:
    """Evaluate breakdown with a conservative one-norm early certificate.

    ``||A d|| <= upper_bound * ||d||`` implies that an inner product above
    ``eps * upper_bound * ||d||**2`` cannot satisfy the frozen relative
    breakdown predicate.  Only the unresolved boundary case pays for the
    complete two-vector measurement.
    """

    flattened = direction.reshape(-1)
    square_sum = torch.vdot(flattened, flattened).real
    threshold_factor = (
        torch.finfo(_channel_pressure_real_dtype(direction.dtype)).eps
        * operator_norm_upper_bound
    )
    # A NaN, Inf, zero, negative or unresolved inner product makes this
    # sufficient-condition comparison false and falls through to the full
    # fail-closed vector gate below.
    certified_nonbreakdown = bool(
        (inner_product.real > threshold_factor * square_sum).item()
    )
    if certified_nonbreakdown:
        return False, True
    return _channel_pressure_working_breakdown(
        direction,
        operator_direction,
        inactive_mask,
        inner_product=inner_product,
    )


def _channel_pressure_rms_interval(
    magnitude: ChannelPressureMagnitude,
    *,
    real_dtype: torch.dtype,
) -> tuple[float, float] | None:
    """Return a conservative host interval for an observed working RMS.

    The ordinary production path obtains its RMS from one device reduction.
    Inflate that observation by the same epsilon margin used by the frozen
    constructor certificates, then round the endpoints outwards.  Values that
    cannot be represented as positive finite host floats deliberately disable
    the bound-only fast path and retain the exact vector fallback.
    """

    value = magnitude.float64_value
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    relative_margin = (
        _CHANNEL_PRESSURE_BREAKDOWN_CERTIFICATE_EPSILON_MARGIN
        * torch.finfo(real_dtype).eps
    )
    if not math.isfinite(relative_margin) or not 0.0 < relative_margin < 1.0:
        return None
    lower = math.nextafter(value * (1.0 - relative_margin), -math.inf)
    upper = math.nextafter(value * (1.0 + relative_margin), math.inf)
    if not 0.0 < lower <= value <= upper or not math.isfinite(upper):
        return None
    return lower, upper


def _channel_pressure_inflate_upper_bound(
    value: float,
    *,
    real_dtype: torch.dtype,
) -> float | None:
    """Round one positive finite host upper bound conservatively upward."""

    if not math.isfinite(value) or value <= 0.0:
        return None
    relative_margin = (
        _CHANNEL_PRESSURE_BREAKDOWN_CERTIFICATE_EPSILON_MARGIN
        * torch.finfo(real_dtype).eps
    )
    inflated = math.nextafter(
        value * (1.0 + relative_margin),
        math.inf,
    )
    return inflated if math.isfinite(inflated) and inflated > 0.0 else None


def _channel_pressure_initial_direction_rms_upper_bound(
    residual_interval: tuple[float, float] | None,
    *,
    preconditioner_inverse_max: float,
    real_dtype: torch.dtype,
) -> float | None:
    """Bound ``rms(M^-1 r)`` for the first PCG direction."""

    if residual_interval is None:
        return None
    _residual_lower, residual_upper = residual_interval
    return _channel_pressure_inflate_upper_bound(
        preconditioner_inverse_max * residual_upper,
        real_dtype=real_dtype,
    )


def _channel_pressure_next_direction_rms_upper_bound(
    direction_upper: float | None,
    previous_residual_interval: tuple[float, float] | None,
    next_residual_interval: tuple[float, float] | None,
    *,
    preconditioner_inverse_min: float,
    preconditioner_inverse_max: float,
    real_dtype: torch.dtype,
) -> float | None:
    """Propagate a conservative PCG direction RMS upper bound.

    For the positive diagonal preconditioner ``P=M^-1``, the PCG recurrence
    gives ``d+ = P r+ + beta d`` and

    ``beta <= (p_max / p_min) * (||r+|| / ||r||)**2``.

    Only host observations that already exist for the convergence policy are
    used.  If any bound is unavailable, the caller falls back to the exact
    frozen relative-breakdown check.
    """

    if (
        direction_upper is None
        or previous_residual_interval is None
        or next_residual_interval is None
        or not math.isfinite(preconditioner_inverse_min)
        or not math.isfinite(preconditioner_inverse_max)
        or preconditioner_inverse_min <= 0.0
        or preconditioner_inverse_max < preconditioner_inverse_min
    ):
        return None
    previous_lower, _previous_upper = previous_residual_interval
    _next_lower, next_upper = next_residual_interval
    if previous_lower <= 0.0 or next_upper <= 0.0:
        return None
    beta_upper = (
        preconditioner_inverse_max
        / preconditioner_inverse_min
        * (next_upper / previous_lower) ** 2
    )
    preconditioned_residual_upper = preconditioner_inverse_max * next_upper
    return _channel_pressure_inflate_upper_bound(
        preconditioned_residual_upper + beta_upper * direction_upper,
        real_dtype=real_dtype,
    )


def _channel_pressure_bounded_operator_nonbreakdown(
    inner_product: torch.Tensor,
    *,
    direction_rms_upper_bound: float,
    active_count: int,
    operator_norm_upper_bound: float,
) -> tuple[bool, bool]:
    """Prove non-breakdown without another per-iteration vector reduction.

    ``||A d|| <= U ||d||`` and ``rms(d) <= B`` imply

    ``eps ||d|| ||A d|| <= eps U active_count B**2``.

    A positive Hermitian inner product above that conservative host threshold
    cannot satisfy the frozen relative-breakdown predicate.  ``False, True``
    means the proof was inconclusive and the exact vector gate must run;
    ``False, False`` preserves the existing non-finite failure path.
    """

    observed = complex(inner_product.item())
    finite = math.isfinite(observed.real) and math.isfinite(observed.imag)
    if not finite:
        return False, False
    if (
        observed.real <= 0.0
        or not math.isfinite(direction_rms_upper_bound)
        or direction_rms_upper_bound <= 0.0
        or not isinstance(active_count, int)
        or isinstance(active_count, bool)
        or active_count <= 0
        or not math.isfinite(operator_norm_upper_bound)
        or operator_norm_upper_bound <= 0.0
    ):
        return False, True
    threshold = (
        torch.finfo(_channel_pressure_real_dtype(inner_product.dtype)).eps
        * operator_norm_upper_bound
        * active_count
        * direction_rms_upper_bound
        * direction_rms_upper_bound
    )
    if not math.isfinite(threshold) or threshold <= 0.0:
        return False, True
    return observed.real > threshold, True


def _channel_pressure_working_residual_observation(
    residual: torch.Tensor,
    preconditioned: torch.Tensor,
    inactive_mask: torch.Tensor,
    *,
    inner_product: torch.Tensor,
    active_count: int,
    finite_companions: tuple[torch.Tensor, ...] = (),
    breakdown_certified: bool = False,
    preconditioner_inverse_max: float | None = None,
) -> tuple[ChannelPressureMagnitude, bool, bool, bool]:
    """Observe residual magnitude and preconditioned breakdown in one sync."""

    if active_count <= 0:
        raise ValueError("active pressure degree count must be positive")
    if breakdown_certified and not finite_companions:
        residual_norm_value = float(
            torch.linalg.vector_norm(residual.reshape(-1)).item()
        )
        inverse_max = (
            float(preconditioner_inverse_max)
            if preconditioner_inverse_max is not None
            else math.inf
        )
        maximum_safe_residual_norm = (
            math.sqrt(
                torch.finfo(
                    _channel_pressure_real_dtype(residual.dtype)
                ).max
                / inverse_max
            )
            if math.isfinite(inverse_max) and inverse_max > 0.0
            else 0.0
        )
        if (
            math.isfinite(residual_norm_value)
            and residual_norm_value > 0.0
            and residual_norm_value <= maximum_safe_residual_norm
        ):
            # The constructor proves a positive finite diagonal
            # preconditioner.  Together with this norm bound,
            # 0 < r^H M^-1 r <= ||r||^2 * ||M^-1|| < finfo.max, so the
            # frozen relative-breakdown predicate is false without reading a
            # second CUDA scalar.  Unresolved ranges use the exact fallback.
            rms_value = residual_norm_value / math.sqrt(active_count)
            mantissa, exponent = math.frexp(rms_value)
            return (
                _channel_pressure_magnitude_from_binary(mantissa, exponent),
                True,
                False,
                True,
            )
        magnitude, residual_finite = _channel_pressure_working_magnitude(
            residual,
            inactive_mask,
            active_count=active_count,
        )
        if not residual_finite or magnitude.is_zero:
            return magnitude, residual_finite, magnitude.is_zero, residual_finite
        breakdown, pair_finite = _channel_pressure_working_breakdown(
            residual,
            preconditioned,
            inactive_mask,
            inner_product=inner_product,
        )
        return magnitude, True, breakdown, pair_finite
    residual_flattened = residual.reshape(-1)
    residual_square_sum = torch.vdot(
        residual_flattened,
        residual_flattened,
    ).real
    metrics = [inner_product.real, residual_square_sum]
    if not breakdown_certified:
        preconditioned_flattened = preconditioned.reshape(-1)
        metrics.append(
            torch.vdot(
                preconditioned_flattened,
                preconditioned_flattened,
            ).real
        )
    metrics.extend(
        torch.maximum(
            companion.real.abs(),
            companion.imag.abs(),
        ).amax()
        for companion in finite_companions
    )
    observed = torch.stack(metrics).tolist()
    inner_value = float(observed[0])
    residual_square_value = float(observed[1])
    metric_count = 2 if breakdown_certified else 3
    preconditioned_square_value = (
        None if breakdown_certified else float(observed[2])
    )
    companions_finite = all(
        math.isfinite(item) for item in observed[metric_count:]
    )
    ordinary = (
        companions_finite
        and math.isfinite(inner_value)
        and math.isfinite(residual_square_value)
        and residual_square_value > 0.0
        and (
            breakdown_certified
            or (
                math.isfinite(float(preconditioned_square_value))
                and float(preconditioned_square_value) > 0.0
            )
        )
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
    if breakdown_certified:
        return magnitude, True, inner_value <= 0.0, True
    threshold = (
        torch.finfo(_channel_pressure_real_dtype(residual.dtype)).eps
        * math.sqrt(residual_square_value)
        * math.sqrt(float(preconditioned_square_value))
    )
    if not math.isfinite(threshold):
        breakdown, pair_finite = _channel_pressure_working_breakdown(
            residual,
            preconditioned,
            inactive_mask,
            inner_product=inner_product,
        )
        return magnitude, True, breakdown, pair_finite
    return magnitude, True, inner_value <= threshold, True


def channel_pressure_scale_power_of_two(
    value: torch.Tensor,
    binary_exponent: int,
) -> torch.Tensor:
    """Multiply a real or complex tensor by 2**exponent without scalar loss."""

    if not isinstance(binary_exponent, int) or isinstance(binary_exponent, bool):
        raise TypeError("binary exponent must be an integer")
    if binary_exponent == 0:
        return value

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
    return _channel_pressure_working_vector_from_projected(
        projected,
        component_amax,
        active_count,
    )


def _channel_pressure_working_vector_from_projected(
    projected: torch.Tensor,
    component_amax: float,
    active_count: int | torch.Tensor,
) -> tuple[torch.Tensor, int | None, ChannelPressureMagnitude]:
    """Scale an already projected pressure vector into its safe range."""

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
    epsilon = torch.finfo(_channel_pressure_real_dtype(left.dtype)).eps
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

    ``pressure_fixed_iterations`` selects fixed work.  In that mode
    ``pressure_max_iterations`` is inactive and relative tolerance is
    diagnostic-only.  Exact zero or a certified represented-precision floor
    reached immediately before breakdown may terminate early.
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
        if pressure_fixed_iterations is None:
            minimum_relative_tolerance = (
                channel_pressure_minimum_relative_tolerance(real_dtype)
            )
            if pressure_relative_tolerance < minimum_relative_tolerance:
                raise ValueError(
                    "pressure_relative_tolerance is below the supported "
                    f"{real_dtype} tolerance-mode floor "
                    f"({minimum_relative_tolerance!r}); use a tolerance at "
                    "or above that floor, or select fixed-work mode"
                )
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
        self.register_buffer(
            "_pressure_inactive_mask",
            null_mask | nyquist_mask,
            persistent=False,
        )
        self._active_pressure_degrees_of_freedom = int(
            (~self._pressure_inactive_mask).sum().item()
        )
        active = ~self._pressure_inactive_mask
        active_pressure_q2 = pressure_q2.unsqueeze(0).masked_select(active)
        active_schur_diag = schur_safe.masked_select(active)
        operator_angle_lower_bound = (
            float(active_pressure_q2.amin().item())
            * float(a_diag.amin().item())
            / (
                float(active_pressure_q2.amax().item())
                * float(a_diag.amax().item())
            )
        )
        preconditioner_angle_lower_bound = (
            float(active_schur_diag.amin().item())
            / float(active_schur_diag.amax().item())
        )
        certificate_floor = (
            _CHANNEL_PRESSURE_BREAKDOWN_CERTIFICATE_EPSILON_MARGIN
            * torch.finfo(real_dtype).eps
        )
        self._pressure_operator_angle_lower_bound = (
            operator_angle_lower_bound
        )
        self._pressure_operator_norm_upper_bound = (
            float(active_pressure_q2.amax().item())
            / float(a_diag.amin().item())
        )
        self._pressure_preconditioner_angle_lower_bound = (
            preconditioner_angle_lower_bound
        )
        self._pressure_preconditioner_inverse_max = (
            1.0 / float(active_schur_diag.amin().item())
        )
        self._pressure_preconditioner_inverse_min = (
            1.0 / float(active_schur_diag.amax().item())
        )
        self._pressure_unscaled_rhs_norm_min = (
            _CHANNEL_PRESSURE_BREAKDOWN_CERTIFICATE_EPSILON_MARGIN
            * math.sqrt(
                torch.finfo(real_dtype).tiny
                * float(active_schur_diag.amax().item())
            )
        )
        self._pressure_unscaled_rhs_norm_max = (
            math.sqrt(
                torch.finfo(real_dtype).max
                / self._pressure_preconditioner_inverse_max
            )
            / _CHANNEL_PRESSURE_BREAKDOWN_CERTIFICATE_EPSILON_MARGIN
        )
        self._pressure_operator_breakdown_certified = bool(
            math.isfinite(operator_angle_lower_bound)
            and operator_angle_lower_bound > certificate_floor
        )
        self._pressure_preconditioner_breakdown_certified = bool(
            math.isfinite(preconditioner_angle_lower_bound)
            and preconditioner_angle_lower_bound > certificate_floor
        )
        self.pressure_rel_tol = pressure_relative_tolerance
        self.pressure_max_iter = pressure_max_iterations
        self.pressure_fixed_iterations = pressure_fixed_iterations
        self.pressure_guess = None
        self._pressure_solve_sequence_id = 0
        zero = _channel_pressure_zero_magnitude()
        self._record_pressure_outcome(
            rhs_norm=zero,
            initial_residual=zero,
            residual=zero,
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
        return self._pressure_inactive_mask

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
        self._last_pressure_residual_magnitude = residual
        self.last_pressure_termination_reason = termination_reason
        self.last_pressure_acceptable = bool(acceptable)
        self.last_pressure_working_scale_binary_exponent = (
            working_scale_binary_exponent
        )
        self._last_pressure_rhs_norm = rhs_norm
        self._last_pressure_initial_residual = initial_residual
        self._last_pressure_achieved_residual = residual
        self._last_pressure_solve_phase = solve_phase
        self._last_pressure_diagnostics_cache = None

    @property
    def last_pressure_residual_magnitude(self) -> dict[str, object]:
        """Return the latest scale-safe residual metadata on demand."""

        return self._last_pressure_residual_magnitude.to_metadata()

    @property
    def last_pressure_diagnostics(self) -> dict[str, object]:
        """Materialize the public diagnostics outside the pressure hot path."""

        if self._last_pressure_diagnostics_cache is None:
            self._last_pressure_diagnostics_cache = (
                self._pressure_diagnostics_metadata(
                    rhs_norm=self._last_pressure_rhs_norm,
                    initial_residual=self._last_pressure_initial_residual,
                    achieved_residual=self._last_pressure_achieved_residual,
                    relative_residual=self.last_pressure_relative_residual,
                    iterations=self.last_pressure_iterations,
                    termination_reason=self.last_pressure_termination_reason,
                    acceptable=self.last_pressure_acceptable,
                    working_scale_binary_exponent=(
                        self.last_pressure_working_scale_binary_exponent
                    ),
                    solve_phase=self._last_pressure_solve_phase,
                )
            )
        return self._last_pressure_diagnostics_cache

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
            "zero_rhs_or_exact_zero_residual_or_exact_requested_fixed_work_or_"
            "fixed_work_roundoff_saturation"
            if fixed_work
            else (
                "zero_rhs_or_exact_zero_residual_or_pcg_recurrence_normalized_"
                "residual_not_above_requested_relative_tolerance"
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
        try:
            active_count = (
                self._active_pressure_degrees_of_freedom
                * rhs_hat.numel()
                // self._pressure_inactive_mask.numel()
            )
            rhs_norm_value = float(
                torch.linalg.vector_norm(rhs_hat.reshape(-1)).item()
            )
            if rhs_norm_value == 0.0:
                component_amax = float(
                    torch.maximum(
                        rhs_hat.real.abs(),
                        rhs_hat.imag.abs(),
                    ).amax().item()
                )
                if component_amax == 0.0:
                    working_rhs = rhs_hat
                    working_exponent = None
                    rhs_magnitude = _channel_pressure_zero_magnitude()
                else:
                    working_rhs, working_exponent, rhs_magnitude = (
                        _channel_pressure_working_vector_from_projected(
                            rhs_hat,
                            component_amax,
                            active_count,
                        )
                    )
            elif (
                math.isfinite(rhs_norm_value)
                and self._pressure_unscaled_rhs_norm_min
                <= rhs_norm_value
                <= self._pressure_unscaled_rhs_norm_max
            ):
                rhs_rms_value = rhs_norm_value / math.sqrt(active_count)
                rhs_mantissa, rhs_exponent = math.frexp(rhs_rms_value)
                working_rhs = rhs_hat
                working_exponent = 0
                rhs_magnitude = _channel_pressure_magnitude_from_binary(
                    rhs_mantissa,
                    rhs_exponent,
                )
            else:
                component_amax = float(
                    torch.maximum(
                        rhs_hat.real.abs(),
                        rhs_hat.imag.abs(),
                    ).amax().item()
                )
                if not math.isfinite(component_amax):
                    raise ValueError(
                        "Channel pressure vector contains NaN or Inf"
                    )
                working_rhs, working_exponent, rhs_magnitude = (
                    _channel_pressure_working_vector_from_projected(
                        rhs_hat,
                        component_amax,
                        active_count,
                    )
                )
        except ValueError as error:
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
            ) from error
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
        # Both operands are already in the projected pressure subspace, so a
        # second mask launch here cannot change any coefficient.
        residual = working_rhs - self._pressure_operator(pressure_work)
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
            finite_companions=(
                ()
                if self._pressure_preconditioner_breakdown_certified
                else (pressure_work,)
            ),
            breakdown_certified=(
                self._pressure_preconditioner_breakdown_certified
            ),
            preconditioner_inverse_max=(
                self._pressure_preconditioner_inverse_max
            ),
        )
        residual_rms_interval = _channel_pressure_rms_interval(
            residual_work_magnitude,
            real_dtype=_channel_pressure_real_dtype(residual.dtype),
        )
        direction_rms_upper_bound = (
            _channel_pressure_initial_direction_rms_upper_bound(
                residual_rms_interval,
                preconditioner_inverse_max=(
                    self._pressure_preconditioner_inverse_max
                ),
                real_dtype=_channel_pressure_real_dtype(residual.dtype),
            )
            if self._pressure_preconditioner_breakdown_certified
            else None
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
        fixed_work_floor_pressure = None
        fixed_work_floor_iterations = None
        fixed_work_floor_relative_residual = None
        saturation_failure_reason = (
            "iteration_limit_reached_without_postcondition"
        )
        if fixed_work and channel_pressure_fixed_work_roundoff_saturated(
            relative_residual,
            residual.dtype,
        ):
            fixed_work_floor_pressure = pressure_work.clone()
            fixed_work_floor_iterations = 0
            fixed_work_floor_relative_residual = float(relative_residual)
        while iterations < iteration_limit:
            operator_direction = self._pressure_operator(direction)
            direction_operator_inner = torch.sum(
                torch.conj(direction) * operator_direction
            )
            if self._pressure_operator_breakdown_certified:
                operator_breakdown, operator_pair_finite = (
                    _channel_pressure_certified_inner_observation(
                        direction_operator_inner
                    )
                )
            elif direction_rms_upper_bound is not None:
                (
                    bounded_nonbreakdown,
                    operator_pair_finite,
                ) = _channel_pressure_bounded_operator_nonbreakdown(
                    direction_operator_inner,
                    direction_rms_upper_bound=direction_rms_upper_bound,
                    active_count=active_count,
                    operator_norm_upper_bound=(
                        self._pressure_operator_norm_upper_bound
                    ),
                )
                if bounded_nonbreakdown:
                    operator_breakdown = False
                elif operator_pair_finite:
                    operator_breakdown, operator_pair_finite = (
                        _channel_pressure_operator_breakdown(
                            direction,
                            operator_direction,
                            self.pressure_inactive_mask,
                            inner_product=direction_operator_inner,
                            operator_norm_upper_bound=(
                                self._pressure_operator_norm_upper_bound
                            ),
                        )
                    )
                else:
                    operator_breakdown = False
            else:
                operator_breakdown, operator_pair_finite = (
                    _channel_pressure_operator_breakdown(
                        direction,
                        operator_direction,
                        self.pressure_inactive_mask,
                        inner_product=direction_operator_inner,
                        operator_norm_upper_bound=(
                            self._pressure_operator_norm_upper_bound
                        ),
                    )
                )
            if not operator_pair_finite:
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_work = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = "nonfinite_intermediate"
                else:
                    termination_reason = "nonfinite_intermediate"
                break
            if operator_breakdown:
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_work = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = "operator_breakdown"
                else:
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
                finite_companions=(
                    ()
                    if self._pressure_preconditioner_breakdown_certified
                    else (pressure_work,)
                ),
                breakdown_certified=(
                    self._pressure_preconditioner_breakdown_certified
                ),
                preconditioner_inverse_max=(
                    self._pressure_preconditioner_inverse_max
                ),
            )
            if not state_finite:
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_work = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = "nonfinite_intermediate"
                else:
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
            if fixed_work:
                if channel_pressure_fixed_work_roundoff_saturated(
                    relative_residual,
                    residual.dtype,
                ):
                    # Retain the best recurrence state at the dtype floor,
                    # rather than the last one: the latter may already carry
                    # a growing roundoff component before the recurrence
                    # rises back above the floor.
                    candidate_relative_residual = float(relative_residual)
                    if (
                        fixed_work_floor_relative_residual is None
                        or candidate_relative_residual
                        < fixed_work_floor_relative_residual
                    ):
                        fixed_work_floor_pressure = pressure_work.clone()
                        fixed_work_floor_iterations = iterations
                        fixed_work_floor_relative_residual = (
                            candidate_relative_residual
                        )
                elif fixed_work_floor_pressure is not None:
                    pressure_work = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = (
                        "iteration_limit_reached_without_postcondition"
                    )
                    break
            if not fixed_work and relative_residual <= self.pressure_rel_tol:
                # A recursively updated PCG residual can differ slightly from
                # the residual of the represented pressure.  Recompute from
                # the actual return candidate before accepting the public
                # tolerance contract.  If it is not yet sufficient, restart
                # the recurrence from that represented residual.
                residual = self._project_pressure_gauge(
                    working_rhs - self._pressure_operator(pressure_work)
                )
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
                    breakdown_certified=(
                        self._pressure_preconditioner_breakdown_certified
                    ),
                    preconditioner_inverse_max=(
                        self._pressure_preconditioner_inverse_max
                    ),
                )
                residual_magnitude = residual_work_magnitude.shifted(
                    int(working_exponent)
                )
                relative_residual = channel_pressure_magnitude_ratio(
                    residual_work_magnitude,
                    rhs_work_magnitude,
                )
                if not state_finite or not preconditioned_pair_finite:
                    termination_reason = "nonfinite_intermediate"
                    break
                if residual_work_magnitude.is_zero:
                    termination_reason = "exact_zero_residual"
                    acceptable = True
                    break
                if relative_residual <= self.pressure_rel_tol:
                    termination_reason = "relative_tolerance_met"
                    acceptable = True
                    break
                if preconditioned_breakdown:
                    termination_reason = "preconditioned_residual_breakdown"
                    break
                residual_rms_interval = _channel_pressure_rms_interval(
                    residual_work_magnitude,
                    real_dtype=_channel_pressure_real_dtype(residual.dtype),
                )
                direction_rms_upper_bound = (
                    _channel_pressure_initial_direction_rms_upper_bound(
                        residual_rms_interval,
                        preconditioner_inverse_max=(
                            self._pressure_preconditioner_inverse_max
                        ),
                        real_dtype=_channel_pressure_real_dtype(residual.dtype),
                    )
                    if self._pressure_preconditioner_breakdown_certified
                    else None
                )
                direction = preconditioned.clone()
                rz_old = rz_new
                continue
            if iterations == iteration_limit:
                if fixed_work:
                    termination_reason = "fixed_work_completed"
                    acceptable = True
                break
            if not preconditioned_pair_finite:
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_work = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = "nonfinite_intermediate"
                else:
                    termination_reason = "nonfinite_intermediate"
                break
            if preconditioned_breakdown:
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_work = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = (
                        "preconditioned_residual_breakdown"
                    )
                else:
                    termination_reason = "preconditioned_residual_breakdown"
                break
            next_residual_rms_interval = _channel_pressure_rms_interval(
                residual_work_magnitude,
                real_dtype=_channel_pressure_real_dtype(residual.dtype),
            )
            direction_rms_upper_bound = (
                _channel_pressure_next_direction_rms_upper_bound(
                    direction_rms_upper_bound,
                    residual_rms_interval,
                    next_residual_rms_interval,
                    preconditioner_inverse_min=(
                        self._pressure_preconditioner_inverse_min
                    ),
                    preconditioner_inverse_max=(
                        self._pressure_preconditioner_inverse_max
                    ),
                    real_dtype=_channel_pressure_real_dtype(residual.dtype),
                )
                if self._pressure_preconditioner_breakdown_certified
                else None
            )
            residual_rms_interval = next_residual_rms_interval
            direction = preconditioned + (rz_new / rz_old) * direction
            rz_old = rz_new
        if (
            acceptable
            and fixed_work_floor_pressure is not None
            and termination_reason
            in {
                "fixed_work_completed",
                CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION,
            }
        ):
            # The recursive PCG residual is only a trigger.  Check the
            # represented state that would actually be returned.  A normal
            # completion keeps its pre-S-06 result when it is within the
            # documented two-floor roundoff margin.  If it exceeds the exact
            # floor, first prefer the saved recurrence-minimum floor state;
            # actual rollback remains subject to the strict one-floor
            # represented-residual postcondition.
            completed_pressure = (
                pressure_work.clone()
                if termination_reason == "fixed_work_completed"
                else None
            )
            try:
                residual = self._project_pressure_gauge(
                    working_rhs - self._pressure_operator(pressure_work)
                )
                residual_work_magnitude = channel_pressure_active_rms(
                    residual,
                    self.pressure_inactive_mask,
                )
                residual_magnitude = residual_work_magnitude.shifted(
                    int(working_exponent)
                )
                relative_residual = channel_pressure_magnitude_ratio(
                    residual_work_magnitude,
                    rhs_work_magnitude,
                )
            except ValueError:
                acceptable = False
                termination_reason = "nonfinite_intermediate"
            else:
                represented_at_floor = (
                    channel_pressure_fixed_work_roundoff_saturated(
                        relative_residual,
                        residual.dtype,
                    )
                )
                completed_residual_magnitude = (
                    residual_magnitude
                    if completed_pressure is not None
                    else None
                )
                completed_relative_residual = (
                    relative_residual
                    if completed_pressure is not None
                    else None
                )
                if (
                    not represented_at_floor
                    and completed_pressure is not None
                    and not torch.equal(
                        completed_pressure,
                        fixed_work_floor_pressure,
                    )
                ):
                    pressure_work = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    try:
                        residual = self._project_pressure_gauge(
                            working_rhs
                            - self._pressure_operator(pressure_work)
                        )
                        residual_work_magnitude = (
                            channel_pressure_active_rms(
                                residual,
                                self.pressure_inactive_mask,
                            )
                        )
                        residual_magnitude = residual_work_magnitude.shifted(
                            int(working_exponent)
                        )
                        relative_residual = channel_pressure_magnitude_ratio(
                            residual_work_magnitude,
                            rhs_work_magnitude,
                        )
                    except ValueError:
                        acceptable = False
                        termination_reason = "nonfinite_intermediate"
                    else:
                        represented_at_floor = (
                            channel_pressure_fixed_work_roundoff_saturated(
                                relative_residual,
                                residual.dtype,
                            )
                        )
                        if represented_at_floor:
                            termination_reason = (
                                CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                            )
                if (
                    not represented_at_floor
                    and completed_pressure is not None
                    and completed_relative_residual is not None
                    and math.isfinite(float(completed_relative_residual))
                    and float(completed_relative_residual)
                    <= 2.0
                    * channel_pressure_minimum_relative_tolerance(
                        residual.dtype
                    )
                ):
                    pressure_work = completed_pressure
                    iterations = iteration_limit
                    residual_magnitude = completed_residual_magnitude
                    relative_residual = completed_relative_residual
                    termination_reason = "fixed_work_completed"
                    represented_at_floor = True
                if not represented_at_floor:
                    acceptable = False
                    termination_reason = saturation_failure_reason
                elif (
                    termination_reason
                    == CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    and not channel_pressure_fixed_work_roundoff_saturated(
                    relative_residual,
                    residual.dtype,
                    )
                ):
                    acceptable = False
                    termination_reason = saturation_failure_reason
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
    "CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION",
    "CHANNEL_VELOCITY_BOUNDARY_CONDITIONS",
    "ChannelPressureConvergenceError",
    "channel_pressure_minimum_relative_tolerance",
    "ChannelPressureMagnitude",
    "ChannelNoSlipModalStokesSolver",
    "channel_pressure_active_rms",
    "channel_pressure_fixed_work_roundoff_saturated",
    "channel_pressure_magnitude_ratio",
    "channel_pressure_relative_breakdown",
    "channel_pressure_scale_power_of_two",
    "channel_pressure_working_vector",
]
