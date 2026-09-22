"""Local convergence qualification for the Phase 4 scalar reference path."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from pssolver.core.integrators import IntegratorScheme, IntegratorSpec
from pssolver.integrators.scalar_reference import (
    ScalarPeriodicReactionDiffusion,
    ScalarPeriodicReferenceStepper,
)


TEMPORAL_STEP_RATIO_LADDER = (1.0, 0.5, 0.25, 0.125)


@dataclass(frozen=True, slots=True)
class TemporalErrorSample:
    """One final-time error measurement on the frozen temporal ladder."""

    dt: float
    steps: int
    final_time: float
    l2_error: float
    linf_error: float
    finite: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "steps": self.steps,
            "final_time": self.final_time,
            "l2_error": self.l2_error,
            "linf_error": self.linf_error,
            "finite": self.finite,
        }


@dataclass(frozen=True, slots=True)
class TemporalConvergenceResult:
    """Errors, pairwise orders, and threshold decision for one scheme."""

    scheme: IntegratorScheme
    samples: tuple[TemporalErrorSample, ...]
    pairwise_l2_orders: tuple[float, ...]
    minimum_required_order: float
    passed: bool

    @property
    def minimum_observed_order(self) -> float:
        return min(self.pairwise_l2_orders)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scheme": self.scheme.value,
            "step_ratio_ladder": list(TEMPORAL_STEP_RATIO_LADDER),
            "samples": [sample.to_metadata() for sample in self.samples],
            "pairwise_l2_orders": list(self.pairwise_l2_orders),
            "minimum_observed_order": self.minimum_observed_order,
            "minimum_required_order": self.minimum_required_order,
            "finite_required": True,
            "passed": self.passed,
        }


def _positive_finite(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{description} must be positive and finite")
    return float(value)


def _integral_step_count(final_time: float, dt: float) -> int:
    quotient = final_time / dt
    rounded = round(quotient)
    if rounded <= 0 or not math.isclose(quotient, rounded, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("final_time must be an integer multiple of every dt")
    return int(rounded)


def measure_temporal_convergence(
    *,
    model: ScalarPeriodicReactionDiffusion,
    scheme: IntegratorScheme,
    base_dt: float = 0.04,
    final_time: float = 0.4,
) -> TemporalConvergenceResult:
    """Measure the frozen four-level final-time L2 and Linf errors."""

    if not isinstance(model, ScalarPeriodicReactionDiffusion):
        raise TypeError("model must be ScalarPeriodicReactionDiffusion")
    if not isinstance(scheme, IntegratorScheme):
        raise TypeError("scheme must be an IntegratorScheme")
    normalized_base_dt = _positive_finite(base_dt, "base_dt")
    normalized_final_time = _positive_finite(final_time, "final_time")
    exact = model.exact_physical(
        normalized_final_time,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    samples = []
    for ratio in TEMPORAL_STEP_RATIO_LADDER:
        dt = normalized_base_dt * ratio
        steps = _integral_step_count(normalized_final_time, dt)
        spec = (
            IntegratorSpec.projected_semi_implicit_euler(dt=dt)
            if scheme is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
            else IntegratorSpec.sbdf2(dt=dt)
        )
        stepper = ScalarPeriodicReferenceStepper(model=model, spec=spec)
        state = stepper.run(steps=steps)
        difference = state.physical - exact
        l2_error = float(
            torch.linalg.vector_norm(difference)
            / math.sqrt(difference.numel())
        )
        linf_error = float(torch.max(torch.abs(difference)))
        finite = math.isfinite(l2_error) and math.isfinite(linf_error)
        samples.append(
            TemporalErrorSample(
                dt=dt,
                steps=steps,
                final_time=state.time(dt),
                l2_error=l2_error,
                linf_error=linf_error,
                finite=finite,
            )
        )

    orders = tuple(
        math.log(coarse.l2_error / fine.l2_error, 2.0)
        for coarse, fine in zip(samples[:-1], samples[1:], strict=True)
    )
    required = (
        0.9
        if scheme is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
        else 1.8
    )
    passed = (
        all(sample.finite for sample in samples)
        and all(sample.l2_error > 0.0 for sample in samples)
        and min(orders) >= required
    )
    return TemporalConvergenceResult(
        scheme=scheme,
        samples=tuple(samples),
        pairwise_l2_orders=orders,
        minimum_required_order=required,
        passed=passed,
    )


__all__ = [
    "TEMPORAL_STEP_RATIO_LADDER",
    "TemporalConvergenceResult",
    "TemporalErrorSample",
    "measure_temporal_convergence",
]
