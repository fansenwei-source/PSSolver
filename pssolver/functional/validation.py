"""Fail-closed validation for the qualified periodic functional runtime.

These are qualification tools, not objective or checkpoint-scheduling APIs.
They use fixed audit functionals and frozen tolerances, so callers cannot turn
a failed check into a pass by supplying looser thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Protocol

import torch

from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.periodic_beris_edwards import PeriodicRuntimeAdapterProtocol

from .contracts import FunctionalControls, FunctionalState


PERIODIC_GRADIENT_VALIDATION_VERSION = 1
PERIODIC_CONSISTENCY_VALIDATION_VERSION = 1


class _PeriodicFunctionalRuntime(Protocol):
    @property
    def state_spec(self): ...
    @property
    def control_specs(self): ...
    def step_and_observe(self, state, controls, step_index): ...


class FunctionalValidationError(RuntimeError):
    """Raised when a frozen functional-runtime qualification gate fails."""

    def __init__(self, message: str, report: object) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class DirectionalDerivativeCheck:
    input_name: str
    epsilon: float
    autograd: float
    finite_difference: float
    absolute_error: float
    relative_error: float
    tolerance: float
    passed: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "input": self.input_name,
            "epsilon": self.epsilon,
            "autograd": self.autograd,
            "finite_difference": self.finite_difference,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "relative_tolerance": self.tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class GradientPathCheck:
    path: str
    gradient_norm: float
    finite: bool
    nonzero: bool
    passed: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "path": self.path,
            "gradient_norm": self.gradient_norm,
            "finite": self.finite,
            "nonzero": self.nonzero,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class PeriodicGradientValidationReport:
    format_version: int
    device: str
    real_dtype: str
    directional_derivatives: tuple[DirectionalDerivativeCheck, ...]
    gradient_paths: tuple[GradientPathCheck, ...]
    passed: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "device": self.device,
            "real_dtype": self.real_dtype,
            "directional_derivatives": [
                value.to_metadata() for value in self.directional_derivatives
            ],
            "gradient_paths": [
                value.to_metadata() for value in self.gradient_paths
            ],
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class TensorConsistencyCheck:
    name: str
    shape: tuple[int, ...]
    dtype: str
    absolute_l2: float
    relative_l2: float
    linf: float
    atol: float
    rtol: float
    bitwise_equal: bool
    finite: bool
    passed: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "absolute_l2": self.absolute_l2,
            "relative_l2": self.relative_l2,
            "linf": self.linf,
            "atol": self.atol,
            "rtol": self.rtol,
            "bitwise_equal": self.bitwise_equal,
            "finite": self.finite,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class PeriodicProductionConsistencyReport:
    format_version: int
    device: str
    real_dtype: str
    tolerances: Mapping[str, float]
    checks: tuple[TensorConsistencyCheck, ...]
    passed: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "device": self.device,
            "real_dtype": self.real_dtype,
            "tolerances": dict(self.tolerances),
            "checks": [value.to_metadata() for value in self.checks],
            "passed": self.passed,
        }


def _frozen_gradient_contract(dtype: torch.dtype) -> tuple[float, float, float]:
    if dtype is torch.float64:
        return 1.0e-6, 1.0e-4, 2.0e-5
    if dtype is torch.float32:
        return 2.0e-3, 1.0e-2, 2.0e-2
    raise TypeError("gradient validation requires float32 or float64 state")


def _frozen_consistency_contract(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float64:
        return 5.0e-13, 5.0e-12
    if dtype is torch.float32:
        return 2.0e-6, 2.0e-5
    raise TypeError("consistency validation requires float32 or float64 state")


def _direction(value: torch.Tensor, phase: float) -> torch.Tensor:
    real_dtype = value.real.dtype if value.is_complex() else value.dtype
    indices = torch.arange(
        value.numel(), dtype=real_dtype, device=value.device
    ).reshape(value.shape)
    real = torch.sin(indices * 0.173 + phase)
    if value.is_complex():
        result = torch.complex(real, torch.cos(indices * 0.119 + phase))
    else:
        result = real
    scale = result.abs().amax()
    if not bool(torch.isfinite(scale).item()) or float(scale.item()) == 0.0:
        raise RuntimeError("failed to construct a finite audit direction")
    return result / scale


def _audit_scalars(runtime, state, controls):
    next_state, observations = runtime.step_and_observe(state, controls, 0)
    dynamics = next_state[0].square().mean() + next_state[1].abs().square().mean()
    observation = (
        observations["velocity"].square().mean()
        + observations["pressure"].square().mean()
    )
    return dynamics, observation


def _real_directional_product(gradient, direction) -> float:
    if gradient is None:
        return 0.0
    value = (gradient.conj() * direction).real.sum()
    return float(value.detach().cpu().item())


def _gradient_path(name: str, gradient) -> GradientPathCheck:
    if gradient is None:
        return GradientPathCheck(name, 0.0, True, False, False)
    finite = bool(torch.isfinite(gradient).all().item())
    norm = float(torch.linalg.vector_norm(gradient).detach().cpu().item())
    nonzero = norm > 0.0
    return GradientPathCheck(name, norm, finite, nonzero, finite and nonzero)


def _requires_grad_state(state: FunctionalState) -> FunctionalState:
    return tuple(value.detach().clone().requires_grad_(True) for value in state)


def _requires_grad_controls(controls: FunctionalControls) -> dict[str, torch.Tensor]:
    return {
        "activity": controls["activity"].detach().clone().requires_grad_(True)
    }


def _safe_grad(output, inputs):
    if not isinstance(output, torch.Tensor) or not output.requires_grad:
        return tuple(None for _ in inputs)
    return torch.autograd.grad(output, inputs, allow_unused=True)


def evaluate_periodic_activity_gradients(
    runtime: _PeriodicFunctionalRuntime,
    state: FunctionalState,
    controls: FunctionalControls,
) -> PeriodicGradientValidationReport:
    """Evaluate fixed autograd, finite-difference, and hidden-detach gates."""

    runtime.state_spec.validate(state)
    activity = controls["activity"]
    state_epsilon, control_epsilon, tolerance = _frozen_gradient_contract(
        state[0].dtype
    )
    if not bool(torch.isfinite(activity).all().item()):
        raise ValueError("activity contains NaN or Inf")

    # Separate graphs prevent an observation's direct state dependence from
    # concealing a detach inside the timestep map.
    dynamic_state = _requires_grad_state(state)
    dynamic_controls = _requires_grad_controls(controls)
    dynamics, _ = _audit_scalars(runtime, dynamic_state, dynamic_controls)
    dynamic_gradients = _safe_grad(
        dynamics, (*dynamic_state, dynamic_controls["activity"])
    )

    observation_state = _requires_grad_state(state)
    observation_controls = _requires_grad_controls(controls)
    _, observation = _audit_scalars(runtime, observation_state, observation_controls)
    observation_gradients = _safe_grad(
        observation, (*observation_state, observation_controls["activity"])
    )
    paths = (
        _gradient_path("dynamics<-q_physical", dynamic_gradients[0]),
        _gradient_path("dynamics<-q_spectral", dynamic_gradients[1]),
        _gradient_path("dynamics<-activity", dynamic_gradients[2]),
        _gradient_path("observations<-q_physical", observation_gradients[0]),
        _gradient_path("observations<-q_spectral", observation_gradients[1]),
        _gradient_path("observations<-activity", observation_gradients[2]),
    )

    state_for_ad = _requires_grad_state(state)
    controls_for_ad = _requires_grad_controls(controls)
    dynamics, observation = _audit_scalars(runtime, state_for_ad, controls_for_ad)
    gradients = _safe_grad(
        dynamics + observation,
        (*state_for_ad, controls_for_ad["activity"]),
    )
    state_directions = tuple(
        _direction(value, 0.31 + index) for index, value in enumerate(state)
    )
    control_direction = _direction(activity, 2.71)
    if bool(
        (activity - control_epsilon * control_direction.abs() < 0.0)
        .any()
        .item()
    ):
        raise ValueError(
            "activity must be interior to its admissible range for the "
            "frozen central finite-difference audit"
        )
    state_autograd = sum(
        _real_directional_product(gradient, direction)
        for gradient, direction in zip(gradients[:2], state_directions, strict=True)
    )
    control_autograd = _real_directional_product(gradients[2], control_direction)

    def objective(test_state, test_controls) -> float:
        dynamic_value, observation_value = _audit_scalars(
            runtime, test_state, test_controls
        )
        return float((dynamic_value + observation_value).detach().cpu().item())

    plus_state = tuple(
        value + state_epsilon * direction
        for value, direction in zip(state, state_directions, strict=True)
    )
    minus_state = tuple(
        value - state_epsilon * direction
        for value, direction in zip(state, state_directions, strict=True)
    )
    state_fd = (
        objective(plus_state, controls) - objective(minus_state, controls)
    ) / (2.0 * state_epsilon)
    plus_controls = {
        "activity": activity + control_epsilon * control_direction
    }
    minus_controls = {
        "activity": activity - control_epsilon * control_direction
    }
    control_fd = (
        objective(state, plus_controls) - objective(state, minus_controls)
    ) / (2.0 * control_epsilon)

    def derivative_check(name, epsilon, autograd_value, finite_difference_value):
        absolute = abs(autograd_value - finite_difference_value)
        scale = max(
            abs(autograd_value),
            abs(finite_difference_value),
            torch.finfo(state[0].dtype).tiny,
        )
        relative = absolute / scale
        finite = all(
            math.isfinite(value)
            for value in (autograd_value, finite_difference_value, relative)
        )
        return DirectionalDerivativeCheck(
            name,
            epsilon,
            autograd_value,
            finite_difference_value,
            absolute,
            relative,
            tolerance,
            finite and relative <= tolerance,
        )

    derivatives = (
        derivative_check("state", state_epsilon, state_autograd, state_fd),
        derivative_check(
            "activity", control_epsilon, control_autograd, control_fd
        ),
    )
    passed = all(value.passed for value in (*paths, *derivatives))
    return PeriodicGradientValidationReport(
        PERIODIC_GRADIENT_VALIDATION_VERSION,
        str(state[0].device),
        str(state[0].dtype).removeprefix("torch."),
        derivatives,
        paths,
        passed,
    )


def validate_periodic_activity_gradients(
    runtime: _PeriodicFunctionalRuntime,
    state: FunctionalState,
    controls: FunctionalControls,
) -> PeriodicGradientValidationReport:
    report = evaluate_periodic_activity_gradients(runtime, state, controls)
    if not report.passed:
        raise FunctionalValidationError(
            "periodic activity gradient qualification failed", report
        )
    return report


def _consistency_check(name, functional, production, *, atol, rtol):
    if functional.shape != production.shape:
        raise ValueError(f"production/functional shape differs for {name}")
    if functional.dtype is not production.dtype:
        raise TypeError(f"production/functional dtype differs for {name}")
    if functional.device != production.device:
        raise ValueError(f"production/functional device differs for {name}")
    finite = bool(
        torch.isfinite(functional).all().item()
        and torch.isfinite(production).all().item()
    )
    difference = functional - production
    absolute_l2 = float(torch.linalg.vector_norm(difference).detach().cpu().item())
    reference_l2 = float(torch.linalg.vector_norm(production).detach().cpu().item())
    relative_l2 = absolute_l2 / max(
        reference_l2, torch.finfo(functional.real.dtype).tiny
    )
    linf = float(difference.abs().amax().detach().cpu().item())
    passed = finite and bool(
        torch.allclose(functional, production, atol=atol, rtol=rtol)
    )
    return TensorConsistencyCheck(
        name,
        tuple(functional.shape),
        str(functional.dtype).removeprefix("torch."),
        absolute_l2,
        relative_l2,
        linf,
        atol,
        rtol,
        bool(torch.equal(functional, production)),
        finite,
        passed,
    )


def evaluate_periodic_production_consistency(
    runtime: _PeriodicFunctionalRuntime,
    production: PeriodicRuntimeAdapterProtocol,
    state: FunctionalState,
    controls: FunctionalControls,
) -> PeriodicProductionConsistencyReport:
    """Compare one functional step with one production step.

    The production adapter is consumed by one timestep and must initially be
    at ``state``.  The tolerance contract is internal and has no override.
    """

    if not isinstance(production, PeriodicRuntimeAdapterProtocol):
        raise TypeError("production must implement PeriodicRuntimeAdapterProtocol")
    runtime.state_spec.validate(state)
    if not isinstance(controls, Mapping) or set(controls) != {"activity"}:
        raise ValueError("controls must contain only activity")
    activity = controls["activity"]
    runtime.control_specs[0].validate(activity)
    if not bool(torch.isfinite(activity).all().item()):
        raise ValueError("activity contains NaN or Inf")
    atol, rtol = _frozen_consistency_contract(state[0].dtype)
    fields = production.fields
    production_state = (
        torch.stack(tuple(fields[name] for name in Q_COMPONENTS), dim=0),
        torch.stack(tuple(fields[f"{name}.hat"] for name in Q_COMPONENTS), dim=0),
    )
    checks = [
        _consistency_check(
            "input.q_physical", state[0], production_state[0], atol=atol, rtol=rtol
        ),
        _consistency_check(
            "input.q_spectral", state[1], production_state[1], atol=atol, rtol=rtol
        ),
    ]
    if not all(value.passed for value in checks):
        return PeriodicProductionConsistencyReport(
            PERIODIC_CONSISTENCY_VALIDATION_VERSION,
            str(state[0].device),
            str(state[0].dtype).removeprefix("torch."),
            MappingProxyType({"atol": atol, "rtol": rtol}),
            tuple(checks),
            False,
        )
    original_alpha = production.solver.model.parameters["alpha"]
    try:
        production.solver.model.parameters["alpha"] = activity
        production.synchronize_for_observation()
        functional_next, functional_observations = runtime.step_and_observe(
            state, controls, production.completed_steps
        )
        production_observations = {
            "Q": torch.stack(tuple(fields[name] for name in Q_COMPONENTS), dim=0),
            "velocity": fields.spatial[5:8].clone(),
            "pressure": fields.spatial[8].clone(),
        }
        for name in ("Q", "velocity", "pressure"):
            checks.append(
                _consistency_check(
                    f"observation.{name}",
                    functional_observations[name],
                    production_observations[name],
                    atol=atol,
                    rtol=rtol,
                )
            )
        production.advance(1)
        checks.extend(
            (
                _consistency_check(
                    "next.q_physical",
                    functional_next[0],
                    fields.spatial[: len(Q_COMPONENTS)],
                    atol=atol,
                    rtol=rtol,
                ),
                _consistency_check(
                    "next.q_spectral",
                    functional_next[1],
                    fields.spectral[: len(Q_COMPONENTS)],
                    atol=atol,
                    rtol=rtol,
                ),
            )
        )
    finally:
        production.solver.model.parameters["alpha"] = original_alpha
        production.synchronize_for_observation()

    passed = all(value.passed for value in checks)
    return PeriodicProductionConsistencyReport(
        PERIODIC_CONSISTENCY_VALIDATION_VERSION,
        str(state[0].device),
        str(state[0].dtype).removeprefix("torch."),
        MappingProxyType({"atol": atol, "rtol": rtol}),
        tuple(checks),
        passed,
    )


def validate_periodic_production_consistency(
    runtime: _PeriodicFunctionalRuntime,
    production: PeriodicRuntimeAdapterProtocol,
    state: FunctionalState,
    controls: FunctionalControls,
) -> PeriodicProductionConsistencyReport:
    report = evaluate_periodic_production_consistency(
        runtime, production, state, controls
    )
    if not report.passed:
        raise FunctionalValidationError(
            "periodic production/functional consistency qualification failed",
            report,
        )
    return report


__all__ = [
    "DirectionalDerivativeCheck",
    "FunctionalValidationError",
    "GradientPathCheck",
    "PERIODIC_CONSISTENCY_VALIDATION_VERSION",
    "PERIODIC_GRADIENT_VALIDATION_VERSION",
    "PeriodicGradientValidationReport",
    "PeriodicProductionConsistencyReport",
    "TensorConsistencyCheck",
    "evaluate_periodic_activity_gradients",
    "evaluate_periodic_production_consistency",
    "validate_periodic_activity_gradients",
    "validate_periodic_production_consistency",
]
