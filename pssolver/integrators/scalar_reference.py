"""Disconnected scalar reference path for Phase 4 SBDF2 qualification.

This module deliberately does not connect to ``RuntimeState``, ``StepProgram``,
Plane, or the production checkpoint format.  It provides a small functional
CPU oracle whose input state is never modified by a timestep.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch

from pssolver.core.integrators import IntegratorScheme, IntegratorSpec
from pssolver.integrators.history import SBDF2History


_SBDF2_OPERATIONS = (
    "require_current_state_and_history",
    "prepare_algebraic",
    "pre_update_callback",
    "evaluate_current_explicit_rhs",
    "assemble_sbdf2_implicit_rhs",
    "solve_implicit_operator",
    "project_dynamic_spectra",
    "inverse_dynamic_spectra",
    "scheduled_spectral_refresh",
    "commit_progress",
    "commit_integrator_history",
)

_STARTUP_OPERATIONS = (
    "require_current_state_and_history",
    "prepare_algebraic",
    "pre_update_callback",
    "evaluate_current_explicit_rhs",
    "assemble_euler_implicit_rhs",
    "solve_implicit_operator",
    "project_dynamic_spectra",
    "inverse_dynamic_spectra",
    "scheduled_spectral_refresh",
    "commit_progress",
    "commit_integrator_history",
)

_EULER_OPERATIONS = _STARTUP_OPERATIONS[:-1]


def _positive_finite(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{description} must be positive and finite")
    return float(value)


def _finite(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{description} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class ScalarPeriodicReactionDiffusion:
    """One Fourier-mode periodic reaction--diffusion reference problem."""

    point_count: int = 32
    length: float = 2.0 * math.pi
    diffusivity: float = 0.2
    reaction_rate: float = 0.15
    initial_mode: int = 3
    initial_amplitude: float = 0.75

    def __post_init__(self) -> None:
        if (
            not isinstance(self.point_count, int)
            or isinstance(self.point_count, bool)
            or self.point_count < 8
            or self.point_count % 2
        ):
            raise ValueError("point_count must be an even integer of at least 8")
        object.__setattr__(self, "length", _positive_finite(self.length, "length"))
        diffusivity = _finite(self.diffusivity, "diffusivity")
        if diffusivity < 0.0:
            raise ValueError("diffusivity must be non-negative")
        object.__setattr__(self, "diffusivity", diffusivity)
        object.__setattr__(
            self,
            "reaction_rate",
            _finite(self.reaction_rate, "reaction_rate"),
        )
        if (
            not isinstance(self.initial_mode, int)
            or isinstance(self.initial_mode, bool)
            or self.initial_mode < 0
            or self.initial_mode >= self.point_count // 2
        ):
            raise ValueError("initial_mode must be a retained non-Nyquist mode")
        object.__setattr__(
            self,
            "initial_amplitude",
            _finite(self.initial_amplitude, "initial_amplitude"),
        )

    @property
    def analytic_growth_rate(self) -> float:
        wave_number = 2.0 * math.pi * self.initial_mode / self.length
        return self.reaction_rate - self.diffusivity * wave_number**2

    def coordinates(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return (
            torch.arange(self.point_count, dtype=dtype, device=device)
            * (self.length / self.point_count)
        )

    def initial_physical(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        coordinate = self.coordinates(dtype=dtype, device=device)
        wave_number = 2.0 * math.pi * self.initial_mode / self.length
        return self.initial_amplitude * torch.cos(wave_number * coordinate)

    def exact_physical(
        self,
        time: float,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not math.isfinite(float(time)) or float(time) < 0.0:
            raise ValueError("time must be finite and non-negative")
        return self.initial_physical(dtype=dtype, device=device) * math.exp(
            self.analytic_growth_rate * float(time)
        )

    def linear_eigenvalues(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        frequencies = torch.fft.rfftfreq(
            self.point_count,
            d=self.length / self.point_count,
            dtype=dtype,
            device=device,
        )
        wave_numbers = 2.0 * math.pi * frequencies
        return -self.diffusivity * wave_numbers.square()

    def explicit_native_spectral_rhs(
        self,
        native_spectrum: torch.Tensor,
    ) -> torch.Tensor:
        return self.reaction_rate * native_spectrum

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "identity": "scalar_periodic_reaction_diffusion",
            "point_count": self.point_count,
            "length": self.length,
            "diffusivity": self.diffusivity,
            "reaction_rate": self.reaction_rate,
            "initial_mode": self.initial_mode,
            "initial_amplitude": self.initial_amplitude,
            "analytic_growth_rate": self.analytic_growth_rate,
            "split": {
                "implicit": "diffusion",
                "explicit": "linear_reaction",
            },
        }


@dataclass(frozen=True, slots=True)
class ScalarReferenceState:
    """One complete functional state of the scalar reference workflow."""

    physical: torch.Tensor
    native_spectrum: torch.Tensor
    completed_steps: int
    refresh_interval: int | None
    refresh_step_count: int
    refresh_count: int
    history: SBDF2History | None

    def __post_init__(self) -> None:
        if not isinstance(self.physical, torch.Tensor):
            raise TypeError("physical must be a tensor")
        if not isinstance(self.native_spectrum, torch.Tensor):
            raise TypeError("native_spectrum must be a tensor")
        if self.physical.ndim != 1 or self.native_spectrum.ndim != 1:
            raise ValueError("scalar reference tensors must be one-dimensional")
        if self.physical.device != self.native_spectrum.device:
            raise ValueError("physical and spectral devices differ")
        if self.physical.dtype is not torch.float64:
            raise ValueError("scalar reference physical dtype must be float64")
        if self.native_spectrum.dtype is not torch.complex128:
            raise ValueError("scalar reference spectral dtype must be complex128")
        for value, description in (
            (self.completed_steps, "completed_steps"),
            (self.refresh_step_count, "refresh_step_count"),
            (self.refresh_count, "refresh_count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{description} must be a non-negative integer")
        if self.refresh_interval is None:
            valid_clock = (
                self.refresh_count == 0
                and self.refresh_step_count == self.completed_steps
            )
        else:
            if (
                not isinstance(self.refresh_interval, int)
                or isinstance(self.refresh_interval, bool)
                or self.refresh_interval <= 0
            ):
                raise ValueError("refresh_interval must be positive or None")
            valid_clock = (
                self.refresh_step_count < self.refresh_interval
                and self.refresh_count * self.refresh_interval
                + self.refresh_step_count
                == self.completed_steps
            )
        if not valid_clock:
            raise ValueError("spectral-refresh counters are inconsistent")
        if self.history is not None and not isinstance(self.history, SBDF2History):
            raise TypeError("history must be SBDF2History or None")
        if not bool(torch.isfinite(self.physical).all()):
            raise ValueError("physical state must be finite")
        if not bool(torch.isfinite(self.native_spectrum).all()):
            raise ValueError("spectral state must be finite")

    @property
    def refresh_due_after_next_step(self) -> bool:
        return (
            self.refresh_interval is not None
            and self.refresh_step_count + 1 >= self.refresh_interval
        )

    def time(self, dt: float) -> float:
        return self.completed_steps * float(dt)

    def to_metadata(self, *, dt: float) -> dict[str, object]:
        return {
            "schema_version": 1,
            "completed_steps": self.completed_steps,
            "time": self.time(dt),
            "physical": {
                "shape": list(self.physical.shape),
                "dtype": str(self.physical.dtype),
                "device": str(self.physical.device),
            },
            "native_spectrum": {
                "shape": list(self.native_spectrum.shape),
                "dtype": str(self.native_spectrum.dtype),
                "device": str(self.native_spectrum.device),
            },
            "spectral_refresh": {
                "interval": self.refresh_interval,
                "step_count": self.refresh_step_count,
                "refresh_count": self.refresh_count,
            },
            "history": None if self.history is None else self.history.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class ScalarReferenceStepResult:
    """Successful state transition and auditable operation trace."""

    state: ScalarReferenceState
    scheme_used: IntegratorScheme
    startup_step: bool
    refreshed: bool
    operation_order: tuple[str, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scheme_used": self.scheme_used.value,
            "startup_step": self.startup_step,
            "refreshed": self.refreshed,
            "operation_order": list(self.operation_order),
        }


class ScalarPeriodicReferenceStepper:
    """Functional CPU reference for semi-implicit Euler and SBDF2."""

    __slots__ = ("_device", "_linear_eigenvalues", "_model", "_spec")

    def __init__(
        self,
        *,
        model: ScalarPeriodicReactionDiffusion,
        spec: IntegratorSpec,
        device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(model, ScalarPeriodicReactionDiffusion):
            raise TypeError("model must be ScalarPeriodicReactionDiffusion")
        if not isinstance(spec, IntegratorSpec):
            raise TypeError("spec must be IntegratorSpec")
        normalized_device = torch.device(device)
        if normalized_device.type != "cpu":
            raise ValueError("P4.2 scalar reference execution is CPU-only")
        self._model = model
        self._spec = spec
        self._device = normalized_device
        self._linear_eigenvalues = model.linear_eigenvalues(
            dtype=torch.float64,
            device=normalized_device,
        )

    @property
    def model(self) -> ScalarPeriodicReactionDiffusion:
        return self._model

    @property
    def spec(self) -> IntegratorSpec:
        return self._spec

    def initial_state(
        self,
        *,
        refresh_interval: int | None = None,
    ) -> ScalarReferenceState:
        physical = self._model.initial_physical(
            dtype=torch.float64,
            device=self._device,
        )
        spectrum = self._project(torch.fft.rfft(physical))
        return ScalarReferenceState(
            physical=physical,
            native_spectrum=spectrum,
            completed_steps=0,
            refresh_interval=refresh_interval,
            refresh_step_count=0,
            refresh_count=0,
            history=None,
        )

    @staticmethod
    def _project(spectrum: torch.Tensor) -> torch.Tensor:
        imaginary = spectrum.imag.clone()
        imaginary[0] = 0.0
        imaginary[-1] = 0.0
        return torch.complex(spectrum.real, imaginary)

    def validate_state(self, state: ScalarReferenceState) -> None:
        """Validate a complete state without advancing or writing output."""

        if not isinstance(state, ScalarReferenceState):
            raise TypeError("state must be ScalarReferenceState")
        if state.physical.shape != (self._model.point_count,):
            raise ValueError("physical state shape does not match model")
        if state.native_spectrum.shape != (self._model.point_count // 2 + 1,):
            raise ValueError("spectral state shape does not match model")
        if state.physical.device != self._device:
            raise ValueError("state device does not match reference stepper")
        if self._spec.scheme is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER:
            if state.history is not None:
                raise ValueError("Euler reference state must not carry SBDF2 history")
        elif state.completed_steps == 0:
            if state.history is not None:
                raise ValueError("fresh SBDF2 state must not carry history")
        else:
            if state.history is None:
                raise ValueError("advanced SBDF2 state requires complete history")
            state.history.validate_for_current(
                state.native_spectrum,
                completed_steps=state.completed_steps,
                dt=self._spec.dt,
            )

    @staticmethod
    def _mark(
        operation: str,
        operations: list[str],
        stage_observer: Callable[[str], None] | None,
    ) -> None:
        operations.append(operation)
        if stage_observer is not None:
            stage_observer(operation)

    def step(
        self,
        state: ScalarReferenceState,
        *,
        pre_update_callback: Callable[[], None] | None = None,
        stage_observer: Callable[[str], None] | None = None,
    ) -> ScalarReferenceStepResult:
        """Return one new state, leaving ``state`` unchanged on every failure."""

        if pre_update_callback is not None and not callable(pre_update_callback):
            raise TypeError("pre_update_callback must be callable or None")
        if stage_observer is not None and not callable(stage_observer):
            raise TypeError("stage_observer must be callable or None")

        operations: list[str] = []
        self.validate_state(state)
        self._mark(
            "require_current_state_and_history",
            operations,
            stage_observer,
        )

        startup = (
            self._spec.scheme is IntegratorScheme.SBDF2
            and state.completed_steps == 0
        )
        use_euler = (
            self._spec.scheme is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
            or startup
        )
        if use_euler:
            denominator = 1.0 - self._spec.dt * self._linear_eigenvalues
        else:
            denominator = 3.0 / (2.0 * self._spec.dt) - self._linear_eigenvalues
        if not bool(torch.isfinite(denominator).all()) or bool(
            torch.any(denominator == 0.0)
        ):
            raise RuntimeError("implicit scalar denominator is invalid")
        self._mark("prepare_algebraic", operations, stage_observer)

        if pre_update_callback is not None:
            pre_update_callback()
        self._mark("pre_update_callback", operations, stage_observer)

        current_rhs = self._model.explicit_native_spectral_rhs(
            state.native_spectrum
        )
        if not bool(torch.isfinite(current_rhs).all()):
            raise RuntimeError("explicit spectral RHS is non-finite")
        self._mark(
            "evaluate_current_explicit_rhs",
            operations,
            stage_observer,
        )

        if use_euler:
            assembled_rhs = (
                state.native_spectrum + self._spec.dt * current_rhs
            )
            assembly_operation = "assemble_euler_implicit_rhs"
        else:
            history = state.history
            if history is None:  # Defensive narrowing after validation.
                raise RuntimeError("SBDF2 history disappeared after validation")
            assembled_rhs = (
                2.0 / self._spec.dt * state.native_spectrum
                - 0.5 / self._spec.dt
                * history.previous_evolved_native_spectrum
                + 2.0 * current_rhs
                - history.previous_explicit_native_spectral_rhs
            )
            assembly_operation = "assemble_sbdf2_implicit_rhs"
        self._mark(assembly_operation, operations, stage_observer)

        candidate_spectrum = assembled_rhs / denominator
        if not bool(torch.isfinite(candidate_spectrum).all()):
            raise RuntimeError("implicit scalar solve produced non-finite values")
        self._mark("solve_implicit_operator", operations, stage_observer)

        candidate_spectrum = self._project(candidate_spectrum)
        self._mark("project_dynamic_spectra", operations, stage_observer)

        candidate_physical = torch.fft.irfft(
            candidate_spectrum,
            n=self._model.point_count,
        )
        if not bool(torch.isfinite(candidate_physical).all()):
            raise RuntimeError("inverse transform produced non-finite values")
        self._mark("inverse_dynamic_spectra", operations, stage_observer)

        refreshed = state.refresh_due_after_next_step
        if refreshed:
            candidate_spectrum = self._project(
                torch.fft.rfft(candidate_physical)
            )
        self._mark(
            "scheduled_spectral_refresh",
            operations,
            stage_observer,
        )

        completed_steps = state.completed_steps + 1
        refresh_step_count = state.refresh_step_count + 1
        refresh_count = state.refresh_count
        if refreshed:
            refresh_step_count = 0
            refresh_count += 1
        self._mark("commit_progress", operations, stage_observer)

        history_for_next: SBDF2History | None = None
        if self._spec.scheme is IntegratorScheme.SBDF2:
            history_for_next = SBDF2History(
                previous_evolved_native_spectrum=state.native_spectrum,
                previous_explicit_native_spectral_rhs=current_rhs,
                source_completed_steps=state.completed_steps,
                dt=self._spec.dt,
            )
            self._mark(
                "commit_integrator_history",
                operations,
                stage_observer,
            )

        next_state = ScalarReferenceState(
            physical=candidate_physical,
            native_spectrum=candidate_spectrum,
            completed_steps=completed_steps,
            refresh_interval=state.refresh_interval,
            refresh_step_count=refresh_step_count,
            refresh_count=refresh_count,
            history=history_for_next,
        )
        scheme_used = (
            IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
            if use_euler
            else IntegratorScheme.SBDF2
        )
        return ScalarReferenceStepResult(
            state=next_state,
            scheme_used=scheme_used,
            startup_step=startup,
            refreshed=refreshed,
            operation_order=tuple(operations),
        )

    def run(
        self,
        *,
        steps: int,
        state: ScalarReferenceState | None = None,
        refresh_interval: int | None = None,
    ) -> ScalarReferenceState:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if state is not None and refresh_interval is not None:
            raise ValueError("refresh_interval is only valid for a fresh run")
        current = (
            self.initial_state(refresh_interval=refresh_interval)
            if state is None
            else state
        )
        for _ in range(steps):
            current = self.step(current).state
        return current

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "identity": "scalar_periodic_reference_stepper",
            "reference_only": True,
            "cpu_only": True,
            "functional_state_transition": True,
            "failure_atomic": True,
            "callback_timing": "after_prepare_before_explicit_rhs",
            "history_commit_position": "last",
            "spec": self._spec.to_metadata(),
            "model": self._model.to_metadata(),
            "startup_operation_order": list(_STARTUP_OPERATIONS),
            "sbdf2_operation_order": list(_SBDF2_OPERATIONS),
            "euler_operation_order": list(_EULER_OPERATIONS),
            "connection": {
                "runtime_state": False,
                "step_program": False,
                "plane": False,
                "production_checkpoint": False,
            },
        }


__all__ = [
    "ScalarPeriodicReactionDiffusion",
    "ScalarPeriodicReferenceStepper",
    "ScalarReferenceState",
    "ScalarReferenceStepResult",
]
