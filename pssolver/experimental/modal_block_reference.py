"""Opt-in two-component SBDF2/modal-block reference canary for Phase 4.5.

This module deliberately remains disconnected from runtime state, Plane, and
the production checkpoint path.  It combines the P4.2 integration contract
with the P4.4 qualified two-by-two modal solve in one auditable reference
workflow.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch

from pssolver.core.integrators import IntegratorScheme, IntegratorSpec
from pssolver.integrators.history import SBDF2History
from pssolver.operators.modal_block import (
    BoundTwoComponentModalOperator,
    TwoComponentModalSolveWorkspace,
)
from pssolver.operators.modal_block_reference import (
    TwoComponentPeriodicReactionDiffusionReference,
)


STARTUP_OPERATIONS = (
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

SBDF2_OPERATIONS = (
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


def _finite(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{description} must be finite")
    return float(value)


def _storage_identity(value: torch.Tensor) -> tuple[str, int]:
    return str(value.device), int(value.untyped_storage().data_ptr())


@dataclass(frozen=True, slots=True)
class TwoComponentSBDF2CanaryModel:
    """One analytic ``L + rho I`` split used by the combined canary."""

    implicit_model: TwoComponentPeriodicReactionDiffusionReference
    explicit_growth_rate: float = 0.07

    def __post_init__(self) -> None:
        if not isinstance(
            self.implicit_model,
            TwoComponentPeriodicReactionDiffusionReference,
        ):
            raise TypeError(
                "implicit_model must be a two-component reference model"
            )
        object.__setattr__(
            self,
            "explicit_growth_rate",
            _finite(self.explicit_growth_rate, "explicit_growth_rate"),
        )

    @property
    def point_count(self) -> int:
        return self.implicit_model.point_count

    def initial_physical(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return self.implicit_model.initial_physical(dtype=dtype, device=device)

    def exact_physical(
        self,
        time: float,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return self.implicit_model.exact_physical(
            time,
            dtype=dtype,
            device=device,
        ) * math.exp(self.explicit_growth_rate * float(time))

    def explicit_rhs_into(
        self,
        native_spectrum: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if output.shape != native_spectrum.shape:
            raise ValueError("explicit RHS output shape mismatch")
        if output.dtype != native_spectrum.dtype:
            raise ValueError("explicit RHS output dtype mismatch")
        if output.device != native_spectrum.device:
            raise ValueError("explicit RHS output device mismatch")
        output.copy_(native_spectrum).mul_(self.explicit_growth_rate)
        return output

    def bind_operator(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        implementation: str,
    ) -> BoundTwoComponentModalOperator:
        return self.implicit_model.bind_operator(
            dtype=dtype,
            device=device,
            implementation=implementation,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "identity": "two_component_sbdf2_modal_block_canary",
            "implicit_model": self.implicit_model.to_metadata(),
            "explicit_growth_rate": self.explicit_growth_rate,
            "split": {
                "implicit": "two_component_reaction_diffusion_modal_block",
                "explicit": "scalar_identity_growth",
            },
        }


@dataclass(frozen=True, slots=True)
class TwoComponentReferenceState:
    """Complete persistent state for the disconnected block canary."""

    physical: torch.Tensor
    native_spectrum: torch.Tensor
    completed_steps: int
    refresh_interval: int | None
    refresh_step_count: int
    refresh_count: int
    history: SBDF2History | None

    def __post_init__(self) -> None:
        if not isinstance(self.physical, torch.Tensor) or not isinstance(
            self.native_spectrum, torch.Tensor
        ):
            raise TypeError("physical and native_spectrum must be tensors")
        if self.physical.ndim != 2 or self.physical.shape[-1] != 2:
            raise ValueError("physical state must have trailing component size 2")
        if self.native_spectrum.ndim != 2 or self.native_spectrum.shape[-1] != 2:
            raise ValueError("spectral state must have trailing component size 2")
        expected_spectral = (
            torch.complex64
            if self.physical.dtype is torch.float32
            else torch.complex128
            if self.physical.dtype is torch.float64
            else None
        )
        if (
            expected_spectral is None
            or self.native_spectrum.dtype is not expected_spectral
        ):
            raise ValueError("physical and spectral dtypes are incompatible")
        if self.physical.device != self.native_spectrum.device:
            raise ValueError("physical and spectral devices differ")
        for value, name in (
            (self.completed_steps, "completed_steps"),
            (self.refresh_step_count, "refresh_step_count"),
            (self.refresh_count, "refresh_count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
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
        if self.history is not None:
            if not isinstance(self.history, SBDF2History):
                raise TypeError("history must be SBDF2History or None")
            identities = {
                _storage_identity(self.native_spectrum),
                _storage_identity(
                    self.history.previous_evolved_native_spectrum
                ),
                _storage_identity(
                    self.history.previous_explicit_native_spectral_rhs
                ),
            }
            if len(identities) != 3:
                raise ValueError("current state and history must own distinct storage")
        if not bool(torch.isfinite(self.physical).all()) or not bool(
            torch.isfinite(self.native_spectrum).all()
        ):
            raise ValueError("reference state must be finite")

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
class TwoComponentSBDF2Workspace:
    """Fixed current-RHS, assembly, and modal-solve scratch ownership."""

    current_explicit_rhs: torch.Tensor
    assembled_implicit_rhs: torch.Tensor
    modal_solve: TwoComponentModalSolveWorkspace

    def __post_init__(self) -> None:
        if not isinstance(self.current_explicit_rhs, torch.Tensor) or not isinstance(
            self.assembled_implicit_rhs, torch.Tensor
        ):
            raise TypeError("SBDF2 workspace entries must be tensors")
        if (
            self.current_explicit_rhs.shape
            != self.assembled_implicit_rhs.shape
            or self.current_explicit_rhs.dtype
            != self.assembled_implicit_rhs.dtype
            or self.current_explicit_rhs.device
            != self.assembled_implicit_rhs.device
        ):
            raise ValueError("SBDF2 workspace tensor layouts differ")
        if self.current_explicit_rhs.shape[-1] != 2:
            raise ValueError("SBDF2 workspace requires two components")
        identities = {
            _storage_identity(self.current_explicit_rhs),
            _storage_identity(self.assembled_implicit_rhs),
            _storage_identity(self.modal_solve.output),
            _storage_identity(self.modal_solve.diagonal_0),
            _storage_identity(self.modal_solve.diagonal_1),
            _storage_identity(self.modal_solve.determinant),
            _storage_identity(self.modal_solve.temporary),
        }
        if len(identities) != 7:
            raise ValueError("combined workspace tensors must own distinct storage")

    @property
    def allocated_tensor_count(self) -> int:
        return 2 + self.modal_solve.allocated_tensor_count

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "bounded": True,
            "allocated_tensor_count": self.allocated_tensor_count,
            "current_explicit_rhs_shape": list(self.current_explicit_rhs.shape),
            "assembled_implicit_rhs_shape": list(
                self.assembled_implicit_rhs.shape
            ),
            "modal_solve": self.modal_solve.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class TwoComponentReferenceStepResult:
    state: TwoComponentReferenceState
    scheme_used: IntegratorScheme
    startup_step: bool
    refreshed: bool
    operation_order: tuple[str, ...]


class TwoComponentSBDF2ReferenceStepper:
    """Combined SBDF2 and 2-by-2 modal-block opt-in reference workflow."""

    __slots__ = (
        "_device",
        "_dtype",
        "_implementation",
        "_model",
        "_operator",
        "_spec",
        "_workspace",
    )

    def __init__(
        self,
        *,
        model: TwoComponentSBDF2CanaryModel,
        spec: IntegratorSpec,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        implementation: str = "closed_form_2x2",
    ) -> None:
        if not isinstance(model, TwoComponentSBDF2CanaryModel):
            raise TypeError("model must be TwoComponentSBDF2CanaryModel")
        if not isinstance(spec, IntegratorSpec):
            raise TypeError("spec must be IntegratorSpec")
        if spec.scheme is not IntegratorScheme.SBDF2:
            raise ValueError("combined P4.5 canary requires SBDF2")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64")
        self._model = model
        self._spec = spec
        self._dtype = dtype
        requested_device = torch.device(device)
        self._implementation = implementation
        self._operator = model.bind_operator(
            dtype=dtype,
            device=requested_device,
            implementation=implementation,
        )
        self._device = self._operator.device
        modal_workspace = self._operator.allocate_workspace()
        self._workspace = TwoComponentSBDF2Workspace(
            current_explicit_rhs=torch.empty_like(modal_workspace.output),
            assembled_implicit_rhs=torch.empty_like(modal_workspace.output),
            modal_solve=modal_workspace,
        )

    @property
    def model(self) -> TwoComponentSBDF2CanaryModel:
        return self._model

    @property
    def spec(self) -> IntegratorSpec:
        return self._spec

    @property
    def workspace(self) -> TwoComponentSBDF2Workspace:
        return self._workspace

    @property
    def operator(self) -> BoundTwoComponentModalOperator:
        return self._operator

    @staticmethod
    def _project_in_place(spectrum: torch.Tensor) -> None:
        spectrum.imag[0].zero_()
        spectrum.imag[-1].zero_()

    @staticmethod
    def _mark(
        operation: str,
        operations: list[str],
        observer: Callable[[str], None] | None,
    ) -> None:
        operations.append(operation)
        if observer is not None:
            observer(operation)

    def initial_state(
        self,
        *,
        refresh_interval: int | None = None,
    ) -> TwoComponentReferenceState:
        physical = self._model.initial_physical(
            dtype=self._dtype,
            device=self._device,
        )
        spectrum = torch.fft.rfft(physical, dim=0)
        self._project_in_place(spectrum)
        return TwoComponentReferenceState(
            physical=physical,
            native_spectrum=spectrum,
            completed_steps=0,
            refresh_interval=refresh_interval,
            refresh_step_count=0,
            refresh_count=0,
            history=None,
        )

    def validate_state(self, state: TwoComponentReferenceState) -> None:
        if not isinstance(state, TwoComponentReferenceState):
            raise TypeError("state must be TwoComponentReferenceState")
        if state.physical.shape != (self._model.point_count, 2):
            raise ValueError("physical state shape does not match model")
        if state.native_spectrum.shape != (
            self._model.point_count // 2 + 1,
            2,
        ):
            raise ValueError("spectral state shape does not match model")
        if (
            state.physical.dtype is not self._dtype
            or state.physical.device != self._device
        ):
            raise ValueError("state dtype or device does not match stepper")
        if state.completed_steps == 0:
            if state.history is not None:
                raise ValueError("fresh SBDF2 state must not carry history")
        else:
            if state.history is None:
                raise ValueError("advanced SBDF2 state requires history")
            state.history.validate_for_current(
                state.native_spectrum,
                completed_steps=state.completed_steps,
                dt=self._spec.dt,
            )

    def step(
        self,
        state: TwoComponentReferenceState,
        *,
        pre_update_callback: Callable[[], None] | None = None,
        stage_observer: Callable[[str], None] | None = None,
    ) -> TwoComponentReferenceStepResult:
        if pre_update_callback is not None and not callable(pre_update_callback):
            raise TypeError("pre_update_callback must be callable or None")
        if stage_observer is not None and not callable(stage_observer):
            raise TypeError("stage_observer must be callable or None")
        operations: list[str] = []
        self.validate_state(state)
        self._mark("require_current_state_and_history", operations, stage_observer)

        startup = state.completed_steps == 0
        alpha = (
            1.0 / self._spec.dt
            if startup
            else 3.0 / (2.0 * self._spec.dt)
        )
        self._mark("prepare_algebraic", operations, stage_observer)
        if pre_update_callback is not None:
            pre_update_callback()
        self._mark("pre_update_callback", operations, stage_observer)

        current_rhs = self._model.explicit_rhs_into(
            state.native_spectrum,
            self._workspace.current_explicit_rhs,
        )
        if not bool(torch.isfinite(current_rhs).all()):
            raise RuntimeError("explicit spectral RHS is non-finite")
        self._mark("evaluate_current_explicit_rhs", operations, stage_observer)

        assembled = self._workspace.assembled_implicit_rhs
        if startup:
            assembled.copy_(state.native_spectrum).mul_(1.0 / self._spec.dt)
            assembled.add_(current_rhs)
            assembly_operation = "assemble_euler_implicit_rhs"
        else:
            history = state.history
            if history is None:
                raise RuntimeError("SBDF2 history disappeared after validation")
            assembled.copy_(state.native_spectrum).mul_(2.0 / self._spec.dt)
            assembled.add_(
                history.previous_evolved_native_spectrum,
                alpha=-0.5 / self._spec.dt,
            )
            assembled.add_(current_rhs, alpha=2.0)
            assembled.sub_(history.previous_explicit_native_spectral_rhs)
            assembly_operation = "assemble_sbdf2_implicit_rhs"
        self._mark(assembly_operation, operations, stage_observer)

        candidate = self._operator.solve_into(
            assembled,
            alpha=alpha,
            workspace=self._workspace.modal_solve,
        )
        self._mark("solve_implicit_operator", operations, stage_observer)
        self._project_in_place(candidate)
        self._mark("project_dynamic_spectra", operations, stage_observer)

        physical = torch.fft.irfft(
            candidate,
            n=self._model.point_count,
            dim=0,
        )
        if not bool(torch.isfinite(physical).all()):
            raise RuntimeError("inverse transform produced non-finite values")
        self._mark("inverse_dynamic_spectra", operations, stage_observer)

        refreshed = state.refresh_due_after_next_step
        if refreshed:
            owned_spectrum = torch.fft.rfft(physical, dim=0)
            self._project_in_place(owned_spectrum)
        else:
            owned_spectrum = candidate.clone()
        self._mark("scheduled_spectral_refresh", operations, stage_observer)

        completed_steps = state.completed_steps + 1
        refresh_step_count = state.refresh_step_count + 1
        refresh_count = state.refresh_count
        if refreshed:
            refresh_step_count = 0
            refresh_count += 1
        self._mark("commit_progress", operations, stage_observer)

        history = SBDF2History(
            previous_evolved_native_spectrum=state.native_spectrum.clone(),
            previous_explicit_native_spectral_rhs=current_rhs.clone(),
            source_completed_steps=state.completed_steps,
            dt=self._spec.dt,
        )
        self._mark("commit_integrator_history", operations, stage_observer)
        next_state = TwoComponentReferenceState(
            physical=physical,
            native_spectrum=owned_spectrum,
            completed_steps=completed_steps,
            refresh_interval=state.refresh_interval,
            refresh_step_count=refresh_step_count,
            refresh_count=refresh_count,
            history=history,
        )
        return TwoComponentReferenceStepResult(
            state=next_state,
            scheme_used=(
                IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
                if startup
                else IntegratorScheme.SBDF2
            ),
            startup_step=startup,
            refreshed=refreshed,
            operation_order=tuple(operations),
        )

    def run(
        self,
        *,
        steps: int,
        state: TwoComponentReferenceState | None = None,
        refresh_interval: int | None = None,
    ) -> TwoComponentReferenceState:
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
            "identity": "two_component_sbdf2_modal_block_reference_stepper",
            "reference_only": True,
            "opt_in": True,
            "functional_state_transition": True,
            "failure_atomic": True,
            "dtype": str(self._dtype),
            "device": str(self._device),
            "implementation": self._implementation,
            "spec": self._spec.to_metadata(),
            "model": self._model.to_metadata(),
            "workspace": self._workspace.to_metadata(),
            "startup_operation_order": list(STARTUP_OPERATIONS),
            "sbdf2_operation_order": list(SBDF2_OPERATIONS),
            "history_commit_position": "last",
            "connection": {
                "runtime_state": False,
                "step_program": False,
                "plane": False,
                "production_checkpoint": False,
                "package_root_export": False,
            },
        }


__all__ = [
    "SBDF2_OPERATIONS",
    "STARTUP_OPERATIONS",
    "TwoComponentReferenceState",
    "TwoComponentReferenceStepResult",
    "TwoComponentSBDF2CanaryModel",
    "TwoComponentSBDF2ReferenceStepper",
    "TwoComponentSBDF2Workspace",
]
