"""Pre-bound timestep programs for the Phase 3 migration.

This module mirrors the qualified projected semi-implicit Euler operation
order over explicit state/workspace contracts.  It is not connected to either
Plane runtime in P3.3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from pssolver.execution.state import RuntimeState
from pssolver.execution.workspace import RuntimeWorkspace


class StepOperation(Protocol):
    """One pre-bound operation in a timestep program."""

    def __call__(
        self,
        state: RuntimeState,
        workspace: RuntimeWorkspace,
        generation: int,
    ) -> None: ...


class ExplicitRHSOperation(Protocol):
    """Pre-bound explicit right-hand-side operation."""

    def __call__(
        self,
        state: RuntimeState,
        workspace: RuntimeWorkspace,
        generation: int,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class ProjectedSemiImplicitEulerStepProgram:
    """Fixed projected-IMEX sequence with explicit mutable ownership.

    The supplied operations are already resolved to one model, geometry,
    backend, and layout.  The program performs no registry lookup, dependency
    discovery, metadata construction, or string-based field dispatch.
    """

    denominator: torch.Tensor
    prepare_algebraic: StepOperation
    explicit_rhs: ExplicitRHSOperation
    project_dynamic_spectra: StepOperation
    inverse_dynamic_spectra: StepOperation
    refresh_dynamic_spectra: StepOperation

    def __post_init__(self) -> None:
        if not isinstance(self.denominator, torch.Tensor):
            raise TypeError("denominator must be a tensor")
        if not bool(torch.isfinite(self.denominator).all().item()):
            raise ValueError("denominator must be finite")
        if bool((self.denominator == 0).any().item()):
            raise ValueError("denominator must be nonzero")
        for name in (
            "prepare_algebraic",
            "explicit_rhs",
            "project_dynamic_spectra",
            "inverse_dynamic_spectra",
            "refresh_dynamic_spectra",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def _validate_bound_state(self, state: RuntimeState) -> None:
        spectral = state.spectral
        if self.denominator.shape != spectral.shape:
            raise ValueError("denominator shape does not match evolved spectrum")
        if torch.promote_types(
            self.denominator.dtype,
            spectral.dtype,
        ) != spectral.dtype:
            raise ValueError("denominator dtype is incompatible with spectrum")
        if self.denominator.device != spectral.device:
            raise ValueError("denominator device does not match evolved spectrum")

    @staticmethod
    def _validate_rhs(rhs: object, state: RuntimeState) -> torch.Tensor:
        if not isinstance(rhs, torch.Tensor):
            raise TypeError("explicit RHS must be a tensor")
        spectral = state.spectral
        if (
            rhs.shape != spectral.shape
            or rhs.dtype != spectral.dtype
            or rhs.device != spectral.device
        ):
            raise ValueError("explicit RHS does not match evolved spectrum")
        return rhs

    def step(
        self,
        state: RuntimeState,
        workspace: RuntimeWorkspace,
        *,
        pre_update_callback: Callable[[], None] | None = None,
    ) -> None:
        """Execute one qualified-order step and commit progress last."""

        if not isinstance(state, RuntimeState):
            raise TypeError("state must be a RuntimeState")
        if not isinstance(workspace, RuntimeWorkspace):
            raise TypeError("workspace must be a RuntimeWorkspace")
        if pre_update_callback is not None and not callable(pre_update_callback):
            raise TypeError("pre_update_callback must be callable or None")
        self._validate_bound_state(state)
        state.representations.require_spectral_current()

        generation = workspace.begin_generation()
        try:
            self.prepare_algebraic(state, workspace, generation)
            if pre_update_callback is not None:
                pre_update_callback()

            rhs = self._validate_rhs(
                self.explicit_rhs(state, workspace, generation),
                state,
            )
            state.spectral.add_(state.progress.dt * rhs)
            state.spectral.div_(self.denominator)
            state.representations.mark_spectral_updated()

            self.project_dynamic_spectra(state, workspace, generation)
            self.inverse_dynamic_spectra(state, workspace, generation)
            state.representations.mark_physical_synchronized()

            refresh_due = state.progress.refresh_due_after_next_step
            if refresh_due:
                self.refresh_dynamic_spectra(state, workspace, generation)
            state.progress.commit_step(refreshed=refresh_due)
        except BaseException:
            workspace.abort_generation(token=generation)
            raise
        workspace.end_generation(token=generation)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scheme": "projected_semi_implicit_euler",
            "operation_order": [
                "prepare_algebraic",
                "pre_update_callback",
                "explicit_rhs",
                "spectral_add_dt_rhs",
                "spectral_divide_by_denominator",
                "project_dynamic_spectra",
                "inverse_dynamic_spectra",
                "scheduled_spectral_refresh",
                "commit_progress",
            ],
            "registry_lookup_in_step": False,
            "metadata_construction_in_step": False,
            "workspace_allocation_in_step": False,
            "connected_runtime": None,
        }


__all__ = ["ProjectedSemiImplicitEulerStepProgram"]
