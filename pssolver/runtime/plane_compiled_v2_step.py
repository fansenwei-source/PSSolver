"""Disconnected compiled projected-Euler execution for Plane.

P5.3 consumes the construction-time references frozen by P5.2.  The program
is deliberately private and is not reachable from a runtime selector or an
application entry point.  It performs no configuration, registry, or field
name lookup inside ``step``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from pssolver.execution.state import RuntimeState
from pssolver.execution.workspace import RuntimeWorkspace, WorkspacePlan
from pssolver.planning.plane_compiled_v2 import PlaneCompiledStage
from pssolver.runtime.plane_compiled_v2_binding import (
    PlaneCompiledV2BindingPlan,
)


_OPERATION_ORDER = tuple(stage.value for stage in PlaneCompiledStage)


@dataclass(frozen=True, slots=True)
class PlaneCompiledFailureSemantics:
    """Exact failure boundary inherited from the qualified Euler oracle."""

    progress_commit_last: bool = True
    workspace_generation_aborted: bool = True
    tensor_rollback: bool = False
    algebraic_side_effect_rollback: bool = False
    callback_side_effect_rollback: bool = False

    def to_metadata(self) -> dict[str, bool]:
        return {
            "progress_commit_last": self.progress_commit_last,
            "workspace_generation_aborted": self.workspace_generation_aborted,
            "tensor_rollback": self.tensor_rollback,
            "algebraic_side_effect_rollback": (
                self.algebraic_side_effect_rollback
            ),
            "callback_side_effect_rollback": self.callback_side_effect_rollback,
        }


class PlaneCompiledEulerStepProgram:
    """Fixed Plane Euler program over pre-bound tensors and operators.

    Construction performs all identity and layout checks through the P5.2
    binding plan.  The hot path uses direct references only.  Its zero-slot
    workspace supplies a re-entrancy guard and generation failure boundary;
    scientific kernels retain their already-qualified, fixed-shape scratch.
    """

    __slots__ = (
        "_denominator",
        "_dt",
        "_failure_semantics",
        "_operators",
        "_state",
        "_workspace",
    )

    connected_runtime = None

    def __init__(self, binding: PlaneCompiledV2BindingPlan) -> None:
        if not isinstance(binding, PlaneCompiledV2BindingPlan):
            raise TypeError("binding must be a PlaneCompiledV2BindingPlan")
        if binding.connected_runtime is not None:
            raise ValueError("compiled Plane binding must remain disconnected")
        if tuple(
            stage.value for stage in binding.declaration.operation_order
        ) != _OPERATION_ORDER:
            raise ValueError("compiled Plane operation order is incompatible")
        if (
            binding.legacy_workspace_bytes != 0
            or binding.additional_workspace_bytes != 0
        ):
            raise ValueError("P5.3 requires a zero-byte integration workspace")

        scalar_values = {
            scalar.name: scalar.value for scalar in binding.scalar_bindings
        }
        try:
            dt = scalar_values["dt"]
        except KeyError as exc:
            raise ValueError("compiled Plane binding is missing dt") from exc
        if dt != binding.state.progress.dt:
            raise ValueError("compiled Plane timestep binding is inconsistent")

        denominator = binding.tensors.denominator
        spectral = binding.state.spectral
        if (
            denominator.shape != spectral.shape
            or torch.promote_types(denominator.dtype, spectral.dtype)
            != spectral.dtype
            or denominator.device != spectral.device
        ):
            raise ValueError("compiled denominator is incompatible with state")
        if not bool(torch.isfinite(denominator).all().item()):
            raise ValueError("compiled denominator must be finite")
        if bool((denominator == 0).any().item()):
            raise ValueError("compiled denominator must be nonzero")

        self._state = binding.state
        self._operators = binding.operators
        self._denominator = denominator
        self._dt = dt
        self._workspace = WorkspacePlan(
            slots=(),
            device=spectral.device,
            maximum_bytes=0,
        ).allocate()
        self._failure_semantics = PlaneCompiledFailureSemantics()

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def workspace(self) -> RuntimeWorkspace:
        return self._workspace

    @property
    def failure_semantics(self) -> PlaneCompiledFailureSemantics:
        return self._failure_semantics

    def _require_rhs(self, value: object) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("explicit RHS must be a tensor")
        spectral = self._state.spectral
        if (
            value.shape != spectral.shape
            or value.dtype != spectral.dtype
            or value.device != spectral.device
        ):
            raise ValueError("explicit RHS does not match evolved spectrum")
        return value

    def step(
        self,
        *,
        pre_update_callback: Callable[[], None] | None = None,
    ) -> None:
        """Advance one step using the frozen production operation order."""

        if pre_update_callback is not None and not callable(
            pre_update_callback
        ):
            raise TypeError("pre_update_callback must be callable or None")

        state = self._state
        workspace = self._workspace
        operators = self._operators
        state.representations.require_spectral_current()

        generation = workspace.begin_generation()
        try:
            operators.prepare_algebraic(state, workspace, generation)
            if pre_update_callback is not None:
                pre_update_callback()

            rhs = self._require_rhs(
                operators.explicit_rhs(state, workspace, generation)
            )
            state.spectral.add_(self._dt * rhs)
            state.spectral.div_(self._denominator)
            state.representations.mark_spectral_updated()

            operators.project_dynamic_spectra(state, workspace, generation)
            operators.inverse_dynamic_spectra(state, workspace, generation)
            state.representations.mark_physical_synchronized()

            refresh_due = state.progress.refresh_due_after_next_step
            if refresh_due:
                operators.refresh_dynamic_spectra(
                    state,
                    workspace,
                    generation,
                )
            state.progress.commit_step(refreshed=refresh_due)
        except BaseException:
            workspace.abort_generation(token=generation)
            raise
        workspace.end_generation(token=generation)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "identity": "compiled_v2_projected_euler",
            "scheme": "projected_semi_implicit_euler",
            "operation_order": list(_OPERATION_ORDER),
            "callback_position": "after_prepare_algebraic_before_explicit_rhs",
            "static_field_invalidation": (
                "consume_restored_static_once_then_invalidate"
            ),
            "scheduled_spectral_refresh": True,
            "progress_commit": "last_successful_stage",
            "failure_semantics": self._failure_semantics.to_metadata(),
            "workspace": self._workspace.to_metadata(),
            "prebound_tensor_references": True,
            "prebound_operator_references": True,
            "registry_lookup_in_step": False,
            "configuration_lookup_in_step": False,
            "string_field_discovery_in_step": False,
            "metadata_construction_in_step": False,
            "implicit_fallback": False,
            "connected_runtime": self.connected_runtime,
        }


def build_plane_compiled_v2_step_program(
    binding: PlaneCompiledV2BindingPlan,
) -> PlaneCompiledEulerStepProgram:
    """Build the private disconnected P5.3 step program."""

    return PlaneCompiledEulerStepProgram(binding)


__all__ = [
    "PlaneCompiledEulerStepProgram",
    "PlaneCompiledFailureSemantics",
    "build_plane_compiled_v2_step_program",
]
