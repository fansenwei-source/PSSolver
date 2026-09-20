"""Shared state-backed projected integration for the Phase 3 migration.

The mixin in this module changes mutable ownership only.  Concrete integrator
facades retain their qualified projection, inverse-storage, and refresh
operations while this layer supplies ``RuntimeState``, bounded workspace,
counter compatibility, and the frozen projected-IMEX ``StepProgram``.
"""

from __future__ import annotations

from pssolver.execution.state import (
    IntegratorProgress,
    RepresentationLedger,
    RuntimeState,
)
from pssolver.execution.workspace import WorkspacePlan
from pssolver.integrators.step_program import (
    ProjectedSemiImplicitEulerStepProgram,
)


class StateBackedProjectedIntegratorMixin:
    """Adopt a projected integrator's evolved arrays without copying them.

    Concrete classes must provide ``_inverse_dynamic_spectra`` and the
    projected-integrator surface initialized by ``super().__init__``.  The
    mixin deliberately has no model-, geometry-, or backend-specific branch
    in its timestep path.
    """

    _phase3_connection_stage: str

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        connection_stage = getattr(self, "_phase3_connection_stage", None)
        if not isinstance(connection_stage, str) or not connection_stage:
            raise RuntimeError(
                "state-backed integrator requires a connection-stage label"
            )
        self._runtime_state = self._create_runtime_state(
            completed_steps=0,
            refresh_interval=self._spectral_refresh_interval,
            refresh_step_count=self._bootstrap_step_count,
            refresh_count=self._bootstrap_refresh_count,
        )
        del self._bootstrap_step_count
        del self._bootstrap_refresh_count
        self._runtime_workspace = WorkspacePlan(
            slots=(),
            device=self.model.fields.spatial.device,
            maximum_bytes=0,
        ).allocate()
        self._step_program = ProjectedSemiImplicitEulerStepProgram(
            denominator=self.denom,
            prepare_algebraic=self._prepare_algebraic,
            explicit_rhs=self._explicit_rhs,
            project_dynamic_spectra=self._project_dynamic_spectra,
            inverse_dynamic_spectra=self._inverse_dynamic_spectra,
            refresh_dynamic_spectra=self._refresh_dynamic_spectra_operation,
        )

    def _dynamic_component_names(self) -> tuple[str, ...]:
        by_index = sorted(
            (index, name)
            for name, index in self.model.fields.name_to_idx.items()
            if index < self.dyn_count
        )
        names = tuple(name for _, name in by_index)
        if len(names) != self.dyn_count:
            raise RuntimeError("dynamic component layout is incomplete")
        return names

    def _create_runtime_state(
        self,
        *,
        completed_steps: int,
        refresh_interval: int | None,
        refresh_step_count: int,
        refresh_count: int,
    ) -> RuntimeState:
        return RuntimeState(
            component_names=self._dynamic_component_names(),
            physical=self.model.fields.spatial[: self.dyn_count],
            spectral=self.model.fields.spectral[: self.dyn_count],
            progress=IntegratorProgress(
                dt=self.dt,
                completed_steps=completed_steps,
                refresh_interval=refresh_interval,
                refresh_step_count=refresh_step_count,
                refresh_count=refresh_count,
            ),
            representations=RepresentationLedger(
                generation=completed_steps,
                physical_generation=completed_steps,
                spectral_generation=completed_steps,
            ),
        )

    @property
    def runtime_state(self) -> RuntimeState:
        return self._runtime_state

    @property
    def runtime_workspace(self):
        return self._runtime_workspace

    @property
    def spectral_refresh_interval(self):
        state = getattr(self, "_runtime_state", None)
        if state is None:
            return self._spectral_refresh_interval
        return state.progress.refresh_interval

    @spectral_refresh_interval.setter
    def spectral_refresh_interval(self, interval):
        if interval is not None and (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval <= 0
        ):
            raise ValueError(
                "spectral_refresh_interval must be a positive integer or None"
            )
        state = getattr(self, "_runtime_state", None)
        if state is None:
            self._spectral_refresh_interval = interval
            return
        completed_steps = state.progress.completed_steps
        if interval is None:
            refresh_count = 0
            refresh_step_count = completed_steps
        else:
            refresh_count, refresh_step_count = divmod(
                completed_steps,
                interval,
            )
        state.replace_progress(
            IntegratorProgress(
                dt=self.dt,
                completed_steps=completed_steps,
                refresh_interval=interval,
                refresh_step_count=refresh_step_count,
                refresh_count=refresh_count,
            )
        )

    @property
    def step_count(self):
        state = getattr(self, "_runtime_state", None)
        if state is None:
            return self._bootstrap_step_count
        return state.progress.refresh_step_count

    @step_count.setter
    def step_count(self, value):
        state = getattr(self, "_runtime_state", None)
        if state is None:
            self._bootstrap_step_count = value
            return
        self._replace_public_counters(step_count=value)

    @property
    def refresh_count(self):
        state = getattr(self, "_runtime_state", None)
        if state is None:
            return self._bootstrap_refresh_count
        return state.progress.refresh_count

    @refresh_count.setter
    def refresh_count(self, value):
        state = getattr(self, "_runtime_state", None)
        if state is None:
            self._bootstrap_refresh_count = value
            return
        self._replace_public_counters(refresh_count=value)

    def _replace_public_counters(
        self,
        *,
        step_count=None,
        refresh_count=None,
    ) -> None:
        progress = self._runtime_state.progress
        step_count = (
            progress.refresh_step_count
            if step_count is None
            else step_count
        )
        refresh_count = (
            progress.refresh_count
            if refresh_count is None
            else refresh_count
        )
        interval = progress.refresh_interval
        completed_steps = (
            step_count
            if interval is None
            else refresh_count * interval + step_count
        )
        self._runtime_state.replace_progress(
            IntegratorProgress(
                dt=self.dt,
                completed_steps=completed_steps,
                refresh_interval=interval,
                refresh_step_count=step_count,
                refresh_count=refresh_count,
            )
        )

    def rebind_runtime_state(self) -> None:
        """Adopt arrays reallocated by a legacy-compatible reset facade."""

        progress = self._runtime_state.progress
        self._runtime_state = self._create_runtime_state(
            completed_steps=progress.completed_steps,
            refresh_interval=progress.refresh_interval,
            refresh_step_count=progress.refresh_step_count,
            refresh_count=progress.refresh_count,
        )

    def restore_progress(
        self,
        completed_steps,
        *,
        static_fields_are_current=False,
    ):
        if not isinstance(completed_steps, int) or isinstance(
            completed_steps,
            bool,
        ):
            raise TypeError("completed_steps must be an integer")
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        interval = self.spectral_refresh_interval
        if interval is None:
            refresh_count = 0
            refresh_step_count = completed_steps
        else:
            refresh_count, refresh_step_count = divmod(
                completed_steps,
                interval,
            )
        self._runtime_state.replace_progress(
            IntegratorProgress(
                dt=self.dt,
                completed_steps=completed_steps,
                refresh_interval=interval,
                refresh_step_count=refresh_step_count,
                refresh_count=refresh_count,
            )
        )
        self._runtime_state.replace_representations(
            RepresentationLedger(
                generation=completed_steps,
                physical_generation=completed_steps,
                spectral_generation=completed_steps,
            )
        )
        self._static_fields_are_current = bool(static_fields_are_current)

    def _prepare_algebraic(self, state, workspace, generation):
        if self._static_fields_are_current:
            self._static_fields_are_current = False
        else:
            self.model.update_static_fields()

    def _explicit_rhs(self, state, workspace, generation):
        return self.model.compute_nonlinear()

    def _project_dynamic_spectra(self, state, workspace, generation):
        self.spectral_projector.project_dynamic_fields(
            self.model.fields,
            sync_spatial=False,
        )

    def _refresh_dynamic_spectra_operation(self, state, workspace, generation):
        self._refresh_dynamic_spectra()

    def _advance_spectral_refresh_clock(self):
        refresh_due = self._runtime_state.progress.refresh_due_after_next_step
        if refresh_due:
            self._refresh_dynamic_spectra()
        self._runtime_state.progress.commit_step(refreshed=refresh_due)
        return refresh_due

    def step(self, pre_update_callback=None):
        self._step_program.step(
            self._runtime_state,
            self._runtime_workspace,
            pre_update_callback=pre_update_callback,
        )

    def phase3_execution_metadata(self):
        return {
            "schema_version": 1,
            "connection_stage": self._phase3_connection_stage,
            "runtime_state": self._runtime_state.to_metadata(),
            "workspace": self._runtime_workspace.to_metadata(),
            "step_program": self._step_program.to_metadata(),
            "legacy_storage_adopted_without_copy": True,
            "production_default_changed": False,
        }


__all__ = ["StateBackedProjectedIntegratorMixin"]
