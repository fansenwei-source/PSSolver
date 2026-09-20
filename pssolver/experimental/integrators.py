"""Opt-in time integrators used by the architecture migration runtime."""

from __future__ import annotations

from pssolver.execution.state import (
    IntegratorProgress,
    RepresentationLedger,
    RuntimeState,
)
from pssolver.execution.workspace import WorkspacePlan
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.integrators.step_program import (
    ProjectedSemiImplicitEulerStepProgram,
)

from .performance import RuntimePerformanceRecorder, performance_region


class ProjectedSemiImplicitEulerIntegrator(SemiImplicitEulerIntegrator):
    """Semi-implicit Euler with a projected evolved spectrum.

    The update order matches the qualified Plane benchmark integrator: update
    the native spectral state, project it before inversion, and use a projected
    physical-to-spectral rebuild whenever the roundoff-refresh clock fires.
    The class remains experimental until the coupled trajectory gates pass.
    """

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        self.spectral_projector = model.spectral_projector
        if not self.spectral_projector.enabled:
            raise ValueError(
                "ProjectedSemiImplicitEulerIntegrator requires dealiasing"
            )

    def _refresh_dynamic_spectra(self):
        self.spectral_projector.refresh_dynamic_fields(
            self.model.fields,
            sync_spatial=True,
        )

    def step(self, pre_update_callback=None):
        if self._static_fields_are_current:
            self._static_fields_are_current = False
        else:
            self.model.update_static_fields()

        if pre_update_callback is not None:
            pre_update_callback()

        nonlinear_hats = self.model.compute_nonlinear()
        dynamic_fields = self.model.fields.spectral[: self.dyn_count]
        dynamic_fields.add_(self.dt * nonlinear_hats)
        dynamic_fields.div_(self.denom)
        self.spectral_projector.project_dynamic_fields(
            self.model.fields,
            sync_spatial=False,
        )

        for group in self.dynamic_transform_groups:
            boundary_conditions = self.model.fields.get_boundary_conditions(
                group[0]
            )
            self.model.fields.spatial[group] = (
                self.spectral_projector.inverse_transform(
                    self.model.fields.spectral[group],
                    boundary_conditions,
                )
            )

        self._advance_spectral_refresh_clock()


class InstrumentedProjectedSemiImplicitEulerIntegrator(
    ProjectedSemiImplicitEulerIntegrator
):
    """Stage N diagnostic variant, never selected by the default path."""

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        self.performance_recorder = None

    def step(self, pre_update_callback=None):
        recorder = self.performance_recorder
        if not isinstance(recorder, RuntimePerformanceRecorder):
            raise RuntimeError(
                "instrumented integrator requires a performance recorder"
            )
        with performance_region(recorder, "timestep.total"):
            with performance_region(recorder, "timestep.algebraic_update"):
                if self._static_fields_are_current:
                    self._static_fields_are_current = False
                else:
                    self.model.update_static_fields()

            if pre_update_callback is not None:
                with performance_region(
                    recorder,
                    "timestep.pre_update_callback",
                ):
                    pre_update_callback()

            with performance_region(recorder, "timestep.explicit_rhs"):
                nonlinear_hats = self.model.compute_nonlinear()
            with performance_region(recorder, "timestep.spectral_update"):
                dynamic_fields = self.model.fields.spectral[: self.dyn_count]
                dynamic_fields.add_(self.dt * nonlinear_hats)
                dynamic_fields.div_(self.denom)
            with performance_region(
                recorder,
                "timestep.dynamic_projection",
            ):
                self.spectral_projector.project_dynamic_fields(
                    self.model.fields,
                    sync_spatial=False,
                )

            with performance_region(recorder, "timestep.dynamic_inverse"):
                for group in self.dynamic_transform_groups:
                    boundary_conditions = (
                        self.model.fields.get_boundary_conditions(group[0])
                    )
                    self.model.fields.spatial[group] = (
                        self.spectral_projector.inverse_transform(
                            self.model.fields.spectral[group],
                            boundary_conditions,
                        )
                    )

            interval = self.spectral_refresh_interval
            refresh_due = (
                interval is not None and self.step_count + 1 >= interval
            )
            if refresh_due:
                with performance_region(
                    recorder,
                    "timestep.spectral_refresh",
                ):
                    self._advance_spectral_refresh_clock()
            else:
                self._advance_spectral_refresh_clock()


class StateBackedProjectedSemiImplicitEulerIntegrator(
    ProjectedSemiImplicitEulerIntegrator
):
    """P3.4 canary integrator backed by explicit state and step contracts.

    This class is selected only by the explicit Plane ``separated_canary``
    construction edge.  Its public counter and refresh attributes remain
    compatible with the legacy integrator while ``RuntimeState`` is their
    authority.
    """

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
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
            (
                index,
                name,
            )
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
        """Adopt arrays reallocated by the legacy-compatible reset facade."""

        progress = self._runtime_state.progress
        self._runtime_state = self._create_runtime_state(
            completed_steps=progress.completed_steps,
            refresh_interval=progress.refresh_interval,
            refresh_step_count=progress.refresh_step_count,
            refresh_count=progress.refresh_count,
        )

    def restore_progress(self, completed_steps, *, static_fields_are_current=False):
        if not isinstance(completed_steps, int) or isinstance(completed_steps, bool):
            raise TypeError("completed_steps must be an integer")
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        interval = self.spectral_refresh_interval
        if interval is None:
            refresh_count = 0
            refresh_step_count = completed_steps
        else:
            refresh_count, refresh_step_count = divmod(completed_steps, interval)
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

    def _inverse_dynamic_spectra(self, state, workspace, generation):
        for group in self.dynamic_transform_groups:
            boundary_conditions = self.model.fields.get_boundary_conditions(
                group[0]
            )
            self.model.fields.spatial[group] = (
                self.spectral_projector.inverse_transform(
                    self.model.fields.spectral[group],
                    boundary_conditions,
                )
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
            "connection_stage": "P3.4_separated_canary",
            "runtime_state": self._runtime_state.to_metadata(),
            "workspace": self._runtime_workspace.to_metadata(),
            "step_program": self._step_program.to_metadata(),
            "legacy_storage_adopted_without_copy": True,
            "production_default_changed": False,
        }


__all__ = [
    "InstrumentedProjectedSemiImplicitEulerIntegrator",
    "ProjectedSemiImplicitEulerIntegrator",
    "StateBackedProjectedSemiImplicitEulerIntegrator",
]
