"""Opt-in time integrators used by the architecture migration runtime."""

from __future__ import annotations

from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.integrators.state_backed import (
    StateBackedProjectedIntegratorMixin,
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
    StateBackedProjectedIntegratorMixin,
    ProjectedSemiImplicitEulerIntegrator
):
    """P3.4 canary integrator backed by explicit state and step contracts.

    This class is selected only by the explicit Plane ``separated_canary``
    construction edge.  Its public counter and refresh attributes remain
    compatible with the legacy integrator while ``RuntimeState`` is their
    authority.
    """

    _phase3_connection_stage = "P3.4_separated_canary"

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

__all__ = [
    "InstrumentedProjectedSemiImplicitEulerIntegrator",
    "ProjectedSemiImplicitEulerIntegrator",
    "StateBackedProjectedSemiImplicitEulerIntegrator",
]
