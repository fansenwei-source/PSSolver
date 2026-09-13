"""Opt-in time integrators used by the architecture migration runtime."""

from __future__ import annotations

from pssolver.integrator import SemiImplicitEulerIntegrator


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


__all__ = ["ProjectedSemiImplicitEulerIntegrator"]
