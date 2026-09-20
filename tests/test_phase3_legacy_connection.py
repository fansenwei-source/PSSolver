"""P3.5 local gates for the production-facade state connection."""

from __future__ import annotations

import torch

import pssolver
import pssolver.integrators as integrators
from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime import (
    DealiasedSemiImplicitEulerIntegrator,
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)
from pssolver.runtime import plane_legacy
from pssolver.workflows import (
    capture_plane_checkpoint,
    restore_plane_checkpoint,
)


class _PrePhase3DealiasedIntegrator(SemiImplicitEulerIntegrator):
    """Frozen pre-P3.5 operation order used only as a local oracle."""

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        self.spectral_projector = model.spectral_projector

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
            self.model.fields.store_spatial_group(
                group,
                self.spectral_projector.inverse_transform(
                    self.model.fields.select_spectral_group(group),
                    boundary_conditions,
                ),
            )
        self._advance_spectral_refresh_clock()


def _spec(tmp_path):
    return create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "unused",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        nx=8,
        ny=8,
        nz=8,
        lx=8.0,
        ly=9.0,
        height=20.0,
        steps=6,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        spectral_refresh_steps=3,
    )


def _initial_values():
    coordinate = torch.arange(8**3, dtype=torch.float64).reshape(8, 8, 8)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


def _request(spec):
    return PlaneRuntimeBuildRequest(
        run_spec=spec,
        production_metadata={
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        },
        initial_values=_initial_values(),
        device="cpu",
    )


def _build_with_integrator(spec, monkeypatch, integrator_class):
    monkeypatch.setattr(
        plane_legacy,
        "DealiasedSemiImplicitEulerIntegrator",
        integrator_class,
    )
    return build_plane_beris_edwards_runtime(_request(spec))


def _assert_storage_equal(left, right):
    assert torch.equal(left.solver.fields.spatial, right.solver.fields.spatial)
    assert torch.equal(left.solver.fields.spectral, right.solver.fields.spectral)


def test_p35_production_facade_matches_frozen_step_oracle(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    reference = _build_with_integrator(
        spec,
        monkeypatch,
        _PrePhase3DealiasedIntegrator,
    )
    candidate = _build_with_integrator(
        spec,
        monkeypatch,
        DealiasedSemiImplicitEulerIntegrator,
    )

    assert type(reference.solver.integrator) is _PrePhase3DealiasedIntegrator
    assert type(candidate.solver.integrator) is (
        DealiasedSemiImplicitEulerIntegrator
    )
    _assert_storage_equal(reference, candidate)

    state = candidate.solver.integrator.runtime_state
    assert state.physical.untyped_storage().data_ptr() == (
        candidate.solver.fields.spatial.untyped_storage().data_ptr()
    )
    assert state.spectral.untyped_storage().data_ptr() == (
        candidate.solver.fields.spectral.untyped_storage().data_ptr()
    )

    for completed_steps in range(1, 7):
        reference.advance(1)
        candidate.advance(1)
        _assert_storage_equal(reference, candidate)
        assert candidate.completed_steps == completed_steps
        assert state.progress.completed_steps == completed_steps
        assert state.representations.generation == completed_steps
        assert candidate.solver.integrator.step_count == (
            reference.solver.integrator.step_count
        )
        assert candidate.solver.integrator.refresh_count == (
            reference.solver.integrator.refresh_count
        )


def test_p35_keeps_public_facade_and_metadata_surface(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    adapter = _build_with_integrator(
        spec,
        monkeypatch,
        DealiasedSemiImplicitEulerIntegrator,
    )
    integrator = adapter.solver.integrator

    assert adapter.runtime_path.value == "legacy_production"
    assert adapter.to_metadata() == {
        "requested": "legacy_production",
        "effective": "legacy_production",
        "adapter": "LegacyPlaneRuntimeAdapter",
        "fallback_used": False,
        "separated_architecture": None,
    }
    assert integrator.phase3_execution_metadata()["connection_stage"] == (
        "P3.5_legacy_production_facade"
    )
    assert integrator.runtime_workspace.plan.required_bytes == 0
    assert "StateBackedProjectedIntegratorMixin" not in integrators.__all__
    assert not hasattr(pssolver, "StateBackedProjectedIntegratorMixin")


def test_p35_solver_reset_rebinds_reallocated_storage(tmp_path, monkeypatch):
    adapter = _build_with_integrator(
        _spec(tmp_path),
        monkeypatch,
        DealiasedSemiImplicitEulerIntegrator,
    )
    adapter.advance(2)
    integrator = adapter.solver.integrator
    old_state = integrator.runtime_state
    old_physical = old_state.physical
    old_spectral = old_state.spectral

    adapter.solver.reset(_initial_values())
    rebound = integrator.runtime_state

    assert rebound.physical is not old_physical
    assert rebound.spectral is not old_spectral
    assert rebound.physical.untyped_storage().data_ptr() == (
        adapter.solver.fields.spatial.untyped_storage().data_ptr()
    )
    assert rebound.spectral.untyped_storage().data_ptr() == (
        adapter.solver.fields.spectral.untyped_storage().data_ptr()
    )
    assert rebound.progress.completed_steps == 2


def test_p35_checkpoint_v1_restarts_in_both_version_directions(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path)

    old_source = _build_with_integrator(
        spec,
        monkeypatch,
        _PrePhase3DealiasedIntegrator,
    )
    old_source.advance(2)
    old_checkpoint = capture_plane_checkpoint(
        old_source,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    new_target = _build_with_integrator(
        spec,
        monkeypatch,
        DealiasedSemiImplicitEulerIntegrator,
    )
    assert restore_plane_checkpoint(
        new_target,
        old_checkpoint,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    ) == 2

    new_source = _build_with_integrator(
        spec,
        monkeypatch,
        DealiasedSemiImplicitEulerIntegrator,
    )
    new_source.advance(2)
    new_checkpoint = capture_plane_checkpoint(
        new_source,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    old_target = _build_with_integrator(
        spec,
        monkeypatch,
        _PrePhase3DealiasedIntegrator,
    )
    assert restore_plane_checkpoint(
        old_target,
        new_checkpoint,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    ) == 2

    old_source.advance(2)
    new_target.advance(2)
    new_source.advance(2)
    old_target.advance(2)
    _assert_storage_equal(old_source, new_target)
    _assert_storage_equal(new_source, old_target)
    assert old_checkpoint.format_version == new_checkpoint.format_version == 1
