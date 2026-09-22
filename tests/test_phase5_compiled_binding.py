"""P5.2 construction-time Plane compiled-v2 binding and audit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import pssolver.runtime as runtime_api
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)
from pssolver.diagnostics import (
    build_tensor_inventory,
    compare_tensor_inventories,
)
from pssolver.execution.workspace import WorkspacePlan, WorkspaceSlotSpec
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.plane_compiled_v2_binding import (
    bind_plane_compiled_v2,
)
from pssolver.runtime.plane_legacy import build_legacy_plane_runtime


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / (
    "notes/architecture_v0_2/phase_5_p52_construction_binding.json"
)


def _spec(tmp_path: Path, **overrides):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / "unused",
        "device": "cpu",
        "dtype": "float64",
        "pointwise_execution": "eager",
        "nx": 6,
        "ny": 6,
        "nz": 6,
        "lx": 8.0,
        "ly": 9.0,
        "height": 20.0,
        "steps": 2,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "spectral_refresh_steps": 3,
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _initial_values():
    coordinate = torch.arange(6**3, dtype=torch.float64).reshape(6, 6, 6)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


def _build(tmp_path: Path, **overrides):
    spec = _spec(tmp_path, **overrides)
    solver, projector = build_legacy_plane_runtime(
        spec,
        device="cpu",
        initial_values=_initial_values(),
    )
    return spec, solver, projector


def test_p52_binds_exact_production_layout_operators_and_coefficients(tmp_path):
    spec, solver, projector = _build(tmp_path)
    plan = bind_plane_compiled_v2(
        spec,
        solver=solver,
        projector=projector,
    )

    assert plan.connected_runtime is None
    assert plan.fields is solver.fields
    assert plan.state is solver.integrator.runtime_state
    assert plan.operators.q_rhs_kernel is solver.model.nlmodel
    assert plan.operators.stokes_kernel is solver.model.static_model
    assert plan.operators.projector is projector
    assert plan.operators.prepare_algebraic.__self__ is solver.model
    assert plan.operators.explicit_rhs.__self__ is solver.model
    assert plan.operators.project_dynamic_spectra.__self__ is solver.integrator
    assert plan.operators.inverse_dynamic_spectra.__self__ is solver.integrator
    assert plan.operators.refresh_dynamic_spectra.__self__ is solver.integrator

    assert tuple(binding.name for binding in plan.scalar_bindings) == (
        "dt",
        "ldg_a",
        "ldg_b",
        "ldg_c",
        "ldg_l1",
        "rotational_viscosity",
        "flow_alignment",
        "viscosity",
        "friction",
        "beta",
        "zeta",
    )
    assert plan.legacy_workspace_bytes == 0
    assert plan.additional_workspace_bytes == 0


def test_p52_distinguishes_semantic_and_actual_execution_groups(tmp_path):
    spec, solver, projector = _build(tmp_path)
    plan = bind_plane_compiled_v2(
        spec,
        solver=solver,
        projector=projector,
    )

    assert [group.components for group in plan.semantic_transform_groups] == [
        ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz"),
        ("ux", "uy"),
        ("uz",),
        ("p",),
    ]
    assert [group.components for group in plan.execution_transform_groups] == [
        ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz"),
        ("ux", "uy", "p"),
        ("uz",),
    ]
    assert [
        group.indexing_mode for group in plan.execution_transform_groups
    ] == ["contiguous_slice", "advanced", "contiguous_slice"]
    assert [
        group.transform_kinds for group in plan.execution_transform_groups
    ] == [
        ("fft", "fft", "dct"),
        ("fft", "fft", "dct"),
        ("fft", "fft", "dst"),
    ]


def test_p52_binding_is_read_only_and_retains_no_new_tensor_storage(tmp_path):
    spec, solver, projector = _build(tmp_path)
    spatial_before = solver.fields.spatial.clone()
    spectral_before = solver.fields.spectral.clone()
    before = build_tensor_inventory(
        {"projector": projector, "solver": solver},
        device="cpu",
    )

    plan = bind_plane_compiled_v2(
        spec,
        solver=solver,
        projector=projector,
    )
    after = build_tensor_inventory(
        {"binding": plan, "projector": projector, "solver": solver},
        device="cpu",
    )
    comparison = compare_tensor_inventories(before, after)

    assert comparison["unique_storage_identity_equal"] is True
    assert comparison["added_storage_identity_count"] == 0
    assert comparison["removed_storage_identity_count"] == 0
    assert torch.equal(solver.fields.spatial, spatial_before)
    assert torch.equal(solver.fields.spectral, spectral_before)
    assert solver.integrator.runtime_state.progress.completed_steps == 0
    assert plan.tensors.evolved_physical.untyped_storage().data_ptr() == (
        solver.fields.spatial.untyped_storage().data_ptr()
    )
    assert plan.tensors.algebraic_spectral.untyped_storage().data_ptr() == (
        solver.fields.spectral.untyped_storage().data_ptr()
    )


def test_p52_metadata_records_dataflow_and_storage_contract(tmp_path):
    spec, solver, projector = _build(tmp_path)
    metadata = bind_plane_compiled_v2(
        spec,
        solver=solver,
        projector=projector,
    ).to_metadata()

    assert metadata["identity"] == "compiled_v2"
    assert metadata["connected_runtime"] is None
    assert metadata["declaration"]["operation_order"] == [
        "prepare_algebraic",
        "pre_update_callback",
        "explicit_rhs",
        "spectral_add_dt_rhs",
        "spectral_divide_by_denominator",
        "project_dynamic_spectra",
        "inverse_dynamic_spectra",
        "scheduled_spectral_refresh",
        "commit_progress",
    ]
    assert metadata["storage"]["owned_root_storage_count"] == 5
    assert metadata["storage"]["reuses_existing_solver_storage"] is True
    assert metadata["storage"]["duplicate_workspace_retained"] is False
    assert metadata["operators"]["all_operators_prebound"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda solver, projector: solver.fields.name_to_idx.pop("p"),
            "field names or indices",
        ),
        (
            lambda solver, projector: solver.fields.name_to_idx.update(
                {"p": 7}
            ),
            "field names or indices",
        ),
        (
            lambda solver, projector: solver.fields.boundary_conditions.__setitem__(
                8, ("periodic", "periodic", "dirichlet")
            ),
            "boundary mismatch",
        ),
        (
            lambda solver, projector: setattr(
                solver.integrator,
                "denom",
                solver.integrator.denom.to(torch.float32),
            ),
            "dtype",
        ),
        (
            lambda solver, projector: setattr(
                solver.integrator,
                "denom",
                solver.integrator.denom[:4],
            ),
            "shape",
        ),
        (
            lambda solver, projector: setattr(
                solver.integrator,
                "denom",
                torch.empty_like(solver.integrator.denom, device="meta"),
            ),
            "expected cpu",
        ),
        (
            lambda solver, projector: setattr(
                solver.integrator,
                "denom",
                solver.fields.L_hat,
            ),
            "unexpected tensor alias",
        ),
        (
            lambda solver, projector: setattr(
                solver.model.nlmodel,
                "spectral_projector",
                object(),
            ),
            "share one projector identity",
        ),
        (
            lambda solver, projector: setattr(
                solver.model,
                "static_transform_groups",
                [[5, 6], [7], [8]],
            ),
            "static transform grouping changed",
        ),
    ],
)
def test_p52_rejects_invalid_field_tensor_and_operator_bindings(
    tmp_path,
    mutation,
    message,
):
    spec, solver, projector = _build(tmp_path)
    mutation(solver, projector)
    with pytest.raises((TypeError, ValueError), match=message):
        bind_plane_compiled_v2(
            spec,
            solver=solver,
            projector=projector,
        )


def test_p52_stops_if_legacy_assembly_retains_workspace_storage(tmp_path):
    spec, solver, projector = _build(tmp_path)
    solver.integrator._runtime_workspace = WorkspacePlan(
        slots=(
            WorkspaceSlotSpec(
                name="unexpected",
                shape=(1,),
                dtype=torch.float64,
                purpose="negative gate",
            ),
        ),
        device="cpu",
        maximum_bytes=8,
    ).allocate()

    with pytest.raises(RuntimeError, match="shared assembly factory"):
        bind_plane_compiled_v2(
            spec,
            solver=solver,
            projector=projector,
        )


def test_p52_rejects_binding_after_the_runtime_has_evolved(tmp_path):
    spec, solver, projector = _build(tmp_path)
    solver.run(1)
    with pytest.raises(ValueError, match="untouched construction-time state"):
        bind_plane_compiled_v2(
            spec,
            solver=solver,
            projector=projector,
        )


def test_p52_remains_private_and_does_not_add_a_runtime_selector():
    assert not hasattr(runtime_api, "bind_plane_compiled_v2")
    assert {member.value for member in PlaneRuntimePath} == {
        "legacy_production",
        "separated_canary",
    }


def test_p52_qualification_record_freezes_local_scope():
    record = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_P5_2_CONSTRUCTION_BINDING_AND_DATAFLOW_AUDIT"
    )
    assert record["authorization"] == {
        "p5_2_implementation_authorized": True,
        "p5_3_implementation_authorized": False,
        "phase_6_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
    assert record["connection"] == {
        "runtime_selector_added": False,
        "production_import_added": False,
        "timestep_execution_added": False,
        "fallback_allowed": False,
    }
