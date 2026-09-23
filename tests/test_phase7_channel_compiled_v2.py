"""P7.3 contracts for direct-import compiled Channel execution."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

import pssolver
import pssolver.configuration as configuration
import pssolver.experimental as experimental
import pssolver.planning as planning
from pssolver.channel import Q_COMPONENTS, build_active_nematic_channel
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    ChannelRuntimePath,
)
from pssolver.experimental.channel_compiled_v2 import (
    ChannelCompiledV2Runtime,
    build_channel_compiled_v2_runtime,
)
from pssolver.linear_solvers.stokes.channel_no_slip import (
    ChannelNoSlipModalStokesSolver,
)
from pssolver.planning.channel_compiled_v2 import (
    CHANNEL_COMPILED_V2_IDENTITY,
    ChannelCompiledStage,
    ChannelCompiledV2Declaration,
    ChannelFieldRole,
    ChannelTransformFamily,
    ChannelTransformGroupDeclaration,
    ChannelWorkspaceLifetime,
    ChannelWorkspaceRequirement,
    channel_compiled_v2_declaration,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
SHAPE = (8, 6, 5)
LENGTHS = (8.0, 6.0, 5.0)


def _spec(**overrides):
    values = {
        "shape": SHAPE,
        "lengths": LENGTHS,
        "dt": 1.0e-3,
        "steps": 5,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "activity": 0.2,
        "pressure_relative_tolerance": 1.0e-8,
        "pressure_max_iterations": 100,
    }
    values.update(overrides)
    return ChannelActiveNematicRunSpec(**values)


def _initial_q():
    coordinate = torch.arange(
        SHAPE[0] * SHAPE[1] * SHAPE[2],
        dtype=torch.float32,
    ).reshape(SHAPE)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


def _legacy(spec, initial_q):
    material = spec.components.material
    pressure = spec.components.pressure_solver
    solver = build_active_nematic_channel(
        spec.shape,
        spec.lengths,
        spec.dt,
        initial_q,
        device="cpu",
        batchsize=spec.batch_size,
        rho=material.rho,
        elastic_constant=material.elastic_constant,
        beta=material.beta,
        friction=material.friction,
        viscosity=material.viscosity,
        pressure_rel_tol=pressure.relative_tolerance,
        pressure_max_iter=pressure.max_iterations,
        pressure_fixed_iterations=pressure.fixed_iterations,
    )
    solver.parameters["alpha"] = torch.full(
        (spec.batch_size, *spec.shape),
        material.activity,
        dtype=solver.dtype,
        device=solver.device,
    )
    return solver


def test_channel_compiled_declaration_freezes_layout_and_modal_parity():
    declaration = channel_compiled_v2_declaration()

    assert declaration.identity == CHANNEL_COMPILED_V2_IDENTITY
    assert declaration.production_connection is False
    assert declaration.component_order == (
        "Qxx",
        "Qxy",
        "Qxz",
        "Qyy",
        "Qyz",
        "ux",
        "uy",
        "uz",
        "p",
    )
    assert declaration.to_metadata()["transform_groups"] == [
        {
            "name": "q_neumann_neumann",
            "components": ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"],
            "role": "evolved",
            "transform": "fft_dct_dct",
        },
        {
            "name": "velocity_dirichlet_dirichlet",
            "components": ["ux", "uy", "uz"],
            "role": "algebraic",
            "transform": "fft_dst_dst",
        },
        {
            "name": "pressure_neumann_neumann",
            "components": ["p"],
            "role": "algebraic",
            "transform": "fft_dct_dct",
        },
    ]
    assert "pressure_guess" in declaration.persistent_state.items
    assert tuple(stage.value for stage in declaration.operation_order) == (
        "prepare_algebraic",
        "pre_update_callback",
        "explicit_rhs",
        "spectral_add_dt_rhs",
        "spectral_divide_by_denominator",
        "project_dynamic_spectra",
        "inverse_dynamic_spectra",
        "scheduled_spectral_refresh",
        "commit_progress",
    )


def test_channel_workspace_declarations_are_bounded_and_immutable():
    declaration = channel_compiled_v2_declaration()

    assert {
        item.name for item in declaration.workspace_requirements
    } == {
        "explicit_q_rhs",
        "constitutive_scratch",
        "active_force",
        "bounded_axis_basis_change",
        "pressure_pcg_vectors",
        "inverse_transform_scratch",
    }
    assert all(
        item.bounded and item.construction_bound
        for item in declaration.workspace_requirements
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.identity = "legacy_channel"
    with pytest.raises(ValueError, match="bounded and construction-bound"):
        ChannelWorkspaceRequirement(
            name="bad_workspace",
            owner="stokes",
            lifetime=ChannelWorkspaceLifetime.RUNTIME,
            bounded=False,
        )


def test_channel_declaration_rejects_layout_and_stage_corruption():
    declaration = channel_compiled_v2_declaration()
    q_group = ChannelTransformGroupDeclaration(
        name="q_group",
        components=("Qxx",),
        role=ChannelFieldRole.EVOLVED,
        transform=ChannelTransformFamily.FFT_DCT_DCT,
    )
    with pytest.raises(ValueError, match="exactly cover"):
        dataclasses.replace(
            declaration,
            evolved_components=("Qxx",),
            algebraic_components=("p",),
            transform_groups=(q_group,),
        )
    reordered = declaration.operation_order[:-2] + (
        ChannelCompiledStage.COMMIT_PROGRESS,
        ChannelCompiledStage.SCHEDULED_SPECTRAL_REFRESH,
    )
    with pytest.raises(ValueError, match="operation_order"):
        dataclasses.replace(declaration, operation_order=reordered)


def test_channel_declaration_is_tensor_free_and_not_exported():
    path = ROOT / "pssolver/planning/channel_compiled_v2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )

    assert imported_roots == {"dataclasses", "enum"}
    assert not hasattr(planning, "channel_compiled_v2_declaration")
    assert not hasattr(pssolver, "channel_compiled_v2_declaration")


def test_direct_runtime_binds_state_workspace_program_and_pressure_state():
    runtime = build_channel_compiled_v2_runtime(
        _spec(),
        initial_q=_initial_q(),
        device="cpu",
    )
    state = runtime.state
    workspace = runtime.workspace
    pressure_solver = runtime.solver.model.static_model

    assert isinstance(runtime, ChannelCompiledV2Runtime)
    assert isinstance(runtime.declaration, ChannelCompiledV2Declaration)
    assert isinstance(pressure_solver, ChannelNoSlipModalStokesSolver)
    assert state.physical.untyped_storage().data_ptr() == (
        runtime.solver.fields.spatial.untyped_storage().data_ptr()
    )
    assert state.spectral.untyped_storage().data_ptr() == (
        runtime.solver.fields.spectral.untyped_storage().data_ptr()
    )
    assert state.persistent_algebraic["pressure_guess"] is (
        pressure_solver.pressure_guess
    )
    assert workspace.plan.required_bytes == 0
    assert workspace.plan.slots == ()
    assert not workspace.active
    assert runtime.step_program.to_metadata()["operation_order"] == [
        stage.value for stage in runtime.declaration.operation_order
    ]


def test_direct_runtime_is_byte_identical_to_legacy_channel_trajectory():
    spec = _spec()
    initial_q = _initial_q()
    legacy = _legacy(spec, initial_q)
    compiled = build_channel_compiled_v2_runtime(
        spec,
        initial_q=initial_q,
        device="cpu",
    )
    workspace = compiled.workspace

    assert torch.equal(legacy.fields.spatial, compiled.solver.fields.spatial)
    assert torch.equal(legacy.fields.spectral, compiled.solver.fields.spectral)
    for completed_steps in range(1, 6):
        legacy.run(1)
        compiled.advance(1)
        assert torch.equal(
            legacy.fields.spatial,
            compiled.solver.fields.spatial,
        )
        assert torch.equal(
            legacy.fields.spectral,
            compiled.solver.fields.spectral,
        )
        assert torch.equal(
            legacy.model.static_model.pressure_guess,
            compiled.capture_pressure_guess(),
        )
        assert compiled.completed_steps == completed_steps
        assert workspace is compiled.workspace
        assert workspace.generation == completed_steps
        assert not workspace.active
        assert compiled.state.persistent_algebraic["pressure_guess"] is (
            compiled.solver.model.static_model.pressure_guess
        )


def test_pressure_state_restore_is_identity_bound_and_fail_closed():
    runtime = build_channel_compiled_v2_runtime(
        _spec(),
        initial_q=_initial_q(),
    )
    runtime.advance(2)
    captured = runtime.capture_pressure_guess()
    runtime.advance(1)

    runtime.restore_pressure_guess(captured)

    assert torch.equal(runtime.capture_pressure_guess(), captured)
    assert runtime.state.persistent_algebraic["pressure_guess"] is (
        runtime.solver.model.static_model.pressure_guess
    )
    with pytest.raises(ValueError, match="does not match"):
        runtime.restore_pressure_guess(captured[..., :-1])
    invalid = captured.clone()
    invalid.reshape(-1)[0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        runtime.restore_pressure_guess(invalid)


def test_runtime_metadata_is_explicit_opt_in_and_json_serializable():
    runtime = build_channel_compiled_v2_runtime(
        _spec(),
        initial_q=_initial_q(),
    )
    metadata = runtime.to_metadata()

    assert metadata["runtime_selection"] == {
        "requested": "compiled_channel_v2",
        "effective": "compiled_channel_v2",
        "fallback_used": False,
        "production_connection": False,
    }
    assert metadata["construction"] == {
        "registry_lookup_in_step": False,
        "capability_lookup_in_step": False,
        "configuration_parsing_in_step": False,
        "runtime_fallback": False,
    }
    assert metadata["state"]["persistent_algebraic_names"] == [
        "pressure_guess"
    ]
    json.dumps(metadata, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("initial_q", "match"),
    (
        ({}, "exactly the five"),
        ({name: 1.0 for name in Q_COMPONENTS}, "must be a tensor"),
    ),
)
def test_runtime_rejects_invalid_initial_state(initial_q, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build_channel_compiled_v2_runtime(_spec(), initial_q=initial_q)


def test_runtime_rejects_unqualified_precision_and_invalid_step_count():
    with pytest.raises(ValueError, match="float32 legacy Channel"):
        build_channel_compiled_v2_runtime(
            _spec(dtype="float64"),
            initial_q=_initial_q(),
        )
    runtime = build_channel_compiled_v2_runtime(
        _spec(),
        initial_q=_initial_q(),
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        runtime.advance(-1)


def test_p73_remains_direct_import_only_and_does_not_change_a_selector():
    assert {item.value for item in ChannelRuntimePath} == {"legacy_channel"}
    assert not hasattr(configuration, "ChannelActiveNematicRunSpec")
    assert not hasattr(experimental, "build_channel_compiled_v2_runtime")
    assert "channel_active_nematics" not in (
        ROOT / "Channel.py"
    ).read_text(encoding="utf-8")


def test_p73_record_binds_sources_and_limits_authorization():
    record = json.loads(
        (NOTES / "phase_7_p73_channel_compiled_execution.json").read_text(
            encoding="utf-8"
        )
    )

    def sha256(relative):
        return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

    assert record["classification"] == (
        "PASS_P7_3_CHANNEL_COMPILED_EXECUTION"
    )
    assert sha256("pssolver/planning/channel_compiled_v2.py") == record[
        "implementation"
    ]["declaration_module_sha256"]
    assert sha256("pssolver/experimental/channel_compiled_v2.py") == record[
        "implementation"
    ]["execution_adapter_sha256"]
    assert sha256("pssolver/channel.py") == record["compatibility"][
        "legacy_channel_module_sha256"
    ]
    assert sha256("Channel.py") == record["compatibility"][
        "channel_entry_point_sha256"
    ]
    assert record["authorization"] == {
        "p7_3_complete": True,
        "p7_4_planning_eligible": True,
        "p7_4_implementation_authorized": False,
        "phase_7_h100_authorized": False,
        "production_connection_authorized": False,
        "production_default_changed": False,
        "new_boundary_law_authorized": False,
        "plane_evidence_modified": False,
    }
