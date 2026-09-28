"""P9.1 periodic-friction, planar-embedding, and functional contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import pssolver
from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
    run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.functional import (
    FUNCTIONAL_API_VERSION,
    FUNCTIONAL_OBSERVATION_TIME,
    FunctionalCapabilitySet,
    FunctionalControlFieldSpec,
    FunctionalObservationSpec,
    FunctionalRuntimeConstructionRequest,
    FunctionalRuntimeIdentity,
    FunctionalRuntimeProtocol,
    FunctionalStateSpec,
    FunctionalTensorSpec,
)
from pssolver.geometries import PeriodicBox
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    embed_z_invariant_planar_q,
    project_z_invariant_planar_q,
)


def _model(*, friction=0.0, policy="zero_mean"):
    return CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
        friction=friction,
        tangential_zero_mode_policy=policy,
    )


def _simulation(
    tmp_path: Path,
    *,
    name: str,
    z_points: int,
    friction: float,
    policy: str,
) -> Simulation:
    model = _model(friction=friction, policy=policy)
    geometry = PeriodicBox(
        shape=(8, 8, z_points),
        lengths=(8.0, 8.0, float(z_points)),
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    generator = np.random.default_rng(20260927)
    planar = np.empty((3, 1, 8, 8), dtype=np.float64)
    planar[0] = 0.2 + generator.normal(scale=1.0e-3, size=(1, 8, 8))
    planar[1] = generator.normal(scale=1.0e-3, size=(1, 8, 8))
    planar[2] = -0.1 + generator.normal(scale=1.0e-3, size=(1, 8, 8))
    embedded = embed_z_invariant_planar_q(planar, z_points=z_points)
    snapshot = tmp_path / f"{name}_snapshot"
    snapshot.mkdir()
    np.save(
        snapshot / "Q_0.npy",
        np.moveaxis(embedded[:, 0], 0, -1),
        allow_pickle=False,
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / name,
            steps=1,
            save_interval=10,
            diagnostic_interval=1,
        ),
    )


def test_public_complete_stress_declares_both_periodic_zero_mode_policies():
    default = _model()
    friction = _model(friction=0.2, policy="friction")

    default_stokes = default.parameters["stokes"]["parameters"]
    friction_stokes = friction.parameters["stokes"]["parameters"]
    assert default_stokes["friction"] == 0.0
    assert default_stokes["tangential_zero_mode_policy"] == "zero_mean"
    assert friction_stokes["friction"] == 0.2
    assert friction_stokes["tangential_zero_mode_policy"] == "friction"
    with pytest.raises(ValueError, match="requires friction == 0"):
        _model(friction=0.2, policy="zero_mean")
    with pytest.raises(ValueError, match="requires friction > 0"):
        _model(friction=0.0, policy="friction")
    with pytest.raises(ValueError, match="zero_mean or friction"):
        _model(policy="not_applicable")


def test_public_periodic_friction_compiles_and_runs_at_nz_one(tmp_path):
    simulation = _simulation(
        tmp_path,
        name="friction_nz1",
        z_points=1,
        friction=0.2,
        policy="friction",
    )
    compiled = compile_simulation(simulation)

    assert compiled.lowering_plan.nullspace.uniform_mode_action == (
        "retain_all_uniform_velocity_resolved_by_friction"
    )
    assert compiled.normalization["uniform_velocity_mode_action"] == (
        "retain_all_uniform_velocity_resolved_by_friction"
    )
    result = run_simulation(compiled)
    assert result.final_step == 1
    assert result.final_observation.q.shape == (8, 8, 1, 5)
    assert np.isfinite(result.final_observation.q).all()
    metadata = json.loads(
        (result.output_directory / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["friction"] == 0.2
    assert metadata["tangential_zero_mode_policy"] == "friction"
    assert metadata["runtime"]["friction"] == 0.2
    assert metadata["runtime"]["tangential_zero_mode_policy"] == "friction"
    assert metadata["runtime"]["uniform_velocity_mode_action"] == (
        "retain_all_uniform_velocity_resolved_by_friction"
    )


def test_planar_q_embedding_is_explicit_exact_and_differentiable():
    planar = torch.linspace(
        -0.2,
        0.3,
        3 * 2 * 4 * 5,
        dtype=torch.float64,
    ).reshape(3, 2, 4, 5)
    planar.requires_grad_(True)
    embedded = embed_z_invariant_planar_q(planar, z_points=3)

    assert embedded.shape == (5, 2, 4, 5, 3)
    assert embedded.dtype is planar.dtype
    assert embedded.device == planar.device
    assert torch.count_nonzero(embedded[2]) == 0
    assert torch.count_nonzero(embedded[4]) == 0
    projected = project_z_invariant_planar_q(embedded)
    assert torch.equal(projected, planar)
    projected.square().sum().backward()
    assert planar.grad is not None
    assert torch.isfinite(planar.grad).all()

    broken = embedded.detach().clone()
    broken[0, 0, 0, 0, 1] += 1.0
    with pytest.raises(ValueError, match="invariant along z"):
        project_z_invariant_planar_q(broken)
    broken = embedded.detach().clone()
    broken[2, 0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="Qxz == Qyz"):
        project_z_invariant_planar_q(broken)


def test_periodic_step_preserves_the_planar_z_invariant_subspace(tmp_path):
    result = run_simulation(
        _simulation(
            tmp_path,
            name="z_invariant",
            z_points=3,
            friction=0.2,
            policy="friction",
        )
    )
    q = result.final_observation.q

    np.testing.assert_allclose(
        q,
        np.repeat(q[:, :, :1, :], q.shape[2], axis=2),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(q[..., 2], 0.0, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(q[..., 4], 0.0, rtol=0.0, atol=2.0e-15)


def test_provisional_functional_declarations_are_fail_closed_and_json_safe(
    tmp_path,
):
    simulation = _simulation(
        tmp_path,
        name="declaration_only",
        z_points=1,
        friction=0.2,
        policy="friction",
    )
    state_tensor = FunctionalTensorSpec(
        name="q",
        shape=(5, 1, 8, 8, 1),
        dtype="float64",
        device="cpu",
        batch_axis=1,
        layout="component_batch_xyz",
        meaning="five independent symmetric-traceless Q components",
        component_names=("Qxx", "Qxy", "Qxz", "Qyy", "Qyz"),
    )
    control_tensor = FunctionalTensorSpec(
        name="activity",
        shape=(1, 8, 8, 1),
        dtype="float64",
        device="cpu",
        batch_axis=0,
        layout="batch_xyz",
        meaning="cell-centered activity coefficient alpha",
    )
    observation_tensor = FunctionalTensorSpec(
        name="velocity",
        shape=(3, 1, 8, 8, 1),
        dtype="float64",
        device="cpu",
        batch_axis=1,
        layout="component_batch_xyz",
        meaning="physical incompressible velocity",
        component_names=("ux", "uy", "uz"),
    )
    state = FunctionalStateSpec((state_tensor,))
    control = FunctionalControlFieldSpec(
        name="activity",
        tensor=control_tensor,
        equation_term="div(beta * alpha * Q)",
        injection_order="form_alpha_times_q_before_divergence",
        dealiasing_identity="project_product_then_spectral_divergence",
        grid_location="cell_centered",
        broadcast_rules=("exact_batch_and_spatial_shape",),
        admissible_min=0.0,
    )
    observation = FunctionalObservationSpec(
        name="velocity",
        tensor=observation_tensor,
        convention="physical_incompressible_velocity",
        control_dependent=True,
        terminal_available_without_control=False,
    )
    capabilities = FunctionalCapabilitySet()
    request = FunctionalRuntimeConstructionRequest(
        simulation=simulation.specification,
        control_fields=(control,),
        observations=(observation,),
    )
    identity = FunctionalRuntimeIdentity(
        scientific={"equations": "complete_stress_beris_edwards"},
        discretization={"shape": [8, 8, 1], "dealias": "cubic_half"},
        execution={"device": "cpu", "dtype": "float64"},
        state_layout=state.to_metadata(),
    )

    q = torch.zeros(state_tensor.shape, dtype=torch.float64)
    state.validate((q,))
    control.validate(torch.zeros(control_tensor.shape, dtype=torch.float64))
    assert state.batch_size == request.batch_size == 1
    assert request.api_version == FUNCTIONAL_API_VERSION
    assert observation.time_alignment == FUNCTIONAL_OBSERVATION_TIME
    assert len(identity.canonical_sha256()) == 64
    assert identity.to_metadata()["state_layout"]["container"] == "flat_tuple"
    assert capabilities.to_metadata() == {
        "supported_batch_sizes": [1],
        "pure_step": False,
        "combined_step_and_observe": False,
        "deterministic_replay": "not_qualified",
        "differentiability": "not_qualified",
        "durable_checkpoint_bridge": False,
        "explicit_jvp": False,
        "explicit_vjp": False,
        "inner_solve_gradient": "not_applicable",
        "differentiable_inputs": [],
    }
    json.dumps(
        {
            "state": state.to_metadata(),
            "control": control.to_metadata(),
            "observation": observation.to_metadata(),
            "capabilities": capabilities.to_metadata(),
            "identity": identity.to_metadata(),
            "request": request.to_metadata(),
        },
        allow_nan=False,
        sort_keys=True,
    )
    assert not hasattr(pssolver, "FunctionalRuntimeProtocol")
    assert isinstance(FunctionalRuntimeProtocol, type)

    with pytest.raises(TypeError, match="dtype"):
        state_tensor.validate(torch.zeros(state_tensor.shape, dtype=torch.float32))
    with pytest.raises(ValueError, match="batch_size=1"):
        FunctionalRuntimeConstructionRequest(
            simulation=simulation.specification,
            control_fields=(control,),
            observations=(observation,),
            batch_size=2,
        )
    with pytest.raises(ValueError, match="requires an index"):
        FunctionalTensorSpec(
            name="q",
            shape=(5, 1, 8, 8, 1),
            dtype="float64",
            device="cuda",
            batch_axis=1,
            layout="component_batch_xyz",
            meaning="five independent symmetric-traceless Q components",
        )
