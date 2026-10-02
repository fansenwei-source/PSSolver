"""P9.7.1 non-executable Channel functional declarations."""

from __future__ import annotations

from dataclasses import replace
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
)
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.simulation import ExecutionSpec
from pssolver.functional import (
    CHANNEL_ACTIVITY_FUNCTIONAL_KIND,
    CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT,
    CHANNEL_FUNCTIONAL_PRESSURE_WARM_START,
    FunctionalCapabilitySet,
    FunctionalRuntimeDeclaration,
    build_channel_activity_functional_runtime,
    build_functional_runtime,
    channel_activity_functional_declaration,
    channel_activity_functional_request,
)
from pssolver.geometries import RectangularChannel
from pssolver.models.active_nematics import CompleteStressBerisEdwards


def _simulation(tmp_path: Path, *, pointwise_execution: str = "eager") -> Simulation:
    model = CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
    )
    shape = (8, 6, 4)
    geometry = RectangularChannel(shape=shape, lengths=(8.0, 6.0, 4.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": no_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    values = np.zeros((*shape, 5), dtype=np.float64)
    values[..., 0] = 0.2
    values[..., 3] = -0.1
    np.save(snapshot / "Q_0.npy", values, allow_pickle=False)
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical",
                "pointwise_execution": pointwise_execution,
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / "output",
            steps=1,
            save_interval=1,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        },
    )


def test_p971_declares_exact_channel_state_control_and_observations(tmp_path):
    simulation = _simulation(tmp_path)
    declaration = channel_activity_functional_declaration(
        simulation.specification
    )

    assert isinstance(declaration, FunctionalRuntimeDeclaration)
    assert declaration.executable is True
    assert [item.name for item in declaration.state_spec.components] == [
        "q_physical",
        "q_spectral",
    ]
    physical, spectral = declaration.state_spec.components
    assert physical.shape == (5, 1, 8, 6, 4)
    assert physical.dtype == "float64"
    assert physical.layout == "component_batch_xyz"
    assert spectral.shape == (5, 1, 8, 6, 4)
    assert spectral.dtype == "complex128"
    assert spectral.layout == (
        "component_batch_periodic_x_neumann_y_neumann_z"
    )

    assert declaration.request == channel_activity_functional_request(
        simulation.specification
    )
    assert [item.name for item in declaration.control_specs] == ["activity"]
    control = declaration.control_specs[0]
    assert control.tensor.shape == (1, 8, 6, 4)
    assert control.equation_term == "div(beta * alpha * Q)"
    assert control.injection_order == (
        "form_beta_alpha_q_product_before_divergence"
    )
    assert control.admissible_min == 0.0
    assert [item.name for item in declaration.observation_specs] == [
        "Q",
        "velocity",
        "pressure",
    ]
    assert declaration.observation_specs[0].terminal_available_without_control
    assert all(
        value.control_dependent
        for value in declaration.observation_specs[1:]
    )


def test_p971_identity_exposes_pressure_and_boundary_policy_without_claims(
    tmp_path,
):
    declaration = channel_activity_functional_declaration(
        _simulation(tmp_path).specification
    )
    metadata = declaration.to_metadata()
    json.dumps(metadata, allow_nan=False, sort_keys=True)

    functional = metadata["identity"]["execution"][
        "functional_runtime_declaration"
    ]
    assert functional["kind"] == CHANNEL_ACTIVITY_FUNCTIONAL_KIND
    assert functional["stage"] == "P9.7.4"
    assert functional["executable"] is True
    assert functional["activity_product_before_divergence"] is True
    assert functional["q_gradient_cache"] == "derived_rebuilt_not_state"
    assert functional["boundary_spaces"] == {
        "Q": ["periodic", "neumann", "neumann"],
        "velocity": ["periodic", "dirichlet", "dirichlet"],
        "pressure": ["periodic", "neumann", "neumann"],
    }
    pressure = functional["pressure_solver"]
    assert pressure["kind"] == "channel_no_slip_modal_stokes_pcg"
    assert pressure["gauge"] == "zero_mean"
    assert pressure["production_warm_start"] is True
    assert pressure["functional_warm_start"] is False
    assert pressure["functional_warm_start_policy"] == (
        CHANNEL_FUNCTIONAL_PRESSURE_WARM_START
    )
    assert pressure["functional_initial_guess"] == "zero"
    assert pressure["stopping_precedence"] == (
        "fixed_iterations_overrides_max_iterations;"
        "relative_tolerance_is_early_exit_only"
    )
    assert pressure["transpose_action"] == (
        "explicit_reverse_dataflow_conjugate_transpose"
    )
    assert pressure["gradient"] == CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT
    assert len(metadata["identity_sha256"]) == 64

    capabilities = declaration.capabilities
    assert capabilities == FunctionalCapabilitySet(
        pure_step=True,
        combined_step_and_observe=True,
        deterministic_replay="bitwise",
        differentiability="validated_custom_adjoint",
        durable_checkpoint_bridge=True,
        inner_solve_gradient=CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT,
        differentiable_inputs=("state", "activity"),
    )
    assert capabilities.pure_step is True
    assert capabilities.combined_step_and_observe is True
    assert capabilities.deterministic_replay == "bitwise"
    assert capabilities.differentiability == "validated_custom_adjoint"
    assert capabilities.differentiable_inputs == ("state", "activity")
    assert capabilities.durable_checkpoint_bridge is True


def test_p971_declaration_does_not_allocate_execute_or_write_output(tmp_path):
    simulation = _simulation(tmp_path)
    output = Path(simulation.specification.workflow.options["output_dir"])
    (tmp_path / "snapshot" / "Q_0.npy").unlink()

    declaration = channel_activity_functional_declaration(
        simulation.specification
    )

    assert declaration.executable is True
    assert not output.exists()
    with pytest.raises(FileNotFoundError):
        build_channel_activity_functional_runtime(declaration.request)
    with pytest.raises(ValueError, match="periodic"):
        build_functional_runtime(declaration.request)
    assert not output.exists()


def test_p971_rejects_noncanonical_requests_before_execution(tmp_path):
    simulation = _simulation(tmp_path, pointwise_execution="compile")
    with pytest.raises(ValueError, match="eager pointwise"):
        channel_activity_functional_declaration(simulation.specification)
    with pytest.raises(TypeError, match="SimulationSpec"):
        channel_activity_functional_declaration(object())

    valid = _simulation(tmp_path / "valid")
    specification = valid.specification
    options = dict(specification.execution.options)
    options["device"] = "cuda"
    cuda_specification = replace(
        specification,
        execution=ExecutionSpec(
            backend=specification.execution.backend,
            runtime_path=specification.execution.runtime_path,
            options=options,
        ),
    )
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="requires CUDA"):
            channel_activity_functional_declaration(cuda_specification)


def test_executable_declaration_cannot_be_downgraded_with_capabilities(tmp_path):
    declaration = channel_activity_functional_declaration(
        _simulation(tmp_path).specification
    )
    with pytest.raises(ValueError, match="cannot claim runtime capabilities"):
        replace(
            declaration,
            executable=False,
        )


def test_p971_remains_provisional_and_has_no_control_project_dependency():
    assert not hasattr(pssolver, "channel_activity_functional_declaration")
    source = (
        Path(__file__).resolve().parents[1]
        / "pssolver"
        / "functional"
        / "channel_activity.py"
    ).read_text(encoding="utf-8")
    assert "pssolver.control" not in source
    assert "PSSolver-Control" not in source
