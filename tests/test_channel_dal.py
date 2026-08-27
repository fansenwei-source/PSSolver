import json

import numpy as np
import pytest
import torch

from Channel_dal import localized_transverse_path_envelope, q_input_summary

from pssolver import write_run_metadata
from pssolver.models.active_nematics import (
    Q_convention_metadata,
    create_initial_condition,
)
from pssolver.channel import Q_BC, Q_COMPONENTS, build_active_nematic_channel
from pssolver.control import (
    DiscreteAdjointLoop,
    FunctionalSemiImplicitStep,
    QTrackingObjective,
    TemporalMaskControl,
    partition_mask_along_axis,
    smooth_box_mask,
)


def build_channel_problem(num_steps=3, num_masks=1):
    shape = (12, 8, 8)
    initial_fields = create_initial_condition(
        "aligned_x_smooth_noise",
        shape,
        boundary_conditions=Q_BC,
        S_initial=2.0 / 3.0,
        seed=12,
        noise_theta=0.02,
        noise_phi=0.02,
    )
    solver = build_active_nematic_channel(
        shape,
        lengths=(12.0, 8.0, 8.0),
        dt=1e-3,
        initial_q=initial_fields,
        device="cpu",
        pressure_rel_tol=1e-7,
    )
    initial_q = torch.stack([solver.fields[name] for name in Q_COMPONENTS]).detach()
    mask = smooth_box_mask(shape, ((3, 9), (1, 7), (1, 7)), transition_width=0.8)
    masks = partition_mask_along_axis(
        mask,
        num_masks,
        axis=0,
        bounds=(3, 9),
        overlap=0.75,
    )
    target_q = initial_q.clone()
    target_q[0] = target_q[0] - 0.03 * mask
    target_q[3] = target_q[3] + 0.03 * mask
    control = TemporalMaskControl(
        masks,
        num_steps=num_steps,
        block_size=1,
        alpha_min=0.0,
        alpha_max=5.0,
        initial_alpha=2.5,
    )
    objective = QTrackingObjective(
        target_q,
        dt=solver.dt,
        spatial_mask=mask,
        running_weight=0.1,
        terminal_weight=1.0,
        control_weight=1e-5,
    )
    stepper = FunctionalSemiImplicitStep(solver)
    dal = DiscreteAdjointLoop(
        stepper,
        objective,
        control,
        num_steps=num_steps,
        checkpoint_stride=1,
        temporal_control_weight=1e-4,
        spatial_control_weight=1e-4,
    )
    return solver, stepper, control, dal, initial_q


def test_channel_functional_step_matches_integrator():
    solver, stepper, control, _, initial_q = build_channel_problem()
    alpha = control.field_for_step(0).detach()
    expected = stepper(initial_q, alpha).detach()

    solver.fields.spatial[:5] = initial_q
    solver.fields.spectral[:5] = solver.fields.fftn()[:5]
    solver.parameters["alpha"] = alpha
    solver.model.static_model.pressure_guess = None
    solver.run(1)
    torch.testing.assert_close(solver.fields.spatial[:5], expected, rtol=3e-5, atol=3e-6)


def test_channel_multistep_functional_matches_integrator_with_time_varying_control():
    production, _, _, _, _ = build_channel_problem(num_steps=5)
    functional, stepper, control, _, functional_q = build_channel_problem(num_steps=5)

    for step in range(5):
        amplitude = 1.0 + 0.5 * step
        alpha = (amplitude * control.masks[0]).unsqueeze(0)

        functional_q = stepper(functional_q, alpha).detach()
        functional_static = functional.fields.spatial[5:].detach().clone()

        production.parameters["alpha"] = alpha
        production.model.static_model.pressure_guess = None
        production.integrator.step()

        torch.testing.assert_close(
            production.fields.spatial[:5],
            functional_q,
            rtol=1e-4,
            atol=1e-5,
        )
        torch.testing.assert_close(
            production.fields.spatial[5:],
            functional_static,
            rtol=1e-4,
            atol=1e-5,
        )


def test_channel_discrete_adjoint_matches_finite_difference():
    _, _, _, dal, initial_q = build_channel_problem(num_masks=3)
    direction = torch.linspace(-1.0, 1.0, dal.control.logits.numel()).reshape_as(
        dal.control.logits
    )
    check = dal.directional_derivative_check(
        initial_q,
        epsilon=5e-2,
        direction=direction,
    )
    absolute_error = abs(check["adjoint"] - check["finite_difference"])
    assert check["relative_error"] < 2e-2 or absolute_error < 5e-9, check


def test_localized_transverse_path_envelope_is_local_and_wall_tapered():
    envelope = localized_transverse_path_envelope(
        (64, 20, 20),
        (128.0, 10.0, 10.0),
        (64.0, 4.5, 5.0),
        (62.0, 5.5, 5.0),
        1.5,
        margin=1.5,
        transition_width=0.5,
        wall_buffer=0.5,
        device="cpu",
    )
    assert envelope.shape == (20, 20)
    assert torch.all((0.0 <= envelope) & (envelope <= 1.0))
    assert envelope[9, 9] > 0.95
    assert envelope[0].max() < 0.35
    assert envelope[:, 0].max() < 0.35


def _write_Q_input(
    directory,
    *,
    actual_S=0.4,
    declared_S=None,
    model_name="active_nematics",
    convention=None,
):
    directory.mkdir()
    Q = np.zeros((6, 4, 3, 5), dtype=np.float32)
    Q[..., 0] = actual_S
    Q[..., 3] = -actual_S / 2.0
    Q_path = directory / "Q.npy"
    np.save(Q_path, Q)
    metadata = {
        "schema_version": 1,
        "model": {
            "name": model_name,
            "Q_convention": (
                Q_convention_metadata() if convention is None else convention
            ),
            "parameters": {
                "S_initial": actual_S if declared_S is None else declared_S,
                "S_bulk": actual_S if declared_S is None else declared_S,
            },
        },
        "classification": {
            "type": "pure_splay_loop",
            "classification_regime": "ordered",
        },
        "loop": {"center": [3.0, 2.0, 1.5], "radius": 1.0},
        "validation": {
            "defect_plaquette_counts": {"xy": 0, "xz": 8, "yz": 8},
            "yz_plane_topology": True,
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return Q_path


def test_q_input_summary_accepts_canonical_metadata_and_observed_S(tmp_path):
    Q_path = _write_Q_input(tmp_path / "canonical")

    summary = q_input_summary(Q_path, expected_S_bulk=0.4)

    assert summary["Q_convention"] == Q_convention_metadata()
    assert summary["S_initial"] == pytest.approx(0.4)
    assert summary["S_bulk"] == pytest.approx(0.4)
    assert summary["observed_ordered_S"] == pytest.approx(0.4)
    assert summary["center"] == [3.0, 2.0, 1.5]
    json.dumps(summary)


def test_q_input_summary_rejects_wrong_model_name(tmp_path):
    Q_path = _write_Q_input(
        tmp_path / "wrong_model",
        model_name="another_model",
    )

    with pytest.raises(ValueError, match="model.name='active_nematics'"):
        q_input_summary(Q_path, expected_S_bulk=0.4)


def test_q_input_summary_rejects_incomplete_canonical_convention(tmp_path):
    convention = Q_convention_metadata()
    convention["definition"] = "Q=S(nn-I/3)"
    Q_path = _write_Q_input(
        tmp_path / "wrong_convention",
        convention=convention,
    )

    with pytest.raises(ValueError, match="complete canonical Q convention"):
        q_input_summary(Q_path, expected_S_bulk=0.4)

def test_q_input_summary_requires_S_initial(tmp_path):
    Q_path = _write_Q_input(tmp_path / "missing_S_initial")
    metadata_path = Q_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["model"]["parameters"]["S_initial"]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="S_initial"):
        q_input_summary(Q_path, expected_S_bulk=0.4)



def test_q_input_summary_rejects_observed_S_mismatch(tmp_path):
    Q_path = _write_Q_input(
        tmp_path / "wrong_observed_S",
        actual_S=0.3,
        declared_S=0.5,
    )

    with pytest.raises(ValueError, match="ordered-region median S"):
        q_input_summary(Q_path, expected_S_bulk=0.5)


def test_q_input_summary_rejects_run_S_mismatch(tmp_path):
    Q_path = _write_Q_input(tmp_path / "wrong_run_S", actual_S=0.5)

    with pytest.raises(ValueError, match="this DAL run uses S_bulk"):
        q_input_summary(Q_path, expected_S_bulk=0.4)


def test_q_input_summary_has_no_missing_metadata_fallback(tmp_path):
    directory = tmp_path / "missing_metadata"
    directory.mkdir()
    Q_path = directory / "Q.npy"
    np.save(Q_path, np.zeros((2, 2, 2, 5), dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="Canonical Q input metadata"):
        q_input_summary(Q_path, expected_S_bulk=0.4)


def test_run_metadata_status_updates_are_atomic_and_json_serializable(tmp_path):
    payload = {
        "schema_version": 1,
        "model": {
            "name": "active_nematics",
            "Q_convention": Q_convention_metadata(),
            "parameters": {"S_initial": 0.4, "S_bulk": 0.4},
        },
    }

    metadata_path = write_run_metadata(tmp_path, payload, status="running")
    assert json.loads(metadata_path.read_text())["status"] == "running"
    assert not (tmp_path / "metadata.json.tmp").exists()

    write_run_metadata(tmp_path, payload, status="complete")
    complete = json.loads(metadata_path.read_text())
    assert complete["status"] == "complete"
    assert complete["model"] == payload["model"]
    assert not (tmp_path / "metadata.json.tmp").exists()
