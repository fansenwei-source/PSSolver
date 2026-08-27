import argparse
import json

import numpy as np
import pytest
import torch

from Channel_dal import q_input_summary
from pssolver import write_run_metadata as atomic_write_run_metadata
from pssolver.models.active_nematics import (
    Q_convention_metadata,
    positive_equilibrium_S,
)
from scripts_channel import calibrate_pure_splay_forward as calibration
from scripts_channel import validate_dal_forward as forward_validation


S_BULK = positive_equilibrium_S(-1.0, -6.0, 6.0)
S_INITIAL = 2.0 / 3.0
SHAPE = (8, 6, 6)


def write_canonical_q(
    directory,
    *,
    actual_S=S_INITIAL,
    S_initial=None,
    S_bulk=S_BULK,
    convention=None,
):
    directory.mkdir()
    q = np.zeros((*SHAPE, 5), dtype=np.float32)
    q[..., 0] = actual_S
    q[..., 3] = -0.5 * actual_S
    q_path = directory / "Q.npy"
    np.save(q_path, q)
    metadata = {
        "schema_version": 1,
        "model": {
            "name": "active_nematics",
            "Q_convention": (
                Q_convention_metadata() if convention is None else convention
            ),
            "parameters": {
                "S_initial": actual_S if S_initial is None else S_initial,
                "S_bulk": S_bulk,
            },
        }
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return q_path, q


@pytest.mark.parametrize(
    "validate_inputs",
    [
        calibration.validate_canonical_q_inputs,
        forward_validation.validate_canonical_q_inputs,
    ],
)
def test_auxiliary_entries_strictly_validate_target_convention(
    tmp_path,
    validate_inputs,
):
    initial_path, _ = write_canonical_q(tmp_path / "initial")
    incomplete = Q_convention_metadata()
    incomplete.pop("version")
    target_path, _ = write_canonical_q(
        tmp_path / "target",
        convention=incomplete,
    )

    with pytest.raises(ValueError, match="complete canonical Q convention"):
        validate_inputs(initial_path, target_path)


@pytest.mark.parametrize(
    "validate_inputs",
    [
        calibration.validate_canonical_q_inputs,
        forward_validation.validate_canonical_q_inputs,
    ],
)
def test_auxiliary_entries_require_one_consistent_positive_S_bulk(
    tmp_path,
    validate_inputs,
):
    initial_path, _ = write_canonical_q(tmp_path / "initial")
    target_path, _ = write_canonical_q(tmp_path / "target", S_bulk=0.4)

    with pytest.raises(ValueError, match="this DAL run uses S_bulk"):
        validate_inputs(initial_path, target_path)


@pytest.mark.parametrize(
    "validate_inputs",
    [
        calibration.validate_canonical_q_inputs,
        forward_validation.validate_canonical_q_inputs,
    ],
)
def test_auxiliary_entries_require_declared_S_initial(
    tmp_path,
    validate_inputs,
):
    initial_path, _ = write_canonical_q(tmp_path / "initial")
    target_path, _ = write_canonical_q(tmp_path / "target")
    metadata_path = initial_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model"]["parameters"].pop("S_initial")
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="S_initial"):
        validate_inputs(initial_path, target_path)


@pytest.mark.parametrize(
    "validate_inputs",
    [
        calibration.validate_canonical_q_inputs,
        forward_validation.validate_canonical_q_inputs,
    ],
)
def test_auxiliary_entries_compare_observed_order_only_with_source_S_initial(
    tmp_path,
    validate_inputs,
):
    initial_path, _ = write_canonical_q(tmp_path / "initial")
    target_path, _ = write_canonical_q(
        tmp_path / "target",
        actual_S=0.4,
        S_initial=0.5,
        S_bulk=S_BULK,
    )

    with pytest.raises(ValueError, match="metadata S_initial"):
        validate_inputs(initial_path, target_path)


@pytest.mark.parametrize(
    "output_summary",
    [
        calibration.dynamic_q_output_summary,
        forward_validation.dynamic_q_output_summary,
    ],
)
def test_dynamic_output_summary_does_not_claim_final_order_equals_S_initial(
    tmp_path,
    output_summary,
):
    q_path, _ = write_canonical_q(
        tmp_path / "dynamic_output",
        actual_S=0.2,
        S_initial=S_INITIAL,
        S_bulk=S_BULK,
    )

    summary = output_summary(
        q_path,
        expected_S_initial=S_INITIAL,
        expected_S_bulk=S_BULK,
    )

    assert summary["artifact_role"] == "dynamic_Q_output"
    assert summary["S_initial"] == pytest.approx(S_INITIAL)
    assert summary["S_bulk"] == pytest.approx(S_BULK)
    assert summary["observed_final_ordered_S"] == pytest.approx(0.2)

def validation_args(tmp_path):
    initial_path, _ = write_canonical_q(tmp_path / "initial")
    target_path, _ = write_canonical_q(tmp_path / "target")
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.ones(SHAPE, dtype=np.float32))
    return argparse.Namespace(
        initial=initial_path,
        target=target_path,
        mask=mask_path,
        stride=(1, 1, 1),
        steps=1,
        dt=1.0e-4,
        alpha_mean=0.0,
        alpha_amplitude=0.0,
        auto_mask_margin=(0, 0, 0),
        auto_mask_relative_threshold=0.2,
        mask_transition_width=1.5,
        pressure_rel_tol=1.0e-7,
        pressure_max_iter=30,
        rtol=1.0e-4,
        atol=1.0e-5,
        report_steps=(),
        device="cpu",
        output=tmp_path / "output",
    )


def test_forward_validation_writes_atomic_canonical_output_metadata(
    tmp_path,
    monkeypatch,
):
    args = validation_args(tmp_path)
    statuses = []

    def recording_writer(output_directory, metadata, *, status=None):
        statuses.append(status)
        return atomic_write_run_metadata(
            output_directory,
            metadata,
            status=status,
        )

    monkeypatch.setattr(forward_validation, "write_run_metadata", recording_writer)
    summary = forward_validation.run_validation(args)

    metadata_path = args.output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert summary["passed"]
    assert statuses == ["running", "complete"]
    assert metadata["status"] == "complete"
    assert metadata["model"]["name"] == "active_nematics"
    assert metadata["model"]["Q_convention"] == Q_convention_metadata()
    parameters = metadata["model"]["parameters"]
    assert parameters["S_initial"] == pytest.approx(S_INITIAL)
    assert parameters["S_bulk"] == pytest.approx(S_BULK)
    assert metadata["numerics"]["grid_transfer"]["method"] == (
        "conservative_block_average"
    )
    assert not (args.output / "metadata.json.tmp").exists()
    for name in ("Q_functional_final", "Q_production_final"):
        output = metadata["results"][name]
        assert output["artifact_role"] == "dynamic_Q_output"
        assert output["S_initial"] == pytest.approx(S_INITIAL)
        assert output["S_bulk"] == pytest.approx(S_BULK)
        assert np.isfinite(output["observed_final_ordered_S"])


def test_calibration_branch_writes_atomic_canonical_output_metadata(
    tmp_path,
    monkeypatch,
):
    initial_path, initial = write_canonical_q(tmp_path / "initial")
    target_path, _ = write_canonical_q(tmp_path / "target")
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.ones(SHAPE, dtype=np.float32))
    initial_input = q_input_summary(initial_path, expected_S_bulk=S_BULK)
    target_input = q_input_summary(target_path, expected_S_bulk=S_BULK)
    initial_q = torch.from_numpy(initial).movedim(-1, 0).unsqueeze(1)
    mask = torch.ones(SHAPE, dtype=torch.float32)
    args = argparse.Namespace(
        initial=initial_path,
        target=target_path,
        mask=mask_path,
        stride=(1, 1, 1),
        steps=1,
        sample_every=1,
        dt=1.0e-4,
        pressure_rel_tol=1.0e-7,
        pressure_max_iter=30,
        core_deficit_threshold=0.2,
        motion_speed_threshold=1.0e-3,
        save_snapshots=False,
        output=tmp_path / "output",
    )
    statuses = []

    def recording_writer(output_directory, metadata, *, status=None):
        statuses.append(status)
        return atomic_write_run_metadata(
            output_directory,
            metadata,
            status=status,
        )

    def diagnostic_stub(*, step, solver, **_kwargs):
        return {
            "step": step,
            "time": step * solver.dt,
            "finite": True,
            "loop_detected": True,
            "yz_plane_topology": True,
            "radius_rms": 1.0,
            "template_shift_x": 0.0,
            "template_correlation": 1.0,
            "normalized_target_mismatch": 0.0,
        }

    monkeypatch.setattr(calibration, "write_run_metadata", recording_writer)
    monkeypatch.setattr(calibration, "diagnostic_row", diagnostic_stub)
    calibration.run_branch(
        alpha=0.0,
        initial_q=initial_q,
        target_q=initial_q.clone(),
        mask=mask,
        shape=SHAPE,
        args=args,
        device=torch.device("cpu"),
        reference_profile=np.zeros(SHAPE[0]),
        S_bulk=S_BULK,
        mismatch_normalization=1.0,
        target_displacement=0.0,
        initial_input=initial_input,
        target_input=target_input,
        mask_metadata={"mode": "file", "path": str(mask_path)},
    )

    branch = args.output / "alpha_0"
    metadata = json.loads((branch / "metadata.json").read_text())
    assert statuses == ["running", "complete"]
    assert metadata["status"] == "complete"
    assert metadata["model"]["name"] == "active_nematics"
    assert metadata["model"]["Q_convention"] == Q_convention_metadata()
    parameters = metadata["model"]["parameters"]
    assert parameters["S_initial"] == pytest.approx(S_INITIAL)
    assert parameters["S_bulk"] == pytest.approx(S_BULK)
    assert metadata["initial_condition"]["name"] == "external_Q"
    assert metadata["target"]["name"] == "external_Q"
    assert not (branch / "metadata.json.tmp").exists()
    output = metadata["results"]["Q_final"]
    assert output["artifact_role"] == "dynamic_Q_output"
    assert output["S_initial"] == pytest.approx(S_INITIAL)
    assert output["S_bulk"] == pytest.approx(S_BULK)
    assert np.isfinite(output["observed_final_ordered_S"])
