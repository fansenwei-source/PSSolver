import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from pssolver.models.active_nematics import aligned_x_band_limited_noise_2d
from scripts_plane.qualify_v3_2d_mother import qualify


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "Plane_beris_edwards_stokes.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_command(output: Path) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--activity-number", "18",
        "--height", "20",
        "--output-dir", str(output),
        "--device", "cpu",
        "--nx", "8",
        "--ny", "8",
        "--nz", "4",
        "--dt", "0.01",
        "--steps", "1",
        "--save-start-step", "0",
        "--save-interval", "1",
        "--diagnostic-interval", "1",
        "--dtype", "float64",
        "--disable-spectral-refresh",
    ]


def test_v3_mother_runs_z_independent_and_saves_q2d(tmp_path):
    output = tmp_path / "mother"
    result = subprocess.run(
        [
            *_base_command(output),
            "--initialization-protocol", "v3-mother",
            "--v3-mother-noise-rms", "0.01",
            "--v3-mother-max-mode-x", "2",
            "--v3-mother-max-mode-y", "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["initialization_protocol"] == "v3-mother"
    assert metadata["solver"]["save_layout"] == "q2d"
    assert metadata["q2d_z_invariance"]["all_saved_frames_passed"] is True
    assert np.load(output / "Q2D_1.npy").shape == (8, 8, 5)
    assert not list(output.glob("Q_*.npy"))


def test_v3_mother_is_height_independent_for_the_same_k_and_zeta(tmp_path):
    outputs = []
    for height, nz in ((10.0, 4), (20.0, 8)):
        output = tmp_path / f"mother_H{height:g}"
        activity = height * np.sqrt(0.03 / 0.02)
        command = _base_command(output)
        activity_index = command.index("--activity-number") + 1
        height_index = command.index("--height") + 1
        nz_index = command.index("--nz") + 1
        command[activity_index] = f"{activity:.17g}"
        command[height_index] = f"{height:.17g}"
        command[nz_index] = str(nz)
        result = subprocess.run(
            [
                *command,
                "--parameterization", "fixed-k",
                "--frank-k", "0.02",
                "--initialization-protocol", "v3-mother",
                "--v3-mother-max-mode-x", "2",
                "--v3-mother-max-mode-y", "2",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(np.load(output / "Q2D_1.npy"))

    np.testing.assert_allclose(outputs[0], outputs[1], rtol=2e-12, atol=2e-14)


def test_v3_mother_output_is_consumable_by_conservative_qualifier(tmp_path):
    output = tmp_path / "mother_for_qualification"
    command = _base_command(output)
    steps_index = command.index("--steps") + 1
    command[steps_index] = "2"
    result = subprocess.run(
        [*command, "--initialization-protocol", "v3-mother"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    report, rows = qualify(
        run_dir=output,
        start_time=0.0,
        end_time=0.02,
        minimum_frames=3,
        minimum_ess=1.0,
        maximum_drift_over_std=10.0,
        maximum_half_difference_over_std=10.0,
        maximum_birth_death_relative_imbalance=1.0,
    )
    assert len(rows) == 3
    assert report["candidate_2d_statistical_steady_state"] is False
    assert report["gates"]["nonzero_dynamic_defect_population"] is False
    assert report["recommended_checkpoint"] is None


def test_v3_mother_continuation_preserves_absolute_steps_and_restart_input(tmp_path):
    first = tmp_path / "mother_first"
    first_result = subprocess.run(
        [*_base_command(first), "--initialization-protocol", "v3-mother"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first_result.returncode == 0, first_result.stderr
    restart = first / "Q2D_1.npy"
    restart_identity = _sha256(restart)

    continuation = tmp_path / "mother_continuation"
    command = _base_command(continuation)
    save_start_index = command.index("--save-start-step") + 1
    command[save_start_index] = "1"
    continuation_result = subprocess.run(
        [
            *command,
            "--initialization-protocol", "v3-mother",
            "--v3-mother-restart-q2d", str(restart),
            "--v3-mother-start-step", "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert continuation_result.returncode == 0, continuation_result.stderr
    metadata = json.loads((continuation / "metadata.json").read_text())
    assert metadata["solver"]["start_step"] == 1
    assert metadata["completed_steps"] == 2
    assert metadata["initial_condition"]["continuation"] is True
    assert metadata["initial_condition"]["restart_input_unchanged_at_completion"] is True
    assert _sha256(restart) == restart_identity
    np.testing.assert_allclose(
        np.load(continuation / "Q2D_1.npy"),
        np.load(restart),
        rtol=2e-12,
        atol=2e-14,
    )
    joint_report, joint_rows = qualify(
        run_dir=(first, continuation),
        start_time=0.0,
        end_time=0.02,
        minimum_frames=3,
        minimum_ess=1.0,
        maximum_drift_over_std=10.0,
        maximum_half_difference_over_std=10.0,
        maximum_birth_death_relative_imbalance=1.0,
    )
    assert [row["step"] for row in joint_rows] == [0, 1, 2]
    assert len(joint_report["continuation_overlap_checks"]) == 1


def test_v3_extruded_requires_and_binds_target_matched_manifest(tmp_path):
    source = aligned_x_band_limited_noise_2d(
        (8, 8),
        S_initial=1.0 / 3.0,
        angle_rms=0.01,
        max_mode_x=2,
        max_mode_y=2,
        dtype=torch.float64,
    )
    checkpoint = tmp_path / "Q2D_100.npy"
    np.save(
        checkpoint,
        np.stack([source[name].numpy() for name in source], axis=-1),
    )
    qualification_report = tmp_path / "qualification.json"
    qualification_report.write_text(json.dumps({
        "candidate_2d_statistical_steady_state": True,
        "parameters": {
            "frank_k": 0.012345679012345678,
            "zeta": 0.01,
        },
        "recommended_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
        },
    }))
    manifest = tmp_path / "v3_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "protocol": "V3",
        "qualified": True,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
        },
        "parameters": {
            "frank_k": 0.012345679012345678,
            "zeta": 0.01,
        },
        "qualification_report": {
            "path": str(qualification_report.resolve()),
            "sha256": _sha256(qualification_report),
        },
    }))
    output = tmp_path / "three_dimensional"
    result = subprocess.run(
        [
            *_base_command(output),
            "--initialization-protocol", "v3-extruded",
            "--initial-q2d", str(checkpoint),
            "--v3-mother-manifest", str(manifest),
            "--v3-rotation-rms", "0.001",
            "--v3-rotation-max-mode-x", "2",
            "--v3-rotation-max-mode-y", "2",
            "--v3-rotation-z-modes", "1", "2", "3",
            "--v3-perturbation-seed", "91",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output / "metadata.json").read_text())
    initial = metadata["initial_condition"]
    assert metadata["initialization_protocol"] == "v3-extruded"
    assert initial["qualified_mother"]["checkpoint_sha256"] == _sha256(checkpoint)
    assert initial["rotation_rms_radians"] == 0.001
    assert initial["contains_kz_zero"] is False
    assert initial["bound_inputs_unchanged_at_completion"] is True
    q0 = np.load(output / "Q_0.npy")
    assert q0.shape == (8, 8, 4, 5)
    assert np.isfinite(q0).all()
