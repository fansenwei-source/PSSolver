import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "Plane_beris_edwards_navier_stokes.py"


def _run(*arguments):
    return subprocess.run(
        [sys.executable, str(RUNNER), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _base_arguments(output):
    return [
        "--activity-number", "9",
        "--height", "10",
        "--lx", "32",
        "--ly", "32",
        "--nx", "8",
        "--ny", "8",
        "--nz", "6",
        "--dt", "0.001",
        "--steps", "2",
        "--save-start-step", "0",
        "--save-interval", "1",
        "--diagnostic-interval", "1",
        "--seed", "24",
        "--num-defect-pairs", "1",
        "--defect-min-separation", "4",
        "--defect-core-radius", "1",
        "--twist-modes", "1", "2",
        "--initialization-protocol", "v1",
        "--output-dir", str(output),
        "--device", "cpu",
        "--dtype", "float64",
        "--dealias-rule", "cubic_half",
    ]


def test_dry_run_records_inertial_equation_and_paper_defaults(tmp_path):
    result = _run(
        *_base_arguments(tmp_path / "unused"),
        "--dry-run",
        "--disable-spectral-refresh",
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads(result.stdout)

    assert metadata["script"] == "Plane_beris_edwards_navier_stokes.py"
    assert metadata["initialization_protocol"] == "v1"
    assert metadata["density"] == 1.0
    assert metadata["friction"] == 0.0
    assert metadata["mean_flow_policy"] == "evolve"
    assert metadata["transform_execution_order"] == "real_first"
    assert metadata["solver"]["transform_execution_order"] == "real_first"
    assert metadata["numerics"]["transforms"] == {
        "execution_order": "real_first",
        "spectral_storage": "full_complex",
        "basis_and_normalization_changed": False,
    }
    flow = metadata["model"]["flow_dynamics"]
    assert flow["regime"] == "incompressible_navier_stokes_beris_edwards"
    assert "partial_t*u+u.grad(u)" in flow["momentum_equation"]
    assert metadata["numerics"]["implicit_momentum_helmholtz"] == (
        "rho/dt+fric+eta*k^2"
    )
    assert metadata["numerics"]["explicit_terms"] == [
        "nematic_force",
        "u_dot_grad_u",
    ]


def test_legacy_transform_order_remains_an_explicit_opt_in(tmp_path):
    result = _run(
        *_base_arguments(tmp_path / "unused_legacy"),
        "--transform-execution-order", "legacy",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads(result.stdout)
    assert metadata["transform_execution_order"] == "legacy"
    assert metadata["solver"]["transform_execution_order"] == "legacy"
    assert metadata["numerics"]["transforms"]["execution_order"] == "legacy"


def test_invalid_density_and_friction_are_rejected(tmp_path):
    bad_density = _run(
        *_base_arguments(tmp_path / "bad_density"),
        "--density", "0",
        "--dry-run",
    )
    bad_friction = _run(
        *_base_arguments(tmp_path / "bad_friction"),
        "--fric", "-0.1",
        "--dry-run",
    )
    assert bad_density.returncode != 0
    assert "density" in bad_density.stderr
    assert bad_friction.returncode != 0
    assert "--fric" in bad_friction.stderr


def test_tiny_v1_run_advances_velocity_and_preserves_incompressibility(tmp_path):
    output = tmp_path / "tiny"
    result = _run(
        *_base_arguments(output),
        "--spectral-refresh-steps", "1",
        "--diagnostics",
        "--save-hydrodynamics",
    )
    assert result.returncode == 0, result.stderr

    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["completed_steps"] == 2
    assert metadata["numerics"]["spectral_refresh"]["actual_count"] == 2
    assert metadata["initial_condition"]["velocity"]["name"] == "quiescent"

    q0 = np.load(output / "Q_0.npy")
    u0 = np.load(output / "u_0.npy")
    u1 = np.load(output / "u_1.npy")
    p1 = np.load(output / "p_1.npy")
    assert q0.shape == (8, 8, 6, 5)
    assert u0.shape == u1.shape == (8, 8, 6, 3)
    assert p1.shape == (8, 8, 6)
    assert np.isfinite(q0).all()
    assert np.isfinite(u1).all()
    assert np.isfinite(p1).all()
    assert np.count_nonzero(u0) == 0
    assert np.linalg.norm(u1) > 0

    diagnostics = np.load(output / "diagnostics.npy")
    assert diagnostics[-1]["div_max"] < 1.0e-12
    assert diagnostics[-1]["time_discrete_momentum_residual_max"] < 1.0e-12
    assert diagnostics[-1]["schur_rel_residual"] < 1.0e-12
