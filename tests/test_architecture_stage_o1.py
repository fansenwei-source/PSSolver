"""Characterization tests for Stage O.1 configuration extraction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pssolver.configuration import (
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneBerisEdwardsRunSpec,
    create_plane_beris_edwards_run_spec,
    parse_plane_beris_edwards_run_spec,
)
from pssolver.core import BoundaryKind, SpectralStorage
from pssolver.models.active_nematics import (
    PLANE_DISTORTION_ODD_BOUNDARY_CONDITIONS,
    PLANE_NORMAL_VELOCITY_BOUNDARY_CONDITIONS,
    PLANE_PRESSURE_BOUNDARY_CONDITIONS,
    PLANE_Q_BOUNDARY_CONDITIONS,
    PLANE_TANGENTIAL_VELOCITY_BOUNDARY_CONDITIONS,
)
from pssolver.presets import resolve_shendruk_plane_preset


PROJECT_ROOT = Path(__file__).parents[1]
PRODUCTION_SCRIPT = PROJECT_ROOT / "Plane_beris_edwards_stokes.py"


def _spec(tmp_path, **overrides):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / "run",
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _dry_run(*arguments):
    return subprocess.run(
        [
            sys.executable,
            str(PRODUCTION_SCRIPT),
            "--activity-number",
            "18",
            "--output-dir",
            "/tmp/pssolver_stage_o1_dry_run_unused",
            "--device",
            "cpu",
            "--pointwise-execution",
            "eager",
            "--dry-run",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_shendruk_paper_window_preset_preserves_a18_mapping():
    preset = resolve_shendruk_plane_preset(
        activity_number=18.0,
        height=20.0,
        parameterization="paper-window",
        frank_k=0.01,
        coefficient_min=0.01,
        coefficient_max=0.05,
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        gamma=2.94,
    )

    assert preset.activity_ratio == pytest.approx(0.81)
    assert preset.zeta == 0.01
    assert preset.frank_k == pytest.approx(0.012345679012345678)
    assert preset.equilibrium_s == pytest.approx(1.0 / 3.0)
    assert preset.equilibrium_q_amplitude == pytest.approx(0.5)
    assert preset.ldg_l1 == pytest.approx(0.024691358024691357)
    assert preset.ldg_l1_over_gamma == pytest.approx(
        0.008398421096833794
    )


def test_shendruk_fixed_k_preset_preserves_explicit_frank_coefficient():
    preset = resolve_shendruk_plane_preset(
        activity_number=30.0,
        height=20.0,
        parameterization="fixed-k",
        frank_k=0.02,
        coefficient_min=0.01,
        coefficient_max=0.05,
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        gamma=2.94,
    )

    assert preset.activity_ratio == pytest.approx(2.25)
    assert preset.frank_k == 0.02
    assert preset.zeta == pytest.approx(0.045)
    assert preset.ldg_l1 == pytest.approx(0.04)


def test_shendruk_paper_window_rejects_out_of_range_activity():
    with pytest.raises(ValueError, match="outside the paper-window interval"):
        resolve_shendruk_plane_preset(
            activity_number=2.0,
            height=20.0,
            parameterization="paper-window",
            frank_k=0.01,
            coefficient_min=0.01,
            coefficient_max=0.05,
            ldg_a=0.0,
            ldg_b=-0.3,
            ldg_c=0.3,
            gamma=2.94,
        )


def test_run_spec_is_frozen_complete_and_uses_legacy_only(tmp_path):
    spec = _spec(tmp_path)
    assert isinstance(spec, PlaneBerisEdwardsRunSpec)
    assert spec.geometry.name == "plane_slab"
    assert spec.geometry.bounded_axes == (2,)
    assert spec.domain.shape == (256, 256, 64)
    assert spec.domain.lengths == (100.0, 100.0, 20.0)
    assert spec.numerics.spectral_storage is SpectralStorage.HERMITIAN_HALF
    assert spec.spectral_refresh.mode == "physical_time"
    assert spec.spectral_refresh.effective_interval_steps == 20
    assert spec.to_metadata()["runtime_path"] == "legacy_production"
    assert spec.identity_metadata()["runtime_path"] == "legacy_production"
    assert len(spec.canonical_sha256()) == 64

    with pytest.raises(FrozenInstanceError):
        spec.dt = 0.02


def test_run_spec_canonical_identity_is_stable_and_complete(tmp_path):
    left = _spec(tmp_path)
    right = _spec(tmp_path)
    changed = _spec(tmp_path, dt=0.005)

    assert left.to_metadata() == right.to_metadata()
    assert left.canonical_sha256() == right.canonical_sha256()
    assert left.canonical_sha256() != changed.canonical_sha256()
    json.dumps(left.to_metadata(), allow_nan=False, sort_keys=True)


def test_physical_boundaries_match_the_qualified_legacy_plane_bases():
    legacy = PLANE_FREE_SLIP_BOUNDARIES.to_legacy()
    assert legacy == {
        "q": PLANE_Q_BOUNDARY_CONDITIONS,
        "tangential_velocity": (
            PLANE_TANGENTIAL_VELOCITY_BOUNDARY_CONDITIONS
        ),
        "normal_velocity": PLANE_NORMAL_VELOCITY_BOUNDARY_CONDITIONS,
        "pressure_modal": PLANE_PRESSURE_BOUNDARY_CONDITIONS,
        "distortion_odd_z": PLANE_DISTORTION_ODD_BOUNDARY_CONDITIONS,
    }
    assert PLANE_FREE_SLIP_BOUNDARIES.q.axes[-1].kind is BoundaryKind.NEUMANN
    assert (
        PLANE_FREE_SLIP_BOUNDARIES.normal_velocity.axes[-1].kind
        is BoundaryKind.DIRICHLET
    )


def test_programmatic_and_cli_specs_share_one_configuration_authority(
    tmp_path,
):
    spec = parse_plane_beris_edwards_run_spec(
        [
            "--activity-number",
            "18",
            "--output-dir",
            str(tmp_path / "run"),
            "--dtype",
            "float64",
            "--spectral-refresh-steps",
            "7",
            "--twist-modes",
            "1",
            "3",
        ]
    )

    assert isinstance(spec, PlaneBerisEdwardsRunSpec)
    assert spec.dtype == "float64"
    assert spec.twist_modes == (1, 3)
    assert spec.S_initial == spec.initial_s
    assert spec.spectral_refresh.mode == "steps"
    assert spec.spectral_refresh_interval_steps == 7
    assert spec.spectral_refresh_effective_time == pytest.approx(0.07)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--dt", "0.03"), "integer multiple"),
        (
            (
                "--dealias-rule",
                "none",
                "--projected-transform-execution",
                "truncated",
            ),
            "requires enabled dealiasing",
        ),
        (
            (
                "--spectral-storage",
                "hermitian_half",
                "--transform-execution-order",
                "legacy",
            ),
            "requires --transform-execution-order real_first",
        ),
    ),
)
def test_cli_validation_errors_remain_at_the_parse_boundary(
    arguments,
    message,
):
    result = _dry_run(*arguments)
    assert result.returncode != 0
    assert message in result.stderr


def test_production_dry_run_records_additive_o1_identity_only():
    result = _dry_run("--disable-spectral-refresh", "--dtype", "float64")
    assert result.returncode == 0, result.stderr
    metadata = json.loads(result.stdout)

    assert metadata["configuration"] == {
        "schema_version": 1,
        "authority": "pssolver.configuration.PlaneBerisEdwardsRunSpec",
        "runtime_path": "legacy_production",
        "canonical_sha256": metadata["configuration"]["canonical_sha256"],
    }
    assert len(metadata["configuration"]["canonical_sha256"]) == 64
    assert metadata["model"]["parameters"]["zeta"] == 0.01
    assert metadata["model"]["parameters"]["frank_K"] == pytest.approx(
        0.012345679012345678
    )
    assert metadata["boundary_conditions"]["Q"] == [
        "periodic",
        "periodic",
        "neumann",
    ]


def test_o1_production_path_does_not_import_experimental_architecture():
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/configuration/plane_beris_edwards.py",
        "pssolver/presets/shendruk.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "pssolver.experimental" not in source
    source = PRODUCTION_SCRIPT.read_text(encoding="utf-8")
    assert "ratio_min =" not in source
    assert 'Q_BC = ("periodic", "periodic", "neumann")' not in source
