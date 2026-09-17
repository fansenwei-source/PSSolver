"""Tests for the periodic fast-path default-promotion smoke runner."""

import pytest

from benchmarks.smoke_periodic_fast_path_default import (
    DefaultSmokeConfig,
    run_default_smoke,
)


def _config(**overrides):
    values = {
        "shape": (8, 6),
        "lengths": (4.0, 3.0),
        "field_count": 3,
        "batch_size": 1,
        "device": "cpu",
        "dtype": "float64",
        "warmup_steps": 1,
        "profile_steps": 2,
        "trajectory_steps": 3,
        "trials": 1,
        "seed": 17,
    }
    values.update(overrides)
    return DefaultSmokeConfig(**values)


def test_default_smoke_matches_implicit_and_explicit_candidate():
    result = run_default_smoke(_config())

    trajectory = result["trajectory"]
    assert trajectory["implicit_candidate"]["spatial_bytewise_equal"]
    assert trajectory["implicit_candidate"]["spectral_bytewise_equal"]
    assert trajectory["rollback_candidate"]["spatial_relative_l2"] <= 1.0e-12
    assert trajectory["rollback_candidate"]["spectral_relative_l2"] <= 1.0e-12
    assert set(result["aggregate"]) == {
        "implicit_default",
        "explicit_candidate",
        "explicit_rollback",
    }
    implicit = trajectory["implicit_default_metadata"]
    assert implicit["periodic_transform_execution"]["effective"] == "multidim"
    assert implicit["transform_group_indexing"]["effective"] == "contiguous_slice"


def test_default_smoke_rejects_invalid_trajectory_steps():
    with pytest.raises(ValueError, match="trajectory_steps"):
        run_default_smoke(_config(trajectory_steps=0))
