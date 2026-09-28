"""CPU contract tests for the P9.5 fresh-process profiler."""

from __future__ import annotations

import json

import numpy as np
import pytest

import benchmarks.profile_periodic_functional as profiler
from benchmarks.profile_periodic_functional import (
    PeriodicFunctionalProfileConfig,
    _atomic_json,
    run_profile,
)


def _initial_q(tmp_path):
    path = tmp_path / "Q_0.npy"
    rng = np.random.default_rng(20260928)
    values = rng.normal(scale=0.01, size=(6, 6, 4, 5)).astype(np.float64)
    values[..., 0] += 0.18
    values[..., 3] -= 0.09
    np.save(path, values, allow_pickle=False)
    return path


def _config(path, role, *, trial=2):
    return PeriodicFunctionalProfileConfig(
        role=role,
        trial=trial,
        shape=(6, 6, 4),
        lengths=(6.0, 6.0, 4.0),
        initial_q_path=str(path),
        device="cpu",
        warmup_steps=0,
        profile_steps=1,
    )


def test_production_and_functional_profiles_are_byte_identical(tmp_path):
    path = _initial_q(tmp_path)

    production = run_profile(_config(path, "production_forward"))
    functional = run_profile(_config(path, "functional_forward"))

    assert production["result"]["state_sha256"] == (
        functional["result"]["state_sha256"]
    )
    assert production["result"]["finite"] is True
    assert functional["result"]["finite"] is True
    assert functional["functional_identity"]["execution"][
        "functional_runtime"
    ]["fallback_used"] is False
    assert production["timing"]["mean_seconds"] > 0.0
    assert functional["timing"]["mean_seconds"] > 0.0


def test_production_profile_constructs_exactly_one_runtime(tmp_path, monkeypatch):
    calls = 0
    original = profiler._production

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(profiler, "_production", counted)
    run_profile(_config(_initial_q(tmp_path), "production_forward"))

    assert calls == 1


def test_vjp_profile_is_finite_nonzero_and_bitwise_replayable(tmp_path):
    report = run_profile(
        _config(_initial_q(tmp_path), "functional_vjp", trial=1)
    )

    assert report["result"]["finite"] is True
    assert report["result"]["nonzero"] is True
    assert report["result"]["replay_bitwise_equal"] is True
    assert len(report["result"]["gradient_sha256"]) == 64
    assert len(report["result"]["gradient_norms"]) == 3
    assert report["correctness"]["r12"]["passed"] is True
    assert report["correctness"]["gradient"]["passed"] is True
    json.dumps(report, allow_nan=False)


def test_profiler_rejects_bad_initial_q_and_atomic_overwrite(tmp_path):
    path = tmp_path / "Q_0.npy"
    np.save(path, np.zeros((6, 6, 4, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="shape or dtype"):
        run_profile(_config(path, "production_forward"))

    output = tmp_path / "profile.json"
    _atomic_json(output, {"ok": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        _atomic_json(output, {"ok": False})


def test_profiler_rejects_invalid_role_and_missing_canonical_filename(tmp_path):
    path = _initial_q(tmp_path)
    with pytest.raises(ValueError, match="role"):
        run_profile(_config(path, "unknown"))

    other = tmp_path / "initial.npy"
    other.write_bytes(path.read_bytes())
    with pytest.raises(FileNotFoundError, match="Q_0.npy"):
        run_profile(_config(other, "production_forward"))
