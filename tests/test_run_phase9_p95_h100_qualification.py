"""Tests for the repository-owned P9.5 H100 matrix runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.run_phase9_p95_h100_qualification import (
    build_commands,
    validate_inputs,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan():
    return {
        "phase": "P9.5",
        "grids": [
            {"id": "small", "shape": [4, 4, 2], "lengths": [4.0, 4.0, 2.0]}
        ],
        "roles": [
            "production_forward",
            "functional_forward",
            "functional_vjp",
        ],
        "trials": 3,
        "profile_count": 9,
        "dt": 0.001,
        "base_activity": 0.013,
        "warmup_steps": 5,
        "profile_steps": 20,
    }


def _inputs(tmp_path):
    root = tmp_path / "inputs"
    directory = root / "small"
    directory.mkdir(parents=True)
    path = directory / "Q_0.npy"
    np.save(path, np.zeros((4, 4, 2, 5), dtype=np.float64))
    manifest = {
        "phase": "P9.5",
        "grids": [
            {
                "id": "small",
                "shape": [4, 4, 2],
                "path": "small/Q_0.npy",
                "sha256": _sha256(path),
            }
        ],
    }
    (root / "input_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def test_runner_builds_balanced_unique_fresh_process_matrix(tmp_path):
    plan = _plan()
    paths = validate_inputs(plan, _inputs(tmp_path))

    commands = build_commands(
        plan,
        initial_paths=paths,
        profile_directory=tmp_path / "profiles",
        python_executable="/fixed/python",
    )

    assert len(commands) == 9
    assert [item["role"] for item in commands[:3]] == plan["roles"]
    assert [item["role"] for item in commands[3:6]] == [
        "functional_forward",
        "functional_vjp",
        "production_forward",
    ]
    assert len({item["output"] for item in commands}) == 9
    assert all(item["argv"][:3] == [
        "/fixed/python",
        "-m",
        "benchmarks.profile_periodic_functional",
    ] for item in commands)


def test_runner_rejects_modified_input(tmp_path):
    root = _inputs(tmp_path)
    path = root / "small" / "Q_0.npy"
    with path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="SHA-256 differs"):
        validate_inputs(_plan(), root)
