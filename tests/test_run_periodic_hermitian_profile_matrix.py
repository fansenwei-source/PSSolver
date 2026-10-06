"""Regression tests for the shell-free Periodic profile matrix runner."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.run_periodic_hermitian_profile_matrix import (
    VariantRuntime,
    _plan_metadata,
    build_matrix_plan,
    execute_matrix,
)


BASELINE = "a" * 40
CANDIDATE = "b" * 40


def _runtime(tmp_path: Path, name: str, commit: str) -> VariantRuntime:
    root = tmp_path / name
    root.mkdir()
    return VariantRuntime(
        python=tmp_path / f"{name}-python",
        repository_root=root,
        commit=commit,
    )


def _plan(tmp_path: Path):
    return build_matrix_plan(
        baseline=_runtime(tmp_path, "baseline", BASELINE),
        candidate=_runtime(tmp_path, "candidate", CANDIDATE),
        inputs={
            "R128": tmp_path / "R128" / "Q_0.npy",
            "R320": tmp_path / "R320" / "Q_0.npy",
        },
        output_root=tmp_path / "output",
    )


def test_matrix_has_36_unique_explicit_commands_without_empty_arguments(tmp_path):
    profiles = _plan(tmp_path)

    assert len(profiles) == 36
    assert len({profile.key for profile in profiles}) == 36
    assert [profile.sequence for profile in profiles] == list(range(1, 37))
    assert all(profile.command for profile in profiles)
    assert all(value for profile in profiles for value in profile.command)
    assert all(
        profile.command[profile.command.index("--device") + 1] == "cuda"
        for profile in profiles
    )
    assert all("cuda:0" not in profile.command for profile in profiles)


def test_physical_artifact_argument_is_present_only_for_four_profiles(tmp_path):
    profiles = _plan(tmp_path)
    with_artifact = [
        profile for profile in profiles if "--physical-artifact" in profile.command
    ]
    without_artifact = [
        profile for profile in profiles if "--physical-artifact" not in profile.command
    ]

    assert len(with_artifact) == 4
    assert len(without_artifact) == 32
    for profile in with_artifact:
        assert profile.role == "production_forward"
        assert profile.trial == 1
        index = profile.command.index("--physical-artifact")
        assert profile.command[index + 1] == str(profile.physical_artifact)
    assert all(profile.physical_artifact is None for profile in without_artifact)


def test_plan_metadata_is_strictly_json_compatible(tmp_path):
    metadata = _plan_metadata(_plan(tmp_path))

    assert metadata["profile_count"] == 36
    assert metadata["profiler_cli_device_token"] == "cuda"
    assert metadata["expected_allocated_device_identity"] == "cuda:0"
    assert len(metadata["profiles"]) == 36
    assert sum(
        "--physical-artifact" in profile["command"]
        for profile in metadata["profiles"]
    ) == 4


def test_matrix_rejects_noncanonical_cli_device_token(tmp_path):
    with pytest.raises(ValueError, match="must be cuda"):
        build_matrix_plan(
            baseline=_runtime(tmp_path, "baseline", BASELINE),
            candidate=_runtime(tmp_path, "candidate", CANDIDATE),
            inputs={
                "R128": tmp_path / "R128" / "Q_0.npy",
                "R320": tmp_path / "R320" / "Q_0.npy",
            },
            output_root=tmp_path / "output",
            device="cuda:0",
        )


def test_python_orchestrator_executes_profiles_with_and_without_artifacts(
    monkeypatch,
    tmp_path,
):
    profiles = _plan(tmp_path)
    output_root = tmp_path / "output"
    calls = []

    def fake_run(command, **_kwargs):
        command = list(command)
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        if "analyze_periodic_hermitian_profiles.py" in command[1]:
            output.write_text(json.dumps({"passed": True}), encoding="utf-8")
        else:
            shape_index = command.index("--shape") + 1
            shape = [int(value) for value in command[shape_index : shape_index + 3]]
            role = command[command.index("--role") + 1]
            trial = int(command[command.index("--trial") + 1])
            variant = command[command.index("--variant") + 1]
            output.write_text(
                json.dumps(
                    {
                        "variant": variant,
                        "base_profile": {
                            "config": {
                                "shape": shape,
                                "role": role,
                                "trial": trial,
                                "device": "cuda:0",
                            },
                            "environment": {"device": "cuda:0"},
                        },
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    execute_matrix(
        profiles,
        output_root=output_root,
        candidate=VariantRuntime(
            python=tmp_path / "candidate-python",
            repository_root=tmp_path / "candidate",
            commit=CANDIDATE,
        ),
        baseline_commit=BASELINE,
    )

    assert len(calls) == 37
    assert sum("--physical-artifact" in command for command in calls[:-1]) == 4
    assert json.loads(
        (output_root / "MATRIX_COMPLETE.json").read_text(encoding="utf-8")
    )["passed"] is True
