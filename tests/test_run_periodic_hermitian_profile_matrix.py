"""Regression tests for the shell-free Periodic profile matrix runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import venv

import pytest

from benchmarks.run_periodic_hermitian_profile_matrix import (
    VariantRuntime,
    _child_environment,
    _child_identity,
    _plan_metadata,
    _prepare_import_bootstraps,
    _runtime,
    build_matrix_plan,
    execute_matrix,
)


BASELINE = "a" * 40
CANDIDATE = "b" * 40


def _fake_runtime(tmp_path: Path, name: str, commit: str) -> VariantRuntime:
    root = tmp_path / name
    root.mkdir()
    (root / "benchmarks").mkdir()
    return VariantRuntime(
        python=tmp_path / f"{name}-python",
        repository_root=root,
        commit=commit,
    )


def _plan(tmp_path: Path):
    return build_matrix_plan(
        baseline=_fake_runtime(tmp_path, "baseline", BASELINE),
        candidate=_fake_runtime(tmp_path, "candidate", CANDIDATE),
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
            baseline=_fake_runtime(tmp_path, "baseline", BASELINE),
            candidate=_fake_runtime(tmp_path, "candidate", CANDIDATE),
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

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append((command, kwargs))
        if len(command) > 1 and command[1] == "-c":
            requested_python = Path(command[0]).absolute()
            prefix = requested_python.parent.parent.resolve()
            purelib = prefix / "lib" / "python3.10" / "site-packages"
            bootstrap = Path(kwargs["env"]["PYTHONPATH"])
            repository = tmp_path / bootstrap.name
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "sys_executable": str(requested_python),
                        "sys_prefix": str(prefix),
                        "sys_base_prefix": "/base-python",
                        "purelib": str(purelib),
                        "pssolver_file": str(purelib / "pssolver" / "__init__.py"),
                        "benchmarks_file": str(
                            bootstrap / "benchmarks" / "__init__.py"
                        ),
                        "benchmarks_paths": [str(repository / "benchmarks")],
                    }
                ),
                "",
            )
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
        baseline=VariantRuntime(
            python=tmp_path / "baseline-python",
            repository_root=tmp_path / "baseline",
            commit=BASELINE,
        ),
        candidate=VariantRuntime(
            python=tmp_path / "candidate-python",
            repository_root=tmp_path / "candidate",
            commit=CANDIDATE,
        ),
        baseline_commit=BASELINE,
    )

    assert len(calls) == 39
    assert sum(
        "--physical-artifact" in command
        for command, _kwargs in calls
        if len(command) > 1 and command[1] != "-c"
    ) == 4
    for command, kwargs in calls:
        bootstrap = Path(kwargs["env"]["PYTHONPATH"])
        assert bootstrap.is_relative_to(output_root / "import_bootstrap")
        assert (bootstrap / "benchmarks" / "__init__.py").is_file()
        assert not (bootstrap / "pssolver").exists()
        if command[1] == "-c":
            continue
        if "analyze_periodic_hermitian_profiles.py" not in command[1]:
            variant = command[command.index("--variant") + 1]
            assert bootstrap.name == variant
    assert json.loads(
        (output_root / "MATRIX_COMPLETE.json").read_text(encoding="utf-8")
    )["passed"] is True


def test_import_bootstrap_exposes_benchmarks_but_not_source_pssolver(tmp_path):
    repository = tmp_path / "repository"
    benchmarks = repository / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "probe.py").write_text("VALUE = 17\n", encoding="utf-8")
    entry = benchmarks / "entry.py"
    entry.write_text(
        "import importlib.util, pathlib\n"
        "from benchmarks.probe import VALUE\n"
        "print(VALUE)\n"
        "spec = importlib.util.find_spec('pssolver')\n"
        "print(None if spec is None else pathlib.Path(spec.origin).resolve())\n",
        encoding="utf-8",
    )
    (repository / "pssolver").mkdir()
    (repository / "pssolver" / "__init__.py").write_text(
        "SOURCE_SHADOW = True\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    roots = _prepare_import_bootstraps(
        output_root,
        {"baseline": repository, "candidate": repository},
    )
    environment = _child_environment(
        {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        roots["candidate"],
    )
    completed = subprocess.run(
        [sys.executable, str(entry)],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    output = completed.stdout.splitlines()
    assert output[0] == "17"
    assert output[1] != str((repository / "pssolver" / "__init__.py").resolve())


def test_runtime_preserves_the_requested_python_symlink(tmp_path):
    base_python = tmp_path / "base-python"
    base_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    repository = tmp_path / "repository"
    repository.mkdir()

    runtime = _runtime(str(venv_python), str(repository), CANDIDATE)

    assert runtime.python == venv_python.absolute()
    assert runtime.python.is_symlink()
    assert runtime.python.resolve() == base_python.resolve()


def test_child_identity_uses_real_venv_purelib_and_benchmark_shim(tmp_path):
    venv_root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(venv_root)
    python = venv_root / "bin" / "python"
    completed = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
    )
    purelib = Path(completed.stdout.strip())
    package = purelib / "pssolver"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    repository = tmp_path / "repository"
    (repository / "benchmarks").mkdir(parents=True)
    # A source package with the same name must not shadow the venv package.
    (repository / "pssolver").mkdir()
    (repository / "pssolver" / "__init__.py").write_text(
        "SOURCE_SHADOW = True\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    bootstrap = _prepare_import_bootstraps(
        output_root,
        {"baseline": repository, "candidate": repository},
    )["candidate"]
    runtime = _runtime(str(python), str(repository), CANDIDATE)

    report = _child_identity(
        runtime,
        bootstrap_root=bootstrap,
        base_environment={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )

    assert report["requested_python"] == str(python.absolute())
    assert report["reported_prefix"] == str(venv_root.resolve())
    assert Path(report["pssolver_file"]).is_relative_to(purelib.resolve())
    assert report["source_shadow_import"] is False
