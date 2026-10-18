"""Tests for the qualification-only RC6 CUDA-toolkit bootstrap preflight."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from benchmarks import run_rc6_cuda_toolkit_preflight as preflight


def _completed(*_args, **_kwargs):
    return subprocess.CompletedProcess(
        args=["nvcc", "--version"], returncode=0,
        stdout="Cuda compilation tools, release 12.1, V12.1.66\n", stderr="",
    )


def test_observation_passes_only_with_loaded_module_nvcc_header_and_matching_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    import torch

    root = tmp_path / "cuda-12.1"
    nvcc = root / "bin" / "nvcc"
    header = root / "include" / "cuda.h"
    nvcc.parent.mkdir(parents=True)
    header.parent.mkdir(parents=True)
    nvcc.write_text("nvcc\n", encoding="utf-8")
    header.write_text("cuda\n", encoding="utf-8")
    monkeypatch.setenv("LOADEDMODULES", "gcc/7.3.0:cuda/12.1")
    monkeypatch.setenv("CUDA_PATH", str(root))
    monkeypatch.setattr(preflight.shutil, "which", lambda name: str(nvcc))
    monkeypatch.setattr(preflight.subprocess, "run", _completed)
    monkeypatch.setattr(torch.version, "cuda", "12.1")

    result = preflight.observe("cuda/12.1", "12.1")

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["nvcc_resolved"] == str(nvcc.resolve())
    assert result["cuda_header"] == str(header)
    assert result["cuda_header_regular"] is True
    assert result["torch_cuda_runtime"] == "12.1"


@pytest.mark.parametrize(
    ("loaded", "which", "stdout", "returncode", "message"),
    [
        ("gcc/7.3.0", "present", "release 12.1", 0, "module"),
        ("cuda/12.1", None, "", 0, "discoverable"),
        ("cuda/12.1", "present", "release 12.1", 1, "did not succeed"),
        ("cuda/12.1", "present", "release 12.4", 0, "release differs"),
    ],
)
def test_observation_fails_closed_for_bootstrap_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loaded: str,
    which: str | None,
    stdout: str,
    returncode: int,
    message: str,
):
    root = tmp_path / "cuda-12.1"
    nvcc = root / "bin" / "nvcc"
    header = root / "include" / "cuda.h"
    nvcc.parent.mkdir(parents=True)
    header.parent.mkdir(parents=True)
    nvcc.write_text("nvcc\n", encoding="utf-8")
    header.write_text("cuda\n", encoding="utf-8")
    monkeypatch.setenv("LOADEDMODULES", loaded)
    monkeypatch.setattr(
        preflight.shutil, "which", lambda name: None if which is None else str(nvcc),
    )
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["nvcc", "--version"], returncode=returncode, stdout=stdout, stderr="",
        ),
    )

    result = preflight.observe("cuda/12.1", "12.1")

    assert result["passed"] is False
    assert any(message in failure for failure in result["failures"])


def test_main_persists_failed_observation_before_returning_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    output = tmp_path / "cuda_toolkit.json"
    monkeypatch.setattr(
        preflight,
        "observe",
        lambda *_args: {
            "schema": preflight.SCHEMA,
            "passed": False,
            "failures": ["nvcc is not discoverable through PATH"],
        },
    )

    assert preflight.main([
        "--expected-module", "cuda/12.1",
        "--expected-release", "12.1",
        "--output", str(output),
    ]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preflight.main([
            "--expected-module", "cuda/12.1",
            "--expected-release", "12.1",
            "--output", str(output),
        ])


def test_slurm_script_bootstraps_cuda_before_node_local_and_stage_execution():
    script = (Path(__file__).resolve().parents[1] / "jobs/rc6_single_h100_qualification.sbatch").read_text(
        encoding="utf-8"
    )
    module_position = script.index("module load cuda/12.1")
    toolkit_position = script.rindex(
        '"${RC6_SUPPORT_ROOT}/benchmarks/run_rc6_cuda_toolkit_preflight.py"'
    )
    tmp_position = script.index('node_root="/tmp/pssolver_rc6_${SLURM_JOB_ID}"')
    stage_position = script.rindex(
        '"${RC6_SUPPORT_ROOT}/benchmarks/run_rc6_single_h100_stage_plan.py"'
    )
    assert module_position < tmp_position < toolkit_position < stage_position
    assert "source /opt/ohpc/admin/lmod/lmod/init/bash" in script
    lmod_fallback = """if ! type module >/dev/null 2>&1; then
  # HPCC provides Lmod here for non-login batch shells.
  # Its Bash initializer reads internal variables that may be unset, so keep
  # nounset disabled only while sourcing the site-owned initialization file.
  set +u
  source /opt/ohpc/admin/lmod/lmod/init/bash
  set -u
fi"""
    assert lmod_fallback in script
    assert '--output "${RC6_CONTROL_ROOT}/preflight/cuda_toolkit.json"' in script
