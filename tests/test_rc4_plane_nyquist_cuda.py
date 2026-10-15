from __future__ import annotations

import os

import pytest
import torch

from benchmarks.diagnose_plane_nyquist_storage import diagnose
from pssolver.configuration.execution_device import scoped_tf32_execution


_H100_QUALIFICATION_ENVIRONMENT_VARIABLE = "PSSOLVER_REQUIRE_H100"


def _h100_qualification_requested() -> bool:
    value = os.environ.get(_H100_QUALIFICATION_ENVIRONMENT_VARIABLE, "0")
    if value not in {"0", "1"}:
        raise ValueError(
            f"{_H100_QUALIFICATION_ENVIRONMENT_VARIABLE} must be 0 or 1"
        )
    return value == "1"


def test_h100_identity_is_an_explicit_qualification_policy(monkeypatch):
    monkeypatch.delenv(_H100_QUALIFICATION_ENVIRONMENT_VARIABLE, raising=False)
    assert _h100_qualification_requested() is False
    monkeypatch.setenv(_H100_QUALIFICATION_ENVIRONMENT_VARIABLE, "1")
    assert _h100_qualification_requested() is True
    monkeypatch.setenv(_H100_QUALIFICATION_ENVIRONMENT_VARIABLE, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _h100_qualification_requested()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence():
    with scoped_tf32_execution(
        "off",
        real_dtype=torch.float64,
        device="cuda",
    ):
        report = diagnose(device="cuda")

    assert report["classification"] == (
        "PASS_PLANE_PERIODIC_NYQUIST_STORAGE_EQUIVALENCE"
    )
    assert report["configuration"]["device"] == "cuda"
    assert report["environment"]["cuda_available"] is True
    assert report["environment"]["allocated_device"] == "cuda:0"
    if _h100_qualification_requested():
        assert "H100" in report["environment"]["device_name"]
    assert report["environment"]["cuda_matmul_allow_tf32"] is False
    assert report["environment"]["cudnn_allow_tf32"] is False
    for case in report["cases"]:
        assert case["raw"]["full_complex"]["finite"] is True
        assert case["raw"]["hermitian_half"]["finite"] is True
        assert case["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12
        assert case["summary"]["raw_pressure_relative_l2"] < 1.0e-12
