from __future__ import annotations

import pytest
import torch

from benchmarks.diagnose_plane_nyquist_storage import diagnose


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence():
    report = diagnose(device="cuda")

    assert report["classification"] == (
        "PASS_PLANE_PERIODIC_NYQUIST_STORAGE_EQUIVALENCE"
    )
    assert report["configuration"]["device"] == "cuda"
    assert report["environment"]["cuda_available"] is True
    assert report["environment"]["allocated_device"] == "cuda:0"
    assert "H100" in report["environment"]["device_name"]
    assert report["environment"]["cuda_matmul_allow_tf32"] is False
    assert report["environment"]["cudnn_allow_tf32"] is False
    for case in report["cases"]:
        assert case["raw"]["full_complex"]["finite"] is True
        assert case["raw"]["hermitian_half"]["finite"] is True
        assert case["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12
        assert case["summary"]["raw_pressure_relative_l2"] < 1.0e-12
