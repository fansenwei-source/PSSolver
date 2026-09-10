"""CPU-only tests for the bounded Nsight capture runner."""

from __future__ import annotations

import pytest
import torch

from benchmarks.capture_beris_edwards_nsys import (
    NsysCaptureConfig,
    NvtxRegionTimer,
    _validate_capture_config,
    run_capture,
)


class _FakeCudaEvent:
    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        del other
        return 1.0


class _FakeCudaRuntime:
    def cudaProfilerStart(self):
        pass

    def cudaProfilerStop(self):
        pass


class _FakeIntegrator:
    def step(self):
        pass


class _FakePointwiseKernels:
    def metadata(self):
        return {
            "requested": "eager",
            "effective": "eager",
            "fallback_allowed": False,
        }


class _FakeSolver:
    integrator = _FakeIntegrator()
    pointwise_kernels = _FakePointwiseKernels()


def test_capture_config_maps_to_complete_timestep_profile():
    config = NsysCaptureConfig(
        shape=(12, 10, 8),
        warmup_steps=4,
        capture_steps=2,
        reuse_q_gradients=False,
        molecular_field_linear_space="spectral",
        stress_divergence_sum_space="spectral",
        pointwise_execution="compile",
        transform_execution_order="real_first",
        spectral_storage="hermitian_half",
    )

    profile = config.profile_config()

    assert profile.shape == (12, 10, 8)
    assert profile.device == "cuda"
    assert profile.warmup_steps == 4
    assert profile.profile_steps == 2
    assert profile.reuse_q_gradients is False
    assert profile.molecular_field_linear_space == "spectral"
    assert profile.stress_divergence_sum_space == "spectral"
    assert profile.pointwise_execution == "compile"
    assert profile.transform_execution_order == "real_first"
    assert profile.spectral_storage == "hermitian_half"
    assert profile.snapshot_interval is None
    assert profile.snapshot_directory is None


def test_capture_defaults_to_real_first_transform_execution():
    config = NsysCaptureConfig()

    assert config.transform_execution_order == "real_first"
    assert config.spectral_storage == "full_complex"
    assert config.molecular_field_linear_space == "spectral"
    assert config.stress_divergence_sum_space == "spectral"
    assert config.pointwise_execution == "compile"
    assert config.profile_config().transform_execution_order == "real_first"
    assert config.profile_config().spectral_storage == "full_complex"
    assert config.profile_config().molecular_field_linear_space == "spectral"
    assert config.profile_config().stress_divergence_sum_space == "spectral"
    assert config.profile_config().pointwise_execution == "compile"


def test_capture_config_rejects_nonpositive_capture_steps():
    with pytest.raises(ValueError, match="capture_steps must be positive"):
        _validate_capture_config(NsysCaptureConfig(capture_steps=0))


def test_capture_config_rejects_unknown_stress_divergence_sum_space():
    with pytest.raises(ValueError, match="stress_divergence_sum_space"):
        _validate_capture_config(
            NsysCaptureConfig(stress_divergence_sum_space="unknown")
        )


def test_capture_config_rejects_unknown_pointwise_execution():
    with pytest.raises(ValueError, match="pointwise_execution"):
        _validate_capture_config(
            NsysCaptureConfig(pointwise_execution="unknown")
        )


def test_capture_result_records_effective_pointwise_execution(monkeypatch):
    config = NsysCaptureConfig(
        shape=(4, 4, 4),
        warmup_steps=1,
        capture_steps=1,
        pointwise_execution="eager",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (0, 0))
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **kwargs: _FakeCudaEvent(),
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _FakeCudaRuntime())
    monkeypatch.setattr(
        "benchmarks.capture_beris_edwards_nsys._build_solver",
        lambda profile_config, timer: _FakeSolver(),
    )
    monkeypatch.setattr(
        "benchmarks.capture_beris_edwards_nsys._state_sha256",
        lambda solver: "state",
    )

    result = run_capture(config)

    assert result["pointwise_kernels"]["requested"] == "eager"
    assert result["pointwise_kernels"]["effective"] == "eager"
    assert result["pointwise_kernels"]["fallback_allowed"] is False


def test_disabled_nvtx_timer_does_not_touch_cuda(monkeypatch):
    pushes = []
    pops = []
    monkeypatch.setattr(
        "torch.cuda.nvtx.range_push",
        lambda name: pushes.append(name),
    )
    monkeypatch.setattr(
        "torch.cuda.nvtx.range_pop",
        lambda: pops.append(True),
    )
    timer = NvtxRegionTimer()

    with timer.region("disabled"):
        pass

    assert pushes == []
    assert pops == []


def test_enabled_nvtx_timer_balances_ranges_on_error(monkeypatch):
    pushes = []
    pops = []
    monkeypatch.setattr(
        "torch.cuda.nvtx.range_push",
        lambda name: pushes.append(name),
    )
    monkeypatch.setattr(
        "torch.cuda.nvtx.range_pop",
        lambda: pops.append(True),
    )
    timer = NvtxRegionTimer()
    timer.enabled = True

    with pytest.raises(RuntimeError, match="sentinel"):
        with timer.region("enabled"):
            raise RuntimeError("sentinel")

    assert pushes == ["enabled"]
    assert pops == [True]
