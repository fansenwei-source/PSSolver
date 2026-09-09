"""CPU-only tests for the bounded Nsight capture runner."""

from __future__ import annotations

import pytest

from benchmarks.capture_beris_edwards_nsys import (
    NsysCaptureConfig,
    NvtxRegionTimer,
    _validate_capture_config,
)


def test_capture_config_maps_to_complete_timestep_profile():
    config = NsysCaptureConfig(
        shape=(12, 10, 8),
        warmup_steps=4,
        capture_steps=2,
        reuse_q_gradients=False,
    )

    profile = config.profile_config()

    assert profile.shape == (12, 10, 8)
    assert profile.device == "cuda"
    assert profile.warmup_steps == 4
    assert profile.profile_steps == 2
    assert profile.reuse_q_gradients is False
    assert profile.snapshot_interval is None
    assert profile.snapshot_directory is None


def test_capture_config_rejects_nonpositive_capture_steps():
    with pytest.raises(ValueError, match="capture_steps must be positive"):
        _validate_capture_config(NsysCaptureConfig(capture_steps=0))


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
