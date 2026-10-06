"""Tests for the versioned rc2 G5 installed-wheel smoke harness."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from benchmarks.analyze_rc2_g5_installed_smokes import analyze
from benchmarks.run_rc2_g5_installed_smoke import (
    FROZEN_INPUT_SHA256,
    FROZEN_SHAPE,
    FROZEN_TOTAL_STEPS,
    KIND,
    build_simulation,
)
from pssolver import compile_simulation


def _snapshot(tmp_path: Path, shape=(8, 8, 8)) -> Path:
    root = tmp_path / "snapshot"
    root.mkdir()
    path = root / "Q_0.npy"
    np.save(path, np.zeros((*shape, 5), dtype=np.float64), allow_pickle=False)
    return path


@pytest.mark.parametrize(
    ("application", "expected_application", "runtime_path"),
    (
        ("periodic", "periodic_complete_stress_beris_edwards", "periodic_spectral"),
        ("channel", "channel_complete_stress_beris_edwards", "channel_complete_stress"),
    ),
)
def test_g5_builder_explicitly_disables_refresh_and_compiles(
    tmp_path,
    application,
    expected_application,
    runtime_path,
):
    simulation = build_simulation(
        application=application,
        initial_q_path=_snapshot(tmp_path),
        output_directory=tmp_path / "output",
        steps=2,
        device="cpu",
        shape=(8, 8, 8),
        lengths=(8.0, 8.0, 8.0),
    )

    assert simulation.to_metadata()["time_integration"]["refresh"] == {
        "mode": "disabled"
    }
    assert simulation.execution.runtime_path == runtime_path
    assert compile_simulation(simulation).application == expected_application


def _report(variant: str, application: str, memory: int = 100) -> dict[str, object]:
    runtime = {
        "requested": f"{application}_runtime",
        "effective": f"{application}_runtime",
        "fallback_used": False,
    }
    result = {
        "final_step": FROZEN_TOTAL_STEPS,
        "complete": True,
        "runtime_selection": runtime,
    }
    return {
        "kind": KIND,
        "variant": variant,
        "application": application,
        "passed": True,
        "all_byte_identical": True,
        "all_finite": True,
        "contract": {
            "shape": list(FROZEN_SHAPE),
            "initial_q_sha256": FROZEN_INPUT_SHA256,
            "spectral_refresh": "disabled",
        },
        "continuous": dict(result),
        "segment": {**result, "final_step": 50},
        "resumed": dict(result),
        "environment": {
            "source_shadow_import": False,
            "python_prefix": "/venv",
            "purelib": "/venv/lib/python3.10/site-packages",
            "pssolver_file": "/venv/lib/python3.10/site-packages/pssolver/__init__.py",
            "tf32_matmul": False,
            "tf32_cudnn": False,
        },
        "memory": {
            "peak_allocated_bytes": memory,
            "peak_reserved_bytes": memory,
        },
        "timing": {"continuous_mean_timestep_seconds": 1.0},
        "repository": {"head": variant},
    }


def _reports(candidate_memory: int = 104):
    return {
        (variant, application): _report(
            variant,
            application,
            100 if variant == "baseline" else candidate_memory,
        )
        for variant in ("baseline", "candidate")
        for application in ("periodic", "channel")
    }


def test_g5_analyzer_accepts_complete_matrix_at_frozen_memory_gate():
    result = analyze(_reports(candidate_memory=105))

    assert result["passed"] is True
    assert result["ratios"]["periodic"]["peak_allocated_bytes_ratio"] == 1.05


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["contract"].update(spectral_refresh="enabled"), "refresh"),
        (lambda value: value.update(all_byte_identical=False), "restart"),
        (
            lambda value: value["continuous"]["runtime_selection"].update(
                fallback_used=True
            ),
            "fallback",
        ),
    ),
)
def test_g5_analyzer_rejects_contract_regressions(mutation, message):
    reports = _reports()
    value = deepcopy(reports[("candidate", "periodic")])
    mutation(value)
    reports[("candidate", "periodic")] = value

    with pytest.raises(RuntimeError, match=message):
        analyze(reports)


def test_g5_analyzer_rejects_memory_regression():
    with pytest.raises(RuntimeError, match="ratio exceeds"):
        analyze(_reports(candidate_memory=106))
