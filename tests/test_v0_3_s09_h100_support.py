"""Static and synthetic gates for the S-09 H100 support bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_v0_3_s09_long_horizon import CASES
from benchmarks.run_v0_3_s09_long_horizon import summarize as summarize_long


ROOT = Path(__file__).resolve().parents[1]


JOB = ROOT / "jobs/v0_3_s09_single_h100.sbatch"


def test_job_orders_correctness_and_consumer_before_profiles() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue" in text
    assert "set -euo pipefail" in text
    assert "--gres=gpu:H100:1" in text
    profile = text.index("run_v0_3_s09_profile_matrix.py")
    for helper in (
        "run_v0_3_s09_cuda_only.py",
        "run_v0_3_s09_long_horizon.py",
        "run_v0_3_s09_tf32_backward.py",
        "run_v0_3_s09_checkpoints.py",
        "run_v0_3_s09_numerical.py",
        "run_v0_3_s09_consumer_h100.py",
    ):
        assert text.index(helper) < profile
    assert text.index("analyze_v0_3_s09_h100.py") > profile
    assert "sha256sum --check checksums.sha256" in text
    assert text.index("sha256sum --check checksums.sha256") < text.index("> COMPLETE")
    assert "trap 'rm -rf \"${node_root}\"' EXIT" in text
    assert '--junit "${S09_OUTPUT_ROOT}/logs/cuda_only.junit.xml"' in text


def test_long_horizon_summary_is_fail_closed() -> None:
    reports = []
    for case, storage in CASES:
        reports.append(
            {
                "config": {
                    "case": case,
                    "spectral_storage": storage,
                    "horizon": 200.0,
                    "device": "cuda",
                },
                "completed_time": 200.0,
                "passed": True,
                "samples": [
                    {"finite": True, "hermitian_violation": 0.0, "violation_bound": 1.0e-12}
                ],
            }
        )
    assert summarize_long(reports)["passed"] is True
    reports[0]["samples"][0]["finite"] = False
    with pytest.raises(ValueError, match="finite/Hermitian"):
        summarize_long(reports)


def test_plan_and_job_bind_the_same_auxiliary_stage_order() -> None:
    plan = json.loads((ROOT / "notes/PSSolver_v0_3_0_s09_h100_plan.json").read_text())
    text = JOB.read_text(encoding="utf-8")
    assert plan["job_stage_order"] == [
        "node_local_tmp_environment_and_host_compiler",
        "installed_wheel_cuda_only_tests",
        "cuda_long_horizon_hermitian",
        "tf32_forward_backward_scope",
        "checkpoint_restart_and_released_fixture_matrix",
        "numerical_gradient_taylor_and_replay",
        "independent_control_consumer",
        "paired_performance_memory_profiles",
        "cumulative_analyzer_and_checksums",
    ]
    assert text.index("run_v0_3_s09_consumer_h100.py") < text.index(
        "run_v0_3_s09_profile_matrix.py"
    )
