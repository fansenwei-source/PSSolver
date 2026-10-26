"""Static and synthetic gates for the installed-wheel S-09 CPU rehearsal."""

from __future__ import annotations

import ast
from pathlib import Path

from benchmarks.run_v0_3_s09_cpu_rehearsal import STAGES


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_rehearsal_represents_every_formal_job_stage_in_order() -> None:
    assert STAGES == (
        "installed_wheel_environment",
        "cuda_device_identity_assertions",
        "installed_wheel_cuda_only_tests",
        "cpu_short_horizon_hermitian",
        "tf32_forward_backward_scope",
        "checkpoint_restart_and_released_fixture_matrix",
        "numerical_gradient_taylor_and_replay",
        "independent_control_consumer",
        "formal_profile_matrix",
        "assembler_and_analyzer_import",
    )


def test_rehearsal_is_wheel_only_and_records_record_summaries() -> None:
    source = (ROOT / "benchmarks/run_v0_3_s09_cpu_rehearsal.py").read_text()
    tree = ast.parse(source)
    assert "-I" in source
    assert "pssolver_record" in source
    assert "pssolver_control_record" in source
    assert "site-packages" in source
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pssolver")
        for node in ast.walk(tree)
    )


def test_rehearsal_skips_only_cuda_specific_or_formal_work() -> None:
    source = (ROOT / "benchmarks/run_v0_3_s09_cpu_rehearsal.py").read_text()
    for exact in (
        "CUDA device identity assertions",
        "CUDA-only test execution",
        "cross-device checkpoint cells",
        "formal 192-profile execution",
    ):
        assert exact in source
    assert '"--device", "cpu"' in source
    assert '"--skip-cross-device"' in source
    assert '"--plan-only"' in source


def test_v39_and_v40_recovery_are_exercised_by_cpu_paths() -> None:
    tf32 = (ROOT / "benchmarks/run_v0_3_s09_tf32_backward.py").read_text()
    consumer = (ROOT / "benchmarks/run_v0_3_s09_consumer_h100.py").read_text()
    assert "root.mkdir(parents=True, exist_ok=False)" in tf32
    assert "requires_grad=True" in consumer
    assert "value.abs().square().mean()" in consumer
    assert "channel_fixed" in consumer and "channel_tolerance" in consumer
    assert "execution_identities" in consumer


def test_cuda_only_stage_is_wired_to_a_real_junit_file() -> None:
    job = (ROOT / "jobs/v0_3_s09_single_h100.sbatch").read_text()
    wrapper = (ROOT / "benchmarks/run_v0_3_s09_cuda_only.py").read_text()
    assert '--junit "${S09_OUTPUT_ROOT}/logs/cuda_only.junit.xml"' in job
    assert 'parser.add_argument("--junit", type=Path, required=True)' in wrapper
    assert 'os.environ["PYTEST_ADDOPTS"]' in wrapper
    assert 'f"--junitxml={args.junit.expanduser().absolute()}"' in wrapper
