"""Local contracts for the RC4.2.6 external qualification harness."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.analyze_rc426_cross_device_checkpoint import analyze
from benchmarks.run_rc426_cross_device_checkpoint import (
    COMPLETE_SCHEMA,
    MATRIX,
    PASS_CLASSIFICATION,
    SCHEMA,
    MatrixCell,
    _atomic_json,
    _run_channel,
    _run_functional,
    _run_periodic,
    _run_plane,
    matrix_plan,
)


def _report(cell) -> dict[str, object]:
    source = "cuda:0" if cell.source == "cuda" else "cpu"
    target = "cuda:0" if cell.target == "cuda" else "cpu"
    return {
        "schema": SCHEMA,
        "cell": cell.to_metadata(),
        "checkpoint": {
            "ordinary_non_symlink_directory": True,
            "source_tree_unchanged": True,
            "source_tree_sha256": {"checkpoint.json": "1" * 64},
            "source_payload_sha256": {"state.npy": "2" * 64},
            "restored_payload_sha256": {"state.npy": "2" * 64},
            "compatibility_identity_source": "a" * 64,
            "compatibility_identity_target": "a" * 64,
            "compatibility_identity_equal": True,
            "negative_integrity_gate": {
                "rejected": True,
                "target_unchanged": True,
                "integrity_guard_reached": True,
                "exception_type": "ValueError",
                "message": "checksum mismatch",
                "tampered_file": "state.npy",
            },
        },
        "progress": {
            "expected_completed_steps": 1,
            "restored_completed_steps": 1,
            "exact": True,
            "source_serialized": {"completed_steps": 1},
            "restored_serialized": {"completed_steps": 1},
            "serialized_equal": True,
        },
        "state": {
            "source_devices": [source],
            "target_devices": [target],
            "serialized_payload_equal": True,
            "persistent_backend_state_equal": True,
            "restored_tensor_sha256": "b" * 64,
            "restored_finite": True,
            "post_restore_step_finite": True,
        },
        "runtime": {"fallback_used": False},
        "passed": True,
    }


def _completion() -> dict[str, object]:
    return {
        "schema": COMPLETE_SCHEMA,
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
        "cell_count": 14,
        "ordered_cell_ids": [cell.id for cell in MATRIX],
    }


def test_frozen_matrix_is_exact_and_bidirectional():
    plan = matrix_plan()
    assert plan["count"] == 14
    assert [cell["id"] for cell in plan["cells"]] == [
        f"X{index:02d}" for index in range(1, 15)
    ]
    assert [cell["sequence"] for cell in plan["cells"]] == list(range(1, 15))
    for first, second in zip(MATRIX[::2], MATRIX[1::2], strict=True):
        assert first.family == second.family
        assert first.runtime_path == second.runtime_path
        assert (first.source, first.target) == ("cpu", "cuda")
        assert (second.source, second.target) == ("cuda", "cpu")


def test_plan_claim_boundary_does_not_expand_to_trajectory_or_performance():
    boundary = matrix_plan()["claim_boundary"]
    assert boundary == {
        "exact_serialized_restore": True,
        "finite_post_restore_step": True,
        "post_restore_cross_device_byte_identity": False,
        "performance_or_memory": False,
        "long_run": False,
    }


def test_atomic_json_refuses_overwrite(tmp_path):
    path = tmp_path / "record.json"
    _atomic_json(path, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _atomic_json(path, {"value": 2})
    assert path.read_text(encoding="utf-8") == '{\n  "value": 1\n}\n'


@pytest.mark.parametrize(
    ("runtime_path", "executor"),
    (
        ("legacy_production", _run_plane),
        ("compiled_v2", _run_plane),
        ("separated_canary", _run_plane),
    ),
)
def test_plane_cell_protocol_executes_on_cpu(
    tmp_path,
    runtime_path,
    executor,
):
    cell = MatrixCell(
        1,
        "CPU",
        "plane_production",
        runtime_path,
        "cpu",
        "cpu",
    )
    report = executor(cell, tmp_path / runtime_path)
    assert report["passed"] is True
    assert report["checkpoint"]["negative_integrity_gate"]["rejected"] is True


@pytest.mark.parametrize(
    ("family", "runtime_path", "executor"),
    (
        ("periodic_production", "periodic_spectral", _run_periodic),
        ("channel_production", "channel_complete_stress", _run_channel),
        ("periodic_functional", "periodic_activity_batch_one", _run_functional),
        ("channel_functional", "channel_activity_batch_one", _run_functional),
    ),
)
def test_periodic_channel_cell_protocol_executes_on_cpu(
    tmp_path,
    family,
    runtime_path,
    executor,
):
    cell = MatrixCell(1, "CPU", family, runtime_path, "cpu", "cpu")
    report = executor(cell, tmp_path / family)
    assert report["passed"] is True
    assert report["state"]["serialized_payload_equal"] is True
    assert report["state"]["post_restore_step_finite"] is True


def test_analyzer_accepts_only_the_complete_ordered_matrix():
    result = analyze(matrix_plan(), [_report(cell) for cell in MATRIX], _completion())
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["qualification_complete"] is True
    assert result["matrix"]["directions"] == {
        "cpu_to_cuda": 7,
        "cuda_to_cpu": 7,
    }


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        ((0, "passed"), False, "X01 did not pass"),
        (
            (0, "checkpoint", "compatibility_identity_equal"),
            False,
            "X01 compatibility_identity_equal failed",
        ),
        (
            (0, "checkpoint", "negative_integrity_gate", "target_unchanged"),
            False,
            "target mutated before rejection",
        ),
        (
            (0, "state", "serialized_payload_equal"),
            False,
            "serialized tensors differ",
        ),
        ((0, "runtime", "fallback_used"), True, "X01 fallback"),
    ),
)
def test_analyzer_rejects_failed_subgates(path, value, match):
    reports = [_report(cell) for cell in MATRIX]
    target: object = reports
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=match):
        analyze(matrix_plan(), reports, _completion())


def test_analyzer_rejects_reordered_or_duplicate_cells():
    reports = [_report(cell) for cell in MATRIX]
    reports[0], reports[1] = reports[1], reports[0]
    with pytest.raises(ValueError, match="X01 identity differs"):
        analyze(matrix_plan(), reports, _completion())


def test_analyzer_rejects_scope_expansion_and_incomplete_marker():
    plan = copy.deepcopy(matrix_plan())
    plan["claim_boundary"]["performance_or_memory"] = True
    with pytest.raises(ValueError, match="scope expanded"):
        analyze(plan, [_report(cell) for cell in MATRIX], _completion())
    completion = _completion()
    completion["qualification_complete"] = False
    with pytest.raises(ValueError, match="matrix incomplete"):
        analyze(
            matrix_plan(),
            [_report(cell) for cell in MATRIX],
            completion,
        )


def test_harness_does_not_mutate_runtime_or_checkpoint_sources():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "benchmarks" / "run_rc426_cross_device_checkpoint.py").read_text()
    analyzer = (
        root / "benchmarks" / "analyze_rc426_cross_device_checkpoint.py"
    ).read_text()
    assert "source.copy_(" not in runner
    assert "MEAN_RATIO_LIMIT" not in analyzer
    assert "MEMORY_RATIO_LIMIT" not in analyzer
    assert "automatic_retry" not in runner


def test_single_h100_job_harness_freezes_scheduler_and_execution_order():
    root = Path(__file__).resolve().parents[1]
    script = (root / "jobs" / "rc426_cross_device_checkpoint_h100.sbatch").read_text()
    for directive in (
        "#SBATCH --partition=hagan-gpu",
        "#SBATCH --account=hagan-lab",
        "#SBATCH --qos=medium",
        "#SBATCH --gres=gpu:H100:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --time=01:00:00",
        "#SBATCH --no-requeue",
    ):
        assert script.count(directive) == 1
    assert "sbatch" not in script
    assert "srun" not in script
    assert script.index("capture_rc4_h100_preflight.py") < script.index(
        "run_rc426_cuda_tests.py"
    ) < script.index("run_rc426_cross_device_checkpoint.py") < script.index(
        "analyze_rc426_cross_device_checkpoint.py"
    )
    assert "unset LD_LIBRARY_PATH" in script
    assert "NVIDIA_TF32_OVERRIDE=0" in script
    assert "TRITON_CACHE_DIR" in script
    assert "TORCHINDUCTOR_CACHE_DIR" in script


def test_job_harness_runs_the_exact_seven_cuda_only_nodes_once():
    root = Path(__file__).resolve().parents[1]
    script = (root / "jobs" / "rc426_cross_device_checkpoint_h100.sbatch").read_text()
    expected = (
        "test_bounded_axis_execution_plan.py::test_cuda_plan_key_uses_the_allocated_device_identity",
        "test_benchmark_bounded_axis_applicability.py::test_cuda_smoke_records_actual_device_and_memory",
        "test_phase4_combined_sbdf2_modal_block.py::test_cuda_canary_binds_concrete_device_and_remains_finite",
        "test_phase8_p845_h100_qualification.py::test_unindexed_cuda_request_binds_lifting_to_allocated_device_identity",
        "test_phase8_p856_finite_q_h100_qualification.py::test_cuda_runtime_matches_cpu_and_binds_allocated_device",
        "test_phase8_p856_finite_q_h100_qualification.py::test_cuda_continuous_split_and_file_restart_are_exact",
        "test_rc4_plane_nyquist_cuda.py::test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence",
    )
    assert all(script.count(node) == 1 for node in expected)
    assert script.count("  --node ") == 7
    assert script.count("--expected 7") == 1


def test_execution_support_record_binds_exact_files_and_preserves_scope():
    root = Path(__file__).resolve().parents[1]
    record = json.loads(
        (
            root
            / "notes"
            / "PSSolver_v0_2_0rc4_rc426_execution_support.json"
        ).read_text(encoding="utf-8")
    )
    assert record["classification"] == (
        "READY_RC4_2_6_SINGLE_H100_EXECUTION_NOT_SUBMITTED"
    )
    assert record["matrix_contract"]["ordered_cell_ids"] == [
        f"X{index:02d}" for index in range(1, 15)
    ]
    for item in record["support_files"].values():
        path = root / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert record["scope"] == {
        "qualification_support_only": True,
        "runtime_source_modified": False,
        "checkpoint_reader_or_writer_modified": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "H100_executed": False,
        "slurm_submitted": False,
        "push_performed": False,
    }
    assert record["authorization"]["H100_submission_authorized"] is False
