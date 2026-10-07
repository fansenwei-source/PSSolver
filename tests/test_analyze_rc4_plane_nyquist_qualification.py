from __future__ import annotations

import json

import pytest

from benchmarks.analyze_rc4_plane_nyquist_qualification import (
    FROZEN_INPUT_PATHS,
    GRIDS,
    TRIALS,
    analyze,
    write_json_atomic,
)


BASELINE = "a" * 40
CANDIDATE = "b" * 40


def _profile(grid: str, trial: int, role: str, ratio: float = 1.0):
    shape = GRIDS[grid]
    commit = BASELINE if role == "baseline" else CANDIDATE
    base_time = (0.007 if grid == "R128" else 0.045) * (1 + trial * 0.001)
    timestep = base_time * (ratio if role == "candidate" else 1.0)
    return {
        "schema_version": 1,
        "config": {
            "runtime_path": "legacy_production",
            "trial": trial,
            "shape": list(shape),
            "lengths": [100.0, 100.0, 20.0],
            "device": "cuda",
            "dtype": "float64",
            "dt": 0.005,
            "activity_number": 18.0,
            "dealias_rule": "cubic_half",
            "projected_transform_execution": "truncated",
            "transform_execution_order": "real_first",
            "spectral_storage": "hermitian_half",
            "molecular_field_linear_space": "spectral",
            "stress_divergence_sum_space": "spectral",
            "pointwise_execution": "compile",
            "reuse_q_gradients": True,
            "spectral_refresh_interval": None,
            "warmup_steps": 10,
            "profile_steps": 50,
            "seed": 24,
            "initial_q_path": FROZEN_INPUT_PATHS[grid],
        },
        "runtime_identity": {
            "requested": "legacy_production",
            "effective": "legacy_production",
            "fallback_used": False,
        },
        "environment": {
            "git": {"head": commit, "dirty": False, "status": ""},
            "device": "cuda",
            "device_name": "NVIDIA H100 PCIe",
            "cuda_matmul_allow_tf32": False,
        },
        "throughput": {"mean_timestep_seconds": timestep},
        "transform_calls": {"forward_per_step": 7.0, "inverse_per_step": 32.0},
        "memory": {
            "peak_allocated_bytes": 1020 if role == "candidate" else 1000,
            "peak_reserved_bytes": 2020 if role == "candidate" else 2000,
        },
        "pointwise_compile": {
            phase: {"graph_breaks": 0}
            for phase in (
                "dynamo_during_build",
                "dynamo_during_warmup",
                "dynamo_during_profile",
            )
        },
        "initial_q_sha256": ("1" if grid == "R128" else "2") * 64,
        "final_state_sha256": (
            ("3" if grid == "R128" else "4")
            if role == "baseline"
            else ("5" if grid == "R128" else "6")
        )
        * 64,
        "completed_steps": 60,
        "finite": True,
    }


def _profiles(role: str, ratio: float = 1.01):
    return [
        _profile(grid, trial, role, ratio)
        for grid in GRIDS
        for trial in TRIALS
    ]


def _cuda_report():
    return {
        "schema": "pssolver.rc4_1_1.plane_nyquist_storage_diagnostic.v2",
        "classification": "PASS_PLANE_PERIODIC_NYQUIST_STORAGE_EQUIVALENCE",
        "configuration": {"device": "cuda"},
        "environment": {
            "allocated_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "cases": [
            {
                "name": f"case_{index}",
                "raw": {
                    "full_complex": {"finite": True},
                    "hermitian_half": {"finite": True},
                },
                "summary": {
                    "raw_max_velocity_relative_l2": 4.0e-16,
                    "raw_pressure_relative_l2": 5.0e-16,
                },
            }
            for index in range(5)
        ],
    }


def _analyze(baseline=None, candidate=None, cuda=None):
    return analyze(
        _profiles("baseline") if baseline is None else baseline,
        _profiles("candidate") if candidate is None else candidate,
        _cuda_report() if cuda is None else cuda,
        baseline_commit=BASELINE,
        candidate_commit=CANDIDATE,
    )


def test_analyzer_accepts_complete_correctness_and_non_regression_evidence():
    result = _analyze()

    assert result["classification"] == (
        "PASS_RC4_1_2_PLANE_NYQUIST_SINGLE_H100_NON_REGRESSION"
    )
    assert result["qualification_complete"] is True
    assert set(result["grids"]) == {"R128", "R320"}
    assert result["production_default_changed"] is False
    assert result["pssolver_control_modified"] is False
    assert result["nematics3d_modified"] is False


def test_analyzer_rejects_incomplete_or_duplicate_profile_matrix():
    candidate = _profiles("candidate")
    with pytest.raises(ValueError, match="incomplete"):
        _analyze(candidate=candidate[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        _analyze(candidate=[*candidate, candidate[0]])


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda report: report["environment"].update(allocated_device="cuda:1"), "allocated CUDA"),
        (lambda report: report["cases"][0]["raw"]["full_complex"].update(finite=False), "not finite"),
        (lambda report: report["cases"][0]["summary"].update(raw_pressure_relative_l2=1.0e-6), "pressure storage"),
    ),
)
def test_analyzer_rejects_invalid_cuda_correctness_evidence(mutation, message):
    report = _cuda_report()
    mutation(report)
    with pytest.raises(ValueError, match=message):
        _analyze(cuda=report)


def test_analyzer_rejects_performance_memory_and_graph_regressions():
    candidate = _profiles("candidate")
    candidate[0]["throughput"]["mean_timestep_seconds"] *= 1.05
    with pytest.raises(ValueError, match="paired timestep"):
        _analyze(candidate=candidate)

    candidate = _profiles("candidate")
    candidate[0]["memory"]["peak_allocated_bytes"] = 2000
    with pytest.raises(ValueError, match="allocated-memory"):
        _analyze(candidate=candidate)

    candidate = _profiles("candidate")
    candidate[0]["pointwise_compile"]["dynamo_during_profile"]["graph_breaks"] = 1
    with pytest.raises(ValueError, match="graph break"):
        _analyze(candidate=candidate)


def test_analyzer_rejects_initial_identity_or_role_determinism_drift():
    candidate = _profiles("candidate")
    candidate[0]["initial_q_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="initial identity mismatch"):
        _analyze(candidate=candidate)

    candidate = _profiles("candidate")
    candidate[0]["final_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="nondeterministic"):
        _analyze(candidate=candidate)


def test_atomic_writer_refuses_non_json_values_and_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "analysis.json"
    result = _analyze()
    write_json_atomic(output, result)

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert list(tmp_path.iterdir()) == [output]

    with pytest.raises(ValueError):
        write_json_atomic(tmp_path / "bad.json", {"value": float("nan")})
    assert sorted(path.name for path in tmp_path.iterdir()) == ["analysis.json"]
