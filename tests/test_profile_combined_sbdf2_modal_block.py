"""P4.6 combined-canary profiler and immutable-evidence validator tests."""

from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

import pytest
from setuptools import find_packages

from benchmarks import profile_combined_sbdf2_modal_block as profiler
from benchmarks import validate_phase4_closure as validator


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "a" * 40
RECORD = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_4_p46_closure_qualification.json"
)


def _memory(peak_allocated: int, peak_reserved: int) -> dict[str, int]:
    return {
        "starting_allocated_bytes": 100,
        "starting_reserved_bytes": 140,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": peak_allocated - 100,
        "incremental_peak_reserved_bytes": peak_reserved - 140,
    }


def _records(
    values: list[float],
    orders: list[int],
    *,
    peak_allocated: int,
    peak_reserved: int,
) -> list[dict[str, object]]:
    return [
        {
            "trial": index + 1,
            "order": orders[index],
            "milliseconds_per_step": value,
            "memory": _memory(peak_allocated, peak_reserved),
            "workspace_pointer_stable": True,
            "finite": True,
            "workspace_allocated_tensor_count": 7,
            "implementation": "closed_form_2x2",
        }
        for index, value in enumerate(values)
    ]


def _aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "trials": 3,
        "mean_milliseconds_per_step": statistics.fmean(values),
        "median_milliseconds_per_step": statistics.median(values),
        "sample_standard_deviation_milliseconds_per_step": statistics.stdev(
            values
        ),
    }


def _row(point_count: int) -> dict[str, object]:
    continuous = [1.0, 1.1, 1.0]
    rebound = [1.02, 1.08, 1.01]
    return {
        "point_count": point_count,
        "trial_records": {
            profiler.CONTINUOUS: _records(
                continuous,
                [1, 2, 1],
                peak_allocated=120,
                peak_reserved=160,
            ),
            profiler.REBOUND: _records(
                rebound,
                [2, 1, 2],
                peak_allocated=121,
                peak_reserved=161,
            ),
        },
        "comparisons": [
            {
                "trial": trial,
                "physical_byte_identical": True,
                "spectrum_byte_identical": True,
                "physical_relative_l2": 0.0,
                "physical_linf": 0.0,
            }
            for trial in (1, 2, 3)
        ],
        "aggregate": {
            profiler.CONTINUOUS: _aggregate(continuous),
            profiler.REBOUND: _aggregate(rebound),
        },
        "rebound_over_continuous_median_ratio": (
            statistics.median(rebound) / statistics.median(continuous)
        ),
        "paired_rebound_over_continuous_ratios": [
            right / left
            for left, right in zip(continuous, rebound, strict=True)
        ],
    }


def _evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "identity": "p4_6_combined_sbdf2_modal_block_profile",
        "scope": "qualification_only_no_production_selection",
        "config": {
            "point_counts": [131072, 1048576],
            "device": "cuda",
            "dtype": "float64",
            "dt": 0.005,
            "warmup_steps": 5,
            "measured_steps": 30,
            "trials": 3,
            "maximum_timing_ratio": 1.10,
            "maximum_memory_ratio": 1.10,
        },
        "environment": {
            "git_head": EXPECTED_COMMIT,
            "cuda_available": True,
            "requested_device": "cuda",
            "concrete_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_matmul_effective": False,
            "tf32_cudnn_effective": False,
        },
        "rows": [_row(131072), _row(1048576)],
        "convergence": {
            "passed": True,
            "minimum_observed_l2_order": 2.001,
            "pairwise_l2_orders": [2.01, 2.005, 2.001],
        },
        "eligibility": {
            "profile_complete": True,
            "production_default_changed": False,
            "phase_5_authorized": False,
        },
    }


def test_cpu_profile_executes_balanced_roles_and_convergence():
    result = profiler.run_profile(
        profiler.CombinedCanaryProfileConfig(
            point_counts=(32, 64),
            device="cpu",
            warmup_steps=2,
            measured_steps=3,
            trials=2,
        )
    )
    assert result["identity"] == "p4_6_combined_sbdf2_modal_block_profile"
    assert result["convergence"]["passed"] is True
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert all(
            comparison["physical_byte_identical"]
            and comparison["spectrum_byte_identical"]
            for comparison in row["comparisons"]
        )
        for variant in profiler.VARIANTS:
            assert all(
                record["workspace_pointer_stable"]
                and record["workspace_allocated_tensor_count"] == 7
                for record in row["trial_records"][variant]
            )


@pytest.mark.parametrize(
    "updates",
    [
        {"point_counts": (31,)},
        {"device": "tpu"},
        {"dtype": "float16"},
        {"dt": 0.0},
        {"warmup_steps": 0},
        {"measured_steps": 0},
        {"trials": 0},
        {"maximum_timing_ratio": 0.9},
        {"maximum_memory_ratio": float("nan")},
    ],
)
def test_profile_config_rejects_invalid_inputs(updates):
    values = {
        "point_counts": (32,),
        "device": "cpu",
        "dtype": "float64",
        "dt": 0.005,
        "warmup_steps": 1,
        "measured_steps": 1,
        "trials": 1,
        "maximum_timing_ratio": 1.10,
        "maximum_memory_ratio": 1.10,
    }
    values.update(updates)
    with pytest.raises(ValueError):
        profiler.run_profile(profiler.CombinedCanaryProfileConfig(**values))


def test_validator_accepts_complete_h100_evidence():
    result = validator.validate_evidence(
        _evidence(),
        expected_commit=EXPECTED_COMMIT,
    )
    assert result["classification"] == "PASS_P4_6_H100_PROFILE"
    assert result["gates"] == {
        "h100_identity": True,
        "tf32_disabled": True,
        "finite": True,
        "workspace_identity": True,
        "workspace_tensor_count": 7,
        "continuous_rebound_byte_identity": True,
        "performance": True,
        "memory": True,
        "convergence": True,
    }
    assert result["eligibility"]["phase_4_complete"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["environment"].__setitem__(
                "git_head", "b" * 40
            ),
            "git head",
        ),
        (
            lambda value: value["environment"].__setitem__(
                "device_name", "NVIDIA L40"
            ),
            "H100",
        ),
        (
            lambda value: value["rows"][0]["comparisons"][0].__setitem__(
                "physical_byte_identical", False
            ),
            "physical identity",
        ),
        (
            lambda value: value["rows"][0]["trial_records"][profiler.REBOUND][
                0
            ].__setitem__("workspace_allocated_tensor_count", 8),
            "workspace count",
        ),
        (
            lambda value: value["rows"][0]["trial_records"][profiler.REBOUND][
                0
            ]["memory"].__setitem__("peak_allocated_bytes", 200),
            "incremental allocated",
        ),
        (
            lambda value: value["convergence"].__setitem__(
                "minimum_observed_l2_order", 1.7
            ),
            "below 1.8",
        ),
    ],
)
def test_validator_rejects_tampered_evidence(mutate, message):
    evidence = copy.deepcopy(_evidence())
    mutate(evidence)
    with pytest.raises(validator.ClosureEvidenceError, match=message):
        validator.validate_evidence(evidence, expected_commit=EXPECTED_COMMIT)


def test_loader_rejects_duplicate_and_nonfinite_json(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}\n')
    with pytest.raises(validator.ClosureEvidenceError, match="duplicate"):
        validator.load_evidence(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n')
    with pytest.raises(validator.ClosureEvidenceError, match="non-finite"):
        validator.load_evidence(nonfinite)


def test_profile_cli_refuses_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_text("preserve\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile", "--device", "cpu", "--output", str(output)],
    )
    with pytest.raises(SystemExit):
        profiler.parse_args()
    assert output.read_text() == "preserve\n"


def test_wheel_package_discovery_includes_opt_in_canary_modules():
    packages = set(find_packages(where=str(ROOT)))
    assert "pssolver.experimental" in packages
    assert (ROOT / "pssolver/experimental/modal_block_reference.py").is_file()
    assert (ROOT / "pssolver/experimental/modal_block_checkpoint.py").is_file()


def test_p46_machine_record_freezes_local_and_h100_closure_contract():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["status"] == "P4_6_LOCAL_COMPLETE_H100_CLOSURE_PENDING"
    assert record["baseline_commit"] == (
        "9b4b0463791e1e71427087937c0002954425fd15"
    )
    assert record["h100_gate"] == {
        "single_job": True,
        "automatic_retry": False,
        "dtype": "float64",
        "tf32": False,
        "point_counts": [131072, 1048576],
        "dt": 0.005,
        "warmup_steps": 5,
        "measured_steps": 30,
        "trials": 3,
        "balanced_order": True,
        "maximum_rebound_over_continuous_median_ratio": 1.1,
        "maximum_peak_allocated_ratio": 1.1,
        "maximum_peak_reserved_ratio": 1.1,
        "workspace_allocated_tensor_count": 7,
        "byte_identical_state_required": True,
        "minimum_pairwise_l2_order": 1.8,
        "cuda_only_test": (
            "tests/test_phase4_combined_sbdf2_modal_block.py::"
            "test_cuda_canary_binds_concrete_device_and_remains_finite"
        ),
    }
    assert record["eligibility"] == {
        "p4_6_local_complete": True,
        "h100_closure_pending": True,
        "phase_4_complete": False,
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
