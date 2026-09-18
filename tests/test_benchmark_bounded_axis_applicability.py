"""Tests for the bounded-axis dense/full-FFT applicability benchmark."""

from __future__ import annotations

import csv
import hashlib
import json

import pytest
import torch

import benchmarks.benchmark_bounded_axis_applicability as applicability
from benchmarks.benchmark_bounded_axis_applicability import (
    ALGORITHMS,
    ApplicabilityConfig,
    _retained_count,
    parse_args,
    run_benchmark,
    write_artifacts,
)


def _small_config(**overrides) -> ApplicabilityConfig:
    values = {
        "sizes": (3, 4),
        "retained_fractions": (0.5, 1.0),
        "line_counts": (2,),
        "kinds": ("dct", "dst"),
        "directions": ("forward", "inverse"),
        "dtypes": ("float64",),
        "value_types": ("real", "complex"),
        "device": "cpu",
        "warmup": 0,
        "repeats": 2,
        "trials": 1,
        "seed": 19,
    }
    values.update(overrides)
    return ApplicabilityConfig(**values)


def test_small_sweep_is_complete_finite_and_json_safe():
    result = run_benchmark(_small_config())

    expected_cases = 2 * 2 * 1 * 2 * 2 * 1 * 2
    assert result["schema_version"] == 1
    assert result["summary"]["logical_case_count"] == expected_cases
    assert result["summary"]["algorithm_case_count"] == (
        expected_cases * len(ALGORITHMS)
    )
    assert len(result["cases"]) == expected_cases
    assert result["scope"] == {
        "microbenchmark_only": True,
        "production_solver_imports_full_fft_reference": False,
        "production_default_changed": False,
        "production_cli_changed": False,
        "basis_or_normalization_changed": False,
        "pruned_algorithm_present": False,
        "full_fft_reference_previously_rejected_for_production": True,
    }

    case_ids = {case["case_id"] for case in result["cases"]}
    assert len(case_ids) == expected_cases
    for case in result["cases"]:
        assert set(case["aggregate"]) == set(ALGORITHMS)
        assert set(case["records"]) == set(ALGORITHMS)
        assert set(case["correctness"]) == set(ALGORITHMS)
        assert case["full_fft_reference_semantics"]["is_pruned"] is False
        assert len(case["comparison"]["paired_trial_speedups"]) == 1
        for algorithm in ALGORITHMS:
            correctness = case["correctness"][algorithm]
            assert correctness["all_finite"] is True
            assert correctness["dense_reference_relative_l2"] <= (
                correctness["tolerance_relative_l2"]
            )
            assert correctness["coefficient_identity_relative_l2"] <= (
                correctness["tolerance_relative_l2"]
            )
            if case["retained_count"] == case["physical_size"]:
                assert correctness["full_roundtrip_relative_l2"] <= (
                    correctness["tolerance_relative_l2"]
                )
            else:
                assert correctness["full_roundtrip_relative_l2"] is None
            aggregate = case["aggregate"][algorithm]
            assert aggregate["steady"]["samples"] == 2
            assert aggregate["cold_wall"]["samples"] == 1
            assert aggregate["memory"][
                "persistent_cache_unique_bytes_max"
            ] > 0

    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("value_type", ("real", "complex"))
def test_full_fft_reference_matches_dense_across_dtype_and_value_type(
    dtype,
    value_type,
):
    result = run_benchmark(
        _small_config(
            sizes=(5, 8),
            retained_fractions=(0.5, 1.0),
            kinds=("dct", "dst"),
            directions=("forward", "inverse"),
            dtypes=(dtype,),
            value_types=(value_type,),
            repeats=1,
        )
    )

    for case in result["cases"]:
        candidate = case["correctness"]["full_fft_reference"]
        assert candidate["dense_reference_relative_l2"] <= candidate[
            "tolerance_relative_l2"
        ]
        assert candidate["coefficient_identity_relative_l2"] <= candidate[
            "tolerance_relative_l2"
        ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"sizes": ()}, "sizes"),
        ({"sizes": (4, 4)}, "sizes must not contain duplicates"),
        ({"retained_fractions": (0.0,)}, "retained_fractions"),
        (
            {"sizes": (3,), "retained_fractions": (0.25, 0.5)},
            "duplicate retained counts",
        ),
        ({"line_counts": (0,)}, "line_counts"),
        ({"kinds": ("fft",)}, "kinds"),
        ({"directions": ("sideways",)}, "directions"),
        ({"dtypes": ("float16",)}, "dtypes"),
        ({"value_types": ("integer",)}, "value_types"),
        ({"device": "tpu"}, "device"),
        ({"warmup": -1}, "warmup"),
        ({"repeats": 0}, "repeats"),
        ({"trials": 0}, "trials"),
        ({"tf32": "automatic"}, "tf32"),
        ({"require_device_name": ""}, "require_device_name"),
    ),
)
def test_invalid_sweep_contracts_fail_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        run_benchmark(_small_config(**overrides))


def test_retained_count_uses_a_nonempty_floor_prefix():
    assert _retained_count(20, 1.0 / 16.0) == 1
    assert _retained_count(20, 2.0 / 3.0) == 13
    assert _retained_count(20, 1.0) == 20


def test_json_and_csv_artifacts_have_identical_case_identity(tmp_path):
    result = run_benchmark(
        _small_config(
            sizes=(4,),
            retained_fractions=(0.5, 1.0),
            kinds=("dct",),
            directions=("forward",),
            value_types=("real",),
            repeats=1,
        )
    )
    json_path = tmp_path / "map.json"
    csv_path = tmp_path / "map.csv"
    write_artifacts(
        result,
        json_output=json_path,
        csv_output=csv_path,
        overwrite=False,
    )

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_case_ids = {case["case_id"] for case in loaded["cases"]}
    assert {row["case_id"] for row in rows} == expected_case_ids
    assert len(rows) == len(expected_case_ids) * len(ALGORITHMS)
    assert loaded["artifacts"]["authoritative_completion_artifact"] == "json"
    assert loaded["artifacts"]["csv"]["row_count"] == len(rows)
    assert loaded["artifacts"]["csv"]["sha256"] == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    assert all(
        row["full_fft_faster_in_all_trials"] in {"True", "False"}
        for row in rows
    )
    assert all(
        float(row["paired_trial_speedup_minimum"]) > 0.0
        and float(row["paired_trial_speedup_maximum"]) > 0.0
        for row in rows
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_artifacts(
            result,
            json_output=json_path,
            csv_output=csv_path,
            overwrite=False,
        )


def test_cli_requires_distinct_new_artifact_paths(tmp_path):
    output = tmp_path / "same.out"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--sizes",
                "4",
                "--retained-fractions",
                "1",
                "--json-output",
                str(output),
                "--csv-output",
                str(output),
            ]
        )


def test_cuda_request_fails_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        run_benchmark(
            _small_config(
                sizes=(4,),
                retained_fractions=(1.0,),
                kinds=("dct",),
                directions=("forward",),
                value_types=("real",),
                device="cuda",
            )
        )


def test_default_maximum_float32_size_passes_audited_tolerance():
    result = run_benchmark(
        _small_config(
            sizes=(1024,),
            retained_fractions=(1.0,),
            line_counts=(1,),
            kinds=("dct",),
            directions=("forward",),
            dtypes=("float32",),
            value_types=("real",),
            repeats=1,
        )
    )

    candidate = result["cases"][0]["correctness"]["full_fft_reference"]
    assert candidate["tolerance_model"] == "max(5e-5, 24*eps*sqrt(N))"
    assert candidate["dense_reference_relative_l2"] <= candidate[
        "tolerance_relative_l2"
    ]


def test_nonfinite_correctness_fails_closed(monkeypatch):
    def nan_forward(self, tensor):
        return torch.full(
            (*tensor.shape[:-1], self.key.retained_count),
            float("nan"),
            dtype=tensor.dtype,
            device=tensor.device,
        )

    monkeypatch.setattr(
        applicability.FullFftReferencePlan,
        "forward_last_axis",
        nan_forward,
    )
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        run_benchmark(
            _small_config(
                sizes=(4,),
                retained_fractions=(1.0,),
                kinds=("dct",),
                directions=("forward",),
                value_types=("real",),
            )
        )


def test_nonfinite_timing_fails_closed(monkeypatch):
    monkeypatch.setattr(
        applicability,
        "_timing_samples",
        lambda *_args, **_kwargs: [float("nan")],
    )
    with pytest.raises(RuntimeError, match="timing samples"):
        run_benchmark(
            _small_config(
                sizes=(4,),
                retained_fractions=(1.0,),
                kinds=("dct",),
                directions=("forward",),
                value_types=("real",),
                repeats=1,
            )
        )


def test_two_artifact_publish_rolls_back_when_json_commit_fails(
    tmp_path,
    monkeypatch,
):
    result = run_benchmark(
        _small_config(
            sizes=(4,),
            retained_fractions=(1.0,),
            kinds=("dct",),
            directions=("forward",),
            value_types=("real",),
            repeats=1,
        )
    )
    json_path = tmp_path / "map.json"
    csv_path = tmp_path / "map.csv"
    real_replace = applicability.os.replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected JSON publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(applicability.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected JSON publish failure"):
        write_artifacts(
            result,
            json_output=json_path,
            csv_output=csv_path,
            overwrite=False,
        )

    assert not json_path.exists()
    assert not csv_path.exists()


def test_overwrite_preserves_backup_when_publish_and_restore_fail(
    tmp_path,
    monkeypatch,
):
    result = run_benchmark(
        _small_config(
            sizes=(4,),
            retained_fractions=(1.0,),
            kinds=("dct",),
            directions=("forward",),
            value_types=("real",),
            repeats=1,
        )
    )
    json_path = tmp_path / "map.json"
    csv_path = tmp_path / "map.csv"
    old_json = b"old JSON artifact\n"
    old_csv = b"old CSV artifact\n"
    json_path.write_bytes(old_json)
    csv_path.write_bytes(old_csv)
    real_replace = applicability.os.replace
    calls = 0

    def fail_json_publish_and_restore(source, target):
        nonlocal calls
        calls += 1
        if calls in {4, 5}:
            raise OSError(f"injected replace failure {calls}")
        return real_replace(source, target)

    monkeypatch.setattr(
        applicability.os,
        "replace",
        fail_json_publish_and_restore,
    )
    with pytest.raises(
        RuntimeError,
        match="rollback was incomplete; preserved recovery paths",
    ) as error:
        write_artifacts(
            result,
            json_output=json_path,
            csv_output=csv_path,
            overwrite=True,
        )

    assert csv_path.read_bytes() == old_csv
    assert not json_path.exists()
    backup_json = next(tmp_path.glob(".map.json.backup.*"))
    assert backup_json.read_bytes() == old_json
    assert str(backup_json.resolve()) in str(error.value)
    assert not list(tmp_path.glob(".map.csv.backup.*"))


def test_tf32_policy_is_recorded_and_global_state_is_restored():
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_precision = torch.get_float32_matmul_precision()
    result = run_benchmark(
        _small_config(
            sizes=(4,),
            retained_fractions=(1.0,),
            kinds=("dct",),
            directions=("forward",),
            dtypes=("float32",),
            value_types=("real",),
            repeats=1,
            tf32="off",
        )
    )

    assert result["environment"]["precision"]["tf32_requested"] == "off"
    assert torch.backends.cuda.matmul.allow_tf32 == original_tf32
    assert torch.get_float32_matmul_precision() == original_precision


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_smoke_records_actual_device_and_memory():
    result = run_benchmark(
        _small_config(
            sizes=(8,),
            retained_fractions=(0.5,),
            kinds=("dct",),
            directions=("forward",),
            value_types=("real",),
            device="cuda",
            repeats=2,
        )
    )

    assert result["environment"]["device"].startswith("cuda:")
    for algorithm in ALGORITHMS:
        memory = result["cases"][0]["aggregate"][algorithm]["memory"]
        assert memory["peak_allocated_delta_bytes_max"] >= 0
        assert memory["peak_reserved_bytes_max"] >= 0
