"""Tests for the analysis-only P8.4.5 recovery diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import diagnose_plane_static_lifting_recovery as diagnostic


def _archive(path: Path, **arrays: np.ndarray) -> Path:
    np.savez(path, **arrays)
    return path


def test_compare_states_reports_exact_archives(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    reference = _archive(
        tmp_path / "reference.npz",
        internal__Qxx=values,
        physical__Qxx=values + 1.0,
    )
    actual = _archive(
        tmp_path / "actual.npz",
        internal__Qxx=values.copy(),
        physical__Qxx=values + 1.0,
    )

    report = diagnostic.compare_states(
        reference_path=reference,
        actual_path=actual,
    )

    assert report["analysis_only"] is True
    assert report["qualification_decision"] is None
    assert report["aggregates"]["internal"]["byte_identical"] is True
    assert report["aggregates"]["physical"]["relative_l2"] == 0.0
    assert report["fields"]["internal__Qxx"]["maximum_ulp"] == 0


def test_compare_states_quantifies_one_ulp_without_applying_tolerance(
    tmp_path: Path,
) -> None:
    reference_value = np.array([1.0, -2.0, 0.0], dtype=np.float64)
    actual_value = reference_value.copy()
    actual_value[0] = np.nextafter(actual_value[0], np.inf)
    reference = _archive(
        tmp_path / "reference.npz",
        internal__Qxx=reference_value,
        physical__Qxx=reference_value,
    )
    actual = _archive(
        tmp_path / "actual.npz",
        internal__Qxx=actual_value,
        physical__Qxx=actual_value,
    )

    report = diagnostic.compare_states(
        reference_path=reference,
        actual_path=actual,
    )

    field = report["fields"]["internal__Qxx"]
    assert field["byte_identical"] is False
    assert field["mismatch_count"] == 1
    assert field["maximum_ulp"] == 1
    assert field["linf"] == np.finfo(np.float64).eps
    assert field["relative_l2"] is not None
    assert report["tolerance_applied"] is False


def test_compare_states_rejects_schema_and_dtype_mismatches(tmp_path: Path) -> None:
    reference = _archive(
        tmp_path / "reference.npz",
        internal__Qxx=np.ones(2, dtype=np.float64),
    )
    missing = _archive(
        tmp_path / "missing.npz",
        internal__Qyy=np.ones(2, dtype=np.float64),
    )
    wrong_dtype = _archive(
        tmp_path / "wrong_dtype.npz",
        internal__Qxx=np.ones(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="keys differ"):
        diagnostic.compare_states(
            reference_path=reference,
            actual_path=missing,
        )
    with pytest.raises(ValueError, match="dtypes differ"):
        diagnostic.compare_states(
            reference_path=reference,
            actual_path=wrong_dtype,
        )


def test_capture_state_runs_real_small_cpu_runtime(tmp_path: Path) -> None:
    arrays = tmp_path / "state.npz"
    report = diagnostic.capture_state(
        variant="strong_planar_lifting",
        shape=(8, 8, 6),
        lengths=(8.0, 8.0, 6.0),
        device="cpu",
        steps=1,
        seed=14,
        pointwise_execution="eager",
        arrays_output=arrays,
    )

    assert report["finite"] is True
    assert report["completed_steps"] == 1
    assert report["arrays"]["path"] == str(arrays.resolve())
    assert len(report["arrays"]["sha256"]) == 64
    with np.load(arrays, allow_pickle=False) as archive:
        assert set(archive.files) == {
            *(f"internal__{name}" for name in diagnostic.STATE_FIELDS),
            *(f"physical__{name}" for name in diagnostic.STATE_FIELDS),
        }
        assert archive["internal__Qxx"].shape == (1, 8, 8, 6)


def test_memory_snapshot_selects_only_frozen_diagnostic_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        diagnostic.torch.cuda,
        "memory_stats",
        lambda _device: {
            "allocated_bytes.all.current": 10,
            "reserved_bytes.all.current": 20,
            "unrelated": 99,
        },
    )

    result = diagnostic._memory_snapshot(diagnostic.torch.device("cuda"))

    assert result["allocated_bytes.all.current"] == 10
    assert result["reserved_bytes.all.current"] == 20
    assert result["active_bytes.all.current"] == 0
    assert "unrelated" not in result


def test_cli_comparison_writes_one_json_without_overwrite(tmp_path: Path) -> None:
    state = _archive(
        tmp_path / "state.npz",
        internal__Qxx=np.ones(2, dtype=np.float64),
        physical__Qxx=np.ones(2, dtype=np.float64),
    )
    output = tmp_path / "report.json"

    assert (
        diagnostic.main(
            [
                "compare-states",
                "--reference",
                str(state),
                "--actual",
                str(state),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["analysis_only"] is True
    with pytest.raises(FileExistsError):
        diagnostic.main(
            [
                "compare-states",
                "--reference",
                str(state),
                "--actual",
                str(state),
                "--output",
                str(output),
            ]
        )
