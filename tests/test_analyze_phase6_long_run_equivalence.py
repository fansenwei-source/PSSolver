"""Synthetic coverage for the Phase 6 P6.2 long-run equivalence audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts_plane.analyze_phase6_long_run_equivalence import (
    P61_CLASSIFICATION,
    _write_new,
    analyze_phase6_long_run_equivalence,
)


P61_COMMIT = "a" * 40
EXECUTION_COMMIT = "b" * 40
SHAPE = (2, 2, 2)
LENGTHS = (100.0, 100.0, 20.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _p61(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "p61.json",
        {
            "classification": P61_CLASSIFICATION,
            "performance_class": "performance_equivalent",
            "all_gates_passed": True,
            "eligible_for_long_run_stage": True,
            "eligible_for_default_promotion": False,
            "production_default_changed": False,
            "expected_qualification_commit": P61_COMMIT,
        },
    )


def _metadata(runtime: str) -> dict:
    raw_sha = "c" * 64
    projected_sha = "d" * 64
    return {
        "status": "complete",
        "completed_steps": 2,
        "shape": list(SHAPE),
        "dt": 0.005,
        "steps": 2,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "activity_number": 18.0,
        "seed": 24,
        "dtype": "float64",
        "save_hydrodynamics": True,
        "zero_mode_policy": "zero_mean",
        "friction": 0.0,
        "dealias_rule": "cubic_half",
        "projected_transform_execution": "truncated",
        "transform_execution_order": "real_first",
        "spectral_storage": "hermitian_half",
        "molecular_field_linear_space": "spectral",
        "stress_divergence_sum_space": "spectral",
        "pointwise_execution": "compile",
        "elapsed_seconds": 1.0 if runtime == "legacy_production" else 2.0,
        "validation_config_sha256": "e" * 64 if runtime == "legacy_production" else "f" * 64,
        "configuration": {
            "authority": "synthetic",
            "schema_version": 1,
            "runtime_path": runtime,
            "canonical_sha256": "1" * 64 if runtime == "legacy_production" else "2" * 64,
        },
        "runtime_selection": {
            "requested": runtime,
            "effective": runtime,
            "fallback_used": False,
            "adapter": runtime,
        },
        "workflow": {
            "runtime_path": runtime,
            "start_step": 0,
            "final_step": 2,
            "saved_steps": [0, 1, 2],
            "checkpoint_steps": [],
        },
        "initial_condition": {
            "raw_q_sha256": raw_sha,
            "projected_q_sha256": projected_sha,
        },
        "solver": {"lengths": list(LENGTHS)},
        "implementation_provenance": {"files": {"solver.py": "3" * 64}},
    }


def _run(tmp_path: Path, runtime: str) -> Path:
    directory = tmp_path / runtime
    directory.mkdir()
    (directory / "COMPLETE").write_text("complete\n", encoding="utf-8")
    _write_json(directory / "metadata.json", _metadata(runtime))
    for step in (0, 1, 2):
        base = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
        np.save(directory / f"Q_{step}.npy", np.stack([base + step] * 5, axis=-1))
        np.save(directory / f"u_{step}.npy", np.stack([base + step] * 3, axis=-1))
        np.save(directory / f"p_{step}.npy", base + step)
    np.save(directory / "diagnostics.npy", np.arange(3, dtype=np.float64))
    (directory / "diagnostics.csv").write_text("step,value\n0,0\n", encoding="utf-8")
    return directory


def _inputs(tmp_path: Path) -> dict:
    p61 = _p61(tmp_path)
    legacy = _run(tmp_path, "legacy_production")
    compiled = _run(tmp_path, "compiled_v2")
    provenance = _write_json(
        tmp_path / "execution.json",
        {
            "classification": "PASS_PHASE6_P62_EXECUTION_PROVENANCE",
            "expected_execution_commit": EXECUTION_COMMIT,
            "formal_h100_submission_count": 1,
            "automatic_retry": False,
            "worktree_clean_before_and_after": True,
            "slurm": {"state": "COMPLETED", "exit_code": "0:0"},
            "environment": {"gpu_name": "NVIDIA H100 PCIe", "tf32": False},
            "runs": {
                "legacy_production": {
                    "run_dir": str(legacy),
                    "runtime_path": "legacy_production",
                    "validation_config_sha256": "e" * 64,
                },
                "compiled_v2": {
                    "run_dir": str(compiled),
                    "runtime_path": "compiled_v2",
                    "validation_config_sha256": "f" * 64,
                },
            },
        },
    )
    return {
        "legacy_run_dir": legacy,
        "compiled_run_dir": compiled,
        "p61_report_path": p61,
        "execution_provenance_path": provenance,
        "expected_p61_report_sha256": _sha256(p61),
        "expected_p61_qualification_commit": P61_COMMIT,
        "expected_execution_commit": EXECUTION_COMMIT,
        "shape": SHAPE,
        "lengths": LENGTHS,
        "final_step": 2,
        "save_interval": 1,
        "diagnostic_interval": 1,
    }


def test_complete_byte_identical_long_runs_close_phase6(tmp_path):
    report = analyze_phase6_long_run_equivalence(**_inputs(tmp_path))

    assert report["classification"] == "PASS_PHASE6_LONG_RUN_BYTE_IDENTICAL"
    assert report["phase_6_complete"] is True
    assert report["frame_count"] == 3
    assert report["paired_array_count"] == 9
    assert report["normalized_metadata_identical"] is True
    assert report["raw_initial_q_identical"] is True
    assert report["projected_initial_q_identical"] is True
    assert set(report["scientific_observable_equivalence"]) == {
        "stationarity",
        "defect_lines",
        "sigma_over_H",
        "subgrid_sigma_over_H",
        "wall_normal_DCT_spectrum",
    }
    assert report["eligible_for_default_promotion"] is False
    assert report["eligible_for_phase_7_planning"] is True
    assert report["phase_7_execution_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("p61", "does not authorize"),
        ("p61_hash", "SHA-256"),
        ("runtime", "requested/effective"),
        ("execution_commit", "execution commit identity"),
        ("execution_path", "run path differs"),
        ("execution_slurm", "Slurm execution"),
        ("raw_q", "raw initial-Q"),
        ("metadata", "normalized long-run metadata"),
        ("array", "byte identity"),
        ("nonfinite", "non-finite"),
        ("diagnostics", "diagnostic byte identity"),
        ("missing", "paired observation is missing"),
    ),
)
def test_long_run_audit_fails_closed(tmp_path, mutation, message):
    kwargs = _inputs(tmp_path)
    compiled = kwargs["compiled_run_dir"]
    if mutation == "p61":
        value = json.loads(kwargs["p61_report_path"].read_text())
        value["eligible_for_long_run_stage"] = False
        _write_json(kwargs["p61_report_path"], value)
        kwargs["expected_p61_report_sha256"] = _sha256(kwargs["p61_report_path"])
    elif mutation == "p61_hash":
        kwargs["expected_p61_report_sha256"] = "0" * 64
    elif mutation == "runtime":
        value = json.loads((compiled / "metadata.json").read_text())
        value["runtime_selection"]["effective"] = "legacy_production"
        _write_json(compiled / "metadata.json", value)
    elif mutation == "execution_commit":
        value = json.loads(kwargs["execution_provenance_path"].read_text())
        value["expected_execution_commit"] = "0" * 40
        _write_json(kwargs["execution_provenance_path"], value)
    elif mutation == "execution_path":
        value = json.loads(kwargs["execution_provenance_path"].read_text())
        value["runs"]["compiled_v2"]["run_dir"] = str(tmp_path / "wrong")
        _write_json(kwargs["execution_provenance_path"], value)
    elif mutation == "execution_slurm":
        value = json.loads(kwargs["execution_provenance_path"].read_text())
        value["slurm"]["state"] = "FAILED"
        _write_json(kwargs["execution_provenance_path"], value)
    elif mutation == "raw_q":
        value = json.loads((compiled / "metadata.json").read_text())
        value["initial_condition"]["raw_q_sha256"] = "0" * 64
        _write_json(compiled / "metadata.json", value)
    elif mutation == "metadata":
        value = json.loads((compiled / "metadata.json").read_text())
        value["gamma"] = 2.94
        _write_json(compiled / "metadata.json", value)
    elif mutation == "array":
        value = np.load(compiled / "Q_1.npy")
        value[0, 0, 0, 0] += 1.0
        np.save(compiled / "Q_1.npy", value)
    elif mutation == "nonfinite":
        value = np.load(compiled / "Q_1.npy")
        value[0, 0, 0, 0] = np.nan
        np.save(compiled / "Q_1.npy", value)
    elif mutation == "diagnostics":
        (compiled / "diagnostics.csv").write_text(
            "step,value\n0,1\n", encoding="utf-8"
        )
    elif mutation == "missing":
        (compiled / "u_2.npy").unlink()

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        analyze_phase6_long_run_equivalence(**kwargs)


def test_output_writer_is_exclusive(tmp_path):
    output = tmp_path / "analysis" / "result.json"
    _write_new(output, {"classification": "PASS"})
    assert json.loads(output.read_text())["classification"] == "PASS"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_new(output, {"classification": "PASS"})
