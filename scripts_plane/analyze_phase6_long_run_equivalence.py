#!/usr/bin/env python3
"""Fail-closed Phase 6 P6.2 long-run architecture-equivalence audit."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


BASELINE_RUNTIME = "legacy_production"
CANDIDATE_RUNTIME = "compiled_v2"
FROZEN_SHAPE = (320, 320, 80)
FROZEN_LENGTHS = (100.0, 100.0, 20.0)
FROZEN_DT = 0.005
FROZEN_FINAL_STEP = 80_000
FROZEN_SAVE_INTERVAL = 1_000
FROZEN_DIAGNOSTIC_INTERVAL = 100
FROZEN_STEPS = tuple(range(0, FROZEN_FINAL_STEP + 1, FROZEN_SAVE_INTERVAL))
P61_CLASSIFICATION = "PASS_PHASE6_H100_CANDIDATE_PERFORMANCE_EQUIVALENT"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path, description: str) -> tuple[Path, dict]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{description} is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {source}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must contain a JSON object")
    return source, dict(value)


def _full_sha1(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase full SHA-1")
    return value


def _full_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _p61_record(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_qualification_commit: str,
) -> dict[str, object]:
    source, value = _load_json(path, "P6.1 final analysis")
    _full_sha256(expected_sha256, "expected P6.1 report SHA-256")
    if _sha256(source) != expected_sha256:
        raise ValueError("P6.1 final-analysis SHA-256 differs from the contract")
    required = {
        "classification": P61_CLASSIFICATION,
        "performance_class": "performance_equivalent",
        "all_gates_passed": True,
        "eligible_for_long_run_stage": True,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
        "expected_qualification_commit": expected_qualification_commit,
    }
    if any(value.get(name) != expected for name, expected in required.items()):
        raise ValueError("P6.1 evidence does not authorize the long-run stage")
    return {
        "path": str(source),
        "sha256": expected_sha256,
        "classification": value["classification"],
        "qualification_commit": expected_qualification_commit,
    }


def _runtime_selection(metadata: Mapping, expected_runtime: str) -> None:
    selection = metadata.get("runtime_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("metadata runtime_selection is missing")
    if (
        selection.get("requested") != expected_runtime
        or selection.get("effective") != expected_runtime
        or selection.get("fallback_used") is not False
    ):
        raise ValueError(f"{expected_runtime} requested/effective identity failed")


def _execution_record(
    path: str | Path,
    *,
    expected_commit: str,
    legacy_run_dir: Path,
    compiled_run_dir: Path,
    legacy_metadata: Mapping,
    compiled_metadata: Mapping,
) -> dict[str, object]:
    source, value = _load_json(path, "P6.2 execution provenance")
    slurm = value.get("slurm")
    environment = value.get("environment")
    runs = value.get("runs")
    if value.get("classification") != "PASS_PHASE6_P62_EXECUTION_PROVENANCE":
        raise ValueError("P6.2 execution provenance classification is not PASS")
    if value.get("expected_execution_commit") != expected_commit:
        raise ValueError("P6.2 execution commit identity differs")
    if value.get("formal_h100_submission_count") != 1:
        raise ValueError("P6.2 requires exactly one formal H100 simulation Job")
    if value.get("automatic_retry") is not False:
        raise ValueError("P6.2 execution provenance recorded automatic retry")
    if value.get("worktree_clean_before_and_after") is not True:
        raise ValueError("P6.2 execution worktree was not clean")
    if not isinstance(slurm, Mapping) or (
        slurm.get("state") != "COMPLETED" or slurm.get("exit_code") != "0:0"
    ):
        raise ValueError("P6.2 Slurm execution is not successful")
    if not isinstance(environment, Mapping) or (
        "H100" not in str(environment.get("gpu_name"))
        or environment.get("tf32") is not False
    ):
        raise ValueError("P6.2 GPU or TF32 provenance differs")
    if not isinstance(runs, Mapping) or set(runs) != {
        BASELINE_RUNTIME,
        CANDIDATE_RUNTIME,
    }:
        raise ValueError("P6.2 execution run set differs")
    expected = {
        BASELINE_RUNTIME: (legacy_run_dir, legacy_metadata),
        CANDIDATE_RUNTIME: (compiled_run_dir, compiled_metadata),
    }
    for runtime, (run_dir, metadata) in expected.items():
        record = runs.get(runtime)
        if not isinstance(record, Mapping):
            raise ValueError(f"P6.2 provenance lacks {runtime} run identity")
        if Path(str(record.get("run_dir"))).expanduser().resolve() != run_dir:
            raise ValueError(f"P6.2 provenance {runtime} run path differs")
        if record.get("runtime_path") != runtime:
            raise ValueError(f"P6.2 provenance {runtime} runtime differs")
        if record.get("validation_config_sha256") != metadata.get(
            "validation_config_sha256"
        ):
            raise ValueError(f"P6.2 provenance {runtime} config identity differs")
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "classification": value["classification"],
        "expected_execution_commit": expected_commit,
        "formal_h100_submission_count": 1,
        "slurm": dict(slurm),
        "environment": dict(environment),
        "runs": {name: dict(record) for name, record in runs.items()},
    }


def _metadata_record(
    run_dir: Path,
    *,
    expected_runtime: str,
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
    final_step: int,
    save_interval: int,
    diagnostic_interval: int,
    steps: tuple[int, ...],
) -> tuple[dict, dict[str, object]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory is missing: {run_dir}")
    complete = run_dir / "COMPLETE"
    if not complete.is_file() or complete.read_text(encoding="utf-8").strip() != "complete":
        raise ValueError(f"run COMPLETE marker is invalid: {run_dir}")
    metadata_path, metadata = _load_json(run_dir / "metadata.json", "run metadata")
    _runtime_selection(metadata, expected_runtime)
    workflow = metadata.get("workflow")
    initial = metadata.get("initial_condition")
    if not isinstance(workflow, Mapping) or not isinstance(initial, Mapping):
        raise ValueError("run metadata lacks workflow or initial-condition identity")
    frozen = {
        "status": "complete",
        "completed_steps": final_step,
        "shape": list(shape),
        "dt": FROZEN_DT,
        "steps": final_step,
        "save_start_step": 0,
        "save_interval": save_interval,
        "diagnostic_interval": diagnostic_interval,
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
    }
    for name, expected in frozen.items():
        actual = metadata.get(name)
        if actual != expected:
            raise ValueError(
                f"{expected_runtime} metadata {name} differs: {actual!r} != {expected!r}"
            )
    solver = metadata.get("solver")
    if not isinstance(solver, Mapping) or tuple(solver.get("lengths", ())) != lengths:
        raise ValueError(f"{expected_runtime} solver lengths differ")
    if workflow.get("runtime_path") != expected_runtime:
        raise ValueError(f"{expected_runtime} workflow runtime path differs")
    if workflow.get("start_step") != 0 or workflow.get("final_step") != final_step:
        raise ValueError(f"{expected_runtime} workflow clock differs")
    if tuple(workflow.get("saved_steps", ())) != steps:
        raise ValueError(f"{expected_runtime} saved-step schedule differs")
    if workflow.get("checkpoint_steps") not in ([], None):
        raise ValueError(f"{expected_runtime} long run unexpectedly wrote checkpoints")
    raw_sha = _full_sha256(initial.get("raw_q_sha256"), "raw initial Q identity")
    projected_sha = _full_sha256(
        initial.get("projected_q_sha256"), "projected initial Q identity"
    )
    return metadata, {
        "run_dir": str(run_dir),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "complete_sha256": _sha256(complete),
        "runtime": expected_runtime,
        "raw_initial_q_sha256": raw_sha,
        "projected_initial_q_sha256": projected_sha,
    }


def _normalized_metadata(value: Mapping) -> dict:
    normalized = deepcopy(dict(value))
    normalized.pop("elapsed_seconds", None)
    normalized.pop("validation_config_sha256", None)
    normalized.pop("runtime_selection", None)
    configuration = normalized.get("configuration")
    if isinstance(configuration, dict):
        configuration.pop("runtime_path", None)
        configuration.pop("canonical_sha256", None)
    workflow = normalized.get("workflow")
    if isinstance(workflow, dict):
        workflow.pop("runtime_path", None)
    return normalized


def _finite_mmap(value: np.ndarray, *, chunk_x: int = 8) -> bool:
    if value.ndim == 0:
        return bool(np.isfinite(value))
    for start in range(0, value.shape[0], chunk_x):
        if not np.isfinite(value[start : start + chunk_x]).all():
            return False
    return True


def _array_pair(
    left: Path,
    right: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
) -> dict[str, object]:
    if not left.is_file() or not right.is_file():
        raise FileNotFoundError(f"paired observation is missing: {left}, {right}")
    left_array = np.load(left, mmap_mode="r", allow_pickle=False)
    right_array = np.load(right, mmap_mode="r", allow_pickle=False)
    for description, value in (("legacy", left_array), ("compiled", right_array)):
        if value.shape != expected_shape or value.dtype != expected_dtype:
            raise ValueError(
                f"{description} array contract differs for {left.name}: "
                f"shape={value.shape}, dtype={value.dtype}"
            )
    if not _finite_mmap(left_array) or not _finite_mmap(right_array):
        raise ValueError(f"paired observation is non-finite: {left.name}")
    left_sha = _sha256(left)
    right_sha = _sha256(right)
    if left_sha != right_sha:
        raise ValueError(f"long-run byte identity failed: {left.name}")
    return {
        "name": left.name,
        "sha256": left_sha,
        "size_bytes": left.stat().st_size,
        "shape": list(expected_shape),
        "dtype": str(expected_dtype),
    }


def analyze_phase6_long_run_equivalence(
    *,
    legacy_run_dir: str | Path,
    compiled_run_dir: str | Path,
    p61_report_path: str | Path,
    execution_provenance_path: str | Path,
    expected_p61_report_sha256: str,
    expected_p61_qualification_commit: str,
    expected_execution_commit: str,
    shape: tuple[int, int, int] = FROZEN_SHAPE,
    lengths: tuple[float, float, float] = FROZEN_LENGTHS,
    final_step: int = FROZEN_FINAL_STEP,
    save_interval: int = FROZEN_SAVE_INTERVAL,
    diagnostic_interval: int = FROZEN_DIAGNOSTIC_INTERVAL,
) -> dict[str, object]:
    """Require exact long-run state identity before closing architecture Phase 6."""

    _full_sha1(expected_p61_qualification_commit, "P6.1 qualification commit")
    _full_sha1(expected_execution_commit, "P6.2 execution commit")
    if final_step <= 0 or save_interval <= 0 or final_step % save_interval:
        raise ValueError("final_step must be a positive multiple of save_interval")
    steps = tuple(range(0, final_step + 1, save_interval))
    p61 = _p61_record(
        p61_report_path,
        expected_sha256=expected_p61_report_sha256,
        expected_qualification_commit=expected_p61_qualification_commit,
    )
    legacy_dir = Path(legacy_run_dir).expanduser().resolve()
    compiled_dir = Path(compiled_run_dir).expanduser().resolve()
    if legacy_dir == compiled_dir:
        raise ValueError("legacy and compiled run directories must differ")
    legacy_metadata, legacy = _metadata_record(
        legacy_dir,
        expected_runtime=BASELINE_RUNTIME,
        shape=shape,
        lengths=lengths,
        final_step=final_step,
        save_interval=save_interval,
        diagnostic_interval=diagnostic_interval,
        steps=steps,
    )
    compiled_metadata, compiled = _metadata_record(
        compiled_dir,
        expected_runtime=CANDIDATE_RUNTIME,
        shape=shape,
        lengths=lengths,
        final_step=final_step,
        save_interval=save_interval,
        diagnostic_interval=diagnostic_interval,
        steps=steps,
    )
    if legacy["raw_initial_q_sha256"] != compiled["raw_initial_q_sha256"]:
        raise ValueError("raw initial-Q identities differ")
    if legacy["projected_initial_q_sha256"] != compiled["projected_initial_q_sha256"]:
        raise ValueError("projected initial-Q identities differ")
    if _normalized_metadata(legacy_metadata) != _normalized_metadata(compiled_metadata):
        raise ValueError("normalized long-run metadata differs")
    execution = _execution_record(
        execution_provenance_path,
        expected_commit=expected_execution_commit,
        legacy_run_dir=legacy_dir,
        compiled_run_dir=compiled_dir,
        legacy_metadata=legacy_metadata,
        compiled_metadata=compiled_metadata,
    )

    dtype = np.dtype("float64")
    arrays: list[dict[str, object]] = []
    expected_shapes = {
        "Q": (*shape, 5),
        "u": (*shape, 3),
        "p": shape,
    }
    for step in steps:
        for prefix, array_shape in expected_shapes.items():
            arrays.append(
                _array_pair(
                    legacy_dir / f"{prefix}_{step}.npy",
                    compiled_dir / f"{prefix}_{step}.npy",
                    expected_shape=array_shape,
                    expected_dtype=dtype,
                )
            )
    diagnostic_pairs = []
    for name in ("diagnostics.npy", "diagnostics.csv"):
        left = legacy_dir / name
        right = compiled_dir / name
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"paired diagnostics are missing: {name}")
        left_sha = _sha256(left)
        if left_sha != _sha256(right):
            raise ValueError(f"long-run diagnostic byte identity failed: {name}")
        diagnostic_pairs.append(
            {"name": name, "sha256": left_sha, "size_bytes": left.stat().st_size}
        )

    observable_basis = {
        name: "proven_equal_by_complete_saved_field_byte_identity"
        for name in (
            "stationarity",
            "defect_lines",
            "sigma_over_H",
            "subgrid_sigma_over_H",
            "wall_normal_DCT_spectrum",
        )
    }
    return {
        "schema_version": 1,
        "classification": "PASS_PHASE6_LONG_RUN_BYTE_IDENTICAL",
        "phase_6_complete": True,
        "expected_execution_commit": expected_execution_commit,
        "p61_evidence": p61,
        "execution_provenance": execution,
        "runs": {BASELINE_RUNTIME: legacy, CANDIDATE_RUNTIME: compiled},
        "frame_count": len(steps),
        "paired_array_count": len(arrays),
        "paired_arrays": arrays,
        "diagnostic_pairs": diagnostic_pairs,
        "normalized_metadata_identical": True,
        "raw_initial_q_identical": True,
        "projected_initial_q_identical": True,
        "scientific_observable_equivalence": observable_basis,
        "scientific_interpretation": (
            "Architecture equivalence only; physical stationarity and strict "
            "Shendruk-model reproduction remain separate scientific claims."
        ),
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
        "eligible_for_phase_7_planning": True,
        "phase_7_execution_authorized": False,
    }


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite output: {target}")
    target.write_text(
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--compiled-run-dir", type=Path, required=True)
    parser.add_argument("--p61-report", type=Path, required=True)
    parser.add_argument("--execution-provenance", type=Path, required=True)
    parser.add_argument("--expected-p61-report-sha256", required=True)
    parser.add_argument("--expected-p61-qualification-commit", required=True)
    parser.add_argument("--expected-execution-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_phase6_long_run_equivalence(
        legacy_run_dir=args.legacy_run_dir,
        compiled_run_dir=args.compiled_run_dir,
        p61_report_path=args.p61_report,
        execution_provenance_path=args.execution_provenance,
        expected_p61_report_sha256=args.expected_p61_report_sha256,
        expected_p61_qualification_commit=args.expected_p61_qualification_commit,
        expected_execution_commit=args.expected_execution_commit,
    )
    _write_new(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
