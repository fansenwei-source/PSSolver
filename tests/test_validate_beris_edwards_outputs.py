from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts_plane.run_beris_edwards_validation import (
    RunSpec,
    IMPLEMENTATION_SOURCE_FILES,
    OUTPUT_VALIDATOR,
    PROJECT_ROOT,
    _sha256_file,
    _canonical_sha256,
    _expected_metadata_fields,
)
from scripts_plane.validate_beris_edwards_outputs import (
    DIAGNOSTIC_FIELDS,
    MAX_STEP,
    RUNNER_PATH,
    main,
    parse_args,
    parse_integral_step,
    validate_outputs,
)


DIAGNOSTIC_DTYPE = [
    ("step", np.int64),
    ("div_max", np.float64),
    ("div_rms", np.float64),
    ("div_rel", np.float64),
    ("schur_iterations", np.float64),
    ("schur_abs_residual", np.float64),
    ("schur_rel_residual", np.float64),
    ("wall_normal_momentum_max", np.float64),
    ("wall_normal_momentum_rms", np.float64),
]


def _set_nested(payload, path, value):
    current = payload
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _schedule(final_step, start_step, interval):
    first = (start_step + interval - 1) // interval * interval
    return (*range(first, final_step, interval), final_step)


def _write_synthetic_run(
    directory: Path,
    *,
    final_step: int,
    save_start_step: int,
    save_interval: int,
    diagnostic_interval: int,
    shape: tuple[int, int, int] = (2, 3, 4),
    dtype: str = "float64",
) -> tuple[Path, tuple[int, ...], tuple[int, ...]]:
    directory.mkdir()
    spec = RunSpec(
        run_id=f"synthetic_{final_step}",
        purposes=("synthetic",),
        activity_number=18.0,
        seed=24,
        nx=shape[0],
        ny=shape[1],
        nz=shape[2],
        dt=0.005,
        steps=final_step,
        save_start_step=save_start_step,
        save_interval=save_interval,
        diagnostic_interval=diagnostic_interval,
        dealias_rule="cubic_half",
    )
    config = spec.scientific_config()
    config["dtype"] = dtype
    config["device"] = "cpu"
    config["expected_gpu_name"] = None
    row = {
        "run_id": spec.run_id,
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "output_dir": str(directory),
    }
    implementation = {
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in IMPLEMENTATION_SOURCE_FILES
    }
    metadata = {}
    for path, value in _expected_metadata_fields(row).items():
        _set_nested(metadata, path, value)
    metadata["implementation_provenance"] = {
        "schema_version": 1,
        "files": implementation,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, allow_nan=False), encoding="utf-8"
    )
    (directory / "COMPLETE").write_text("complete\n", encoding="utf-8")

    snapshot_steps = _schedule(final_step, save_start_step, save_interval)
    for step in snapshot_steps:
        np.save(directory / f"Q_{step}.npy", np.zeros((*shape, 5), dtype=dtype))
        np.save(directory / f"u_{step}.npy", np.zeros((*shape, 3), dtype=dtype))
        np.save(directory / f"p_{step}.npy", np.zeros(shape, dtype=dtype))

    diagnostic_steps = _schedule(final_step, 0, diagnostic_interval)
    diagnostics = np.zeros(len(diagnostic_steps), dtype=DIAGNOSTIC_DTYPE)
    diagnostics["step"] = diagnostic_steps
    diagnostics["div_max"] = 2.0e-12
    diagnostics["div_rms"] = 1.0e-12
    diagnostics["div_rel"] = 2.0e-13
    diagnostics["schur_iterations"] = 1.0
    diagnostics["schur_abs_residual"] = 3.0e-13
    diagnostics["schur_rel_residual"] = 4.0e-13
    diagnostics["wall_normal_momentum_max"] = 5.0e-12
    diagnostics["wall_normal_momentum_rms"] = 6.0e-13
    np.save(directory / "diagnostics.npy", diagnostics)
    np.savetxt(
        directory / "diagnostics.csv",
        diagnostics,
        delimiter=",",
        header=",".join(DIAGNOSTIC_FIELDS),
        comments="",
    )

    plan = {
        "schema_version": 1,
        "validation": "beris_edwards_stokes_issue6",
        "output_root": str(directory.parent),
        "implementation_sha256": implementation,
        "validation_tools_sha256": {
            str(OUTPUT_VALIDATOR.relative_to(PROJECT_ROOT)): _sha256_file(
                OUTPUT_VALIDATOR
            ),
            str(RUNNER_PATH.relative_to(PROJECT_ROOT)): _sha256_file(RUNNER_PATH),
        },
        "advisory_gates": {
            "divergence_and_schur_relative_residual_max": 1.0e-10,
        },
        "runs": [row],
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    plan_path = directory.parent / f"plan_{final_step}.json"
    plan_path.write_text(json.dumps(plan, allow_nan=False), encoding="utf-8")
    return plan_path, snapshot_steps, diagnostic_steps


def _make_single(tmp_path):
    run_dir = tmp_path / "single"
    plan, snapshots, diagnostics = _write_synthetic_run(
        run_dir,
        final_step=1,
        save_start_step=1,
        save_interval=1,
        diagnostic_interval=1,
    )
    return run_dir, plan, snapshots, diagnostics


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.000000000000000000e+00", 1),
        ("2e3", 2000),
        (1.0, 1),
        (np.float64(7.0), 7),
        (np.int64(9), 9),
    ],
)
def test_strict_step_parser_accepts_scientific_integer_values(value, expected):
    assert parse_integral_step(value) == expected


@pytest.mark.parametrize(
    "value",
    ["1.5", 1.5, "NaN", np.nan, "Inf", -np.inf, True, "", -1, MAX_STEP + 1],
)
def test_strict_step_parser_rejects_fractional_nonfinite_and_out_of_range(value):
    with pytest.raises(ValueError):
        parse_integral_step(value)


def test_formal_cli_requires_plan(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(SystemExit) as error:
        parse_args(["--run-dir", str(run_dir), "--report", str(tmp_path / "r.json")])
    assert error.value.code == 2


def test_single_step_output_contract_passes_with_plan(tmp_path):
    run_dir, plan, _, _ = _make_single(tmp_path)
    report = validate_outputs(run_dir, plan_path=plan)
    assert report["passed"]
    assert report["validator_plan_binding"] == "bound"
    assert report["runner_plan_binding"] == "bound"
    assert report["final_step"] == 1
    assert report["checks"]["runner_completed_run_contract"]["passed"]
    assert report["snapshots"]["frame_count_per_field"] == 1
    assert report["diagnostics"]["row_count"] == 2


def test_save_interval_may_exceed_final_step_and_still_save_only_final(tmp_path):
    run_dir = tmp_path / "sparse_save"
    plan, snapshot_steps, _ = _write_synthetic_run(
        run_dir,
        final_step=1,
        save_start_step=1,
        save_interval=500,
        diagnostic_interval=1,
    )

    report = validate_outputs(run_dir, plan_path=plan)

    assert report["passed"]
    assert snapshot_steps == (1,)


def test_long_run_checks_all_21_frames_and_exact_201_diagnostics(tmp_path):
    run_dir = tmp_path / "long"
    plan, snapshots, diagnostic_steps = _write_synthetic_run(
        run_dir,
        final_step=20_000,
        save_start_step=10_000,
        save_interval=500,
        diagnostic_interval=100,
        dtype="float32",
    )
    report = validate_outputs(run_dir, plan_path=plan)
    assert report["passed"]
    assert snapshots == (*range(10_000, 20_000, 500), 20_000)
    assert len(snapshots) == 21
    assert len(diagnostic_steps) == 201
    assert report["snapshots"]["expected_steps"] == list(snapshots)
    assert report["diagnostics"]["row_count"] == 201


def test_corrupt_intermediate_long_frame_is_rejected(tmp_path):
    run_dir = tmp_path / "long"
    plan, _, _ = _write_synthetic_run(
        run_dir,
        final_step=20_000,
        save_start_step=10_000,
        save_interval=500,
        diagnostic_interval=100,
    )
    values = np.load(run_dir / "Q_15000.npy")
    values[0, 0, 0, 0] = np.nan
    np.save(run_dir / "Q_15000.npy", values)
    report = validate_outputs(run_dir, plan_path=plan)
    assert not report["passed"]
    assert "NaN or Inf" in json.dumps(report)


@pytest.mark.parametrize("mode", ["missing", "extra", "leading_zero_alias"])
def test_exact_snapshot_set_rejects_missing_extra_or_alias_frames(tmp_path, mode):
    run_dir, plan, _, _ = _make_single(tmp_path)
    if mode == "missing":
        (run_dir / "u_1.npy").unlink()
    elif mode == "extra":
        np.save(run_dir / "p_0.npy", np.zeros((2, 3, 4), dtype=np.float64))
    else:
        np.save(run_dir / "Q_01.npy", np.zeros((2, 3, 4, 5), dtype=np.float64))
    report = validate_outputs(run_dir, plan_path=plan)
    assert not report["passed"]
    serialized = json.dumps(report)
    assert (
        "snapshot steps" in serialized
        or "required regular file" in serialized
        or "noncanonical snapshot filename" in serialized
    )


def test_exact_diagnostic_cadence_is_required(tmp_path):
    run_dir = tmp_path / "cadence"
    plan, _, _ = _write_synthetic_run(
        run_dir,
        final_step=10,
        save_start_step=10,
        save_interval=10,
        diagnostic_interval=2,
    )
    values = np.load(run_dir / "diagnostics.npy")
    values = np.delete(values, 2)
    np.save(run_dir / "diagnostics.npy", values)
    np.savetxt(
        run_dir / "diagnostics.csv", values, delimiter=",",
        header=",".join(DIAGNOSTIC_FIELDS), comments=""
    )
    report = validate_outputs(run_dir, plan_path=plan)
    assert not report["passed"]
    assert "expected schedule" in json.dumps(report)


def test_residual_gate_applies_to_every_diagnostic_row(tmp_path):
    run_dir = tmp_path / "gate"
    plan, _, _ = _write_synthetic_run(
        run_dir,
        final_step=10,
        save_start_step=10,
        save_interval=10,
        diagnostic_interval=2,
    )
    values = np.load(run_dir / "diagnostics.npy")
    values["div_rel"][1] = 2.0e-9
    np.save(run_dir / "diagnostics.npy", values)
    np.savetxt(
        run_dir / "diagnostics.csv", values, delimiter=",",
        header=",".join(DIAGNOSTIC_FIELDS), comments=""
    )
    report = validate_outputs(run_dir, plan_path=plan)
    assert not report["passed"]
    assert "at step 2 exceeds" in json.dumps(report)


def test_npy_csv_numeric_mismatch_is_rejected_exactly(tmp_path):
    run_dir, plan, _, _ = _make_single(tmp_path)
    path = run_dir / "diagnostics.csv"
    rows = list(csv.reader(path.open(encoding="utf-8")))
    rows[-1][3] = "9.000000000000000000e-13"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    report = validate_outputs(run_dir, plan_path=plan)
    assert not report["passed"]
    assert "npy/csv mismatch in field div_rel" in json.dumps(report)


def test_nonfinite_or_relaxed_library_threshold_cannot_formally_pass(tmp_path):
    run_dir, plan, _, _ = _make_single(tmp_path)
    for value in (float("nan"), float("inf"), 1.0e-9):
        report = validate_outputs(run_dir, plan_path=plan, div_rel_max=value)
        assert not report["passed"]
        assert "thresholds:" in json.dumps(report)


def test_fractional_scientific_csv_step_is_rejected_without_truncation(tmp_path):
    run_dir, plan, _, _ = _make_single(tmp_path)
    path = run_dir / "diagnostics.csv"
    rows = list(csv.reader(path.open(encoding="utf-8")))
    rows[-1][0] = "1.500000000000000000e+00"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    report = validate_outputs(run_dir, plan_path=plan)
    assert not report["passed"]
    assert "is not an integer" in json.dumps(report)


@pytest.mark.parametrize("target", ["plan_hash", "config_hash"])
def test_canonical_plan_and_config_hashes_are_enforced(tmp_path, target):
    run_dir, plan_path, _, _ = _make_single(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if target == "plan_hash":
        plan["plan_sha256"] = "0" * 64
    else:
        plan["runs"][0]["config_sha256"] = "0" * 64
        plan["plan_sha256"] = _canonical_sha256(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    report = validate_outputs(run_dir, plan_path=plan_path)
    assert not report["passed"]
    assert "sha256" in json.dumps(report).lower()


@pytest.mark.parametrize(
    ("tool_path", "binding"),
    [
        (OUTPUT_VALIDATOR, "validator_plan_binding"),
        (RUNNER_PATH, "runner_plan_binding"),
    ],
)
def test_validation_tool_hash_mismatch_is_rejected(tmp_path, tool_path, binding):
    run_dir, plan_path, _, _ = _make_single(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    relative = str(tool_path.relative_to(PROJECT_ROOT))
    plan["validation_tools_sha256"][relative] = "0" * 64
    plan["plan_sha256"] = _canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = validate_outputs(run_dir, plan_path=plan_path)

    assert not report["passed"]
    assert binding in json.dumps(report)


def test_missing_validator_binding_cannot_formally_pass(tmp_path):
    run_dir, plan_path, _, _ = _make_single(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    relative = str(OUTPUT_VALIDATOR.relative_to(PROJECT_ROOT))
    del plan["validation_tools_sha256"][relative]
    plan["plan_sha256"] = _canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = validate_outputs(run_dir, plan_path=plan_path)

    assert not report["passed"]
    assert report["validator_plan_binding"] == "missing"


def test_duplicate_json_key_is_rejected(tmp_path):
    run_dir, plan_path, _, _ = _make_single(tmp_path)
    contents = plan_path.read_text(encoding="utf-8")
    contents = contents.replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )
    plan_path.write_text(contents, encoding="utf-8")

    report = validate_outputs(run_dir, plan_path=plan_path)

    assert not report["passed"]
    assert "duplicate JSON key" in json.dumps(report)


def test_cli_writes_failure_report_and_refuses_overwrite(tmp_path):
    run_dir, plan, _, _ = _make_single(tmp_path)
    (run_dir / "COMPLETE").write_text("running\n", encoding="utf-8")
    report_path = tmp_path / "reports" / "postcheck.json"
    args = ["--run-dir", str(run_dir), "--plan", str(plan), "--report", str(report_path)]
    assert main(args) == 1
    first = report_path.read_bytes()
    assert main(args) == 2
    assert report_path.read_bytes() == first
    assert not json.loads(first)["passed"]


def test_report_inside_run_directory_is_rejected(tmp_path):
    run_dir, plan, _, _ = _make_single(tmp_path)
    with pytest.raises(SystemExit) as error:
        main([
            "--run-dir", str(run_dir), "--plan", str(plan),
            "--report", str(run_dir / "postcheck.json")
        ])
    assert error.value.code == 2
    assert not (run_dir / "postcheck.json").exists()
