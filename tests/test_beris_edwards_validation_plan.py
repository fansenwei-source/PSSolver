import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts_plane.run_beris_edwards_validation import (
    CONVERGENCE_ANALYZER,
    DEFECT_CORE_ANALYZER,
    DEFECT_CORE_OUTPUTS,
    NUMERICAL_STAGES,
    _expected_metadata_fields,
    _validate_completed_run,
    analysis_commands,
    build_plan,
    build_runs,
    command_for_run,
    execute_plan,
)


def test_short_numerical_matrix_has_eight_unique_runs_and_reuses_baseline():
    runs = build_runs(NUMERICAL_STAGES)

    assert len(runs) == 8
    shared = [run for run in runs if set(run.purposes) == {"time", "space"}]
    assert len(shared) == 1
    assert shared[0].nx == 256
    assert shared[0].dt == pytest.approx(0.005)
    assert shared[0].final_time == pytest.approx(1.0)

    preflight = [run for run in runs if "preflight" in run.purposes]
    assert len(preflight) == 1
    assert preflight[0].nx == 512
    assert preflight[0].steps == 1
    assert preflight[0].save_start_step == preflight[0].steps
    assert preflight[0].saved_frame_count == 1

    dealias = [run for run in runs if "dealias" in run.purposes]
    assert {(run.nx, run.dealias_rule) for run in dealias} == {
        (320, "cubic_half"),
        (320, "two_thirds"),
        (512, "cubic_half"),
        (512, "two_thirds"),
    }


def test_validation_commands_pin_precision_refresh_zero_mode_and_outputs(tmp_path):
    run = build_runs(("time",))[0]
    command = command_for_run(
        run,
        python_bin="/test/python",
        output_root=tmp_path,
        device="cuda",
    )

    assert command[0] == "/test/python"
    assert command[command.index("--dtype") + 1] == "float64"
    assert command[command.index("--tf32") + 1] == "off"
    assert "--disable-spectral-refresh" in command
    assert command[command.index("--zero-mode-policy") + 1] == "zero_mean"
    assert "--diagnostics" in command
    assert "--save-hydrodynamics" in command
    output_dir = Path(command[command.index("--output-dir") + 1])
    assert output_dir == tmp_path / run.run_id


def test_long_seed_pilot_spans_three_activities_and_three_seeds():
    runs = build_runs(("long_seed",), long_final_time=100.0)
    long_runs = [run for run in runs if "long_seed" in run.purposes]

    assert len(runs) == 10
    assert len(long_runs) == 9
    assert {run.activity_number for run in long_runs} == {15.0, 18.0, 25.0}
    assert {run.seed for run in long_runs} == {24, 41, 73}
    assert {run.final_time for run in long_runs} == {100.0}
    assert all(run.save_start_step == run.steps // 2 for run in long_runs)
    assert all(run.save_interval == 500 for run in long_runs)
    assert len([run for run in runs if "preflight" in run.purposes]) == 1


def test_optional_r512_time_control_has_baseline_and_analysis(tmp_path):
    runs = build_runs(("space",), include_r512_time_control=True)
    controls = [run for run in runs if "r512_time_control" in run.purposes]
    baselines = [run for run in runs if "r512_time_baseline" in run.purposes]

    assert len(controls) == 1
    assert len(baselines) == 1
    assert controls[0].nx == 512
    assert controls[0].dt == pytest.approx(0.0025)
    assert controls[0].final_time == pytest.approx(1.0)
    assert baselines[0].dt == pytest.approx(0.005)
    commands = analysis_commands(
        runs,
        python_bin="python",
        output_root=tmp_path,
    )
    names = [item["name"] for item in commands]
    assert "r512_time_control" in names
    assert names.index("space_defect_core") < names.index(
        "time_defect_core_R512"
    )
    core_time = next(
        item for item in commands if item["name"] == "time_defect_core_R512"
    )
    summary = (
        tmp_path
        / "analysis_defect_core_space"
        / "defect_core_summary.json"
    )
    assert core_time["analyzer"] == str(DEFECT_CORE_ANALYZER)
    assert core_time["dependency_files"] == [str(summary)]
    assert core_time["command"][
        core_time["command"].index("--space-core-summary") + 1
    ] == str(summary)
    assert set(core_time["input_run_ids"]) == {
        baselines[0].run_id,
        controls[0].run_id,
    }


def test_core_time_requires_complete_space_triplet(tmp_path):
    runs = build_runs(("preflight",), include_r512_time_control=True)
    names = {
        item["name"]
        for item in analysis_commands(
            runs, python_bin="python", output_root=tmp_path
        )
    }

    assert "r512_time_control" in names
    assert "space_defect_core" not in names
    assert "time_defect_core_R512" not in names


def test_plan_records_deferrals_hashes_limits_and_analysis_commands(tmp_path):
    runs = build_runs(NUMERICAL_STAGES)
    plan = build_plan(
        runs,
        python_bin="/test/python",
        output_root=tmp_path,
        device="cuda",
    )

    assert plan["validation"] == "beris_edwards_stokes_issue6"
    assert "deferred" in plan["known_scope"]["issue_1"]
    assert "remain open" in plan["known_scope"]["issue_7"]
    assert plan["fixed_numerics"] == {
        "dtype": "float64",
        "tf32": "off",
        "spectral_refresh": "disabled",
        "zero_mode_policy": "zero_mean",
    }
    assert len(plan["implementation_sha256"]) == 12
    assert len(plan["validation_tools_sha256"]) == 3
    assert all(len(value) == 64 for value in plan["implementation_sha256"].values())
    assert all(len(row["config_sha256"]) == 64 for row in plan["runs"])
    for row in plan["runs"]:
        command = row["command"]
        assert command[command.index("--validation-config-sha256") + 1] == row[
            "config_sha256"
        ]
    assert len(plan["plan_sha256"]) == 64
    assert plan["storage_estimate"]["primary_snapshot_gib"] > 0.0
    assert "blocked" not in plan["issue_6_component_status"]["defect_core"]
    assert {item["name"] for item in plan["analysis_commands"]} == {
        "time_convergence",
        "space_global_scalars",
        "space_defect_core",
        "dealias_sensitivity_R320",
        "dealias_sensitivity_R512",
    }
    config_by_run_id = {row["run_id"]: row["config_sha256"] for row in plan["runs"]}
    for analysis in plan["analysis_commands"]:
        assert analysis["input_config_sha256"] == {
            run_id: config_by_run_id[run_id]
            for run_id in analysis["input_run_ids"]
        }
    core_space = next(
        item
        for item in plan["analysis_commands"]
        if item["name"] == "space_defect_core"
    )
    assert core_space["analyzer"] == str(DEFECT_CORE_ANALYZER)
    assert tuple(core_space["required_outputs"]) == DEFECT_CORE_OUTPUTS
    space_analysis = next(
        item
        for item in plan["analysis_commands"]
        if item["name"] == "space_global_scalars"
    )
    assert "defect-core" in space_analysis["limitation"]


def test_analysis_commands_require_complete_triplets(tmp_path):
    runs = build_runs(("time",))[:2]

    assert analysis_commands(
        runs,
        python_bin="python",
        output_root=tmp_path,
    ) == []


def test_standalone_dealias_stage_builds_paired_comparisons(tmp_path):
    runs = build_runs(("dealias",))
    dealias_runs = [run for run in runs if "dealias" in run.purposes]

    assert len(runs) == 5
    assert len([run for run in runs if "preflight" in run.purposes]) == 1
    assert {(run.nx, run.dealias_rule) for run in dealias_runs} == {
        (320, "cubic_half"),
        (320, "two_thirds"),
        (512, "cubic_half"),
        (512, "two_thirds"),
    }
    commands = analysis_commands(
        runs,
        python_bin="python",
        output_root=tmp_path,
    )
    assert {item["name"] for item in commands} == {
        "dealias_sensitivity_R320",
        "dealias_sensitivity_R512",
    }


def test_long_seed_plan_exposes_large_primary_snapshot_estimate(tmp_path):
    runs = build_runs(("long_seed",), long_final_time=100.0)
    long_runs = [run for run in runs if "long_seed" in run.purposes]
    plan = build_plan(
        runs,
        python_bin="python",
        output_root=tmp_path,
        device="cuda",
    )

    assert all(run.saved_frame_count == 21 for run in long_runs)
    assert plan["storage_estimate"]["primary_snapshot_gib"] > 100.0
    assert plan["execution_safety"]["long_seed_requires_allow_large_output"]


def _set_nested(payload, path, value):
    current = payload
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _materialize_completed_run(run_row, implementation_sha256):
    directory = Path(run_row["output_dir"])
    directory.mkdir(parents=True)
    metadata = {}
    for path, value in _expected_metadata_fields(run_row).items():
        _set_nested(metadata, path, value)
    expected_gpu_name = run_row["config"].get("expected_gpu_name")
    if expected_gpu_name is not None:
        metadata["runtime_environment"]["cuda_device_name"] = (
            f"NVIDIA {expected_gpu_name} test GPU"
        )
    metadata["implementation_provenance"] = {
        "schema_version": 1,
        "files": implementation_sha256,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (directory / "COMPLETE").write_text("complete\n", encoding="utf-8")
    final_step = run_row["config"]["steps"]
    for name in (
        f"Q_{final_step}.npy",
        f"u_{final_step}.npy",
        f"p_{final_step}.npy",
        "diagnostics.npy",
        "Q_0.npy",
        "Q2D_initial.npy",
        "Q2D_defects.csv",
    ):
        (directory / name).touch()


def test_completed_run_reuse_requires_exact_metadata_and_implementation(tmp_path):
    plan = build_plan(
        build_runs(("preflight",)),
        python_bin="python",
        output_root=tmp_path,
        device="cpu",
    )
    run_row = plan["runs"][0]
    _materialize_completed_run(run_row, plan["implementation_sha256"])

    assert _validate_completed_run(
        run_row,
        implementation_sha256=plan["implementation_sha256"],
    )

    metadata_path = Path(run_row["output_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dt"] *= 2.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="not reusable"):
        _validate_completed_run(
            run_row,
            implementation_sha256=plan["implementation_sha256"],
        )


def test_completed_run_reuse_rejects_validation_config_hash_mismatch(tmp_path):
    plan = build_plan(
        build_runs(("preflight",)),
        python_bin="python",
        output_root=tmp_path,
        device="cpu",
    )
    run_row = plan["runs"][0]
    _materialize_completed_run(run_row, plan["implementation_sha256"])

    metadata_path = Path(run_row["output_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["validation_config_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="validation_config_sha256"):
        _validate_completed_run(
            run_row,
            implementation_sha256=plan["implementation_sha256"],
        )


def test_gpu_preflight_reuse_requires_expected_gpu_name(tmp_path):
    plan = build_plan(
        build_runs(("preflight",)),
        python_bin="python",
        output_root=tmp_path,
        device="cuda",
        expected_gpu_name="H100",
    )
    run_row = plan["runs"][0]
    _materialize_completed_run(run_row, plan["implementation_sha256"])

    assert _validate_completed_run(
        run_row,
        implementation_sha256=plan["implementation_sha256"],
    )

    metadata_path = Path(run_row["output_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["runtime_environment"]["cuda_device_name"] = "NVIDIA A100"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="GPU name"):
        _validate_completed_run(
            run_row,
            implementation_sha256=plan["implementation_sha256"],
        )


def test_device_is_part_of_validation_config_identity(tmp_path):
    runs = build_runs(("preflight",))
    cpu_plan = build_plan(
        runs,
        python_bin="python",
        output_root=tmp_path,
        device="cpu",
    )
    cuda_plan = build_plan(
        runs,
        python_bin="python",
        output_root=tmp_path,
        device="cuda",
        expected_gpu_name="H100",
    )

    assert cpu_plan["runs"][0]["config_sha256"] != cuda_plan["runs"][0][
        "config_sha256"
    ]


def test_incomplete_existing_run_is_rejected(tmp_path):
    plan = build_plan(
        build_runs(("preflight",)),
        python_bin="python",
        output_root=tmp_path,
        device="cpu",
    )
    run_row = plan["runs"][0]
    directory = Path(run_row["output_dir"])
    directory.mkdir(parents=True)
    (directory / "partial.log").touch()

    with pytest.raises(RuntimeError, match="incomplete existing run"):
        _validate_completed_run(
            run_row,
            implementation_sha256=plan["implementation_sha256"],
        )


def _analysis_by_output_dir(plan):
    return {
        Path(item["command"][item["command"].index("--output-dir") + 1]): item
        for item in plan["analysis_commands"]
    }


def test_execute_reuses_runs_and_runs_then_reuses_analyses(tmp_path, monkeypatch):
    import scripts_plane.run_beris_edwards_validation as validation_runner

    plan = build_plan(
        build_runs(NUMERICAL_STAGES),
        python_bin="python",
        output_root=tmp_path,
        device="cpu",
    )
    for run_row in plan["runs"]:
        _materialize_completed_run(run_row, plan["implementation_sha256"])
    analyses_by_dir = _analysis_by_output_dir(plan)
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        analysis_dir = Path(command[command.index("--output-dir") + 1])
        analysis = analyses_by_dir[analysis_dir]
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for name in analysis["required_outputs"]:
            (analysis_dir / name).write_text("complete\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(validation_runner.subprocess, "run", fake_run)
    execute_plan(plan, output_root=tmp_path)

    assert len(calls) == 5
    for analysis in plan["analysis_commands"]:
        command = analysis["command"]
        analysis_dir = Path(command[command.index("--output-dir") + 1])
        manifest_path = analysis_dir / "validation_analysis_manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        analyzer_relative = str(
            Path(analysis["analyzer"])
            .resolve()
            .relative_to(validation_runner.PROJECT_ROOT.resolve())
        )
        assert manifest["schema_version"] == 2
        assert manifest["analyzer"] == analyzer_relative
        assert manifest["analyzer_sha256"] == plan[
            "validation_tools_sha256"
        ][analyzer_relative]
        assert set(manifest["outputs"]) == set(analysis["required_outputs"])
        first_input = next(iter(manifest["inputs"].values()))["files"]
        if analysis["analyzer"] == str(DEFECT_CORE_ANALYZER):
            assert "Q2D_initial.npy" in first_input
        else:
            assert analysis["analyzer"] == str(CONVERGENCE_ANALYZER)
            assert "Q2D_initial.npy" not in first_input

    calls.clear()
    execute_plan(plan, output_root=tmp_path)
    assert calls == []


def test_analysis_reuse_rejects_changed_required_output(tmp_path, monkeypatch):
    import scripts_plane.run_beris_edwards_validation as validation_runner

    plan = build_plan(
        build_runs(("space",)),
        python_bin="python",
        output_root=tmp_path,
        device="cpu",
    )
    for run_row in plan["runs"]:
        _materialize_completed_run(run_row, plan["implementation_sha256"])
    analyses_by_dir = _analysis_by_output_dir(plan)

    def fake_run(command, **_kwargs):
        analysis_dir = Path(command[command.index("--output-dir") + 1])
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for name in analyses_by_dir[analysis_dir]["required_outputs"]:
            (analysis_dir / name).write_text("complete\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(validation_runner.subprocess, "run", fake_run)
    execute_plan(plan, output_root=tmp_path)
    analysis = plan["analysis_commands"][0]
    analysis_dir = Path(
        analysis["command"][analysis["command"].index("--output-dir") + 1]
    )
    (analysis_dir / analysis["required_outputs"][0]).write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(FileExistsError, match="matching manifest and file hashes"):
        execute_plan(plan, output_root=tmp_path)


def test_analysis_reuse_rejects_changed_input_data(tmp_path, monkeypatch):
    import scripts_plane.run_beris_edwards_validation as validation_runner

    plan = build_plan(
        build_runs(("space",)),
        python_bin="python",
        output_root=tmp_path,
        device="cpu",
    )
    for run_row in plan["runs"]:
        _materialize_completed_run(run_row, plan["implementation_sha256"])
    analyses_by_dir = _analysis_by_output_dir(plan)

    def fake_run(command, **_kwargs):
        analysis_dir = Path(command[command.index("--output-dir") + 1])
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for name in analyses_by_dir[analysis_dir]["required_outputs"]:
            (analysis_dir / name).write_text("complete\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(validation_runner.subprocess, "run", fake_run)
    execute_plan(plan, output_root=tmp_path)
    changed_run = next(
        row for row in plan["runs"] if "space" in row["purposes"]
    )
    final_step = changed_run["config"]["steps"]
    (Path(changed_run["output_dir"]) / f"Q_{final_step}.npy").write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(FileExistsError, match="matching manifest and file hashes"):
        execute_plan(plan, output_root=tmp_path)
