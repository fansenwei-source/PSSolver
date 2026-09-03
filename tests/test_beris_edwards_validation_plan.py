import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import scripts_plane.run_beris_edwards_validation as validation_runner

from scripts_plane.run_beris_edwards_validation import (
    CONVERGENCE_ANALYZER,
    DEFECT_CORE_ANALYZER,
    DEFECT_CORE_OUTPUTS,
    NUMERICAL_STAGES,
    OUTPUT_VALIDATOR,
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


def test_long_seed_matrix_spans_three_activities_and_three_seeds():
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


def test_long_pilot_is_one_a18_seed24_run_without_duplicate_preflight(tmp_path):
    runs = build_runs(("long_pilot",))
    assert runs == build_runs(("long_pilot",), long_final_time=100.0)

    assert len(runs) == 1
    run = runs[0]
    assert run.run_id == "A18_R320_dt0p005_cubic_half_seed24_2647293efa55"
    assert run.purposes == ("long_pilot",)
    assert (run.nx, run.ny, run.nz) == (320, 320, 80)
    assert run.activity_number == pytest.approx(18.0)
    assert run.seed == 24
    assert run.dt == pytest.approx(0.005)
    assert run.steps == 20_000
    assert run.final_time == pytest.approx(100.0)
    assert run.save_start_step == 10_000
    assert run.save_interval == 500
    assert run.diagnostic_interval == 100
    assert run.saved_frame_count == 21
    assert "preflight" not in run.purposes
    storage = run.storage_estimate()
    assert storage["saved_frames"] == 21
    assert storage["bytes_per_frame"] == 589_824_000
    assert storage["primary_snapshot_bytes"] == 12_386_304_000
    assert storage["primary_snapshot_gib"] == pytest.approx(11.53564453125)

    plan = build_plan(
        runs,
        python_bin="python",
        output_root=tmp_path,
        device="cuda",
        expected_gpu_name="H100",
    )
    assert plan["execution_safety"]["simulation_command_count"] == 1
    assert plan["execution_safety"]["long_run_requires_allow_large_output"]
    prerequisite = plan["execution_safety"][
        "long_pilot_external_preflight_prerequisite"
    ]
    assert "does not inspect or certify" in prerequisite
    assert plan["storage_estimate"]["primary_snapshot_bytes"] == 12_386_304_000
    assert plan["storage_estimate"]["primary_snapshot_gib"] == pytest.approx(
        11.53564453125
    )
    assert plan["analysis_commands"] == []
    command = plan["runs"][0]["command"]
    assert command[command.index("--dealias-rule") + 1] == "cubic_half"
    assert command[command.index("--save-start-step") + 1] == "10000"
    assert command[command.index("--save-interval") + 1] == "500"
    assert command[command.index("--diagnostic-interval") + 1] == "100"


def test_long_pilot_t200_builds_one_fresh_a18_seed24_run(tmp_path):
    runs = build_runs(("long_pilot",), long_final_time=200.0)

    assert len(runs) == 1
    run = runs[0]
    assert run.run_id == "A18_R320_dt0p005_cubic_half_seed24_df65b5f8c7d8"
    assert run.purposes == ("long_pilot",)
    assert run.activity_number == pytest.approx(18.0)
    assert run.seed == 24
    assert (run.nx, run.ny, run.nz) == (320, 320, 80)
    assert run.dt == pytest.approx(0.005)
    assert run.steps == 40_000
    assert run.final_time == pytest.approx(200.0)
    assert run.save_start_step == 20_000
    assert run.save_interval == 500
    assert run.diagnostic_interval == 100
    assert run.dealias_rule == "cubic_half"

    config = run.scientific_config()
    assert config["lengths"] == [100.0, 100.0, 20.0]
    assert config["dtype"] == "float64"
    assert config["tf32"] == "off"
    assert config["spectral_refresh"] == "disabled"
    assert config["zero_mode_policy"] == "zero_mean"
    assert config["parameterization"] == "paper-window"
    t100_config = build_runs(("long_pilot",))[0].scientific_config()
    for time_field in ("steps", "final_time", "save_start_step"):
        config.pop(time_field)
        t100_config.pop(time_field)
    assert config == t100_config
    assert "restart" not in config
    assert "resume" not in config

    storage = run.storage_estimate()
    assert storage["saved_frames"] == 41
    assert storage["bytes_per_frame"] == 589_824_000
    assert storage["primary_snapshot_bytes"] == 24_182_784_000
    assert storage["primary_snapshot_gib"] == pytest.approx(22.52197265625)

    plan = build_plan(
        runs,
        python_bin="/test/python",
        output_root=tmp_path,
        device="cuda",
        expected_gpu_name="H100",
    )
    assert plan["execution_safety"]["simulation_command_count"] == 1
    assert plan["analysis_commands"] == []
    assert len(plan["runs"]) == 1
    row = plan["runs"][0]
    assert row["run_id"] == run.run_id
    assert row["purposes"] == ["long_pilot"]
    assert row["config_sha256"] == (
        "a13bd4c961fa41e0098f062fba1fbb99256002ad0a4d6ac1576bb68d2c6635e2"
    )
    assert row["storage_estimate"] == storage

    command = row["command"]
    expected_arguments = {
        "--activity-number": "18",
        "--nx": "320",
        "--ny": "320",
        "--nz": "80",
        "--steps": "40000",
        "--save-start-step": "20000",
        "--save-interval": "500",
        "--diagnostic-interval": "100",
        "--dealias-rule": "cubic_half",
        "--dtype": "float64",
        "--tf32": "off",
        "--zero-mode-policy": "zero_mean",
        "--seed": "24",
        "--validation-config-sha256": row["config_sha256"],
    }
    for option, value in expected_arguments.items():
        assert command[command.index(option) + 1] == value
    assert "--disable-spectral-refresh" in command
    assert not {"--restart", "--resume", "--snapshot"}.intersection(command)


@pytest.mark.parametrize(
    "long_final_time",
    [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 99.995, 100.001],
)
def test_long_pilot_api_rejects_invalid_final_time(long_final_time):
    with pytest.raises(ValueError):
        build_runs(("long_pilot",), long_final_time=long_final_time)


def _long_pilot_cli(tmp_path, *extra, long_final_time="200", execute=True):
    argv = [
        "run_beris_edwards_validation.py",
        "--output-root",
        str(tmp_path / "pilot"),
        "--stage",
        "long_pilot",
        "--long-final-time",
        long_final_time,
        "--device",
        "cuda",
        "--expected-gpu-name",
        "H100",
    ]
    if execute:
        argv.extend(("--execute", "--confirm-direct-execution"))
    return [*argv, *extra]


def test_long_pilot_t200_cli_writes_plan_without_execution(
    tmp_path, monkeypatch, capsys
):
    plan_path = tmp_path / "planning" / "plan.json"

    def unexpected_execute(*_args, **_kwargs):
        pytest.fail("planning-only invocation must not call execute_plan")

    monkeypatch.setattr(validation_runner, "execute_plan", unexpected_execute)
    monkeypatch.setattr(
        sys,
        "argv",
        _long_pilot_cli(
            tmp_path,
            "--write-plan",
            str(plan_path),
            execute=False,
        ),
    )
    assert validation_runner.main() == 0
    capsys.readouterr()

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["execution_safety"]["simulation_command_count"] == 1
    assert len(plan["runs"]) == 1
    assert plan["runs"][0]["purposes"] == ["long_pilot"]
    assert plan["runs"][0]["config"]["final_time"] == pytest.approx(200.0)
    assert not (tmp_path / "pilot").exists()


def test_long_pilot_without_large_output_confirmation_never_executes(
    tmp_path, monkeypatch, capsys
):
    def unexpected_execute(*_args, **_kwargs):
        pytest.fail("execute_plan must not run without --allow-large-output")

    monkeypatch.setattr(validation_runner, "execute_plan", unexpected_execute)
    monkeypatch.setattr(sys, "argv", _long_pilot_cli(tmp_path))
    with pytest.raises(SystemExit, match="--allow-large-output"):
        validation_runner.main()
    capsys.readouterr()


def test_long_pilot_with_confirmation_executes_exactly_one_run(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def capture_execute(plan, *, output_root, skip_analysis):
        calls.append((plan, output_root, skip_analysis))

    monkeypatch.setattr(validation_runner, "execute_plan", capture_execute)
    monkeypatch.setattr(
        sys, "argv", _long_pilot_cli(tmp_path, "--allow-large-output")
    )
    assert validation_runner.main() == 0
    capsys.readouterr()
    assert len(calls) == 1
    plan, output_root, skip_analysis = calls[0]
    assert output_root == (tmp_path / "pilot").resolve()
    assert skip_analysis is False
    assert len(plan["runs"]) == 1
    assert plan["runs"][0]["purposes"] == ["long_pilot"]
    assert plan["runs"][0]["config"]["final_time"] == pytest.approx(200.0)


@pytest.mark.parametrize(
    "extra",
    [
        ("--stage", "long_seed"),
        ("--include-r512-time-control",),
    ],
)
def test_long_pilot_rejects_invalid_execution_scope(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(
        sys,
        "argv",
        _long_pilot_cli(tmp_path, "--allow-large-output", *extra),
    )
    with pytest.raises(SystemExit) as error:
        validation_runner.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "long_final_time",
    ["nan", "inf", "-inf", "0", "-1", "99.995", "100.001"],
)
def test_long_pilot_cli_rejects_invalid_final_time(
    tmp_path, monkeypatch, long_final_time
):
    monkeypatch.setattr(
        sys,
        "argv",
        _long_pilot_cli(tmp_path, long_final_time=long_final_time),
    )
    with pytest.raises(SystemExit) as error:
        validation_runner.parse_args()
    assert error.value.code == 2


def test_long_pilot_cli_keeps_direct_execution_confirmation_gate(
    tmp_path, monkeypatch
):
    argv = _long_pilot_cli(tmp_path, "--allow-large-output")
    argv.remove("--confirm-direct-execution")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        validation_runner.parse_args()
    assert error.value.code == 2


def test_long_pilot_cli_keeps_cuda_gpu_name_gate(tmp_path, monkeypatch):
    argv = _long_pilot_cli(tmp_path, "--allow-large-output")
    gpu_name_index = argv.index("--expected-gpu-name")
    del argv[gpu_name_index : gpu_name_index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        validation_runner.parse_args()
    assert error.value.code == 2


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
    assert "implemented" in plan["known_scope"]["issue_7"]
    assert plan["fixed_numerics"] == {
        "dtype": "float64",
        "tf32": "off",
        "spectral_refresh": "disabled",
        "zero_mode_policy": "zero_mean",
    }
    assert len(plan["implementation_sha256"]) == 13
    assert len(plan["validation_tools_sha256"]) == 4
    validator_relative = str(OUTPUT_VALIDATOR.relative_to(OUTPUT_VALIDATOR.parents[1]))
    assert OUTPUT_VALIDATOR.is_file()
    assert validator_relative in plan["validation_tools_sha256"]
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


def test_core_dealias_stage_reuses_exact_existing_identities_and_is_analysis_only(
    tmp_path,
):
    runs = build_runs(("core_dealias",))
    expected_ids = {
        "A18_R320_dt0p005_cubic_half_seed24_0e49e595d6de",
        "A18_R320_dt0p005_two_thirds_seed24_7270fddf85b2",
        "A18_R512_dt0p005_cubic_half_seed24_18cece2feb21",
        "A18_R512_dt0p005_two_thirds_seed24_e8982189fde0",
    }
    assert {run.run_id for run in runs} == expected_ids
    assert all(run.purposes == ("core_dealias",) for run in runs)
    assert not any("preflight" in run.purposes for run in runs)

    plan = build_plan(
        runs,
        python_bin="python",
        output_root=tmp_path,
        device="cuda",
        expected_gpu_name="NVIDIA H100 PCIe",
    )
    expected_config_hashes = {
        "A18_R320_dt0p005_cubic_half_seed24_0e49e595d6de": (
            "a19ad496d1b0c3026b03aa28090d5e9d3aaa4c827edcdab32ed3a73bc80b5186"
        ),
        "A18_R320_dt0p005_two_thirds_seed24_7270fddf85b2": (
            "2cd7416e264fb54692f0a98c6f458af3e9ae9025168025c7d52984186931e58e"
        ),
        "A18_R512_dt0p005_cubic_half_seed24_18cece2feb21": (
            "3fcbd766a0568490a5e733b86f920ee66134e2b056e75febc39cca51b3f0ac17"
        ),
        "A18_R512_dt0p005_two_thirds_seed24_e8982189fde0": (
            "8d673430e52c609cb05a19815e8b9c4074f03f69c43d3b15d42d737cdaddb455"
        ),
    }
    assert {
        row["run_id"]: row["config_sha256"] for row in plan["runs"]
    } == expected_config_hashes
    assert plan["execution_safety"]["simulation_launch_permitted"] is False
    assert plan["execution_safety"]["simulation_command_count"] == 0
    assert plan["storage_estimate"]["primary_snapshot_bytes"] == 0
    assert all(row["command"] is None for row in plan["runs"])
    assert {item["name"] for item in plan["analysis_commands"]} == {
        "core_dealias_sensitivity_R320",
        "core_dealias_sensitivity_R512",
    }
    for analysis in plan["analysis_commands"]:
        assert analysis["analyzer"] == str(DEFECT_CORE_ANALYZER)
        assert analysis["command"][analysis["command"].index("--mode") + 1] == "dealias"
        assert "analysis_dealias_R" not in " ".join(analysis["command"])
        input_rows = [
            next(row for row in plan["runs"] if row["run_id"] == run_id)
            for run_id in analysis["input_run_ids"]
        ]
        assert [row["config"]["dealias_rule"] for row in input_rows] == [
            "cubic_half",
            "two_thirds",
        ]


def test_core_dealias_stage_is_exclusive():
    with pytest.raises(ValueError, match="analysis-only"):
        build_runs(("core_dealias", "space"))
    with pytest.raises(ValueError, match="R512 time controls"):
        build_runs(("core_dealias",), include_r512_time_control=True)


def _core_dealias_plan(tmp_path):
    return build_plan(
        build_runs(("core_dealias",)),
        python_bin="python",
        output_root=tmp_path,
        device="cuda",
        expected_gpu_name="NVIDIA H100 PCIe",
    )


def test_core_dealias_missing_run_never_launches_subprocess(tmp_path, monkeypatch):
    import scripts_plane.run_beris_edwards_validation as validation_runner

    plan = _core_dealias_plan(tmp_path)

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("analysis-only failure must not launch subprocess")

    monkeypatch.setattr(validation_runner.subprocess, "run", forbidden_run)
    with pytest.raises(RuntimeError, match="analysis-only plan requires"):
        execute_plan(plan, output_root=tmp_path)


def test_core_dealias_missing_sidecar_never_starts_first_analyzer(
    tmp_path, monkeypatch
):
    import scripts_plane.run_beris_edwards_validation as validation_runner

    plan = _core_dealias_plan(tmp_path)
    for run_row in plan["runs"]:
        _materialize_completed_run(run_row, plan["implementation_sha256"])
    r512_two_thirds = next(
        row
        for row in plan["runs"]
        if row["config"]["shape"][0] == 512
        and row["config"]["dealias_rule"] == "two_thirds"
    )
    (Path(r512_two_thirds["output_dir"]) / "Q_0.npy").unlink()
    calls = []

    def forbidden_run(command, **_kwargs):
        calls.append(command)
        raise AssertionError("analyzer started before all sidecars were checked")

    monkeypatch.setattr(validation_runner.subprocess, "run", forbidden_run)
    with pytest.raises(RuntimeError, match="Q_0.npy"):
        execute_plan(plan, output_root=tmp_path)
    assert calls == []


def test_core_dealias_cannot_skip_analysis(tmp_path):
    plan = _core_dealias_plan(tmp_path)
    with pytest.raises(ValueError, match="cannot skip"):
        execute_plan(plan, output_root=tmp_path, skip_analysis=True)


def test_core_dealias_preflights_all_inputs_and_destinations_before_first_analyzer(
    tmp_path, monkeypatch
):
    import scripts_plane.run_beris_edwards_validation as validation_runner

    plan = _core_dealias_plan(tmp_path)
    for run_row in plan["runs"]:
        _materialize_completed_run(run_row, plan["implementation_sha256"])
    conflict = tmp_path / "analysis_defect_core_dealias_R512"
    conflict.mkdir()
    (conflict / "sentinel").write_text("do not replace\n", encoding="utf-8")
    calls = []

    def forbidden_run(command, **_kwargs):
        calls.append(command)
        raise AssertionError("R320 analyzer started before R512 conflict check")

    monkeypatch.setattr(validation_runner.subprocess, "run", forbidden_run)
    with pytest.raises(FileExistsError, match="matching manifest"):
        execute_plan(plan, output_root=tmp_path)
    assert calls == []
    assert (conflict / "sentinel").read_text(encoding="utf-8") == "do not replace\n"


def test_core_dealias_executes_only_two_analyzers_and_reuses_manifests(
    tmp_path, monkeypatch
):
    import scripts_plane.run_beris_edwards_validation as validation_runner

    plan = _core_dealias_plan(tmp_path)
    for run_row in plan["runs"]:
        _materialize_completed_run(run_row, plan["implementation_sha256"])
    old_global_dir = tmp_path / "analysis_dealias_R320"
    old_global_dir.mkdir()
    old_sentinel = old_global_dir / "sentinel"
    old_sentinel.write_text("old global analysis\n", encoding="utf-8")
    analyses_by_dir = _analysis_by_output_dir(plan)
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        assert Path(command[1]).resolve() == DEFECT_CORE_ANALYZER.resolve()
        assert str(validation_runner.MODEL_SCRIPT) not in command
        analysis_dir = Path(command[command.index("--output-dir") + 1])
        analysis = analyses_by_dir[analysis_dir]
        analysis_dir.mkdir(parents=True)
        for name in analysis["required_outputs"]:
            (analysis_dir / name).write_text("complete\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(validation_runner.subprocess, "run", fake_run)
    execute_plan(plan, output_root=tmp_path)
    assert len(calls) == 2
    assert old_sentinel.read_text(encoding="utf-8") == "old global analysis\n"

    for analysis in plan["analysis_commands"]:
        analysis_dir = Path(
            analysis["command"][analysis["command"].index("--output-dir") + 1]
        )
        manifest = json.loads(
            (analysis_dir / "validation_analysis_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["command"] == analysis["command"]
        assert manifest["analysis_code_provenance"]["runner_sha256"] == plan[
            "validation_tools_sha256"
        ]["scripts_plane/run_beris_edwards_validation.py"]
        first_files = next(iter(manifest["inputs"].values()))["files"]
        assert {
            "metadata.json",
            "COMPLETE",
            "Q_0.npy",
            "Q2D_initial.npy",
            "Q2D_defects.csv",
            "diagnostics.npy",
            "diagnostics.csv",
        }.issubset(first_files)
        first_input = next(iter(manifest["inputs"].values()))
        assert first_input["simulation_code_provenance"]["schema_version"] == 1
        assert first_input["simulation_runtime_environment"]["device_type"] == "cuda"

    calls.clear()
    execute_plan(plan, output_root=tmp_path)
    assert calls == []


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
        "diagnostics.csv",
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
