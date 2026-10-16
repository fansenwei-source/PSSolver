"""Contracts for the frozen rc5 G-1 H100 planner and matrix runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.plan_rc5_g1_h100 import (
    CONSUMER_SOURCE,
    CONSUMER_WHEEL_SHA256,
    FAMILIES,
    PROVIDER_BASELINE,
    PROVIDER_CANDIDATE,
    REPAIRED_PERIODIC_FAMILIES,
    SCHEDULED_REFRESH_FAMILIES,
    build_plan,
    main as plan_main,
)
from benchmarks.run_rc5_g1_profile_matrix import build_command_plan
from benchmarks.run_rc5_g1_stage_plan import (
    DRAFT_SCHEMA,
    EXPECTED_SCHEMAS,
    STAGES,
    freeze_plan,
    validate_plan,
)


def test_rc5_g1_plan_has_exact_provider_consumer_and_profile_identities():
    plan = build_plan()
    assert plan["schema"] == "pssolver.rc5.g1.single_h100_plan.v2"
    assert plan["provider"]["baseline_commit"] == PROVIDER_BASELINE
    assert plan["provider"]["candidate_commit"] == PROVIDER_CANDIDATE
    assert plan["consumer"]["qualified_source_commit"] == CONSUMER_SOURCE
    assert CONSUMER_SOURCE == "6d3558b1367da9a3e7d6b3f8fbc5584cf384bfe3"
    assert plan["consumer"]["wheel_sha256"] == CONSUMER_WHEEL_SHA256
    assert CONSUMER_WHEEL_SHA256 == (
        "4e112df06770a572d3395c71c3e6073b8e4ea458229857243ed91845f2f0bf75"
    )
    assert plan["consumer_h100_gate"]["expected_passed"] == 342
    profiles = plan["profile_matrix"]
    assert len(FAMILIES) == 8
    assert len(profiles) == 96
    assert [entry["sequence"] for entry in profiles] == list(range(1, 97))
    assert len({entry["key"] for entry in profiles}) == 96
    assert {entry["provider_commit"] for entry in profiles} == {
        PROVIDER_BASELINE,
        PROVIDER_CANDIDATE,
    }
    assert all(entry["device"] == "cuda" for entry in profiles)
    assert PROVIDER_CANDIDATE == "dabeb9a487a82700ddd364c4233941a175808a3a"
    assert plan["provider"]["candidate_wheel_sha256"] == (
        "954072040dc1681529c29e09d80f2fe9112d73442b98932978e1739c90212b22"
    )


def test_rc5_g1_plan_separates_raw_refresh_and_steady_transform_counts():
    plan = build_plan()
    contract = plan["transform_measurement_window"]
    assert contract == {
        "raw_counts_are_authoritative": True,
        "reported_per_step_is_raw_count_divided_by_timed_steps": True,
        "steady_state_counts_exclude_scheduled_refresh_calls": True,
        "scheduled_refresh_families": sorted(SCHEDULED_REFRESH_FAMILIES),
        "scheduled_refresh_interval": 20,
        "forward_calls_per_scheduled_refresh": 1,
        "inverse_calls_per_scheduled_refresh": 0,
        "source_profiles_are_immutable": True,
    }
    profiles = plan["profile_matrix"]
    refreshed = [
        entry for entry in profiles if entry["family"] in SCHEDULED_REFRESH_FAMILIES
    ]
    ordinary = [
        entry for entry in profiles if entry["family"] not in SCHEDULED_REFRESH_FAMILIES
    ]
    assert len(refreshed) == 24
    assert len(ordinary) == 72
    assert all(entry["scheduled_refreshes_in_timed_window"] == 3 for entry in refreshed)
    assert all(entry["scheduled_forward_calls_in_timed_window"] == 3 for entry in refreshed)
    assert all(entry["expected_forward_raw_calls"] == 663 for entry in refreshed)
    assert all(entry["expected_inverse_raw_calls"] == 2160 for entry in refreshed)
    assert all(entry["scheduled_refreshes_in_timed_window"] == 0 for entry in ordinary)
    assert all(entry["scheduled_forward_calls_in_timed_window"] == 0 for entry in ordinary)
    assert all(entry["scheduled_inverse_calls_in_timed_window"] == 0 for entry in profiles)


def test_rc5_g1_plan_does_not_apply_global_state_hash_equality():
    plan = build_plan()
    comparison = plan["numerical_comparison"]
    assert comparison["global_final_state_sha_equality"] is False
    assert comparison["repaired_periodic_families"] == list(
        REPAIRED_PERIODIC_FAMILIES
    )
    assert set(comparison["byte_identity_families"]) == {
        "plane_legacy_production",
        "plane_compiled_v2",
        "channel_legacy",
        "channel_compiled_v2",
        "channel_complete_stress",
        "channel_functional",
    }
    assert comparison["repaired_periodic_components"] == [
        "Q",
        "velocity",
        "pressure",
    ]
    thresholds = plan["thresholds"]
    assert thresholds["physical_relative_l2_max"] == 1.0e-12
    assert thresholds["physical_linf_max"] == 1.0e-12
    assert thresholds["candidate_hermitian_violation_max"] == 0.0


def test_rc5_g1_plan_keeps_fail_closed_submission_and_gate_counts():
    plan = build_plan()
    assert len(plan["cuda_only_tests"]) == 7
    assert plan["tf32_scope_gate"]["case_count"] == 48
    assert plan["checkpoint_gate"]["released_primary_case_count"] == 13
    assert plan["submission"] == {
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "second_job_without_new_authorization": False,
        "fail_fast": True,
    }


def test_rc5_g1_plan_cli_writes_strict_json(tmp_path):
    output = tmp_path / "plan.json"
    assert plan_main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == build_plan()
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_rc5_g1_matrix_plan_preserves_venv_paths_and_is_shell_free(tmp_path):
    baseline = tmp_path / "baseline" / "bin" / "python"
    candidate = tmp_path / "candidate" / "bin" / "python"
    command = build_command_plan(
        baseline_python=baseline,
        candidate_python=candidate,
        support_root=tmp_path / "support",
        output_root=tmp_path / "output",
    )
    assert command["profile_count"] == 96
    assert command["baseline_python"] == str(baseline)
    assert command["candidate_python"] == str(candidate)
    assert len({row["compiler_cache"] for row in command["rows"]}) == 96
    assert all(isinstance(row["command"], list) for row in command["rows"])
    assert all("--device" in row["command"] for row in command["rows"])
    assert all(
        row["command"][row["command"].index("--device") + 1] == "cuda"
        for row in command["rows"]
    )
    assert all(
        row["support_script"].endswith("profile_rc5_g1_runtime_family.py")
        for row in command["rows"]
    )


def _stage_plan(tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "h100"
    runtime = tmp_path / "runtime" / "python"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_bytes(b"python\n")
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True, exist_ok=True)
    if not executable.exists():
        executable.symlink_to(runtime)
    scripts = tmp_path / "support"
    scripts.mkdir(exist_ok=True)
    for name in STAGES:
        (scripts / f"{name}.py").write_text(f"# {name}\n", encoding="utf-8")
    return {
        "schema": "pssolver.rc5.g1.h100_stage_plan.v1",
        "provider_baseline_commit": PROVIDER_BASELINE,
        "provider_candidate_commit": PROVIDER_CANDIDATE,
        "consumer_source_commit": CONSUMER_SOURCE,
        "stages": [
            {
                "name": name,
                "command": [str(executable), str(scripts / f"{name}.py")],
                "artifact": str(output / "results" / f"{name}.json"),
                "expected_schema": EXPECTED_SCHEMAS[name],
                "environment": {},
                "command_file_bindings": [
                    {
                        "kind": "python_executable",
                        "path": str(executable),
                        "resolved_path": str(executable.resolve()),
                        "sha256": hashlib.sha256(
                            executable.read_bytes()
                        ).hexdigest(),
                    },
                    {
                        "kind": "python_script",
                        "path": str(scripts / f"{name}.py"),
                        "resolved_path": str((scripts / f"{name}.py").resolve()),
                        "sha256": hashlib.sha256(
                            (scripts / f"{name}.py").read_bytes()
                        ).hexdigest(),
                    },
                ],
            }
            for name in STAGES
        ],
    }


def test_rc5_g1_stage_plan_is_ordered_shell_free_and_output_scoped(tmp_path):
    plan = _stage_plan(tmp_path)
    stages = validate_plan(plan, tmp_path / "h100")
    assert [stage["name"] for stage in stages] == list(STAGES)
    assert all(isinstance(stage["command"], list) for stage in stages)

    broken = _stage_plan(tmp_path)
    broken["stages"][0]["command"] = ["/bin/bash", "-c", "true"]
    with pytest.raises(ValueError, match="shell command string"):
        validate_plan(broken, tmp_path / "h100")

    broken = _stage_plan(tmp_path)
    broken["stages"][0]["artifact"] = str(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="escapes output root"):
        validate_plan(broken, tmp_path / "h100")


def test_rc5_g1_stage_plan_freezer_binds_every_command_file(tmp_path):
    plan = _stage_plan(tmp_path)
    draft = dict(plan)
    draft["schema"] = DRAFT_SCHEMA
    for row in draft["stages"]:
        row.pop("command_file_bindings")
    frozen = freeze_plan(draft, tmp_path / "h100")
    assert frozen["schema"] == "pssolver.rc5.g1.h100_stage_plan.v1"
    assert len(frozen["stages"]) == 9
    executable = Path(frozen["stages"][0]["command"][0])
    assert executable.is_symlink()
    assert frozen["stages"][0]["command_file_bindings"][0] == {
        "kind": "python_executable",
        "path": str(executable),
        "resolved_path": str(executable.resolve()),
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }
    assert all(
        len(row["command_file_bindings"]) == 2 for row in frozen["stages"]
    )
    assert validate_plan(frozen, tmp_path / "h100") == frozen["stages"]


def test_rc5_g1_stage_plan_rejects_command_drift_and_inline_python(tmp_path):
    plan = _stage_plan(tmp_path)
    script = Path(plan["stages"][0]["command"][1])
    script.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        validate_plan(plan, tmp_path / "h100")

    draft = _stage_plan(tmp_path / "inline")
    draft["schema"] = DRAFT_SCHEMA
    draft["stages"][0]["command"] = [
        draft["stages"][0]["command"][0],
        "-c",
        "pass",
    ]
    for row in draft["stages"]:
        row.pop("command_file_bindings")
    with pytest.raises(ValueError, match="inline Python"):
        freeze_plan(draft, tmp_path / "inline" / "h100")


def test_rc5_g1_slurm_template_is_fixed_to_one_nonrequeued_h100_job():
    root = Path(__file__).resolve().parents[1]
    script = (root / "jobs/rc5_g1_single_h100.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --partition=hagan-gpu" in script
    assert "#SBATCH --account=hagan-lab" in script
    assert "#SBATCH --qos=medium" in script
    assert "#SBATCH --gres=gpu:H100:1" in script
    assert "#SBATCH --no-requeue" in script
    assert "run_rc5_g1_stage_plan.py" in script
    assert "NVIDIA_TF32_OVERRIDE=0" in script
    assert "sbatch" not in "\n".join(
        line for line in script.splitlines() if not line.startswith("#SBATCH")
    )


def test_rc5_g1_profile_child_enforces_and_restores_both_tf32_flags():
    root = Path(__file__).resolve().parents[1]
    source = (root / "benchmarks/profile_rc5_g1_runtime_family.py").read_text(
        encoding="utf-8"
    )
    assert 'torch.set_float32_matmul_precision("highest")' in source
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
    assert "torch.backends.cudnn.allow_tf32 = False" in source
    assert "torch.set_float32_matmul_precision(before[0])" in source
    assert "torch.backends.cuda.matmul.allow_tf32 = before[1]" in source
    assert "torch.backends.cudnn.allow_tf32 = before[2]" in source
