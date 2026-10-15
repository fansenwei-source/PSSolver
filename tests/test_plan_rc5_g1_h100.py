"""Contracts for the frozen rc5 G-1 H100 planner and matrix runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.plan_rc5_g1_h100 import (
    CONSUMER_SOURCE,
    FAMILIES,
    PROVIDER_BASELINE,
    PROVIDER_CANDIDATE,
    REPAIRED_PERIODIC_FAMILIES,
    build_plan,
    main as plan_main,
)
from benchmarks.run_rc5_g1_profile_matrix import build_command_plan
from benchmarks.run_rc5_g1_stage_plan import EXPECTED_SCHEMAS, STAGES, validate_plan


def test_rc5_g1_plan_has_exact_provider_consumer_and_profile_identities():
    plan = build_plan()
    assert plan["schema"] == "pssolver.rc5.g1.single_h100_plan.v1"
    assert plan["provider"]["baseline_commit"] == PROVIDER_BASELINE
    assert plan["provider"]["candidate_commit"] == PROVIDER_CANDIDATE
    assert plan["consumer"]["qualified_source_commit"] == CONSUMER_SOURCE
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
    return {
        "schema": "pssolver.rc5.g1.h100_stage_plan.v1",
        "provider_baseline_commit": PROVIDER_BASELINE,
        "provider_candidate_commit": PROVIDER_CANDIDATE,
        "consumer_source_commit": CONSUMER_SOURCE,
        "stages": [
            {
                "name": name,
                "command": ["/qualified/venv/bin/python", f"/support/{name}.py"],
                "artifact": str(output / "results" / f"{name}.json"),
                "expected_schema": EXPECTED_SCHEMAS[name],
                "environment": {},
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
