"""Synthetic contracts for the RC6 single-H100 qualification support."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmarks.analyze_rc6_single_h100 import PASS_CLASSIFICATION, analyze
from benchmarks.assemble_rc6_h100_auxiliary import assemble
from benchmarks.plan_rc6_single_h100_qualification import (
    CONSUMER_SOURCE,
    CONSUMER_WHEEL_SHA256,
    PROVIDER_BASELINE,
    PROVIDER_BASELINE_WHEEL_SHA256,
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    build_plan,
)
from benchmarks.run_rc6_profile_matrix import build_command_plan
from benchmarks.run_rc6_profile_matrix import _validate_tf32_bootstrap
from benchmarks.run_rc6_profile_child import SCHEMA as TF32_BOOTSTRAP_SCHEMA
from benchmarks.run_rc6_profile_child import TF32_OFF as CHILD_TF32_OFF
from benchmarks import run_rc6_profile_child
from benchmarks.run_rc6_single_h100_stage_plan import (
    DRAFT_SCHEMA,
    EXPECTED_SCHEMAS,
    STAGES,
    execute,
    freeze_plan,
    validate_plan,
)


OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _component_reports() -> dict[str, dict[str, object]]:
    plan = build_plan()
    return {
        "environment": {
            "schema": "pssolver.rc6.environment.v1",
            "passed": True,
            "cuda_available": True,
            "visible_device_count": 1,
            "allocated_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "source_shadow_import": False,
            "pip_check": "PASS",
            "imports_from_site_packages": True,
            "installed_wheels": {
                "baseline": PROVIDER_BASELINE_WHEEL_SHA256,
                "candidate": PROVIDER_CANDIDATE_WHEEL_SHA256,
                "consumer": CONSUMER_WHEEL_SHA256,
            },
            "preflight": {
                "node_local_tmp_rename_noreplace": True,
                "pytest_tmpdir_is_unique_node_local_tmp": True,
                "host_compiler_stdatomic_and_cuda_compile": True,
                "empty_triton_cache_compile": True,
                "empty_torchinductor_cache_compile": True,
            },
        },
        "cuda_only": {
            "schema": "pssolver.rc6.cuda_only.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
            "passed": True,
            "passed_count": 7,
            "failed_count": 0,
            "skipped_count": 0,
            "deselected_count": 0,
            "node_ids": plan["cuda_only_tests"],
        },
        "contract_smokes": {
            "schema": "pssolver.rc6.contract_smokes.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
            "passed": True,
            "checks": plan["rc6_contract_smokes"],
        },
        "tf32_scope": {
            "schema": "pssolver.rc4_5_8.tf32_scope_h100.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
            "caller_state_count": 6,
            "runtime_family_case_count": 48,
            "original_state_restored": True,
            "passed": True,
        },
        "current_checkpoints": {
            "schema": "pssolver.rc6.current_checkpoint_h100.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "consumer_source_commit": CONSUMER_SOURCE,
            "passed": True,
            "family_count": 4,
            "direction_count": 2,
            "continuous_split_restart_passed": True,
            "transactional_restore_passed": True,
            "negative_gates_passed": True,
            "rejected_restart_output_remains_absent": True,
            "source_checkpoints_unchanged": True,
        },
        "released_checkpoints": {
            "schema": "pssolver.rc6.released_checkpoint_h100.v1",
            "passed": True,
            "primary_case_count": 13,
            "dispositions_match_cpu_record": True,
            "source_checkpoints_unchanged": True,
        },
        "numerical": {
            "schema": "pssolver.rc6.numerical_h100.v1",
            "provider_baseline_commit": PROVIDER_BASELINE,
            "provider_candidate_commit": PROVIDER_CANDIDATE,
            "passed": True,
            "families": [
                {
                    "family": family,
                    "completed_steps": 2,
                    "finite": True,
                    "fallback_used": False,
                    "byte_identical": True,
                    "relative_l2": 0.0,
                    "linf": 0.0,
                    "candidate_hermitian_violation": 0.0,
                }
                for family in plan["numerical_gate"]["runtime_families"]
            ],
        },
        "consumer": {
            "schema": "pssolver.rc6.consumer_h100.v1",
            "passed": True,
            "source_commit": CONSUMER_SOURCE,
            "wheel_sha256": CONSUMER_WHEEL_SHA256,
            "passed_count": 344,
            "skipped_count": 1,
            "failed_count": 0,
            "xfailed_count": 0,
            "deselected_count": 0,
            "skip_node_id": "tests/test_warmup_experiments.py::test_r5_benchmark",
            "identities": {
                "periodic": plan["consumer"]["periodic_identity_sha256"],
                "channel_fixed": plan["consumer"]["channel_fixed_identity_sha256"],
                "channel_tolerance": plan["consumer"]["channel_tolerance_identity_sha256"],
            },
            "guard_order_regression_passed": True,
            "periodic_and_channel_cuda_execution_finite": True,
            "checkpoint_replay_and_gradient_gates_passed": True,
            "scientific_control_result": False,
        },
    }


def _auxiliary() -> dict[str, object]:
    reports = _component_reports()
    return assemble(**reports)


def _write_profiles(root: Path) -> None:
    plan = build_plan()
    root.mkdir()
    for entry in plan["profile_matrix"]:
        role = entry["provider_role"]
        group = f"{entry['family']}:{entry['grid']}:{entry['trial']}"
        mean = 1.01 if role == "candidate" else 1.0
        samples = [mean] * 60
        profile = {
            "schema": "pssolver.rc4_5_8.runtime_family_profile.v1",
            "config": {
                "family": entry["family"],
                "trial": entry["trial"],
                "shape": plan["grid_contracts"][entry["grid"]]["shape"],
                "lengths": plan["grid_contracts"][entry["grid"]]["lengths"],
                "device": "cuda",
                "dtype": entry["dtype"],
            },
            "provider_identity": {
                "role": role,
                "commit": entry["provider_commit"],
                "wheel_sha256": entry["provider_wheel_sha256"],
            },
            "runtime_identity": {
                "requested": entry["family"],
                "effective": entry["family"],
                "fallback_used": False,
            },
            "environment": {
                "allocated_device": "cuda:0",
                "device_name": "NVIDIA H100 PCIe",
                "tf32_caller_before": OFF,
                "tf32_caller_after": OFF,
            },
            "timing": {
                "samples_seconds": samples,
                "mean_timestep_seconds": mean,
                "median_timestep_seconds": mean,
            },
            "memory": {
                "peak_allocated_bytes": 101 if role == "candidate" else 100,
                "peak_reserved_bytes": 101 if role == "candidate" else 100,
            },
            "transform_calls": {
                "forward": entry["expected_forward_raw_calls"],
                "inverse": entry["expected_inverse_raw_calls"],
                "forward_per_step": entry["expected_forward_raw_calls"] / 60,
                "inverse_per_step": entry["expected_inverse_raw_calls"] / 60,
            },
            "compiler": {"graph_breaks": 0, "compile_fallback": False},
            "initial_state_sha256": _hash("initial:" + group),
            "final_state_sha256": _hash("final:" + group),
            "completed_profile_steps": 60,
            "runtime_completed_steps": 70,
            "finite": True,
        }
        (root / f"{entry['key']}.json").write_text(
            json.dumps(profile, allow_nan=False), encoding="utf-8"
        )


def test_rc6_profile_command_plan_is_complete_order_balanced_and_shell_free(tmp_path):
    command = build_command_plan(
        baseline_python=tmp_path / "baseline/bin/python",
        candidate_python=tmp_path / "candidate/bin/python",
        support_root=tmp_path / "support",
        output_root=tmp_path / "output",
    )
    assert command["profile_count"] == 192
    assert [row["sequence"] for row in command["rows"]] == list(range(1, 193))
    assert len({row["key"] for row in command["rows"]}) == 192
    assert len({row["compiler_cache"] for row in command["rows"]}) == 192
    assert len({row["tf32_bootstrap_output"] for row in command["rows"]}) == 192
    assert all(isinstance(row["command"], list) and "-c" not in row["command"] for row in command["rows"])
    plan = build_plan()
    for entry, row in zip(plan["profile_matrix"], command["rows"]):
        assert row["provider_role"] == entry["provider_role"]
        assert row["command"][row["command"].index("--device") + 1] == "cuda"
        assert row["command"][1].endswith("run_rc6_profile_child.py")
        assert row["command"][row["command"].index("--profile-script") + 1].endswith(
            "profile_rc458_runtime_family.py"
        )


def test_rc6_profile_child_persists_tf32_policy_before_profiler(tmp_path, monkeypatch):
    bootstrap = tmp_path / "bootstrap.json"
    called: list[list[str]] = []
    before = run_rc6_profile_child._capture_tf32()
    monkeypatch.setattr(
        run_rc6_profile_child.profile,
        "main",
        lambda argv: called.append(list(argv)) or 0,
    )
    try:
        result = run_rc6_profile_child.main(
            [
                "--tf32-bootstrap-output", str(bootstrap),
                "--profile-script", str(Path(run_rc6_profile_child.profile.__file__)),
                "--family", "plane_legacy_production",
            ]
        )
        assert result == 0
        assert called == [["--family", "plane_legacy_production"]]
        value = json.loads(bootstrap.read_text(encoding="utf-8"))
        assert value["schema"] == TF32_BOOTSTRAP_SCHEMA
        assert value["policy"] == CHILD_TF32_OFF
        assert value["after"] == CHILD_TF32_OFF
        assert value["assignment_errors"] == []
        assert value["passed"] is True
        assert value["profile_start_authorized_after_persisted_enforcement"] is True
        assert _validate_tf32_bootstrap(
            bootstrap, Path(run_rc6_profile_child.profile.__file__)
        ) == value
    finally:
        torch = run_rc6_profile_child.torch
        torch.set_float32_matmul_precision(str(before["float32_matmul_precision"]))
        torch.backends.cuda.matmul.allow_tf32 = bool(before["cuda_matmul_allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(before["cudnn_allow_tf32"])


def test_rc6_profile_child_validator_rejects_tf32_drift(tmp_path):
    profile_script = Path(run_rc6_profile_child.profile.__file__).resolve()
    report = {
        "schema": TF32_BOOTSTRAP_SCHEMA,
        "profile_script": str(profile_script),
        "profile_script_sha256": hashlib.sha256(profile_script.read_bytes()).hexdigest(),
        "policy": CHILD_TF32_OFF,
        "before": CHILD_TF32_OFF,
        "after": {**CHILD_TF32_OFF, "cudnn_allow_tf32": True},
        "assignment_errors": [],
        "passed": True,
        "profile_start_authorized_after_persisted_enforcement": True,
    }
    output = tmp_path / "bootstrap.json"
    output.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="enforced state"):
        _validate_tf32_bootstrap(output, profile_script)


def test_rc6_auxiliary_assembler_accepts_exact_reports_and_rejects_drift():
    result = _auxiliary()
    assert result["passed"] is True
    assert result["checkpoints"]["released_primary_case_count"] == 13
    reports = _component_reports()
    reports["released_checkpoints"]["primary_case_count"] = 12
    with pytest.raises(ValueError, match="released case count"):
        assemble(**reports)


def test_rc6_analyzer_accepts_balanced_192_profile_matrix(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    result = analyze(root, _auxiliary())
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["qualification_complete"] is True
    assert result["profile_count"] == 192
    assert result["group_count"] == 16
    assert all(group["passed"] for group in result["groups"])
    assert all(group["trials_at_or_below_individual_reference"] == 6 for group in result["groups"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("performance", "performance or memory gate failed"),
        ("transform", "raw forward differs"),
        ("fallback", "runtime fallback"),
        ("final", "final state differs"),
    ],
)
def test_rc6_analyzer_rejects_profile_drift(tmp_path, mutation, message):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "candidate_plane_legacy_production_plane_R128_trial1.json"
    value = json.loads(target.read_text(encoding="utf-8"))
    if mutation == "performance":
        value["timing"]["samples_seconds"] = [1.2] * 60
        value["timing"]["mean_timestep_seconds"] = 1.2
        value["timing"]["median_timestep_seconds"] = 1.2
    elif mutation == "transform":
        value["transform_calls"]["forward"] += 1
        value["transform_calls"]["forward_per_step"] = value["transform_calls"]["forward"] / 60
    elif mutation == "fallback":
        value["runtime_identity"]["fallback_used"] = True
    else:
        value["final_state_sha256"] = _hash("different")
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        analyze(root, _auxiliary())


def _draft_stage_plan(tmp_path: Path) -> tuple[dict[str, object], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "h100"
    helper = tmp_path / "stage.py"
    helper.write_text(
        "import json, pathlib, sys\n"
        "schema, output, name = sys.argv[1:4]\n"
        "value = {'schema': schema, 'passed': True}\n"
        "if name == 'analyze':\n"
        "    value.update(classification='PASS_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION', qualification_complete=True)\n"
        "path = pathlib.Path(output)\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_text(json.dumps(value))\n",
        encoding="utf-8",
    )
    stages = []
    for name in STAGES:
        artifact = output / "results" / f"{name}.json"
        stages.append(
            {
                "name": name,
                "command": [sys.executable, str(helper), EXPECTED_SCHEMAS[name], str(artifact), name],
                "artifact": str(artifact),
                "expected_schema": EXPECTED_SCHEMAS[name],
                "environment": {},
            }
        )
    return (
        {
            "schema": DRAFT_SCHEMA,
            "provider_baseline_commit": PROVIDER_BASELINE,
            "provider_candidate_commit": PROVIDER_CANDIDATE,
            "consumer_source_commit": CONSUMER_SOURCE,
            "stages": stages,
        },
        output,
    )


def test_rc6_stage_plan_freezes_hashes_and_executes_in_order(tmp_path):
    draft, output = _draft_stage_plan(tmp_path)
    frozen = freeze_plan(draft, output)
    assert [row["name"] for row in frozen["stages"]] == list(STAGES)
    assert all(len(row["command_file_bindings"]) == 2 for row in frozen["stages"])
    assert validate_plan(frozen, output) == frozen["stages"]
    result = execute(frozen, output)
    assert result["passed"] is True
    assert [row["name"] for row in result["completed"]] == list(STAGES)


def test_rc6_stage_plan_rejects_hash_drift_inline_code_and_output_escape(tmp_path):
    draft, output = _draft_stage_plan(tmp_path)
    frozen = freeze_plan(draft, output)
    helper = Path(frozen["stages"][0]["command"][1])
    helper.write_text("# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        validate_plan(frozen, output)

    draft, output = _draft_stage_plan(tmp_path / "inline")
    draft["stages"][0]["command"] = [sys.executable, "-c", "pass"]
    with pytest.raises(ValueError, match="inline Python"):
        freeze_plan(draft, output)

    draft, output = _draft_stage_plan(tmp_path / "escape")
    draft["stages"][0]["artifact"] = str(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="escapes output root"):
        freeze_plan(draft, output)
