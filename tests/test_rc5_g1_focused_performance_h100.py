"""Contracts for the rc5 G-1 focused performance/memory recovery."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import pytest

from benchmarks.analyze_rc5_g1_focused_performance_h100 import (
    PASS_CLASSIFICATION,
    analyze,
)
from benchmarks.plan_rc5_g1_focused_performance_h100 import (
    FOCUSED_CELLS,
    SCHEMA,
    TIMING_CV_MAX,
    V5_MANIFEST_SHA256,
    V6_MANIFEST_SHA256,
    build_plan,
    main as plan_main,
)
from benchmarks.run_rc5_g1_focused_profile_matrix import (
    _preflight,
    build_command_plan,
)


OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _hash(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_profiles(root: Path) -> None:
    plan = build_plan()
    root.mkdir()
    for entry in plan["profile_matrix"]:
        role = entry["provider_role"]
        group = f"{entry['family']}:{entry['grid']}:trial{entry['trial']}"
        mean = (1.0 + 0.001 * int(entry["trial"])) * (
            1.01 if role == "candidate" else 1.0
        )
        samples = [mean] * 60
        final = _hash("final:" + group)
        if role == "candidate" and entry["family"] == "periodic_functional":
            final = _hash("repaired:" + group)
        profile = {
            "schema": "pssolver.rc4_5_8.runtime_family_profile.v1",
            "config": {
                "family": entry["family"],
                "provider_role": role,
                "provider_commit": entry["provider_commit"],
                "provider_wheel_sha256": entry["provider_wheel_sha256"],
                "trial": entry["trial"],
                "shape": plan["grid_contracts"][entry["grid"]]["shape"],
                "lengths": plan["grid_contracts"][entry["grid"]]["lengths"],
                "device": "cuda",
                "dtype": entry["dtype"],
                "warmup_steps": 10,
                "profile_steps": 60,
            },
            "provider_identity": {
                "role": role,
                "commit": entry["provider_commit"],
                "wheel_sha256": entry["provider_wheel_sha256"],
            },
            "runtime_identity": {
                "requested": entry["runtime_path"],
                "effective": entry["runtime_path"],
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
                "sample_standard_deviation_seconds": 0.0,
                "coefficient_of_variation": 0.0,
            },
            "memory": {
                "peak_allocated_bytes": 101 if role == "candidate" else 100,
                "peak_active_bytes": 101 if role == "candidate" else 100,
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
            "final_state_sha256": final,
            "completed_profile_steps": 60,
            "runtime_completed_steps": 70,
            "finite": True,
        }
        (root / f"{entry['key']}.json").write_text(
            json.dumps(profile, allow_nan=False), encoding="utf-8"
        )


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _replace_samples(profile: dict[str, object], samples: list[float]) -> None:
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    profile["timing"] = {
        "samples_seconds": samples,
        "mean_timestep_seconds": mean,
        "median_timestep_seconds": statistics.median(samples),
        "sample_standard_deviation_seconds": stdev,
        "coefficient_of_variation": stdev / mean,
    }


def test_focused_plan_is_exact_order_balanced_and_keeps_original_thresholds():
    plan = build_plan()
    assert plan["schema"] == SCHEMA
    assert plan["profile_count"] == 48
    assert len(FOCUSED_CELLS) == 4
    assert [entry["sequence"] for entry in plan["profile_matrix"]] == list(
        range(1, 49)
    )
    assert len({entry["key"] for entry in plan["profile_matrix"]}) == 48
    for cell in FOCUSED_CELLS:
        rows = [
            row
            for row in plan["profile_matrix"]
            if row["family"] == cell["family"] and row["grid"] == cell["grid"]
        ]
        assert len(rows) == 12
        orders = [
            [row["provider_role"] for row in rows if row["trial"] == trial]
            for trial in range(1, 7)
        ]
        assert orders == [
            ["baseline", "candidate"],
            ["candidate", "baseline"],
            ["baseline", "candidate"],
            ["candidate", "baseline"],
            ["baseline", "candidate"],
            ["candidate", "baseline"],
        ]
    assert plan["timing_validity"] == {
        "raw_sample_count": 60,
        "sample_standard_deviation": "sample",
        "coefficient_of_variation_max": TIMING_CV_MAX,
        "reported_summary_must_match_raw_samples": True,
        "invalid_profile_disposition": "fail_closed_stop",
        "outlier_removal_allowed": False,
        "trial_replacement_allowed": False,
    }
    assert plan["thresholds"]["mean_timestep_ratio_max"] == 1.03
    assert plan["thresholds"]["median_of_profile_means_ratio_max"] == 1.03
    assert plan["thresholds"]["every_paired_mean_timestep_ratio_max"] == 1.05
    assert plan["thresholds"]["peak_reserved_group_ratio_max"] == 1.03
    assert plan["immutable_inputs"]["v5_manifest_sha256"] == V5_MANIFEST_SHA256
    assert plan["immutable_inputs"]["v6_manifest_sha256"] == V6_MANIFEST_SHA256
    assert not any(plan["authorization"].values())


def test_focused_plan_cli_writes_strict_non_overwriting_json(tmp_path):
    output = tmp_path / "plan.json"
    assert plan_main(["--output", str(output)]) == 0
    assert _load(output) == build_plan()
    with pytest.raises(FileExistsError):
        plan_main(["--output", str(output)])
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_focused_analyzer_accepts_clean_new_evidence_only(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    result = analyze(root)
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["focused_evidence_complete"] is True
    assert result["focused_cell_count"] == 4
    assert result["profile_count"] == 48
    assert result["eligible_for_cumulative_reassembly"] is True
    assert result["qualification_complete"] is False
    assert result["g1_complete"] is False
    assert all(group["passed"] for group in result["groups"])
    assert all(
        pair["execution_order"]
        == (["baseline", "candidate"] if pair["trial"] % 2 else ["candidate", "baseline"])
        for group in result["groups"]
        for pair in group["pairs"]
    )


def test_focused_analyzer_rejects_timing_cv_instead_of_dropping_outlier(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "candidate_plane_compiled_v2_plane_R128_trial1.json"
    profile = _load(target)
    _replace_samples(profile, [1.0] * 59 + [2.0])
    _save(target, profile)
    with pytest.raises(ValueError, match="timing validity failed"):
        analyze(root)


def test_focused_analyzer_rejects_reported_timing_summary_drift(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "baseline_periodic_functional_periodic_R128_trial2.json"
    profile = _load(target)
    profile["timing"]["mean_timestep_seconds"] *= 1.001
    _save(target, profile)
    with pytest.raises(ValueError, match="reported mean does not match"):
        analyze(root)


def test_focused_analyzer_rejects_one_paired_timing_regression(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / (
        "candidate_channel_compiled_v2_channel_R128x20x20_trial3.json"
    )
    profile = _load(target)
    samples = [value * 1.06 for value in profile["timing"]["samples_seconds"]]
    _replace_samples(profile, samples)
    _save(target, profile)
    with pytest.raises(ValueError, match="paired timestep regression"):
        analyze(root)


def test_focused_analyzer_rejects_group_reserved_memory_regression(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / (
        "candidate_channel_functional_channel_R512x40x40_trial6.json"
    )
    profile = _load(target)
    profile["memory"]["peak_reserved_bytes"] = 104
    _save(target, profile)
    with pytest.raises(ValueError, match="reserved memory regression"):
        analyze(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("initial", "initial state differs"),
        ("transform", "raw forward transform count differs"),
        ("runtime", "effective runtime differs"),
        ("tf32", "TF32 after differs"),
    ],
)
def test_focused_analyzer_rejects_identity_transform_and_policy_tamper(
    tmp_path, mutation, message
):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "candidate_plane_compiled_v2_plane_R128_trial4.json"
    profile = _load(target)
    if mutation == "initial":
        profile["initial_state_sha256"] = "f" * 64
    elif mutation == "transform":
        profile["transform_calls"]["forward"] += 1
        profile["transform_calls"]["forward_per_step"] = (
            profile["transform_calls"]["forward"] / 60
        )
    elif mutation == "runtime":
        profile["runtime_identity"]["effective"] = "legacy_production"
    else:
        profile["environment"]["tf32_caller_after"]["cudnn_allow_tf32"] = True
    _save(target, profile)
    with pytest.raises(ValueError, match=message):
        analyze(root)


def test_focused_analyzer_rejects_missing_or_extra_profile_files(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    missing = root / "baseline_plane_compiled_v2_plane_R128_trial1.json"
    missing.unlink()
    with pytest.raises(ValueError, match="profile missing"):
        analyze(root)

    root = tmp_path / "profiles-extra"
    _write_profiles(root)
    (root / "unregistered.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="profile file set differs"):
        analyze(root)


def _command_plan(tmp_path: Path) -> dict[str, object]:
    baseline = tmp_path / "baseline" / "bin" / "python"
    candidate = tmp_path / "candidate" / "bin" / "python"
    baseline.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    baseline.write_bytes(b"python\n")
    candidate.write_bytes(b"python\n")
    support = tmp_path / "support"
    (support / "benchmarks").mkdir(parents=True)
    (support / "benchmarks/profile_rc5_g1_runtime_family.py").write_text(
        "# support\n", encoding="utf-8"
    )
    return build_command_plan(
        baseline_python=baseline,
        candidate_python=candidate,
        support_root=support,
        output_root=tmp_path / "output",
    )


def test_focused_runner_command_plan_is_exact_balanced_and_shell_free(tmp_path):
    plan = _command_plan(tmp_path)
    rows = plan["rows"]
    assert plan["profile_count"] == 48
    assert [row["sequence"] for row in rows] == list(range(1, 49))
    assert len({row["key"] for row in rows}) == 48
    assert len({row["compiler_cache"] for row in rows}) == 48
    assert all(isinstance(row["command"], list) for row in rows)
    assert all("--device" in row["command"] for row in rows)
    assert all(
        row["command"][row["command"].index("--device") + 1] == "cuda"
        for row in rows
    )
    assert all(
        row["support_script"].endswith("profile_rc5_g1_runtime_family.py")
        for row in rows
    )
    _preflight(plan)


def test_focused_runner_preflight_rejects_source_shadow_and_existing_output(tmp_path):
    plan = _command_plan(tmp_path)
    support = Path(plan["support_root"])
    (support / "pssolver").mkdir()
    with pytest.raises(RuntimeError, match="must not contain"):
        _preflight(plan)

    (support / "pssolver").rmdir()
    Path(plan["output_root"]).mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _preflight(plan)


def test_focused_slurm_script_is_single_h100_fail_closed_and_not_g1_complete():
    root = Path(__file__).resolve().parents[1]
    source = (root / "jobs/rc5_g1_focused_performance_h100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=hagan-gpu" in source
    assert "#SBATCH --account=hagan-lab" in source
    assert "#SBATCH --qos=medium" in source
    assert "#SBATCH --gres=gpu:H100:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "run_rc5_g1_checkpoint_tmp_preflight.py" in source
    assert "run_rc5_g1_environment.py" in source
    assert "run_rc5_g1_focused_profile_matrix.py" in source
    assert 'touch "${RC5_G1_OUTPUT_ROOT}/FOCUSED_COMPLETE"' in source
    assert 'touch "${RC5_G1_OUTPUT_ROOT}/COMPLETE"' not in source
    assert "sbatch" not in "\n".join(
        line for line in source.splitlines() if not line.startswith("#SBATCH")
    )
