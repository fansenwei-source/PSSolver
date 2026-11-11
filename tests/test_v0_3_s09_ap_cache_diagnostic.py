from __future__ import annotations

import itertools
import json
from pathlib import Path

from benchmarks.analyze_v0_3_s09_ap_cache_diagnostic import _ratio_summary, analyze
from benchmarks.analyze_v0_3_s09_h100 import _profile_center
from benchmarks.plan_v0_3_s09_ap_cache_diagnostic import (
    BASELINE_COMMIT,
    BASELINE_WHEEL_SHA256,
    CANDIDATE_COMMIT,
    CANDIDATE_WHEEL_SHA256,
    build_plan,
    build_profile_matrix,
)
ROOT = Path(__file__).resolve().parents[1]


def test_h100_plan_has_balanced_72_profile_diagnostic_contract() -> None:
    plan = build_plan(mode="h100")
    profiles = plan["profiles"]
    assert len(profiles) == 72
    assert [item["sequence"] for item in profiles] == list(range(1, 73))
    assert {item["diagnostic_role"] for item in profiles} == {
        "baseline",
        "candidate_cache_on",
        "candidate_revert_proxy",
    }
    assert {item["family"] for item in profiles} == {
        "channel_legacy",
        "channel_compiled_v2",
    }
    assert {item["grid"] for item in profiles} == {
        "channel_R128x20x20",
        "channel_R512x40x40",
    }
    assert all(item["timed_steps"] == 60 for item in profiles)
    assert all(item["samples_per_profile"] == 60 for item in profiles)
    assert all(item["isolated_subprocess"] is True for item in profiles)
    assert all(item["isolated_empty_triton_cache"] is True for item in profiles)
    assert all(item["isolated_empty_torchinductor_cache"] is True for item in profiles)
    assert plan["diagnostic_only"] is True
    assert plan["qualification_complete"] is False
    assert plan["s09_complete"] is False


def test_roles_use_all_six_permutations_and_bind_exact_provider_identities() -> None:
    rows = build_profile_matrix(mode="h100")
    for family in ("channel_legacy", "channel_compiled_v2"):
        for grid in ("channel_R128x20x20", "channel_R512x40x40"):
            group = [row for row in rows if row["family"] == family and row["grid"] == grid]
            for trial in range(1, 7):
                current = [row for row in group if row["trial"] == trial]
                assert sorted(row["order_position"] for row in current) == [1, 2, 3]
                assert {row["diagnostic_role"] for row in current} == {
                    "baseline",
                    "candidate_cache_on",
                    "candidate_revert_proxy",
                }
                for row in current:
                    if row["diagnostic_role"] == "baseline":
                        assert row["provider_commit"] == BASELINE_COMMIT
                        assert row["provider_wheel_sha256"] == BASELINE_WHEEL_SHA256
                    else:
                        assert row["provider_commit"] == CANDIDATE_COMMIT
                        assert row["provider_wheel_sha256"] == CANDIDATE_WHEEL_SHA256
            orders = [
                tuple(
                    row["diagnostic_role"]
                    for row in sorted(
                        (item for item in group if item["trial"] == trial),
                        key=lambda item: item["order_position"],
                    )
                )
                for trial in range(1, 7)
            ]
            assert set(orders) == set(itertools.permutations((
                "baseline", "candidate_cache_on", "candidate_revert_proxy"
            )))
            for role in ("baseline", "candidate_cache_on", "candidate_revert_proxy"):
                assert [order.index(role) for order in orders].count(0) == 2
                assert [order.index(role) for order in orders].count(1) == 2
                assert [order.index(role) for order in orders].count(2) == 2
            for left, right in itertools.combinations(
                ("baseline", "candidate_cache_on", "candidate_revert_proxy"), 2
            ):
                assert sum(order.index(left) < order.index(right) for order in orders) == 3


def test_cpu_rehearsal_preserves_full_role_matrix_but_is_not_measurement() -> None:
    plan = build_plan(mode="cpu_rehearsal")
    assert len(plan["profiles"]) == 72
    assert all(item["device"] == "cpu" for item in plan["profiles"])
    assert all(item["timed_steps"] == 2 for item in plan["profiles"])
    assert plan["measurement_contract"]["formal_samples_per_profile"] == 60
    assert plan["diagnostic_only"] is True
    environment = plan["job_environment_contract"]
    assert environment["unset_ld_library_path"] is True
    assert environment["host_compiler"] == {
        "cc": "/opt/ohpc/pub/compiler/gcc/7.3.0/bin/gcc",
        "cxx": "/opt/ohpc/pub/compiler/gcc/7.3.0/bin/g++",
    }
    assert environment["cuda_module"] == "cuda/12.1"
    assert environment["node_local_tmpdir"] is True
    assert environment["torch_compile_cuda_smoke_before_matrix"] is True


def test_trial_ratio_uses_exact_frozen_profile_center() -> None:
    def row(trial: int, samples: list[float], position: int) -> dict[str, object]:
        return {
            "trial": trial,
            "samples": samples,
            "mean": sum(samples) / len(samples),
            "center": _profile_center(samples, 10),
            "position": position,
            "peak_allocated": 100,
            "peak_reserved": 200,
        }

    reference = []
    observed = []
    expected = []
    for trial in range(1, 7):
        baseline = [1.0] * 50 + [100.0] * 10
        candidate = [2.0] * 50 + [1000.0] * 10
        reference.append(row(trial, baseline, 1 if trial <= 3 else 2))
        observed.append(row(trial, candidate, 2 if trial <= 3 else 1))
        expected.append(_profile_center(candidate, 10) / _profile_center(baseline, 10))
    summary = _ratio_summary(reference, observed)
    assert summary["six_trial_ratios"] == expected
    assert summary["six_trial_ratios"] == [2.0] * 6
    assert summary["six_trial_ratios"][0] != (
        observed[0]["mean"] / reference[0]["mean"]
    )


def test_analyzer_is_descriptive_and_requires_all_cache_proofs(tmp_path: Path) -> None:
    plan = build_plan(mode="cpu_rehearsal")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    manifest_rows = []
    for row in plan["profiles"]:
        role = row["diagnostic_role"]
        trial = int(row["trial"])
        role_scale = {
            "baseline": 1.0,
            "candidate_cache_on": 1.01,
            "candidate_revert_proxy": 1.02,
        }[role]
        output = tmp_path / f"{row['key']}.json"
        state_key = f"{row['family']}-{row['grid']}-{trial}"
        value = {
            "diagnostic_role": role,
            "config": {
                "family": row["family"],
                "trial": trial,
            },
            "timing": {
                "samples_seconds": [role_scale, role_scale],
            },
            "memory": {
                "peak_allocated_bytes": int(1000 * role_scale),
                "peak_reserved_bytes": int(2000 * role_scale),
            },
            "initial_state_sha256": state_key,
            "final_state_sha256": state_key,
            "cache_control": {
                "proved_revert_proxy": role == "candidate_revert_proxy",
            },
            "finite": True,
        }
        output.write_text(json.dumps(value), encoding="utf-8")
        manifest_rows.append({"key": row["key"], "result": str(output)})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"plan": str(plan_path), "profiles": manifest_rows}),
        encoding="utf-8",
    )
    result = analyze(manifest_path)
    assert result["classification"] == "S09_AP_CACHE_DIAGNOSTIC_DATA_COMPLETE"
    assert result["diagnostic_only"] is True
    assert result["qualification_complete"] is False
    assert result["s09_complete"] is False
    assert result["decision_made"] is False
    assert result["profile_count"] == 72
    assert result["all_revert_proxy_proofs_passed"] is True
    assert result["candidate_cache_on_revert_proxy_state_byte_identity"] is True
    assert result["trial_ratio_estimator"] == (
        "frozen_profile_center_six_fixed_block_means_median"
    )
    assert len(result["groups"]) == 4


def test_job_is_single_gpu_diagnostic_without_qualification_claims() -> None:
    job = (ROOT / "jobs/v0_3_s09_ap_cache_diagnostic_h100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:H100:1" in job
    assert "#SBATCH --job-name=pssolver-v03-s09-apdiag" in job
    assert "pssolver_v0_3_s09_ap_cache_diagnostic_h100_85ba3ae_20261026_v1/logs" in job
    assert "#SBATCH --no-requeue" in job
    assert "S09_DIAG_OUTPUT_ROOT" in job
    assert "profile_count" in job
    assert 'assert value["qualification_complete"] is False' in job
    assert 'assert value["s09_complete"] is False' in job
    assert "_record_memory_history" not in job
    assert "run_v0_3_s09_ap_cache_memory_snapshot.py" in job
    assert "candidate_revert_proxy.pickle" not in job
    assert "for role in candidate_cache_on candidate_revert_proxy" in job
    assert "unset LD_LIBRARY_PATH" in job
    assert "export CC=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/gcc" in job
    assert "export CXX=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/g++" in job
    assert "source /opt/ohpc/admin/lmod/lmod/init/bash" in job
    assert "module load cuda/12.1" in job
    assert 'trap \'rm -rf "${node_root}"\' EXIT' in job
    assert 'export TMPDIR="${pytest_tmp}"' in job
    assert "cc-version.txt" in job
    assert "module-list.txt" in job
    assert "nvcc-version.txt" in job
    preflight = "run_v0_3_s09_ap_cache_diagnostic_preflight.py"
    matrix = "run_v0_3_s09_ap_cache_diagnostic.py"
    assert job.index(preflight, job.index("# Reuse the cumulative")) < job.index(
        matrix, job.index("# Reuse the cumulative")
    )
    assert "sbatch" not in job


def test_runner_accepts_normal_venv_python_symlinks_but_support_is_regular() -> None:
    runner = (ROOT / "benchmarks/run_v0_3_s09_ap_cache_diagnostic.py").read_text(
        encoding="utf-8"
    )
    assert "if not python.is_file():" in runner
    assert "if not path.is_file() or path.is_symlink():" in runner
