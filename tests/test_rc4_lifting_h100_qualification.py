"""Frozen contract and fail-closed analyzer tests for RC4.3.7."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.analyze_rc437_lifting_h100 import (
    ACTIVE_RATIO_MAX,
    ALLOCATED_RATIO_MAX,
    GRIDS,
    INPUTS,
    MEAN_RATIO_MAX,
    MEDIAN_RATIO_MAX,
    PASS,
    TRIALS,
    analyze,
)
from benchmarks.analyze_rc437_lifting_h100_recovery import (
    SCHEMA as RECOVERY_ANALYSIS_SCHEMA,
    analyze as analyze_recovery,
)
from benchmarks.profile_rc437_plane_runtime import (
    SCHEMA as PROFILE_SCHEMA,
    _atomic_json as write_profile_json,
)
from benchmarks.qualify_rc437_lifting_cuda import (
    PASS as CUDA_PASS,
    SCHEMA as CUDA_SCHEMA,
    _build,
    _initial,
    _wall_residual,
)
from benchmarks.qualify_rc437_lifting_cuda_recovery import (
    MEASUREMENT_LAYER,
    SCHEMA as RECOVERY_CUDA_SCHEMA,
    _direct_conditioning_evidence,
)
from pssolver.workflows.plane_checkpoint import capture_plane_checkpoint


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "notes/PSSolver_v0_2_0rc4_rc437_h100_qualification_plan.json"
RECOVERY_PLAN = (
    ROOT / "notes/PSSolver_v0_2_0rc4_rc437_h100_recovery_plan.json"
)
PROFILE_SERIALIZATION_RECOVERY_PLAN = (
    ROOT
    / "notes/PSSolver_v0_2_0rc4_rc437_profile_serialization_recovery_plan.json"
)
BASELINE = "1b15bf856cb6597ae709ee7496cde96d0b2cc7d1"
CANDIDATE = "e1dea73c5d0f00e097599c4ff2c94ed7dceaa915"
FAILED_QUALIFICATION = "d9d340d7fe49aa245802a975b6eaa49906815cbb"
CORRECT_LAYER_RECOVERY = "2121cfbffd31857484dc912f9413465d5a2b4806"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha256(revision: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{revision}:{path}"],
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _plan() -> dict[str, object]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _recovery_plan() -> dict[str, object]:
    return json.loads(RECOVERY_PLAN.read_text(encoding="utf-8"))


def _profile_serialization_recovery_plan() -> dict[str, object]:
    return json.loads(
        PROFILE_SERIALIZATION_RECOVERY_PLAN.read_text(encoding="utf-8")
    )


def _profile(
    grid: str,
    trial: int,
    role: str,
    ratio: float = 1.01,
    root: Path | None = None,
) -> dict[str, object]:
    base = (0.007 if grid == "R128" else 0.045) * (1.0 + trial * 0.001)
    value = base * (ratio if role == "candidate" else 1.0)
    commit = CANDIDATE if role == "candidate" else BASELINE
    source_root = root if root is not None else Path("/synthetic") / role
    return {
        "schema": PROFILE_SCHEMA,
        "config": {
            "runtime_path": "legacy_production", "trial": trial, "shape": list(GRIDS[grid]),
            "lengths": [100.0, 100.0, 20.0], "device": "cuda", "dtype": "float64",
            "dt": 0.005, "activity_number": 18.0, "dealias_rule": "cubic_half",
            "projected_transform_execution": "truncated", "transform_execution_order": "real_first",
            "spectral_storage": "hermitian_half", "molecular_field_linear_space": "spectral",
            "stress_divergence_sum_space": "spectral", "pointwise_execution": "compile",
            "reuse_q_gradients": True, "spectral_refresh_interval": None,
            "warmup_steps": 10, "profile_steps": 50, "seed": 24,
            "initial_q_path": INPUTS[grid],
        },
        "runtime_identity": {"requested": "legacy_production", "effective": "legacy_production", "fallback_used": False},
        "environment": {
            "git": {"head": commit, "dirty": False, "status": ""},
            "allocated_device": "cuda:0", "device_name": "NVIDIA H100 PCIe",
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "pssolver_import": str(source_root / "pssolver/__init__.py"),
        },
        "timing": {"mean_timestep_seconds": value, "median_timestep_seconds": value},
        "memory": {"peak_allocated_bytes": 1010 if role == "candidate" else 1000, "peak_active_bytes": 1010 if role == "candidate" else 1000, "peak_reserved_bytes": 2020 if role == "candidate" else 2000},
        "transform_calls": {"forward_per_step": 7.0, "inverse_per_step": 32.0},
        "pointwise_compile": {
            "execution": {"requested": "compile", "effective": "compile", "fallback_allowed": False},
            "during_build": {"graph_breaks": 0}, "during_warmup": {"graph_breaks": 0}, "during_profile": {"graph_breaks": 0},
        },
        "initial_q_sha256": ("1" if grid == "R128" else "2") * 64,
        "final_state_sha256": ("3" if grid == "R128" else "4") * 64,
        "completed_steps": 60, "finite": True,
    }


def _profiles(
    role: str,
    ratio: float = 1.01,
    root: Path | None = None,
) -> list[dict[str, object]]:
    return [
        _profile(grid, trial, role, ratio, root)
        for grid in GRIDS
        for trial in TRIALS
    ]


def _cuda(candidate_root: Path) -> dict[str, object]:
    imported = candidate_root / "pssolver/__init__.py"
    imported.parent.mkdir(parents=True, exist_ok=True)
    imported.write_text("", encoding="utf-8")
    return {
        "schema": CUDA_SCHEMA, "classification": CUDA_PASS, "passed": True,
        "environment": {"device": "cuda:0", "device_name": "NVIDIA H100 PCIe", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "pssolver_import": str(imported)},
        "fresh_conditioning": {"policy": "localized_quintic_wall_transition_v1", "cross_device_relative_l2": 0.0, "interior_byte_identical_to_unconditioned": True, "transition_region_changed": True, "wall_residual_linf": 0.0},
        "restart": {"restored_steps": 2, "all_byte_identical": True, "finite": True, "fresh_construction_provenance": {"conditioning_applied": True}},
        "qualification_boundary": {"internal_lifting_infrastructure_only": True, "public_lifted_plane_reopened": False, "localized_policy_publicly_qualified": False, "production_default_changed": False},
    }


def _analyze(tmp_path: Path, baseline=None, candidate=None, cuda=None):
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    report = _cuda(candidate_root) if cuda is None else cuda
    return analyze(
        _profiles("baseline", root=baseline_root) if baseline is None else baseline,
        _profiles("candidate", root=candidate_root) if candidate is None else candidate,
        report,
        baseline_commit=BASELINE,
        candidate_commit=CANDIDATE,
        baseline_root=baseline_root,
        candidate_root=candidate_root,
    )


def _recovery_cuda(candidate_root: Path) -> dict[str, object]:
    report = _cuda(candidate_root)
    report["schema"] = RECOVERY_CUDA_SCHEMA
    report["fresh_conditioning"].update(
        {
            "measurement_layer": MEASUREMENT_LAYER,
            "cuda": {
                "measurement_layer": MEASUREMENT_LAYER,
                "components": {
                    name: {
                        "interior_byte_identical_to_unconditioned": True,
                        "transition_region_changed": True,
                    }
                    for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
                },
            },
            "post_projection_diagnostic": {
                "measurement_layer": (
                    "adapter_fields_spatial_after_global_truncated_"
                    "spectral_projection"
                ),
                "qualification_gate": False,
                "all_interior_byte_identical_to_unconditioned": False,
            },
        }
    )
    return report


def _analyze_recovery(tmp_path: Path, cuda=None):
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    report = _recovery_cuda(candidate_root) if cuda is None else cuda
    return analyze_recovery(
        _profiles("baseline", root=baseline_root),
        _profiles("candidate", root=candidate_root),
        report,
        baseline_commit=BASELINE,
        candidate_commit=CANDIDATE,
        baseline_root=baseline_root,
        candidate_root=candidate_root,
    )


def test_rc437_plan_freezes_honest_claim_boundary_and_single_job():
    plan = _plan()
    assert plan["source_identity"]["baseline_commit"] == BASELINE
    assert plan["source_identity"]["candidate_runtime_commit"] == CANDIDATE
    assert plan["claim_boundary"]["public_lifted_plane_reopened"] is False
    assert plan["claim_boundary"]["localized_policy_publicly_qualified"] is False
    assert plan["profile_matrix"]["profile_count"] == 12
    assert plan["profile_matrix"]["cross_role_final_sha_identity_required"] is True
    assert plan["slurm"]["formal_submission_limit"] == 1
    assert plan["slurm"]["automatic_retry"] is False
    assert plan["authorization"]["h100_submission_authorized"] is False


def test_rc437_plan_binds_tools_inputs_and_master_contract():
    plan = _plan()
    for name in ("profile_runner", "cuda_correctness_runner", "analyzer"):
        relative = plan["qualification_support"][name]
        assert _git_blob_sha256(FAILED_QUALIFICATION, relative) == plan[
            "qualification_support"
        ][f"{name}_sha256"]
    assert _sha256(ROOT / plan["source_identity"]["master_plan"]) == plan["source_identity"]["master_plan_sha256"]
    assert _sha256(ROOT / plan["source_identity"]["rc4_3_6_record"]) == plan["source_identity"]["rc4_3_6_record_sha256"]
    assert plan["thresholds"] == {
        "mean_candidate_over_baseline_timestep_max": MEAN_RATIO_MAX,
        "median_candidate_over_baseline_timestep_max": MEDIAN_RATIO_MAX,
        "peak_allocated_candidate_over_baseline_max": ALLOCATED_RATIO_MAX,
        "peak_active_candidate_over_baseline_max": ACTIVE_RATIO_MAX,
        "peak_reserved_role": "diagnostic_only",
        "localized_conditioning_cpu_cuda_relative_l2_max": 1e-15,
        "wall_residual_linf_max": 1e-12,
    }


def test_rc437_recovery_plan_preserves_v1_and_binds_v2_support():
    plan = _recovery_plan()
    assert _sha256(ROOT / plan["source_identity"]["original_plan"]) == plan[
        "source_identity"
    ]["original_plan_sha256"]
    for name in ("profile_runner", "failed_cuda_runner", "failed_analyzer"):
        relative = plan["immutable_v1_support"][name]
        assert _git_blob_sha256(FAILED_QUALIFICATION, relative) == plan[
            "immutable_v1_support"
        ][
            f"{name}_sha256"
        ]
    for name in ("cuda_correctness_runner", "analyzer"):
        relative = plan["recovery_support"][name]
        assert _git_blob_sha256(CORRECT_LAYER_RECOVERY, relative) == plan[
            "recovery_support"
        ][
            f"{name}_sha256"
        ]
    assert plan["failure_adjudication"]["scientific_contract_relaxed"] is False
    assert plan["recovery_support"]["post_projection_comparison_role"] == (
        "diagnostic_only"
    )
    assert plan["authorization"]["push_authorized"] is False
    assert plan["authorization"]["h100_recovery_submission_authorized"] is False
    assert plan["future_h100_recovery"]["automatic_retry"] is False


def test_rc437_profile_json_round_trip_normalizes_tuple_values(tmp_path):
    output = tmp_path / "profile.json"
    value = {
        "config": {
            "shape": (128, 128, 32),
            "lengths": (100.0, 100.0, 20.0),
        },
        "completed_steps": 60,
        "finite": True,
    }

    write_profile_json(output, value)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "config": {
            "shape": [128, 128, 32],
            "lengths": [100.0, 100.0, 20.0],
        },
        "completed_steps": 60,
        "finite": True,
    }
    assert not list(tmp_path.glob(".profile.json.tmp-*"))


def test_rc437_profile_serialization_recovery_plan_binds_failure_and_fix():
    plan = _profile_serialization_recovery_plan()
    profile = ROOT / plan["qualification_support"]["profile_runner"]

    assert _sha256(profile) == plan["qualification_support"][
        "profile_runner_sha256"
    ]
    assert plan["failed_job_evidence"]["job_id"] == "10874753"
    assert plan["failed_job_evidence"]["checksums_verified"] == 206
    assert plan["failure_adjudication"]["scientific_failure_supported"] is False
    assert plan["fix_contract"]["runtime_source_changed"] is False
    assert plan["fix_contract"]["profile_schema_changed"] is False
    assert plan["authorization"]["push_authorized"] is False
    assert plan["authorization"]["h100_recovery_submission_authorized"] is False
    assert plan["authorization"]["automatic_retry_authorized"] is False


def test_rc437_analyzer_accepts_complete_evidence(tmp_path):
    result = _analyze(tmp_path)
    assert result["classification"] == PASS
    assert result["rc4_3_complete"] is True
    assert result["public_lifted_plane_reopened"] is False
    assert result["localized_policy_publicly_qualified"] is False
    assert result["production_default_changed"] is False


def test_rc437_cuda_helper_cpu_reference_construction_and_provenance():
    spec, adapter = _build(
        device="cpu",
        apply_fresh=True,
        pointwise_execution="eager",
    )
    checkpoint = capture_plane_checkpoint(adapter, run_spec=spec)

    assert tuple(adapter.fields.spatial.shape) == (9, 1, 8, 8, 8)
    assert _wall_residual(adapter) <= 1.0e-12
    assert checkpoint.fresh_construction_provenance["conditioning"][
        "effective_policy"
    ] == "localized_quintic_wall_transition_v1"
    assert checkpoint.fresh_construction_provenance["conditioning_applied"] is True
    assert "fresh_initial_remainder_conditioning" not in (
        checkpoint.lifting_restart["lifting"]
    )


def test_rc437_recovery_measures_locality_at_the_direct_operator_output():
    _, adapter = _build(
        device="cpu",
        apply_fresh=True,
        pointwise_execution="eager",
    )
    evidence, conditioned = _direct_conditioning_evidence(
        adapter.solver.model.static_lifting_runtime,
        _initial((8, 8, 8)),
    )

    assert evidence["measurement_layer"] == MEASUREMENT_LAYER
    assert evidence["interior_byte_identical_to_unconditioned"] is True
    assert evidence["transition_region_changed"] is True
    assert evidence["interior_cell_count"] == 6
    assert evidence["transition_cell_count"] == 2
    assert tuple(conditioned.shape) == (5, 8, 8, 8)
    assert all(
        component["interior_byte_identical_to_unconditioned"] is True
        for component in evidence["components"].values()
    )


def test_rc437_recovery_analyzer_accepts_nonidentical_post_projection_diagnostic(
    tmp_path,
):
    result = _analyze_recovery(tmp_path)

    assert result["schema"] == RECOVERY_ANALYSIS_SCHEMA
    assert result["classification"] == PASS
    assert result["cuda_correctness"]["measurement_layer"] == MEASUREMENT_LAYER
    assert result["cuda_correctness"]["post_projection_role"] == (
        "diagnostic_only"
    )
    assert result["qualification_recovery"]["runtime_source_changed"] is False
    assert result["qualification_recovery"]["scientific_contract_relaxed"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda report: report["fresh_conditioning"].update(
                {"measurement_layer": "adapter.fields.spatial"}
            ),
            "measurement layer differs",
        ),
        (
            lambda report: report["fresh_conditioning"].update(
                {"interior_byte_identical_to_unconditioned": False}
            ),
            "direct conditioning interior changed",
        ),
        (
            lambda report: report["fresh_conditioning"][
                "post_projection_diagnostic"
            ].update({"qualification_gate": True}),
            "became a qualification gate",
        ),
    ),
)
def test_rc437_recovery_analyzer_rejects_wrong_layer_or_relaxed_gate(
    tmp_path,
    mutation,
    message,
):
    candidate_root = tmp_path / "candidate"
    report = _recovery_cuda(candidate_root)
    mutation(report)
    with pytest.raises(ValueError, match=message):
        _analyze_recovery(tmp_path, cuda=report)


def test_rc437_recovery_analyzer_rejects_legacy_cuda_schema(tmp_path):
    candidate_root = tmp_path / "candidate"
    with pytest.raises(ValueError, match="recovery schema differs"):
        _analyze_recovery(tmp_path, cuda=_cuda(candidate_root))


def test_rc437_analyzer_rejects_missing_duplicate_or_changed_dynamics(tmp_path):
    candidate = _profiles("candidate", root=tmp_path / "candidate")
    with pytest.raises(ValueError, match="incomplete"):
        _analyze(tmp_path, candidate=candidate[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        _analyze(tmp_path, candidate=[*candidate, candidate[0]])
    candidate = _profiles("candidate", root=tmp_path / "candidate")
    candidate[0]["final_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="supported dynamics changed"):
        _analyze(tmp_path, candidate=candidate)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("timing", "mean_timestep_seconds"), 2.0, "mean timestep regression"),
        (("memory", "peak_allocated_bytes"), 2000, "allocated-memory regression"),
        (("memory", "peak_active_bytes"), 2000, "active-memory regression"),
        (("pointwise_compile", "during_profile", "graph_breaks"), 1, "graph break"),
    ),
)
def test_rc437_analyzer_rejects_non_regression_and_compile_failures(tmp_path, path, value, message):
    candidate = _profiles("candidate", root=tmp_path / "candidate")
    target = candidate[0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        _analyze(tmp_path, candidate=candidate)


def test_rc437_analyzer_rejects_cuda_scope_expansion(tmp_path):
    root = tmp_path / "candidate"
    report = _cuda(root)
    report = copy.deepcopy(report)
    report["qualification_boundary"]["public_lifted_plane_reopened"] = True
    with pytest.raises(ValueError, match="boundary expanded"):
        analyze(
            _profiles("baseline", root=tmp_path / "baseline"),
            _profiles("candidate", root=root),
            report,
            baseline_commit=BASELINE,
            candidate_commit=CANDIDATE,
            baseline_root=tmp_path / "baseline",
            candidate_root=root,
        )


def test_rc437_plan_lists_all_current_login_node_cuda_tests():
    frozen = set(_plan()["cuda_only_node_ids"])
    discovered = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if 'reason="CUDA is unavailable"' not in line:
                continue
            for following in lines[index + 1:index + 6]:
                stripped = following.strip()
                if stripped.startswith("def test_"):
                    name = stripped.split("(", 1)[0].removeprefix("def ")
                    discovered.add(f"tests/{path.name}::{name}")
                    break
    assert discovered == frozen
