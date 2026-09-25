"""Local contracts for the P7.7.12 single-H100 qualification."""

from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.analyze_public_simulation_h100_qualification import (
    QualificationEvidenceError,
    analyze,
)
from benchmarks.run_public_simulation_h100_case import build_simulation
from pssolver import compile_simulation


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / (
    "notes/architecture_v0_2/phase_7_p7712_h100_qualification_plan.json"
)
EXPECTED_COMMIT = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _args(tmp_path: Path, case: dict[str, object]) -> Namespace:
    return Namespace(
        application=case["application"],
        runtime_path=case["runtime_path"],
        shape=tuple(case["shape"]),
        lengths=tuple(case["lengths"]),
        dt=case["dt"],
        steps=2,
        save_interval=2,
        diagnostic_interval=2,
        checkpoint_interval=None,
        restart_from=None,
        output_directory=tmp_path / case["case_id"],
        seed=24,
        device="cpu",
        spectral_storage=case["spectral_storage"],
        pointwise_execution="eager",
        activity=5.0,
    )


def _artifact_records(case_id: str, suffix: str = "final") -> dict[str, object]:
    digest = hashlib.sha256(f"{case_id}:{suffix}".encode()).hexdigest()
    return {
        name: {"sha256": digest, "size_bytes": 1}
        for name in (
            "Q_60.npy",
            "u_60.npy",
            "p_60.npy",
            "diagnostics.npy",
            "diagnostics.csv",
            "COMPLETE",
        )
    }


def _report(
    case: dict[str, object],
    *,
    kind: str,
    trial: int,
    entry: str,
    timestep: float = 1.0,
    artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    if kind == "segment":
        start_step, final_step = 0, 30
        records = _artifact_records(case["case_id"], "segment")
        checkpoints = {"30": {"checkpoint.json": {"sha256": "b" * 64}}}
    elif kind == "resume":
        start_step, final_step = 30, 60
        records = _artifact_records(case["case_id"])
        checkpoints = {}
    else:
        start_step, final_step = 0, 60
        records = _artifact_records(case["case_id"])
        checkpoints = {}
    if artifacts is not None:
        records = artifacts
    return {
        "schema_version": 1,
        "phase": "P7.7.12",
        "case": {
            "case_id": case["case_id"],
            "kind": kind,
            "trial": trial,
            "application": case["application"],
            "runtime_path": case["runtime_path"],
            "entry": entry,
            "shape": case["shape"],
            "lengths": case["lengths"],
        },
        "environment": {
            "git": {"head": EXPECTED_COMMIT, "status_porcelain": ""},
            "cuda_available": True,
            "device": "cuda",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_matmul": False,
            "tf32_cudnn": False,
        },
        "result": {
            "start_step": start_step,
            "final_step": final_step,
            "elapsed_seconds": timestep * (final_step - start_step),
            "mean_timestep_seconds": timestep,
            "wall_seconds": timestep * (final_step - start_step) + 1.0,
            "public_metadata": {} if entry == "public" else None,
        },
        "memory": {
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        },
        "artifacts": {
            "records": records,
            "finite": True,
            "completed_steps": final_step,
            "runtime_selection": {
                "requested": case["runtime_path"],
                "effective": case["runtime_path"],
                "fallback_used": False,
            },
        },
        "checkpoints": checkpoints,
    }


def _reports(plan: dict[str, object]) -> list[dict[str, object]]:
    result = []
    for case in plan["cases"]:
        for trial in range(1, 4):
            for entry in ("direct", "public"):
                result.append(
                    _report(
                        case,
                        kind="performance",
                        trial=trial,
                        entry=entry,
                    )
                )
        result.append(_report(case, kind="segment", trial=0, entry="public"))
        result.append(_report(case, kind="resume", trial=0, entry="direct"))
        result.append(_report(case, kind="resume", trial=0, entry="public"))
    return result


def test_plan_freezes_exact_scope_and_reused_evidence():
    plan = _plan()
    assert plan["classification"] == "READY_P7_7_12_SINGLE_H100_NON_REGRESSION"
    assert len(plan["cases"]) == 4
    assert plan["requirements"]["report_count"] == 36
    assert plan["requirements"]["formal_h100_submission_count"] == 1
    assert plan["requirements"]["automatic_retry"] is False
    assert plan["scope"] == {
        "numerical_kernel_changed": False,
        "checkpoint_schema_changed": False,
        "production_default_changed": False,
        "compiled_runtime_promoted": False,
        "phase_8_authorized": False,
        "phase_9_authorized": False,
    }
    for evidence in plan["reused_evidence"]:
        assert _sha256(ROOT / evidence["path"]) == evidence["sha256"]


@pytest.mark.parametrize("case_index", range(4))
def test_frozen_cases_compile_to_the_requested_existing_application(
    tmp_path,
    case_index,
):
    case = _plan()["cases"][case_index]
    compiled = compile_simulation(build_simulation(_args(tmp_path, case)))
    assert compiled.application_request.runtime_path.value == case["runtime_path"]
    assert compiled.application_request.device == "cpu"
    assert compiled.application_specification.geometry.domain.shape == tuple(
        case["shape"]
    )
    assert compiled.normalization["runtime_fallback_allowed"] is False


def test_analyzer_accepts_the_exact_frozen_matrix():
    plan = _plan()
    result = analyze(plan, _reports(plan), expected_commit=EXPECTED_COMMIT)
    assert result["classification"] == (
        "PASS_P7_7_12_SINGLE_H100_NON_REGRESSION"
    )
    assert result["qualification_complete"] is True
    assert result["report_count"] == 36
    assert result["eligible_for_phase_8_planning"] is True
    assert result["phase_8_authorized"] is False
    assert result["production_default_changed"] is False


def test_analyzer_rejects_numerical_drift():
    plan = _plan()
    reports = _reports(plan)
    reports[1]["artifacts"]["records"] = _artifact_records("tampered")
    with pytest.raises(QualificationEvidenceError, match="artifact mismatch"):
        analyze(plan, reports, expected_commit=EXPECTED_COMMIT)


def test_analyzer_rejects_public_timestep_regression():
    plan = _plan()
    reports = _reports(plan)
    for report in reports:
        case = report["case"]
        if (
            case["case_id"] == "plane_legacy_r128"
            and case["kind"] == "performance"
            and case["entry"] == "public"
        ):
            report["result"]["mean_timestep_seconds"] = 1.2
    with pytest.raises(QualificationEvidenceError, match="mean timestep"):
        analyze(plan, reports, expected_commit=EXPECTED_COMMIT)


def test_analyzer_rejects_dirty_or_incomplete_evidence():
    plan = _plan()
    reports = _reports(plan)
    reports[0]["environment"]["git"]["status_porcelain"] = "?? artifact"
    with pytest.raises(QualificationEvidenceError, match="worktree is dirty"):
        analyze(plan, reports, expected_commit=EXPECTED_COMMIT)
    with pytest.raises(QualificationEvidenceError, match="expected 36 reports"):
        analyze(plan, reports[:-1], expected_commit=EXPECTED_COMMIT)
