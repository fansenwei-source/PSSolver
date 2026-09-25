"""Contracts for the attribution-correct P7.7.12 evidence recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.analyze_public_simulation_h100_qualification import (
    QualificationEvidenceError,
)
from benchmarks.recover_public_simulation_h100_qualification import recover


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "notes/architecture_v0_2/phase_7_p7712_h100_qualification_plan.json"
RECOVERY_CONTRACT = ROOT / (
    "notes/architecture_v0_2/phase_7_p7712_h100_evidence_recovery.json"
)
EXPECTED_COMMIT = "d" * 40
DIAGNOSTIC_DTYPE = np.dtype([("step", np.int64), ("value", np.float64)])


def _plan() -> dict[str, object]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_recovery_contract_is_analysis_only_and_keeps_phase8_deferred():
    contract = json.loads(RECOVERY_CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "READY_CPU_ONLY_EVIDENCE_RECOVERY"
    assert contract["recovery_execution"] == {
        "resource": "CPU-only analysis",
        "new_h100_job_authorized": False,
        "simulation_rerun_authorized": False,
        "reuse_original_reports_and_outputs": True,
        "automatic_retry": False,
        "write_new_control_directory": True,
        "modify_original_control_or_scratch": False,
    }
    assert contract["scope"]["solver_source_changed"] is False
    assert contract["scope"]["phase_8_authorized"] is False
    assert contract["scope"]["phase_9_authorized"] is False


def _records(case_id: str, suffix: str = "final") -> dict[str, object]:
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
    output_directory: Path | None = None,
    timestep: float = 1.0,
) -> dict[str, object]:
    if kind == "segment":
        start, final = 0, 30
        records = _records(str(case["case_id"]), "segment")
        checkpoints = {"30": {"checkpoint.json": {"sha256": "b" * 64}}}
    elif kind == "resume":
        start, final = 30, 60
        records = _records(str(case["case_id"]))
        checkpoints = {}
    else:
        start, final = 0, 60
        records = _records(str(case["case_id"]))
        checkpoints = {}
    public_metadata = None
    if entry == "public":
        assert output_directory is not None
        public_metadata = {
            "output_directory": str(output_directory),
            "diagnostic_steps": [final],
        }
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
            "start_step": start,
            "final_step": final,
            "elapsed_seconds": timestep * (final - start),
            "mean_timestep_seconds": timestep,
            "wall_seconds": timestep * (final - start) + 1.0,
            "public_metadata": public_metadata,
        },
        "memory": {
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        },
        "artifacts": {
            "records": records,
            "finite": True,
            "completed_steps": final,
            "runtime_selection": {
                "requested": case["runtime_path"],
                "effective": case["runtime_path"],
                "fallback_used": False,
            },
        },
        "checkpoints": checkpoints,
    }


def _write_diagnostics(directory: Path, values: list[tuple[int, float]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    array = np.array(values, dtype=DIAGNOSTIC_DTYPE)
    np.save(directory / "diagnostics.npy", array, allow_pickle=False)
    np.savetxt(
        directory / "diagnostics.csv",
        array,
        delimiter=",",
        header="step,value",
        comments="",
    )


def _reports(tmp_path: Path, *, slow_public: bool = False) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for case in _plan()["cases"]:
        case_id = str(case["case_id"])
        final_value = float(len(reports) + 1)
        for trial in range(1, 4):
            for entry in ("direct", "public"):
                output = tmp_path / f"{case_id}-performance-{trial}-{entry}"
                if entry == "public":
                    _write_diagnostics(output, [(0, -1.0), (60, final_value)])
                timestep = 1.2 if slow_public and entry == "public" else 1.0
                reports.append(
                    _report(
                        case,
                        kind="performance",
                        trial=trial,
                        entry=entry,
                        output_directory=output if entry == "public" else None,
                        timestep=timestep,
                    )
                )
        segment_output = tmp_path / f"{case_id}-segment"
        _write_diagnostics(segment_output, [(0, -1.0), (30, 0.0)])
        reports.append(
            _report(
                case,
                kind="segment",
                trial=0,
                entry="public",
                output_directory=segment_output,
            )
        )
        reports.append(
            _report(case, kind="resume", trial=0, entry="direct")
        )
        resume_output = tmp_path / f"{case_id}-resume-public"
        _write_diagnostics(resume_output, [(60, final_value)])
        reports.append(
            _report(
                case,
                kind="resume",
                trial=0,
                entry="public",
                output_directory=resume_output,
            )
        )
    return reports


def test_recovery_accepts_h100_matrix_and_common_final_diagnostics(tmp_path):
    result = recover(
        _plan(),
        _reports(tmp_path, slow_public=True),
        expected_commit=EXPECTED_COMMIT,
        repository_root=ROOT,
    )
    assert result["qualification_complete"] is True
    assert result["classification"].startswith("PASS_P7_7_12")
    assert result["public_timestep_boundary"] == {
        **result["public_timestep_boundary"],
        "public_timestep_loop_count": 0,
        "public_wrapper_inside_application_timer": False,
        "application_elapsed_timer_passed_through": True,
    }
    assert result["reused_runtime_qualification"]["qualification_complete"] is True
    assert result["eligible_for_phase_8_planning"] is True
    for value in result["performance"].values():
        assert value["arithmetic_mean_ratio"] == pytest.approx(1.2)
        assert value["timing_adjudication"] == (
            "diagnostic_only_not_attributable_to_public_wrapper"
        )
    for value in result["restart"].values():
        assert value["common_final_diagnostic_record"] == "byte_for_byte"


def test_recovery_rejects_common_final_diagnostic_drift(tmp_path):
    reports = _reports(tmp_path)
    target = next(
        report
        for report in reports
        if report["case"]["case_id"] == "channel_compiled_r128"
        and report["case"]["kind"] == "resume"
        and report["case"]["entry"] == "public"
    )
    directory = Path(target["result"]["public_metadata"]["output_directory"])
    (directory / "diagnostics.npy").unlink()
    (directory / "diagnostics.csv").unlink()
    _write_diagnostics(directory, [(60, 999.0)])
    with pytest.raises(
        QualificationEvidenceError,
        match="final diagnostic differs",
    ):
        recover(
            _plan(),
            reports,
            expected_commit=EXPECTED_COMMIT,
            repository_root=ROOT,
        )


def test_recovery_still_rejects_scientific_state_drift(tmp_path):
    reports = _reports(tmp_path)
    target = next(
        report
        for report in reports
        if report["case"]["case_id"] == "plane_legacy_r128"
        and report["case"]["kind"] == "resume"
        and report["case"]["entry"] == "public"
    )
    target["artifacts"]["records"]["Q_60.npy"]["sha256"] = "f" * 64
    with pytest.raises(QualificationEvidenceError, match="resumed artifacts differ"):
        recover(
            _plan(),
            reports,
            expected_commit=EXPECTED_COMMIT,
            repository_root=ROOT,
        )
