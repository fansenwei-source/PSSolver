"""Contracts for the attribution-correct P7.7.12 evidence recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "notes/architecture_v0_2/phase_7_p7712_h100_qualification_plan.json"
RECOVERY_CONTRACT = ROOT / ("notes/architecture_v0_2/phase_7_p7712_h100_evidence_recovery.json")
EXPECTED_COMMIT = "d" * 40
DIAGNOSTIC_DTYPE = np.dtype([("step", np.int64), ("value", np.float64)])


def _plan() -> dict[str, object]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


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
        reports.append(_report(case, kind="resume", trial=0, entry="direct"))
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
