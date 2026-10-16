#!/usr/bin/env python3
"""Plan or execute the proposed rc5 G-1 paired-block profile matrix.

Planning is CPU-only.  Execution remains fail-closed until an external,
non-symlink authorization record binds the versioned contract and analyzer to
one explicitly authorized H100 submission.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.analyze_rc5_g1_paired_block_replay import (
    FORMAL_PASS_CLASSIFICATION,
    analyze,
)
from benchmarks.plan_rc5_g1_paired_block_contract import build_plan
from benchmarks.rc458_venv_identity import lexical_python_path
from benchmarks.run_rc5_g1_focused_profile_matrix import (
    _atomic_json,
    _preflight as _base_preflight,
    _sha256,
    build_command_plan as _build_command_plan,
)


SCHEMA = "pssolver.rc5.g1.paired_block_command_plan.v1"
PROGRESS_SCHEMA = "pssolver.rc5.g1.paired_block_progress.v1"
COMPLETE_SCHEMA = "pssolver.rc5.g1.paired_block_matrix_complete.v1"
FAILURE_SCHEMA = "pssolver.rc5.g1.paired_block_matrix_failure.v1"
AUTHORIZATION_SCHEMA = "pssolver.rc5.g1.paired_block_execution_authorization.v1"
SUPPORT_FILES = (
    "benchmarks/profile_rc5_g1_runtime_family.py",
    "benchmarks/plan_rc5_g1_paired_block_contract.py",
    "benchmarks/analyze_rc5_g1_paired_block_replay.py",
    "benchmarks/run_rc5_g1_paired_block_profile_matrix.py",
)


def _load_authorization(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"authorization record must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("authorization record must contain an object")
    return {"path": str(path), "sha256": _sha256(path), "record": value}


def build_command_plan(
    *,
    baseline_python: Path,
    candidate_python: Path,
    support_root: Path,
    output_root: Path,
    authorization_record: Path | None = None,
) -> dict[str, object]:
    contract = build_plan()
    command_plan = _build_command_plan(
        baseline_python=baseline_python,
        candidate_python=candidate_python,
        support_root=support_root,
        output_root=output_root,
        plan=contract,
        command_schema=SCHEMA,
    )
    command_plan.update(
        {
            "contract_schema": contract["schema"],
            "contract_classification": contract["classification"],
            "evidence_mode": "new_pre_registered_evidence",
            "progress_schema": PROGRESS_SCHEMA,
            "support_files": list(SUPPORT_FILES),
            "thresholds": contract["thresholds"],
            "raw_sample_contract": contract["raw_sample_contract"],
            "execution_authorization": _load_authorization(authorization_record),
        }
    )
    return command_plan


def _validate_authorization(command_plan: dict[str, object]) -> None:
    authorization = command_plan.get("execution_authorization")
    if not isinstance(authorization, dict):
        raise PermissionError("paired-block execution requires an authorization record")
    record = authorization.get("record")
    if not isinstance(record, dict):
        raise TypeError("authorization payload is missing")
    if record.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("authorization schema differs")
    required_true = (
        "formal_contract_adoption_authorized",
        "single_h100_submission_authorized",
    )
    for name in required_true:
        if record.get(name) is not True:
            raise PermissionError(f"authorization {name} is not true")
    if record.get("automatic_retry_authorized") is not False:
        raise PermissionError("automatic retry must be false")
    if record.get("maximum_formal_h100_submissions") != 1:
        raise PermissionError("exactly one formal H100 submission must be authorized")
    support = Path(str(command_plan["support_root"])).resolve()
    expected_sources = record.get("source_sha256")
    if not isinstance(expected_sources, dict):
        raise TypeError("authorization source_sha256 is missing")
    for relative in (
        "benchmarks/plan_rc5_g1_paired_block_contract.py",
        "benchmarks/analyze_rc5_g1_paired_block_replay.py",
    ):
        path = support / relative
        if expected_sources.get(relative) != _sha256(path):
            raise ValueError(f"authorization source identity differs: {relative}")


def preflight(command_plan: dict[str, object], *, for_execution: bool) -> None:
    _base_preflight(command_plan)
    if command_plan.get("schema") != SCHEMA:
        raise ValueError("paired-block command-plan schema differs")
    if command_plan.get("evidence_mode") != "new_pre_registered_evidence":
        raise ValueError("paired-block evidence mode differs")
    if command_plan.get("progress_schema") != PROGRESS_SCHEMA:
        raise ValueError("paired-block progress schema differs")
    support = Path(str(command_plan["support_root"])).resolve()
    for relative in SUPPORT_FILES:
        path = support / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"paired-block support file missing: {path}")
    rows = command_plan["rows"]
    for row in rows:
        command = row["command"]
        if command.count("--post-flush-rewarm-steps") != 1:
            raise ValueError(f"{row['key']} rewarm option count differs")
        index = command.index("--post-flush-rewarm-steps")
        if command[index + 1] != "3":
            raise ValueError(f"{row['key']} rewarm value differs")
    if for_execution:
        _validate_authorization(command_plan)


def execute(command_plan: dict[str, object]) -> dict[str, object]:
    preflight(command_plan, for_execution=True)
    output_root = Path(str(command_plan["output_root"])).absolute()
    profiles = output_root / "profiles"
    profiles.mkdir(parents=True, exist_ok=False)
    completed: list[dict[str, object]] = []
    for row in command_plan["rows"]:
        cache = Path(str(row["compiler_cache"])).absolute()
        triton = cache / "triton"
        inductor = cache / "torchinductor"
        triton.mkdir(parents=True)
        inductor.mkdir()
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "NVIDIA_TF32_OVERRIDE": "0",
                "TRITON_CACHE_DIR": str(triton),
                "TORCHINDUCTOR_CACHE_DIR": str(inductor),
                "PYTHONPATH": str(Path(str(command_plan["support_root"])).resolve()),
            }
        )
        result = subprocess.run(
            row["command"],
            cwd=output_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        logs = output_root / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{row['key']}.stdout").write_text(result.stdout, encoding="utf-8")
        (logs / f"{row['key']}.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"profile {row['key']} failed with {result.returncode}")
        path = Path(str(row["output"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config", {}).get("family") not in str(row["key"]):
            raise RuntimeError(f"profile identity differs: {row['key']}")
        completed.append(
            {"sequence": row["sequence"], "key": row["key"], "sha256": _sha256(path)}
        )
        _atomic_json(
            output_root / "progress" / f"{int(row['sequence']):03d}.json",
            {"schema": PROGRESS_SCHEMA, "completed": completed},
        )

    analysis = analyze(
        profiles,
        output_root / "progress",
        evidence_mode="new_pre_registered_evidence",
        progress_schema=PROGRESS_SCHEMA,
    )
    _atomic_json(output_root / "analysis.json", analysis)
    if analysis["classification"] != FORMAL_PASS_CLASSIFICATION:
        _atomic_json(
            output_root / "MATRIX_FAILED.json",
            {
                "schema": FAILURE_SCHEMA,
                "classification": analysis["classification"],
                "profile_count": len(completed),
                "automatic_retry": False,
            },
        )
        raise RuntimeError("paired-block performance or memory gate failed")
    result = {
        "schema": COMPLETE_SCHEMA,
        "profile_count": len(completed),
        "completed": completed,
        "analysis_classification": analysis["classification"],
        "passed": len(completed) == 48 and analysis["proposed_contract_passed"] is True,
    }
    _atomic_json(output_root / "MATRIX_COMPLETE.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--authorization-record", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    plan = build_command_plan(
        baseline_python=lexical_python_path(args.baseline_python),
        candidate_python=lexical_python_path(args.candidate_python),
        support_root=args.support_root.resolve(),
        output_root=args.output_root.absolute(),
        authorization_record=args.authorization_record,
    )
    _atomic_json(args.command_plan, plan)
    if args.plan_only:
        preflight(plan, for_execution=False)
    else:
        execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
