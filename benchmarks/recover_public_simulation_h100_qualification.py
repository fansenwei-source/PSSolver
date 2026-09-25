#!/usr/bin/env python3
"""Recover P7.7.12 with attribution-correct performance and restart gates.

The original H100 matrix remains immutable evidence.  This analyzer corrects
two qualification-contract mistakes without changing a numerical runtime:

* an application's workflow timer cannot measure public wrapper work that
  happens before/after the application invocation; and
* a resumed workflow does not reconstruct the uninterrupted diagnostic
  history, so only records at common physical steps are comparable.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import analyze_public_simulation_h100_qualification as base
from pssolver.api import runner as public_runner


RECOVERY_CLASSIFICATION = (
    "PASS_P7_7_12_H100_EVIDENCE_WITH_ATTRIBUTION_CORRECT_RECOVERY"
)
SCIENTIFIC_ARTIFACTS = ("Q_{step}.npy", "u_{step}.npy", "p_{step}.npy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_public_timestep_boundary() -> dict[str, object]:
    """Prove that public dispatch does not execute inside a timestep loop."""

    run_source = inspect.getsource(public_runner.run_simulation)
    run_tree = ast.parse(run_source)
    loops = tuple(
        node
        for node in ast.walk(run_tree)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While))
    )
    invoke_calls = tuple(
        node
        for node in ast.walk(run_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "invoke"
    )
    result_calls = tuple(
        node
        for node in ast.walk(run_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_public_result"
    )
    base._require(not loops, "public run_simulation contains a timestep loop")
    base._require(
        len(invoke_calls) == 1,
        "public run_simulation must invoke exactly one application runner",
    )
    base._require(
        len(result_calls) == 1,
        "public run_simulation must wrap exactly one application result",
    )

    result_source = inspect.getsource(public_runner._public_result)
    result_tree = ast.parse(result_source)
    elapsed_passthrough = any(
        isinstance(node, ast.keyword)
        and node.arg == "elapsed_seconds"
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "application_result"
        and node.value.attr == "elapsed_seconds"
        for node in ast.walk(result_tree)
    )
    base._require(
        elapsed_passthrough,
        "public result does not pass through the application elapsed timer",
    )
    source_path = Path(inspect.getsourcefile(public_runner.run_simulation) or "")
    base._require(source_path.is_file(), "public runner source path is absent")
    return {
        "public_timestep_loop_count": 0,
        "application_invoke_count": 1,
        "public_result_wrap_count": 1,
        "application_elapsed_timer_passed_through": True,
        "public_wrapper_inside_application_timer": False,
        "source": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
    }


def _validate_reused_runtime_qualification(
    plan: dict[str, object],
    *,
    repository_root: Path,
) -> dict[str, object]:
    matches = [
        item
        for item in plan.get("reused_evidence", [])
        if str(item.get("path", "")).endswith("phase_7_p776_h100_closure.json")
    ]
    base._require(len(matches) == 1, "P7.7.6 closure evidence is not unique")
    item = matches[0]
    path = (repository_root / str(item["path"])).resolve()
    base._require(path.is_file(), "P7.7.6 closure evidence is absent")
    observed_sha = _sha256(path)
    base._require(
        observed_sha == item.get("sha256"),
        "P7.7.6 closure evidence SHA-256 mismatch",
    )
    record = base._load_object(path)
    qualification = record.get("qualification")
    profiles = record.get("profiles")
    base._require(
        isinstance(qualification, dict)
        and qualification.get("complete") is True,
        "P7.7.6 production qualification is incomplete",
    )
    base._require(
        isinstance(profiles, dict)
        and str(profiles.get("classification", "")).startswith("PASS_"),
        "P7.7.6 targeted performance adjudication did not pass",
    )
    base._require(
        qualification.get("production_default_changed") is False,
        "P7.7.6 unexpectedly changed a production default",
    )
    return {
        "path": str(path),
        "sha256": observed_sha,
        "classification": record.get("classification"),
        "qualification_complete": True,
        "targeted_performance_classification": profiles.get("classification"),
    }


def _index_reports(
    plan: dict[str, object],
    reports: list[dict[str, object]],
    *,
    expected_commit: str,
) -> tuple[
    dict[str, dict[str, object]],
    dict[tuple[str, str, int, str], dict[str, object]],
]:
    base._require(plan.get("phase") == "P7.7.12", "plan phase is not P7.7.12")
    requirements = plan.get("requirements")
    base._require(isinstance(requirements, dict), "plan requirements are absent")
    base._require(
        len(reports) == int(requirements["report_count"]),
        "H100 report matrix is incomplete",
    )
    cases = {
        str(item["case_id"]): item
        for item in plan.get("cases", [])
    }
    base._require(len(cases) == 4, "plan must contain four cases")
    indexed: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for report in reports:
        key = base._report_key(report)
        base._require(key not in indexed, f"duplicate report key: {key!r}")
        base._require(key[0] in cases, f"unknown case: {key[0]}")
        base._validate_report(
            report,
            case_plan=cases[key[0]],
            expected_commit=expected_commit,
        )
        indexed[key] = report
    return cases, indexed


def _required_scientific_records(
    report: dict[str, object],
    *,
    final_step: int,
) -> dict[str, object]:
    records = base._artifact_identity(report)
    names = tuple(value.format(step=final_step) for value in SCIENTIFIC_ARTIFACTS)
    result: dict[str, object] = {}
    for name in (*names, "COMPLETE"):
        base._require(name in records, f"required artifact is absent: {name}")
        result[name] = records[name]
    return result


def _public_output_directory(report: dict[str, object]) -> Path:
    result = report.get("result")
    base._require(isinstance(result, dict), "report result is absent")
    metadata = result.get("public_metadata")
    base._require(isinstance(metadata, dict), "public result metadata is absent")
    value = metadata.get("output_directory")
    base._require(isinstance(value, str) and value, "public output directory is absent")
    path = Path(value).expanduser().resolve()
    base._require(path.is_dir(), f"public output directory is absent: {path}")
    return path


def _diagnostic_record(
    report: dict[str, object],
    *,
    step: int,
) -> dict[str, object]:
    directory = _public_output_directory(report)
    npy_path = directory / "diagnostics.npy"
    csv_path = directory / "diagnostics.csv"
    base._require(npy_path.is_file(), f"diagnostics NPY is absent: {npy_path}")
    base._require(csv_path.is_file(), f"diagnostics CSV is absent: {csv_path}")
    values = np.load(npy_path, allow_pickle=False)
    base._require(values.dtype.names is not None, "diagnostics NPY is not structured")
    base._require("step" in values.dtype.names, "diagnostics NPY has no step field")
    selected = values[values["step"] == step]
    base._require(
        selected.shape == (1,),
        f"expected one diagnostic record at step {step}, found {selected.shape[0]}",
    )
    csv_values = np.genfromtxt(csv_path, delimiter=",", names=True)
    csv_values = np.atleast_1d(csv_values)
    base._require(
        csv_values.dtype.names is not None and "step" in csv_values.dtype.names,
        "diagnostics CSV has no step column",
    )
    csv_selected = csv_values[csv_values["step"] == step]
    base._require(
        csv_selected.shape == (1,),
        f"expected one CSV diagnostic record at step {step}",
    )
    npy_tuple = tuple(selected[0][name].item() for name in values.dtype.names)
    csv_tuple = tuple(csv_selected[0][name].item() for name in csv_values.dtype.names)
    base._require(npy_tuple == csv_tuple, "diagnostics NPY/CSV final records differ")
    payload = selected.tobytes()
    return {
        "step": step,
        "values": list(npy_tuple),
        "record_sha256": hashlib.sha256(payload).hexdigest(),
        "npy_path": str(npy_path),
        "npy_sha256": _sha256(npy_path),
        "csv_path": str(csv_path),
        "csv_sha256": _sha256(csv_path),
    }


def recover(
    plan: dict[str, object],
    reports: list[dict[str, object]],
    *,
    expected_commit: str,
    repository_root: Path,
) -> dict[str, object]:
    """Apply the P7.7.12 attribution-correct recovery contract."""

    cases, indexed = _index_reports(
        plan,
        reports,
        expected_commit=expected_commit,
    )
    public_boundary = _audit_public_timestep_boundary()
    p776 = _validate_reused_runtime_qualification(
        plan,
        repository_root=repository_root,
    )
    trials = int(plan["performance"]["trials"])
    performance: dict[str, object] = {}
    restart: dict[str, object] = {}
    for case_id in sorted(cases):
        timestep_ratios: list[float] = []
        wall_ratios: list[float] = []
        allocated_ratios: list[float] = []
        reserved_ratios: list[float] = []
        for trial in range(1, trials + 1):
            direct = indexed[(case_id, "performance", trial, "direct")]
            public = indexed[(case_id, "performance", trial, "public")]
            base._require(
                base._artifact_identity(direct) == base._artifact_identity(public),
                f"direct/public artifacts differ for {case_id} trial {trial}",
            )
            direct_result, public_result = direct["result"], public["result"]
            timestep_ratios.append(
                base._finite_positive(
                    public_result["mean_timestep_seconds"], "public timestep"
                )
                / base._finite_positive(
                    direct_result["mean_timestep_seconds"], "direct timestep"
                )
            )
            wall_ratios.append(
                base._finite_positive(public_result["wall_seconds"], "public wall time")
                / base._finite_positive(direct_result["wall_seconds"], "direct wall time")
            )
            allocated_ratios.append(
                int(public["memory"]["peak_allocated_bytes"])
                / int(direct["memory"]["peak_allocated_bytes"])
            )
            reserved_ratios.append(
                int(public["memory"]["peak_reserved_bytes"])
                / int(direct["memory"]["peak_reserved_bytes"])
            )
        max_allocated = max(allocated_ratios)
        max_reserved = max(reserved_ratios)
        limits = plan["performance"]
        base._require(
            max_allocated
            <= float(limits["public_over_direct_peak_allocated_ratio_max"]),
            f"allocated-memory non-regression failed for {case_id}",
        )
        base._require(
            max_reserved
            <= float(limits["public_over_direct_peak_reserved_ratio_max"]),
            f"reserved-memory non-regression failed for {case_id}",
        )
        performance[case_id] = {
            "paired_public_over_direct_workflow_timestep_ratios": timestep_ratios,
            "arithmetic_mean_ratio": statistics.fmean(timestep_ratios),
            "median_ratio": statistics.median(timestep_ratios),
            "paired_public_over_direct_wall_ratios": wall_ratios,
            "peak_allocated_ratio": max_allocated,
            "peak_reserved_ratio": max_reserved,
            "scientific_artifact_identity": "byte_for_byte",
            "timing_adjudication": (
                "diagnostic_only_not_attributable_to_public_wrapper"
            ),
            "authoritative_runtime_performance_evidence": "P7.7.6",
        }

        final_step = int(plan["restart"]["continuous_steps"])
        segment_step = int(plan["restart"]["segment_steps"])
        segment = indexed[(case_id, "segment", 0, "public")]
        direct_resume = indexed[(case_id, "resume", 0, "direct")]
        public_resume = indexed[(case_id, "resume", 0, "public")]
        continuous = indexed[(case_id, "performance", 1, "public")]
        base._require(segment["result"]["final_step"] == segment_step, "segment clock mismatch")
        base._require(str(segment_step) in segment["checkpoints"], "segment checkpoint absent")
        base._require(
            base._artifact_identity(direct_resume)
            == base._artifact_identity(public_resume),
            f"direct/public resumed artifacts differ for {case_id}",
        )
        continuous_state = _required_scientific_records(
            continuous,
            final_step=final_step,
        )
        resumed_state = _required_scientific_records(
            public_resume,
            final_step=final_step,
        )
        base._require(
            continuous_state == resumed_state,
            f"continuous/resumed scientific state differs for {case_id}",
        )
        continuous_diagnostic = _diagnostic_record(continuous, step=final_step)
        resumed_diagnostic = _diagnostic_record(public_resume, step=final_step)
        base._require(
            continuous_diagnostic["record_sha256"]
            == resumed_diagnostic["record_sha256"],
            f"continuous/resumed final diagnostic differs for {case_id}",
        )
        restart[case_id] = {
            "checkpoint_step": segment_step,
            "final_step": final_step,
            "direct_public_resume_artifacts": "byte_for_byte",
            "continuous_resume_q_u_p_complete": "byte_for_byte",
            "diagnostic_history_policy": (
                "compare_common_physical_steps_not_full_history"
            ),
            "common_final_diagnostic_record": "byte_for_byte",
            "continuous_diagnostic": continuous_diagnostic,
            "resumed_diagnostic": resumed_diagnostic,
        }

    base._require(
        len(indexed) == 4 * (trials * 2 + 3),
        "unexpected report classes remain",
    )
    return {
        "schema_version": 1,
        "phase": "P7.7.12",
        "recovery": "attribution_correct_performance_and_diagnostics",
        "classification": RECOVERY_CLASSIFICATION,
        "qualification_complete": True,
        "expected_commit": expected_commit,
        "report_count": len(reports),
        "public_timestep_boundary": public_boundary,
        "reused_runtime_qualification": p776,
        "performance": performance,
        "restart": restart,
        "runtime_fallback_used": False,
        "numerical_kernel_changed": False,
        "checkpoint_schema_changed": False,
        "production_default_changed": False,
        "compiled_runtime_promoted": False,
        "eligible_for_phase_8_planning": True,
        "phase_8_authorized": False,
        "phase_9_authorized": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reports-directory", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    reports = [
        base._load_object(path)
        for path in sorted(args.reports_directory.glob("*.json"))
    ]
    result = recover(
        base._load_object(args.plan),
        reports,
        expected_commit=args.expected_commit,
        repository_root=args.repository_root.expanduser().resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
