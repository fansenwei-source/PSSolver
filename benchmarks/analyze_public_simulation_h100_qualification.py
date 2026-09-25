#!/usr/bin/env python3
"""Fail-closed analyzer for the frozen P7.7.12 H100 report matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


class QualificationEvidenceError(RuntimeError):
    """Raised when P7.7.12 evidence violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationEvidenceError(message)


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationEvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationEvidenceError(f"JSON root must be an object: {path}")
    return value


def _finite_positive(value: object, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QualificationEvidenceError(f"{description} is not numeric") from exc
    _require(math.isfinite(result) and result > 0.0, f"{description} must be finite and positive")
    return result


def _report_key(report: dict[str, object]) -> tuple[str, str, int, str]:
    case = report.get("case")
    _require(isinstance(case, dict), "report case must be an object")
    key = (
        str(case.get("case_id")),
        str(case.get("kind")),
        int(case.get("trial", -1)),
        str(case.get("entry")),
    )
    _require(key[0] and key[0] != "None", "report case_id is absent")
    return key


def _validate_report(
    report: dict[str, object],
    *,
    case_plan: dict[str, object],
    expected_commit: str,
) -> None:
    _require(report.get("schema_version") == 1, "unexpected report schema")
    _require(report.get("phase") == "P7.7.12", "unexpected report phase")
    case = report["case"]
    for name in ("application", "runtime_path", "shape", "lengths"):
        _require(case.get(name) == case_plan.get(name), f"report {name} differs from plan")
    environment = report.get("environment")
    _require(isinstance(environment, dict), "environment is absent")
    git = environment.get("git")
    _require(isinstance(git, dict), "Git provenance is absent")
    _require(git.get("head") == expected_commit, "report Git HEAD differs from target")
    _require(git.get("status_porcelain") == "", "report worktree is dirty")
    _require(environment.get("cuda_available") is True, "CUDA is unavailable")
    _require(environment.get("device") == "cuda", "report did not execute on CUDA")
    _require(environment.get("device_name") == "NVIDIA H100 PCIe", "wrong GPU model")
    _require(environment.get("tf32_matmul") is False, "matmul TF32 is enabled")
    _require(environment.get("tf32_cudnn") is False, "cuDNN TF32 is enabled")
    artifacts = report.get("artifacts")
    _require(isinstance(artifacts, dict), "artifact evidence is absent")
    _require(artifacts.get("finite") is True, "non-finite artifact reported")
    runtime = artifacts.get("runtime_selection")
    _require(isinstance(runtime, dict), "runtime selection is absent")
    expected_runtime = case_plan["runtime_path"]
    _require(runtime.get("requested") == expected_runtime, "requested runtime mismatch")
    _require(runtime.get("effective") == expected_runtime, "effective runtime mismatch")
    _require(runtime.get("fallback_used") is False, "runtime fallback was used")
    records = artifacts.get("records")
    _require(
        isinstance(records, dict) and len(records) == 6,
        "scientific artifact set is incomplete",
    )
    result = report.get("result")
    _require(isinstance(result, dict), "result evidence is absent")
    _finite_positive(result.get("elapsed_seconds"), "workflow elapsed seconds")
    _finite_positive(result.get("mean_timestep_seconds"), "mean timestep seconds")
    _finite_positive(result.get("wall_seconds"), "wall seconds")
    entry = case["entry"]
    _require(entry in {"direct", "public"}, "invalid entry identity")
    _require(
        (result.get("public_metadata") is not None) == (entry == "public"),
        "public result metadata disagrees with entry",
    )
    memory = report.get("memory")
    _require(isinstance(memory, dict), "memory evidence is absent")
    _require(int(memory.get("peak_allocated_bytes", 0)) > 0, "peak allocated memory is absent")
    _require(int(memory.get("peak_reserved_bytes", 0)) > 0, "peak reserved memory is absent")


def _artifact_identity(report: dict[str, object]) -> dict[str, object]:
    return report["artifacts"]["records"]


def analyze(
    plan: dict[str, object],
    reports: list[dict[str, object]],
    *,
    expected_commit: str,
) -> dict[str, object]:
    _require(plan.get("phase") == "P7.7.12", "plan phase is not P7.7.12")
    requirements = plan.get("requirements")
    _require(isinstance(requirements, dict), "plan requirements are absent")
    expected_count = int(requirements["report_count"])
    _require(
        len(reports) == expected_count,
        f"expected {expected_count} reports, found {len(reports)}",
    )
    case_plans = {
        value["case_id"]: value
        for value in plan.get("cases", [])
    }
    _require(len(case_plans) == 4, "plan must contain four unique cases")
    indexed: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for report in reports:
        key = _report_key(report)
        _require(key not in indexed, f"duplicate report key: {key!r}")
        _require(key[0] in case_plans, f"unknown case_id: {key[0]}")
        _validate_report(
            report,
            case_plan=case_plans[key[0]],
            expected_commit=expected_commit,
        )
        indexed[key] = report

    performance_plan = plan["performance"]
    trials = int(performance_plan["trials"])
    performance_summary: dict[str, object] = {}
    restart_summary: dict[str, object] = {}
    for case_id in sorted(case_plans):
        ratios: list[float] = []
        allocated_ratios: list[float] = []
        reserved_ratios: list[float] = []
        for trial in range(1, trials + 1):
            direct_key = (case_id, "performance", trial, "direct")
            public_key = (case_id, "performance", trial, "public")
            _require(direct_key in indexed, f"missing report {direct_key!r}")
            _require(public_key in indexed, f"missing report {public_key!r}")
            direct, public = indexed[direct_key], indexed[public_key]
            _require(
                _artifact_identity(direct) == _artifact_identity(public),
                f"direct/public artifact mismatch for {case_id} trial {trial}",
            )
            direct_result, public_result = direct["result"], public["result"]
            _require(direct_result["start_step"] == 0, "performance run did not start at zero")
            _require(public_result["start_step"] == 0, "performance run did not start at zero")
            expected_final = int(performance_plan["steps"])
            _require(
                direct_result["final_step"] == expected_final,
                "direct performance final step mismatch",
            )
            _require(
                public_result["final_step"] == expected_final,
                "public performance final step mismatch",
            )
            ratios.append(
                _finite_positive(public_result["mean_timestep_seconds"], "public timestep")
                / _finite_positive(direct_result["mean_timestep_seconds"], "direct timestep")
            )
            allocated_ratios.append(
                int(public["memory"]["peak_allocated_bytes"])
                / int(direct["memory"]["peak_allocated_bytes"])
            )
            reserved_ratios.append(
                int(public["memory"]["peak_reserved_bytes"])
                / int(direct["memory"]["peak_reserved_bytes"])
            )
        mean_ratio = statistics.fmean(ratios)
        median_ratio = statistics.median(ratios)
        allocated_ratio = max(allocated_ratios)
        reserved_ratio = max(reserved_ratios)
        _require(
            mean_ratio <= float(performance_plan["public_over_direct_mean_timestep_ratio_max"]),
            f"mean timestep non-regression failed for {case_id}",
        )
        _require(
            median_ratio
            <= float(
                performance_plan[
                    "public_over_direct_median_timestep_ratio_max"
                ]
            ),
            f"median timestep non-regression failed for {case_id}",
        )
        _require(
            allocated_ratio
            <= float(
                performance_plan[
                    "public_over_direct_peak_allocated_ratio_max"
                ]
            ),
            f"allocated-memory non-regression failed for {case_id}",
        )
        _require(
            reserved_ratio
            <= float(
                performance_plan[
                    "public_over_direct_peak_reserved_ratio_max"
                ]
            ),
            f"reserved-memory non-regression failed for {case_id}",
        )
        performance_summary[case_id] = {
            "paired_public_over_direct_timestep_ratios": ratios,
            "mean_timestep_ratio": mean_ratio,
            "median_timestep_ratio": median_ratio,
            "peak_allocated_ratio": allocated_ratio,
            "peak_reserved_ratio": reserved_ratio,
            "artifact_identity": "byte_for_byte",
        }

        segment_key = (case_id, "segment", 0, "public")
        direct_resume_key = (case_id, "resume", 0, "direct")
        public_resume_key = (case_id, "resume", 0, "public")
        for key in (segment_key, direct_resume_key, public_resume_key):
            _require(key in indexed, f"missing restart report {key!r}")
        segment = indexed[segment_key]
        direct_resume = indexed[direct_resume_key]
        public_resume = indexed[public_resume_key]
        segment_step = int(plan["restart"]["segment_steps"])
        final_step = int(plan["restart"]["continuous_steps"])
        _require(segment["result"]["start_step"] == 0, "segment did not start at zero")
        _require(segment["result"]["final_step"] == segment_step, "segment final step mismatch")
        _require(str(segment_step) in segment["checkpoints"], "segment checkpoint is absent")
        continuous_artifacts = _artifact_identity(
            indexed[(case_id, "performance", 1, "public")]
        )
        for entry, resumed in (("direct", direct_resume), ("public", public_resume)):
            _require(
                resumed["result"]["start_step"] == segment_step,
                f"{entry} resume start mismatch",
            )
            _require(
                resumed["result"]["final_step"] == final_step,
                f"{entry} resume final mismatch",
            )
            _require(
                _artifact_identity(resumed) == continuous_artifacts,
                f"{entry} resumed artifacts differ from continuous run for {case_id}",
            )
        _require(
            _artifact_identity(direct_resume) == _artifact_identity(public_resume),
            f"direct/public resumed artifacts differ for {case_id}",
        )
        restart_summary[case_id] = {
            "checkpoint_step": segment_step,
            "final_step": final_step,
            "direct_public_identity": "byte_for_byte",
            "continuous_resume_identity": "byte_for_byte",
        }

    expected_keys = 4 * (trials * 2 + 3)
    _require(len(indexed) == expected_keys, "unexpected report classes remain")
    return {
        "schema_version": 1,
        "phase": "P7.7.12",
        "classification": "PASS_P7_7_12_SINGLE_H100_NON_REGRESSION",
        "qualification_complete": True,
        "expected_commit": expected_commit,
        "report_count": len(reports),
        "performance": performance_summary,
        "restart": restart_summary,
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    report_paths = sorted(args.reports_directory.glob("*.json"))
    result = analyze(
        _load_object(args.plan),
        [_load_object(path) for path in report_paths],
        expected_commit=args.expected_commit,
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
