#!/usr/bin/env python3
"""Build and hash-freeze the shell-free RC6 single-H100 stage plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.plan_rc6_single_h100_qualification import (
    CONSUMER_SOURCE,
    PROVIDER_BASELINE,
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
)
from benchmarks.run_rc6_single_h100_stage_plan import (
    DRAFT_SCHEMA,
    EXPECTED_SCHEMAS,
    STAGES,
    _atomic_json,
    freeze_plan,
)


def _stage(name: str, python: Path, script: Path, command: list[str], artifact: Path) -> dict[str, object]:
    return {
        "name": name,
        "command": [str(python), str(script), *command],
        "environment": {},
        "artifact": str(artifact),
        "expected_schema": EXPECTED_SCHEMAS[name],
    }


def build_draft(args: argparse.Namespace) -> dict[str, object]:
    if args.candidate_wheel_sha256 != PROVIDER_CANDIDATE_WHEEL_SHA256:
        raise ValueError("candidate wheel SHA-256 differs")
    output = args.output_root.expanduser().absolute()
    support = args.provider_support.expanduser().resolve()
    consumer_support = args.consumer_support.expanduser().resolve()
    scripts = support / "benchmarks"
    baseline = args.baseline_python.expanduser().absolute()
    candidate = args.candidate_python.expanduser().absolute()
    consumer = args.consumer_python.expanduser().absolute()
    results = output / "results"
    profile_root = output / "profile_matrix"
    stages = [
        _stage("environment", candidate, scripts / "run_rc6_environment.py", [
            "--baseline-python", str(baseline), "--candidate-python", str(candidate),
            "--consumer-python", str(consumer), "--baseline-wheel", str(args.baseline_wheel.resolve()),
            "--candidate-wheel", str(args.candidate_wheel.resolve()), "--consumer-wheel", str(args.consumer_wheel.resolve()),
            "--provider-support", str(support), "--tmp-preflight", str(args.tmp_preflight.resolve()),
            "--compiler-work-root", str(output / "compiler_preflight"), "--output", str(results / "environment.json"),
        ], results / "environment.json"),
        _stage("cuda_only", candidate, scripts / "run_rc6_cuda_only.py", [
            "--test-root", str(support), "--detail-output", str(results / "cuda_detail.json"),
            "--policy-output", str(results / "cuda_tf32.json"), "--output", str(results / "cuda.json"),
        ], results / "cuda.json"),
        _stage("contract_smokes", candidate, scripts / "run_rc6_contract_smokes.py", [
            "--test-root", str(support), "--work-root", str(output / "contract_work"),
            "--output", str(results / "contract_smokes.json"),
        ], results / "contract_smokes.json"),
        _stage("tf32_scope", candidate, scripts / "qualify_rc458_tf32_scope.py", [
            "--provider-commit", PROVIDER_CANDIDATE,
            "--provider-wheel-sha256", args.candidate_wheel_sha256,
            "--output", str(results / "tf32_scope.json"),
        ], results / "tf32_scope.json"),
        _stage("current_checkpoints", candidate, scripts / "run_rc6_current_checkpoints.py", [
            "--test-root", str(support), "--output-root", str(output / "current_checkpoints"),
            "--output", str(results / "current_checkpoints.json"),
        ], results / "current_checkpoints.json"),
        _stage("released_checkpoints", candidate, scripts / "run_rc6_released_checkpoints.py", [
            "--test-root", str(support), "--released-record", str(args.released_record.resolve()),
            "--output", str(results / "released_checkpoints.json"),
        ], results / "released_checkpoints.json"),
        _stage("numerical", candidate, scripts / "run_rc6_numerical.py", [
            "--baseline-python", str(baseline), "--candidate-python", str(candidate),
            "--support-root", str(support), "--child-root", str(output / "numerical_children"),
            "--output", str(results / "numerical.json"),
        ], results / "numerical.json"),
        _stage("profiles", candidate, scripts / "run_rc6_profile_matrix.py", [
            "--baseline-python", str(baseline), "--candidate-python", str(candidate),
            "--support-root", str(support), "--output-root", str(profile_root),
            "--command-plan", str(output / "profile_command_plan.json"),
        ], profile_root / "MATRIX_COMPLETE.json"),
        _stage("consumer", consumer, scripts / "run_rc6_consumer.py", [
            "--consumer-support", str(consumer_support),
            "--provider-reference", str(args.provider_reference.resolve()),
            "--work-root", str(output / "consumer_work"), "--output", str(results / "consumer.json"),
        ], results / "consumer.json"),
        _stage("assemble", candidate, scripts / "assemble_rc6_h100_auxiliary.py", [
            "--environment", str(results / "environment.json"), "--cuda-only", str(results / "cuda.json"),
            "--contract-smokes", str(results / "contract_smokes.json"), "--tf32-scope", str(results / "tf32_scope.json"),
            "--current-checkpoints", str(results / "current_checkpoints.json"),
            "--released-checkpoints", str(results / "released_checkpoints.json"),
            "--numerical", str(results / "numerical.json"), "--consumer", str(results / "consumer.json"),
            "--output", str(results / "auxiliary.json"),
        ], results / "auxiliary.json"),
        _stage("analyze", candidate, scripts / "analyze_rc6_single_h100.py", [
            "--profile-root", str(profile_root / "profiles"), "--auxiliary", str(results / "auxiliary.json"),
            "--output", str(results / "analysis.json"),
        ], results / "analysis.json"),
    ]
    if [stage["name"] for stage in stages] != list(STAGES):
        raise RuntimeError("stage order differs")
    return {
        "schema": DRAFT_SCHEMA,
        "provider_baseline_commit": PROVIDER_BASELINE,
        "provider_candidate_commit": PROVIDER_CANDIDATE,
        "consumer_source_commit": CONSUMER_SOURCE,
        "stages": stages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-python", "candidate-python", "consumer-python", "baseline-wheel", "candidate-wheel", "consumer-wheel", "provider-support", "consumer-support", "provider-reference", "released-record", "tmp-preflight", "output-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--candidate-wheel-sha256", required=True)
    args = parser.parse_args(argv)
    draft = build_draft(args)
    frozen = freeze_plan(draft, args.output_root)
    _atomic_json(args.output, frozen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
