#!/usr/bin/env python3
"""S-19 adjudication with the exact frozen S-09 performance formulas.

The frozen function runs in a private namespace: only identities, the plan,
auxiliary validation and failure reporting are rebound. Neither its code nor
its thresholds are edited. Performance failures are all retained; other
failed preconditions raise immediately. An N result can never become PASS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import analyze_v0_3_s09_h100 as frozen
from benchmarks._artifact_io import atomic_write_json
from benchmarks._s19_compare import HardQualificationError
from benchmarks.plan_v0_4_s19_h100 import (
    BASELINE_COMMIT, BASELINE_WHEEL_SHA256, CANDIDATE_COMMIT,
    CANDIDATE_WHEEL_SHA256, CONSUMER_RUNTIME_COMMIT, CONSUMER_WHEEL_SHA256,
    build_plan,
)


def validate_auxiliary(value, plan):
    expected = plan["job_stage_order"][:-2]
    stages = value.get("stages")
    if not isinstance(stages, list) or [row.get("name") for row in stages] != expected:
        raise HardQualificationError("missing, duplicate or reordered stage")
    if (value.get("formal_h100_submission_count") != 1
            or value.get("automatic_retry") is not False
            or value.get("second_job_submitted") is not False):
        raise HardQualificationError("submission contract mismatch")
    for row in stages:
        if (row.get("provider_candidate_commit") != CANDIDATE_COMMIT
                or row.get("candidate_wheel_sha256") != CANDIDATE_WHEEL_SHA256
                or row.get("execution_device") != "cuda"
                or row.get("finite") is not True
                or row.get("fallback_used") is not False):
            raise HardQualificationError("stage identity/device/finite/fallback mismatch")
        allowed = ("PASS", "N") if row["name"] == "numerical_gradient_taylor_and_replay" else ("PASS",)
        if row.get("classification") not in allowed:
            raise HardQualificationError("stage failed or unclassified")
    indexed = {row["name"]: row for row in stages}
    environment = stages[0]
    for key in ("installed_wheels_only", "source_shadow_import_false", "pip_check",
                "node_local_tmp", "host_compiler", "empty_compile_caches",
                "cuda_available", "single_visible_h100", "allocated_cuda_0"):
        if environment.get(key) is not True:
            raise HardQualificationError(f"environment gate: {key}")
    if (environment.get("baseline_wheel_sha256") != BASELINE_WHEEL_SHA256
            or environment.get("consumer_wheel_sha256") != CONSUMER_WHEEL_SHA256):
        raise HardQualificationError("environment wheel identity")
    cuda = indexed["installed_wheel_cuda_only_tests"]
    if (sorted(cuda.get("node_ids", [])) != sorted(plan["cuda_only_tests"])
            or cuda.get("passed_count") != 7 or cuda.get("nonpassing_count") != 0
            or cuda.get("junit_written") is not True):
        raise HardQualificationError("CUDA test coverage or JUnit")
    long = indexed["cuda_long_horizon_hermitian"]
    if (long.get("cases") != plan["long_horizon"]["cases"]
            or long.get("formal_h100_contract") is not True
            or long.get("all_finite") is not True or long.get("all_within_bound") is not True):
        raise HardQualificationError("long-horizon contract")
    tf32 = indexed["tf32_forward_backward_scope"]
    if tf32.get("caller_state_count") != 6 or tf32.get("functional_case_count") != 12:
        raise HardQualificationError("TF32 coverage")
    for key in ("forward_observed_policy", "backward_observed_policy", "success_restored",
                "exception_restored", "nested_lifo", "outside_scope_preserved"):
        if tf32.get(key) is not True:
            raise HardQualificationError(f"TF32 gate: {key}")
    checkpoint = indexed["checkpoint_restart_and_released_fixture_matrix"]
    if checkpoint.get("cross_device_cell_count") != 8:
        raise HardQualificationError("checkpoint cross-device coverage")
    for key in ("current_matrix", "released_fixtures", "restart_exact", "negative_gates", "sources_unchanged"):
        if checkpoint.get(key) is not True:
            raise HardQualificationError(f"checkpoint gate: {key}")
    numerical = indexed["numerical_gradient_taylor_and_replay"]
    for key in ("same_absolute_work_path", "independent_empty_caches", "baseline_self_comparison",
                "public_dispatch_passed", "four_pcg_paths", "all_fields_and_steps_recorded"):
        if numerical.get(key) is not True:
            raise HardQualificationError(f"numerical coverage: {key}")
    if numerical.get("trajectory_cases") != 40 or numerical.get("trajectory_steps") != 20:
        raise HardQualificationError("trajectory coverage")
    consumer = indexed["independent_control_consumer"]
    if (consumer.get("consumer_source_commit") != CONSUMER_RUNTIME_COMMIT
            or consumer.get("consumer_wheel_sha256") != CONSUMER_WHEEL_SHA256
            or consumer.get("passed_count") != plan["consumer_h100_gate"]["expected_passed"]
            or consumer.get("skip_node_ids") != [plan["consumer_h100_gate"]["allowed_skip"]]):
        raise HardQualificationError("Control identity/coverage")
    for key in ("installed_wheels_only", "functional_api_1_1", "provider_identity_migrated",
                "periodic_gradient", "channel_gradient", "checkpointed_adjoint", "negative_controls",
                "finite_cuda_smokes"):
        if consumer.get(key) is not True:
            raise HardQualificationError(f"Control gate: {key}")


def analyze(profile_root, auxiliary):
    plan = build_plan()
    validate_auxiliary(auxiliary, plan)
    failures = []
    exact_prefixes = {f"{row['family']}/{row['grid']} performance or memory gate failed: "
                      for row in plan["profile_matrix"]}

    def require(condition, message):
        if condition:
            return
        if any(message.startswith(prefix) for prefix in exact_prefixes):
            failures.append(message)
            return
        raise HardQualificationError(message)

    namespace = dict(frozen.analyze.__globals__)
    namespace.update(build_plan=build_plan, _validate_auxiliary=validate_auxiliary, _require=require,
                     BASELINE_COMMIT=BASELINE_COMMIT, CANDIDATE_COMMIT=CANDIDATE_COMMIT,
                     CONSUMER_RUNTIME_COMMIT=CONSUMER_RUNTIME_COMMIT)
    bound = types.FunctionType(frozen.analyze.__code__, namespace, frozen.analyze.__name__)
    result = bound(profile_root, auxiliary)
    for group in result["groups"]:
        group["passed"] = all(group["checks"].values())
    if len(failures) != sum(not group["passed"] for group in result["groups"]):
        raise HardQualificationError("recorded performance failures disagree with checks")
    numerical_failed = any(row["classification"] == "N" for row in auxiliary["stages"])
    passed = not failures and not numerical_failed
    result.update(schema="pssolver.v0_4.s19.h100_analysis.v1",
                  classification="PASS_S19_H100" if passed else "FAIL_S19_H100",
                  qualification_complete=passed, eligible_for_release_integration_review=passed,
                  recorded_performance_failures=failures, numerical_failed=numerical_failed,
                  all_samples_retained=True, d0013_exception_applies=False,
                  performance_formula="unmodified analyze_v0_3_s09_h100.analyze code object",
                  formal_h100_contract=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.profile_root, json.loads(args.auxiliary.read_text()))
    atomic_write_json(args.output, result, overwrite=False)
    return 0 if result["qualification_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
