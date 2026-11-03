#!/usr/bin/env python3
"""S-19 preparation contract; generating a plan never submits a job."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from benchmarks._artifact_io import atomic_write_json
from benchmarks.plan_v0_3_s09_h100 import build_plan as frozen_s09_plan
from benchmarks._s19_compare import KNOWN_NONFINITE_PCG_RECORDS

BASELINE_COMMIT = "47473880534a66f6da940db82122c79cda48a86b"
CANDIDATE_COMMIT = "c7c940e693719becccef80a918142a27aa050510"
CONSUMER_RUNTIME_COMMIT = "140bd273693c341545ceb023bfd2de2e36458e07"
BASELINE_WHEEL_SHA256 = "8e2bd0c7941220f6c7c9b4cba80d9802e1495db626ad1b330a2e1681645187d5"
CANDIDATE_WHEEL_SHA256 = "33521a8309a7da1e8a3157d54e020264f123c25b3f6f4b149bb150e779b0f82e"
CONSUMER_WHEEL_SHA256 = "c40fc48dcd3c14eaf784abdb564ac2df89c15de3b0c49ab62301267887102a14"
CONTROL_ROOT = "/scratch1/vincent/PSSolver/pssolver_v0_4_s19_h100_c7c940e_20261102_v1_control"
OUTPUT_ROOT = "/scratch1/vincent/PSSolver/pssolver_v0_4_s19_h100_c7c940e_20261102_v1_output"
VENV_ROOT = "/scratch1/vincent/PSSolver/pssolver_v0_4_s19_h100_c7c940e_20261102_v1_venvs"
BASE_INTERPRETER = "/home/fansenwei/miniconda3/envs/Nematics3D/bin/python"
ROLES = ("baseline_a", "baseline_b", "candidate")
REFRESH_INTERVALS = ("default", None, 1, 2, 7)
CALLER_STATES = frozen_s09_plan()["tf32_backward"]["caller_states"]


def build_plan() -> dict[str, object]:
    frozen = frozen_s09_plan()
    matrix = deepcopy(frozen["profile_matrix"])
    for row in matrix:
        baseline = row["provider_role"] == "baseline"
        row["provider_commit"] = BASELINE_COMMIT if baseline else CANDIDATE_COMMIT
        row["provider_wheel_sha256"] = BASELINE_WHEEL_SHA256 if baseline else CANDIDATE_WHEEL_SHA256
    environment = deepcopy(frozen["environment"])
    environment.update(matplotlib="3.7.5", packaging="26.3", typing_extensions="4.16.0",
                       stale_dist_info_resolution="D-0021; preserve original pip freeze")
    consumer = deepcopy(frozen["consumer_h100_gate"])
    # main 140bd27 has 304 tests. Five CUDA-only cases are in addition to the
    # 298 CPU passes; the intentionally opt-in research benchmark stays skipped.
    consumer.update(expected_passed=303, cpu_expected_passed=298, expected_skipped=1)
    return {
        "schema": "pssolver.v0_4.s19.single_h100_plan.v1",
        "classification": "S19_PREPARATION_DRAFT_NOT_SUBMISSION_AUTHORITY",
        "plan_step": "S-19",
        "decisions": ["D-0011", "D-0014", "D-0016", "D-0018", "D-0019", "D-0020", "D-0021", "D-0022"],
        "provider": {
            "baseline": {"commit": BASELINE_COMMIT, "version": "0.3.0", "wheel_sha256": BASELINE_WHEEL_SHA256},
            "candidate": {"commit": CANDIDATE_COMMIT, "version": "0.4.0.dev0", "wheel_sha256": CANDIDATE_WHEEL_SHA256},
        },
        "consumer": {"commit": CONSUMER_RUNTIME_COMMIT, "version": "0.0.1.dev0", "wheel_sha256": CONSUMER_WHEEL_SHA256},
        "scheduler": deepcopy(frozen["scheduler"]),
        "environment": environment,
        "job_stage_order": deepcopy(frozen["job_stage_order"]),
        "phase_numbering": {
            "1": ["node_local_tmp_environment_and_host_compiler", "installed_wheel_cuda_only_tests"],
            "2": ["cuda_long_horizon_hermitian"], "3": ["tf32_forward_backward_scope"],
            "4": ["checkpoint_restart_and_released_fixture_matrix"],
            "5": ["numerical_gradient_taylor_and_replay"],
            "6": ["independent_control_consumer"], "7": ["paired_performance_memory_profiles"],
            "8": ["cumulative_analyzer_and_checksums"],
        },
        "cuda_only_tests": deepcopy(frozen["cuda_only_tests"]),
        "long_horizon": deepcopy(frozen["long_horizon"]),
        "tf32_backward": deepcopy(frozen["tf32_backward"]),
        "consumer_h100_gate": consumer,
        "checkpoint_contract": {
            "cross_device_cells": 8, "directions": ["cpu_to_cuda", "cuda_to_cpu"],
            "import_payload_preserved": True, "same_device_continuous_restart_exact": True,
            "cross_device_contract": "unchanged S-09 device comparison; never assert CPU/CUDA bitwise equivalence",
            "released_versions": ["v0.2.x", "v0.3.0"], "read_only_fixtures": True,
            "current_exports_include": ["functional_periodic", "functional_channel"],
        },
        "numerical": {
            "roles": list(ROLES), "sequential_not_concurrent": True,
            "same_absolute_calculation_output_restart_path": True,
            "preserve_results_before_reset": True,
            "compile_cache_per_process": "unique, initially empty, never reused",
            "families": deepcopy(frozen["profile_contract"]["runtime_families"]),
            "refresh_intervals": list(REFRESH_INTERVALS), "steps": 20,
            "pcg_paths": ["production", "transpose", "implicit", "oracle"],
            "pcg_devices": {"production": "cuda:0", "transpose": "cuda:0", "implicit": "cuda:0", "oracle": "cpu (unchanged public oracle guard)"},
            "known_nonfinite_records": [list(row) for row in sorted(KNOWN_NONFINITE_PCG_RECORDS)],
            "known_nonfinite_comparison": "D-0022: all raw bytes A/A/B, including positions and bit patterns; differences are N, not tolerated",
            "baseline_self_comparison": ["baseline_a", "baseline_b"],
            "candidate_comparisons": [["baseline_a", "candidate"], ["baseline_b", "candidate"]],
            "compare_runtime_and_configuration_identities": True,
            "public_r1_dispatch_coverage_separate_from_runtime_factories": True,
            "bitwise_required": True, "nonzero_tolerance_authorized": False,
            "do_not_reuse_s09_revert_proxy": True,
        },
        "failure_policy": {
            "H": {"kinds": ["identity", "environment", "device", "fallback", "nonfinite", "crash"],
                  "continue": False, "qualification": "FAIL"},
            "N": {"kinds": ["finite_no_fallback_non_bitwise_numerical_difference"],
                  "continue_control_and_performance": True, "qualification": "FAIL",
                  "retain": "every field and step: absolute and relative maxima; all samples"},
            "D0022": {"exact_nonfinite_records": [list(x) for x in sorted(KNOWN_NONFINITE_PCG_RECORDS)],
                      "different_raw_bytes": "N/FAIL, continue; metrics undefined at nonfinite values, retain all raw bytes",
                      "all_other_nonfinite": "H/STOP, including every forward field"},
        },
        "grid_contracts": deepcopy(frozen["grid_contracts"]),
        "profile_matrix": matrix,
        "profile_contract": deepcopy(frozen["profile_contract"]),
        "thresholds": deepcopy(frozen["thresholds"]),
        "performance_adjudication": deepcopy(frozen["performance_adjudication"]),
        "d0013_exception_applies": False,
        "deployment": {
            "base_interpreter": BASE_INTERPRETER,
            "control_root": CONTROL_ROOT, "output_root": OUTPUT_ROOT, "venv_root": VENV_ROOT,
            "all_roots_must_be_absent_before_creation": True,
            "venv_roles": ["baseline", "candidate", "consumer"],
            "creation": "base/bin/python -I -B -m venv --system-site-packages NEW_PREFIX",
            "bootstrap_wheels": {"pip": "25.2", "setuptools": "78.1.1"},
            "bootstrap_reason": "Install exact C versions into each new venv; ensurepip's bundled versions otherwise shadow the base. Original shared environment untouched.",
            "install": "venv/bin/python -I -B -m pip --disable-pip-version-check --no-cache-dir install --no-deps --no-index --no-compile EXACT_WHEEL",
            "verify_uploaded_hashes_before_venv_creation": True,
            "sources_only_from_verified_archives": True,
            "support_contains_provider_or_consumer_package": False,
            "consumer_provenance_reference": "Separate local clone of a verified git bundle containing only Control main 140bd27 and reachable history; src and .git serve immutable provenance tests only, never added to sys.path; installed packages preloaded and every loaded module checked against own venv",
            "verify_each_venv_isolated_import_versions_pip_check_and_RECORD": True,
            "replay_all_job_preconditions_on_login_before_sbatch": True,
            "create_control_logs_before_submission": True,
            "cleanup": "After local evidence retrieval and manifest verification remove only these three venvs and unpacked transient source/support trees; retain inputs, logs, results and manifests; node-local owned tmp is cleaned by EXIT trap.",
            "record_sacct_and_scontrol_raw": True,
        },
        "submission": {"formal_jobs": 1, "automatic_retry": False,
                       "diagnostic_jobs_max": 1, "separate_user_confirmation_required": True},
        "preparation_review_classification": "INTERIM_PASS_*",
        "runtime_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, build_plan(), overwrite=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
