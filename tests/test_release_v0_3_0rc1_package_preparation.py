from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
RECORD = ROOT / "notes" / "pssolver_v0_3_0rc1_package_preparation.json"
SOURCE_RECORD = ROOT / "notes" / "pssolver_v0_3_0rc1.json"
SUCCESSOR = (
    ROOT / "notes" / "PSSolver_v0_3_0_s03_successor_61_rc1_source_freeze.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc1_package_record_binds_the_exact_release_source() -> None:
    record = _read(RECORD)
    source = record["source"]
    assert source["release_source_commit"] == (
        "c031cdae646edc5b202b05238db1eb513b255805"
    )
    assert source["release_source_parent"] == (
        "0e96ee5d05193b0ddae189f6171ba387a8a96b6b"
    )
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", source["release_source_parent"], source["release_source_commit"]],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    assert source["record_commit_is_release_source"] is False
    assert source["runtime_delta_from_development_tip"] == [
        "pssolver/_version.py: 0.3.0.dev0 -> 0.3.0rc1"
    ]


def test_rc1_package_record_binds_source_records_without_rewriting_them() -> None:
    source = _read(RECORD)["source"]
    assert _sha256(SOURCE_RECORD) == source["source_record_sha256"]
    assert _sha256(SUCCESSOR) == source["source_successor_record_sha256"]
    assert _read(SOURCE_RECORD)["status"].endswith(
        "pending_artifacts_and_independent_acceptance"
    )


def test_rc1_artifact_identities_are_exact() -> None:
    build = _read(RECORD)["build"]
    assert build["source_archive"] == {
        "path": "package/PSSolver-0.3.0rc1.tar",
        "sha256": "a8f2c049ef1f46349b30f47bbe917f9382bd4cfccbb8ddbf3261d9191f39b95c",
        "size_bytes": 27842560,
        "prefix": "PSSolver-0.3.0rc1/",
    }
    assert build["wheel"]["sha256"] == (
        "3e52204593c8e09c7f88cbe54488b977800955f285d90fa1f126bd4dc21c6578"
    )
    assert build["sdist"]["sha256"] == (
        "ab9d5f83408e3591063d08e90df1117fe9f95360ff6a6f646148b2f99e8f367a"
    )
    assert build["sdist_rebuilt_wheel_member_content_identical"] is True
    assert build["wheel_python_content_matches_clean_archive"] is True


def test_rc1_source_cpu_gates_bind_both_d0008_baselines() -> None:
    gates = _read(RECORD)["cpu_gates"]
    assert gates["common"] == {
        "passed": 4828,
        "failed": 0,
        "skipped": 1,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
        "optional_skip": "tests/test_analytic_periodic_defect_gas_2d.py::test_nematics3d_recovers_requested_defect_count_charges_and_positions",
        "optional_skip_reason": "optional nematics3d dependency unavailable in both D-0008 baseline clones",
    }
    assert gates["a2_clone"]["torch"] == "2.6.0+cu126"
    assert gates["v0_2_2_clone"]["torch"] == "2.6.0+cu124"
    assert len(gates["cuda_only_deselects"]) == 7


def test_rc1_installed_wheel_gate_is_complete_and_unshadowed() -> None:
    gate = _read(RECORD)["installed_wheel_gate"]
    assert (gate["passed"], gate["failed"], gate["skipped"], gate["xfailed"]) == (
        7,
        0,
        0,
        0,
    )
    assert gate["pssolver_version"] == "0.3.0rc1"
    assert gate["functional_api_version"] == "1.1"
    assert gate["pip_check"] == "PASS"
    assert gate["source_shadow_import"] is False
    assert gate["all_shipped_modules_imported"] is True
    assert gate["legacy_and_compiled_v2_channel_runtimes_built"] is True
    assert gate[
        "plane_periodic_legacy_channel_and_complete_stress_channel_public_compilation_dry_runs_passed"
    ] is True


def test_rc1_cumulative_gate_boundary_is_not_overclaimed() -> None:
    gates = _read(RECORD)["cumulative_gates"]
    assert gates == {
        "G-01": "pending_R-0038_only_for_V-44",
        "G-02": "PASS",
        "G-03": "PASS",
        "G-04": "PASS",
        "G-05": "PASS",
        "G-06": "PASS_WITH_D-0013_BOUNDED_EXCEPTION",
        "G-07": "PASS",
        "G-08": "PASS",
        "G-09": "pending_R-0038_independent_candidate_acceptance",
    }


def test_rc1_package_preparation_does_not_claim_external_authority() -> None:
    authorization = _read(RECORD)["authorization"]
    assert authorization["eligible_for_R-0038_independent_acceptance"] is True
    assert all(
        authorization[key] is False
        for key in (
            "push_authorized",
            "tag_authorized",
            "github_release_authorized",
            "pypi_authorized",
            "hpcc_test_directory_authorized",
            "slurm_or_h100_authorized",
        )
    )


def test_rc1_freeze_changes_only_the_version_in_the_runtime_tree() -> None:
    source = _read(RECORD)["source"]
    assert source["functional_api_version"] == "1.1"
    for key in (
        "solver_equations_changed_by_freeze",
        "numerical_algorithms_changed_by_freeze",
        "checkpoint_schema_changed_by_freeze",
        "functional_api_version_changed_by_freeze",
        "provider_identity_changed_by_freeze",
        "production_default_changed_by_freeze",
        "threshold_changed_by_freeze",
        "skip_or_xfail_added_by_freeze",
    ):
        assert source[key] is False
