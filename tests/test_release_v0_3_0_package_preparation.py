from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
RECORD = ROOT / "notes" / "pssolver_v0_3_0_package_preparation.json"
SOURCE_RECORD = ROOT / "notes" / "pssolver_v0_3_0.json"
SUCCESSOR = (
    ROOT / "notes" / "PSSolver_v0_3_0_s03_successor_64_final_source_freeze.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_package_record_binds_the_exact_release_source() -> None:
    source = _read(RECORD)["source"]
    assert source["release_source_commit"] == (
        "47473880534a66f6da940db82122c79cda48a86b"
    )
    assert source["release_source_parent"] == (
        "72813d753dc9ca52e39240a643781c22605fd187"
    )
    assert subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            source["release_source_parent"],
            source["release_source_commit"],
        ],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    assert source["record_commit_is_release_source"] is False
    assert source["runtime_delta_from_rc1_release_source"] == [
        "pssolver/_version.py: 0.3.0rc1 -> 0.3.0"
    ]


def test_final_package_record_binds_source_records_without_rewriting_them() -> None:
    source = _read(RECORD)["source"]
    assert _sha256(SOURCE_RECORD) == source["source_record_sha256"]
    assert _sha256(SUCCESSOR) == source["source_successor_record_sha256"]
    assert _read(SOURCE_RECORD)["status"].endswith(
        "pending_artifacts_and_independent_acceptance"
    )


def test_final_artifact_identities_are_exact() -> None:
    build = _read(RECORD)["build"]
    assert build["source_archive"] == {
        "path": "package/PSSolver-0.3.0.tar",
        "sha256": "78fbe5e646b16015bfa2874dc072df93db64cd238757e646460069f5bac6f873",
        "size_bytes": 27904000,
        "prefix": "PSSolver-0.3.0/",
    }
    assert build["wheel"]["sha256"] == (
        "8e2bd0c7941220f6c7c9b4cba80d9802e1495db626ad1b330a2e1681645187d5"
    )
    assert build["sdist"]["sha256"] == (
        "ec46471f5c35fc56089255155508b8a23a6d71ee81488d6b4aec933faed4d68d"
    )
    assert build["sdist_rebuilt_wheel_member_content_identical"] is True
    assert build["wheel_python_content_matches_clean_archive"] is True
    assert build["rc1_normalized_member_names_identical"] is True
    assert build["rc1_changed_members"] == [
        "pssolver-0.3.0.dist-info/METADATA",
        "pssolver-0.3.0.dist-info/RECORD",
        "pssolver/_version.py",
    ]


def test_final_source_cpu_gates_bind_both_d0008_baselines() -> None:
    gates = _read(RECORD)["cpu_gates"]
    assert gates["common"] == {
        "passed": 4851,
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


def test_final_installed_wheel_gate_is_complete_and_unshadowed() -> None:
    gate = _read(RECORD)["installed_wheel_gate"]
    assert (gate["passed"], gate["failed"], gate["skipped"], gate["xfailed"]) == (
        7,
        0,
        0,
        0,
    )
    assert gate["pssolver_version"] == "0.3.0"
    assert gate["functional_api_version"] == "1.1"
    assert gate["pip_check"] == "PASS"
    assert gate["source_shadow_import"] is False
    assert gate["all_shipped_modules_imported"] is True
    assert gate["legacy_and_compiled_v2_channel_runtimes_built"] is True
    assert gate[
        "plane_periodic_legacy_channel_and_complete_stress_channel_public_compilation_dry_runs_passed"
    ] is True


def test_final_cumulative_gate_boundary_is_not_overclaimed() -> None:
    gates = _read(RECORD)["cumulative_gates"]
    assert gates["G-01"] == "PASS"
    assert gates["G-06"] == "PASS_WITH_D-0013_BOUNDED_EXCEPTION"
    assert gates["G-09_rc1_candidate_acceptance"] == "PASS_R0038"
    assert gates["G-09_rc1_post_release_acceptance"] == "PASS_R0039"
    assert gates["G-09_final_candidate_acceptance"] == "pending_R0040"
    assert gates["G-09_final_post_release_acceptance"] == "not_started"


def test_final_package_preparation_does_not_claim_external_authority() -> None:
    authorization = _read(RECORD)["authorization"]
    assert authorization["eligible_for_R0040_independent_acceptance"] is True
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


def test_final_freeze_changes_only_the_version_in_the_runtime_tree() -> None:
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
