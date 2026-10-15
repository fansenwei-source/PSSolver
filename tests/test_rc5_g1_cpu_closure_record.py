"""Evidence contract for the rc5 G-1 frozen CPU closure."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_cpu_closure.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_g1_cpu_record_binds_the_qualified_source_without_promoting_record_commit():
    record = _record()
    assert record["phase"] == "G-1.1"
    assert record["classification"] == (
        "PASS_RC5_G1_FROZEN_SOURCE_AND_INSTALLED_WHEEL_CPU"
    )
    source = record["qualified_source"]
    assert source == {
        "branch": "fix/v0.2.0rc5",
        "commit": "674a8121779be72c95aa5dc9e2babca0ce7b93a9",
        "runtime_implementation_commit": (
            "91c702c3deeb3e973fac80b6029e9dd02925937b"
        ),
        "package_version": "0.2.0rc5",
        "record_commit_is_qualified_source": False,
        "requires_cpu_requalification": False,
    }


def test_g1_cpu_record_requires_zero_source_failures_or_skips():
    assert _record()["source_cpu"] == {
        "passed": 3623,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "subtests_passed": 8,
        "seconds": 205.67,
    }


def test_g1_cpu_record_binds_the_archive_wheel_and_installed_results():
    wheel = _record()["installed_wheel"]
    assert wheel["source_archive"]["sha256"] == (
        "76f0a9190255e6e131e820559b3d30bbb51f0ff633c922dda9a49622ed5bc1ee"
    )
    assert wheel["wheel"]["sha256"] == (
        "3097b08b08ba7b2d488fd8af348ec48986d69007a640bd806a225ec9963bb65f"
    )
    assert wheel["source_shadow_import"] is False
    assert wheel["pip_check"] == "PASS"
    assert wheel["targeted"] == {
        "passed": 178,
        "failed": 0,
        "skipped": 0,
        "xml_sha256": (
            "c2befea2a18f0d89063da7773551e9ad2a24c908dafdae19ce31c0eaa1ef35c0"
        ),
    }
    assert wheel["applicable_full"] == {
        "passed": 3609,
        "failed": 0,
        "skipped": 7,
        "xfailed": 0,
        "subtests_passed": 8,
        "excluded_source_layout_tests": 7,
        "seconds": 203.49,
        "xml_sha256": (
            "64d9a81a208e912c259a74ae8bb2b74c9425f7ffd9ee360cb1bc7ac9a216967c"
        ),
    }


def test_g1_cpu_record_names_each_allowed_skip_and_the_source_layout_exclusion():
    wheel = _record()["installed_wheel"]
    assert set(wheel["allowed_skips"]) == {
        "tests/test_benchmark_bounded_axis_applicability.py::test_cuda_smoke_records_actual_device_and_memory",
        "tests/test_bounded_axis_execution_plan.py::test_cuda_plan_key_uses_the_allocated_device_identity",
        "tests/test_phase4_combined_sbdf2_modal_block.py::test_cuda_canary_binds_concrete_device_and_remains_finite",
        "tests/test_phase8_p845_h100_qualification.py::test_unindexed_cuda_request_binds_lifting_to_allocated_device_identity",
        "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_runtime_matches_cpu_and_binds_allocated_device",
        "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_continuous_split_and_file_restart_are_exact",
        "tests/test_rc4_plane_nyquist_cuda.py::test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence",
    }
    assert wheel["excluded_source_layout_file"] == (
        "tests/test_phase8_p861_local_cumulative_audit.py"
    )


def test_g1_cpu_record_rejects_the_cwd_bootstrap_attempt_as_evidence():
    rejected = _record()["rejected_harness_attempt"]
    assert rejected["classification"] == (
        "EXTERNAL_INSTALLED_TEST_CWD_BOOTSTRAP_ERROR"
    )
    assert rejected["used_as_qualification_evidence"] is False
    assert rejected["failed_tests"] == 7
    assert rejected["corrected_contract_preflight"] == "7 passed"


def test_g1_cpu_record_binds_the_complete_external_manifest():
    evidence = _record()["evidence"]
    assert evidence["checksum_entries"] == 1712
    assert evidence["checksum_passed"] == 1712
    assert evidence["checksum_failed"] == 0
    assert evidence["checksums_sha256"] == (
        "6a10414a20d3c5f7c12e21e98afb594fe3008be3010388355a16e6fdb9fbe832"
    )


def test_g1_cpu_closure_remains_fail_closed_for_later_gates():
    record = _record()
    assert record["pending_g1_gates"] == [
        "released_checkpoint_matrix_rc1_through_rc4",
        "provider_identity_impact_review",
        "pssolver_control_installed_wheel_qualification_if_required",
        "single_h100_cumulative_qualification",
        "post_release_verification",
    ]
    assert record["authorization"] == {
        "source_and_installed_wheel_cpu_complete": True,
        "g1_complete": False,
        "eligible_for_push": False,
        "eligible_for_h100_submission": False,
        "h100_submission_count": 0,
        "automatic_retry": False,
        "tag_created": False,
        "release_created": False,
        "pypi_published": False,
        "production_default_changed": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
    }
