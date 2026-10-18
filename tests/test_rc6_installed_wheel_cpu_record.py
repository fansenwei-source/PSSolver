"""Evidence contract for the RC6 frozen installed-wheel CPU gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_installed_wheel_cpu.json"
SOURCE_FREEZE = ROOT / "notes/PSSolver_v0_2_0rc6_source_freeze.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc6_installed_cpu_record_binds_the_qualified_source():
    record = _record()
    assert record["phase"] == "RC6-9"
    assert record["classification"] == "PASS_RC6_FROZEN_INSTALLED_WHEEL_CPU"
    assert record["qualified_source"] == {
        "branch": "fix/v0.2.0rc6",
        "commit": "ea801ec08f5b2465661334e1c126133e55b68e96",
        "latest_runtime_implementation_commit": (
            "80c6e17db5838ef4f142add4955221637efcdf3e"
        ),
        "package_version": "0.2.0rc6",
        "record_commit_is_qualified_source": False,
        "requires_cpu_requalification": False,
    }


def test_rc6_installed_cpu_record_preserves_the_source_freeze():
    assert _sha256(SOURCE_FREEZE) == (
        "2d393b06ae6eaef29012d43b9fc370637da3e4962f2345b7ce82bd09656bca64"
    )
    source_freeze = json.loads(SOURCE_FREEZE.read_text(encoding="utf-8"))
    assert source_freeze["authorization"]["eligible_for_installed_wheel_cpu"] is True
    assert source_freeze["authorization"]["cumulative_qualification_complete"] is False
    assert _record()["source_cpu_reused"]["passed"] == 3936


def test_rc6_installed_cpu_record_binds_artifacts_and_results():
    installed = _record()["installed_wheel"]
    assert installed["source_archive"]["sha256"] == (
        "60fcd0d591e68b4b66b3da8428f354e4f3f02135f6407bd9256c64340bb6f7ff"
    )
    assert installed["sdist"]["sha256"] == (
        "77b19c10523f9aae70c292ec5c7e3db86ca4479960705851181d8b4c0c819be4"
    )
    assert installed["wheel"]["sha256"] == (
        "8840ebfaa44a20a828928256c319edfa2aa4a9b8adf4f1bac014c33964ba0dd6"
    )
    assert installed["wheel"]["installed_with_no_deps"] is True
    assert installed["pip_check"] == "PASS"
    assert installed["source_shadow_import"] is False
    assert installed["targeted"]["passed"] == 166
    assert installed["targeted"]["failed"] == 0
    assert installed["applicable_full"]["passed"] == 3929
    assert installed["applicable_full"]["failed"] == 0
    assert installed["applicable_full"]["errors"] == 0
    assert installed["applicable_full"]["skipped"] == 7
    assert installed["applicable_full"]["subtests_passed"] == 8


def test_rc6_installed_cpu_record_names_exact_skips_and_exclusion():
    installed = _record()["installed_wheel"]
    assert set(installed["allowed_skips"]) == {
        "tests/test_benchmark_bounded_axis_applicability.py::test_cuda_smoke_records_actual_device_and_memory",
        "tests/test_bounded_axis_execution_plan.py::test_cuda_plan_key_uses_the_allocated_device_identity",
        "tests/test_phase4_combined_sbdf2_modal_block.py::test_cuda_canary_binds_concrete_device_and_remains_finite",
        "tests/test_phase8_p845_h100_qualification.py::test_unindexed_cuda_request_binds_lifting_to_allocated_device_identity",
        "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_runtime_matches_cpu_and_binds_allocated_device",
        "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_continuous_split_and_file_restart_are_exact",
        "tests/test_rc4_plane_nyquist_cuda.py::test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence",
    }
    assert installed["excluded_source_layout_file"] == (
        "tests/test_phase8_p861_local_cumulative_audit.py"
    )
    assert installed["applicable_full"]["excluded_source_layout_tests"] == 7


def test_rc6_installed_cpu_record_rejects_setup_attempts_as_evidence():
    rejected = _record()["rejected_harness_attempts"]
    assert [item["classification"] for item in rejected] == [
        "EXTERNAL_ARCHIVE_SUPPORT_ROOT_GIT_METADATA_ERROR",
        "EXTERNAL_FRESH_VENV_NUMPY_METADATA_MISMATCH",
    ]
    assert all(item["used_as_qualification_evidence"] is False for item in rejected)


def test_rc6_installed_cpu_record_binds_complete_external_manifest():
    evidence = _record()["evidence"]
    assert evidence["checksum_entries"] == 1898
    assert evidence["checksum_passed"] == 1898
    assert evidence["checksum_failed"] == 0
    assert evidence["checksums_sha256"] == (
        "e05b1d9d26a898a58e7634505045642c41dee1a6f1ac18a5062d33dba48db6c4"
    )


def test_rc6_installed_cpu_record_remains_fail_closed_for_later_gates():
    record = _record()
    assert record["pending_rc6_gates"] == [
        "released_checkpoint_compatibility_and_tamper",
        "provider_identity_impact_review",
        "pssolver_control_migration_and_installed_wheel_qualification_if_required",
        "single_h100_cumulative_qualification",
        "independent_post_release_verification",
    ]
    authorization = record["authorization"]
    assert authorization["source_and_installed_wheel_cpu_complete"] is True
    assert authorization["rc6_cumulative_qualification_complete"] is False
    assert authorization["eligible_for_push"] is False
    assert authorization["eligible_for_h100_submission"] is False
    assert authorization["h100_submission_count"] == 0
    assert authorization["automatic_retry"] is False
    assert authorization["tag_created"] is False
    assert authorization["release_created"] is False
    assert authorization["pypi_published"] is False
    assert authorization["production_default_changed"] is False
    assert authorization["PSSolver_Control_modified"] is False
    assert authorization["nematics3d_modified"] is False
