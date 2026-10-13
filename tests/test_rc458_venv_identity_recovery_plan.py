"""Static contract for the RC4.5.8 venv-identity recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc458_venv_identity_recovery_plan.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_failed_h100_attempt_is_bound_without_reclassification():
    record = _record()
    failed = record["failed_attempt"]

    assert failed == {
        "job_id": 10879343,
        "state": "FAILED",
        "exit_code": "1:0",
        "control_directory": "/home/fansenwei/pssolver_rc458_tf32_h100_5c286f8_b8539e0_20261013_v1",
        "checksum_entries": 4443,
        "checksum_failures": 0,
        "checksum_manifest_sha256": "c31f1ccc06ef815c582d8cd0f784aad2864e3569d710cfa10e8af337e9c3595b",
        "classification": "FAIL_RC4_5_8_H100_EXTERNAL_VENV_IDENTITY_HARNESS",
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "scientific_failure": False,
        "solver_failure": False,
        "profiles_completed": 0,
        "profiles_expected": 96,
        "complete_marker_present": False,
        "archive_read_only": True,
    }


def test_recovery_is_qualification_only_and_fail_closed():
    record = _record()
    contract = record["recovery_contract"]

    assert record["qualification_complete"] is False
    assert all(
        contract[key] is True
        for key in (
            "preserve_lexical_interpreter_path",
            "record_resolved_interpreter_target_separately",
            "require_pyvenv_cfg",
            "require_runtime_prefix_matches_lexical_venv",
            "require_purelib_within_runtime_prefix",
            "require_import_within_purelib",
            "reject_source_shadow_import",
            "validate_candidate_provider_import",
            "validate_consumer_provider_import",
            "validate_consumer_package_import",
            "preserve_baseline_profile_interpreter_path",
            "preserve_candidate_profile_interpreter_path",
        )
    )
    assert contract["runtime_numerics_modified"] is False
    assert contract["thresholds_modified"] is False
    assert contract["production_default_changed"] is False


def test_recovery_uses_new_directories_and_no_current_authorization():
    record = _record()
    harness = record["harness_identity"]
    authorization = record["authorization"]

    assert harness["recovery_control_directory"].endswith("_recovery_v2")
    assert harness["recovery_scratch_directory"].endswith("_recovery_v2")
    assert harness["failed_control_directory_reused"] is False
    assert harness["failed_scratch_directory_reused"] is False
    assert authorization == {
        "push_authorized": False,
        "h100_recovery_submission_authorized": False,
        "automatic_retry": False,
        "second_recovery_job_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }


def test_versioned_recovery_sources_match_the_recorded_hashes():
    record = _record()

    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected
