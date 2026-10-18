"""Forward-only release-integration review for PSSolver 0.2.0rc5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pssolver.functional.api import functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_release_integration_review.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc5_integration_review_binds_the_qualified_runtime_and_g1_closure():
    record = _record()
    assert record["classification"] == (
        "PASS_RC5_RELEASE_INTEGRATION_REVIEW_READY_FOR_FINAL_PACKAGE_PREPARATION"
    )
    source = record["source_identity"]
    assert source["qualified_runtime_candidate_commit"] == (
        "dabeb9a487a82700ddd364c4233941a175808a3a"
    )
    assert source["reviewed_parent_commit"] == (
        "dcd59df27bb00863b46ca29f5c326fce4ec1da77"
    )
    assert source["record_commit_is_reviewed_source"] is False
    assert source["runtime_candidate_is_ancestor_of_reviewed_parent"] is True

    closure = record["g1_closure"]
    assert closure["qualification_complete"] is True
    assert closure["g1_complete"] is True
    assert closure["performance_memory_cells_passed"] == 16
    assert closure["performance_memory_cells_expected"] == 16
    assert closure["archive_checksum_passed"] == 30
    assert closure["archive_checksum_expected"] == 30
    assert _sha256(closure["path"]) == closure["sha256"]


def test_rc5_integration_review_preserves_the_historical_freeze_records():
    frozen = _record()["frozen_release_metadata"]
    for item in ("machine_record", "release_notes"):
        assert _sha256(frozen[item]["path"]) == frozen[item]["sha256"]
    assert frozen["historical_source_freeze_record_rewritten"] is False
    assert frozen["historical_pending_fields_preserved"] is True


def test_rc5_integration_review_records_only_qualification_scope_after_runtime():
    scope = _record()["post_runtime_candidate_scope"]
    assert scope["changed_file_count"] == 101
    assert scope["top_level_counts"] == {
        "benchmarks": 29,
        "jobs": 3,
        "notes": 37,
        "tests": 32,
    }
    assert set(scope["allowed_top_level_paths"]) == {
        "benchmarks",
        "jobs",
        "notes",
        "tests",
    }
    for key in (
        "pssolver_package_changed",
        "setup_py_changed",
        "pyproject_toml_changed",
        "manifest_in_changed",
        "version_changed_after_runtime_candidate",
        "runtime_or_numerical_implementation_changed",
    ):
        assert scope[key] is False
    assert scope["qualification_or_evidence_only"] is True


def test_rc5_integration_review_keeps_the_public_contract_stable():
    contract = _record()["public_contract"]
    assert contract["package_version"] == "0.2.0rc5"
    assert functional_protocol_provenance()["functional_protocol"]["current"] == (
        contract["functional_api_version"]
    )
    assert contract["functional_batch_sizes"] == [1]
    for key in (
        "production_default_changed",
        "compiled_runtime_promoted",
        "checkpoint_compatibility_removed",
        "PSSolver_Control_bundled",
        "nematics3d_modified",
    ):
        assert contract[key] is False


def test_rc5_integration_review_authorizes_preparation_but_not_release_actions():
    decision = _record()["integration_decision"]
    assert decision["release_integration_review_complete"] is True
    assert decision["eligible_for_branch_push"] is True
    assert decision["eligible_for_final_release_package_preparation"] is True
    assert decision["post_release_verification_required"] is True
    assert decision["final_tag_target_commit"] is None
    assert decision["final_sdist_sha256"] is None
    assert decision["final_wheel_sha256"] is None
    assert decision["final_installed_wheel_cpu_smoke"] == "pending"
    for key in (
        "eligible_for_automatic_merge",
        "eligible_for_default_promotion",
        "tag_authorized",
        "github_prerelease_authorized",
        "pypi_publication_authorized",
        "push_executed",
        "tag_created",
        "github_prerelease_created",
        "pypi_published",
    ):
        assert decision[key] is False
