"""Forward-only release-integration review for PSSolver 0.2.0rc6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pssolver.functional.api import functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_release_integration_review.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc6_integration_review_binds_the_qualified_runtime_and_closure():
    record = _record()
    assert record["classification"] == (
        "PASS_RC6_RELEASE_INTEGRATION_REVIEW_READY_FOR_FINAL_PACKAGE_PREPARATION"
    )
    source = record["source_identity"]
    assert source["immutable_baseline_tag"] == "v0.2.0rc5"
    assert source["immutable_baseline_commit"] == (
        "e2e9d05bfac858d8e1ae1653ac86a2187c9917ef"
    )
    assert source["qualified_runtime_candidate_commit"] == (
        "ea801ec08f5b2465661334e1c126133e55b68e96"
    )
    assert source["qualification_support_commit"] == (
        "3228d3d723393818ee6149e9cf56021ffaaed411"
    )
    assert source["cumulative_closure_record_commit"] == (
        "5fbf556c142a29896fe41d7cd991dc65a8149e1e"
    )
    assert source["reviewed_parent_commit"] == source["cumulative_closure_record_commit"]
    assert source["record_commit_is_reviewed_source"] is False
    assert source["runtime_candidate_is_ancestor_of_reviewed_parent"] is True
    assert source["qualification_support_is_ancestor_of_reviewed_parent"] is True


def test_rc6_integration_review_preserves_the_frozen_release_records():
    frozen = _record()["frozen_release_metadata"]
    for item in ("machine_record", "release_notes", "source_freeze"):
        assert _sha256(frozen[item]["path"]) == frozen[item]["sha256"]
    assert frozen["historical_source_freeze_record_rewritten"] is False
    assert frozen["historical_pending_fields_preserved"] is True


def test_rc6_integration_review_binds_the_cumulative_h100_evidence():
    closure = _record()["cumulative_h100_closure"]
    assert _sha256(closure["path"]) == closure["sha256"]
    persisted = _json(closure["path"])
    assert persisted["classification"] == closure["classification"]
    assert persisted["authorization"]["qualification_complete"] is True
    assert persisted["authorization"]["rc6_complete"] is True
    assert persisted["slurm"]["job_id"] == closure["job_id"]
    gates = persisted["cumulative_gates"]
    assert (gates["completed_stages"], gates["planned_stages"]) == (11, 11)
    assert gates["profile_count"] == closure["profile_count"] == 192
    assert gates["profile_child_tf32_bootstrap_passed"] == 192
    assert gates["performance_memory_groups_passed"] == 16
    assert (gates["consumer_passed"], gates["consumer_skipped"], gates["consumer_failed"]) == (
        344,
        1,
        0,
    )
    control = persisted["archive_identities"]["control"]
    formal = persisted["archive_identities"]["formal"]
    assert (control["checksum_passed"], control["checksum_entries"]) == (24, 24)
    assert control["checksums_sha256"] == closure["control_manifest_sha256"]
    assert (formal["checksum_passed"], formal["checksum_entries"]) == (4763, 4763)
    assert formal["checksums_sha256"] == closure["formal_manifest_sha256"]


def test_rc6_integration_review_records_only_qualification_scope_after_runtime():
    scope = _record()["post_runtime_candidate_scope"]
    assert scope["changed_file_count"] == 77
    assert scope["top_level_counts"] == {
        "benchmarks": 21,
        "jobs": 4,
        "notes": 30,
        "tests": 22,
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


def test_rc6_integration_review_keeps_the_public_contract_stable():
    contract = _record()["public_contract"]
    machine = _json("notes/pssolver_v0_2_0_rc6.json")["public_contract"]
    for key in (
        "package_version",
        "functional_api_version",
        "models",
        "geometries",
        "boundary_policies",
        "qualified_combinations",
        "runtime_paths",
        "functional_batch_sizes",
        "production_default_changed",
        "compiled_runtime_promoted",
        "lifting_reopened",
    ):
        assert contract[key] == machine[key]
    assert functional_protocol_provenance()["functional_protocol"]["current"] == (
        contract["functional_api_version"]
    )
    assert contract["functional_batch_sizes"] == [1]
    for key in (
        "production_default_changed",
        "compiled_runtime_promoted",
        "checkpoint_schema_changed",
        "lifting_reopened",
        "PSSolver_Control_bundled",
        "nematics3d_modified",
    ):
        assert contract[key] is False


def test_rc6_integration_review_allows_preparation_but_not_release_actions():
    decision = _record()["integration_decision"]
    assert decision["release_integration_review_complete"] is True
    assert decision["eligible_for_branch_push"] is True
    assert decision["eligible_for_final_release_package_preparation"] is True
    assert decision["independent_post_release_verification_required"] is True
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
        "closure_record_push_executed",
        "integration_review_push_executed",
        "tag_created",
        "github_prerelease_created",
        "pypi_published",
    ):
        assert decision[key] is False
