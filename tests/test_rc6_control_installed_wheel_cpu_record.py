"""Evidence contract for the RC6 PSSolver-Control installed-wheel CPU closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_control_installed_wheel_cpu.json"
IDENTITY = ROOT / "notes/PSSolver_v0_2_0rc6_provider_identity_impact.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc6_control_cpu_record_binds_the_qualified_sources():
    record = _record()
    assert record["classification"] == (
        "PASS_RC6_PSSOLVER_CONTROL_GUARD_ORDER_INSTALLED_WHEEL_CPU"
    )
    assert record["qualified_provider_source"]["commit"] == (
        "ea801ec08f5b2465661334e1c126133e55b68e96"
    )
    assert record["qualified_consumer_source"]["commit"] == (
        "208edd03e6b26a0e788c07d295fddd7a67237189"
    )
    assert record["qualified_consumer_source"]["runtime_implementation_commit"] == (
        "6d3558b1367da9a3e7d6b3f8fbc5584cf384bfe3"
    )


def test_rc6_control_cpu_record_succeeds_the_identity_review_without_rewriting_it():
    record = _record()
    assert _sha256(IDENTITY) == record["prerequisites"][
        "provider_identity_record_sha256"
    ]
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    assert identity["consumer_impact"][
        "installed_wheel_cpu_against_frozen_rc6_provider_complete"
    ] is False
    assert record["prerequisites"]["provider_identity_migration_required"] is False
    consumer_record = record["consumer_evidence_record"]
    assert consumer_record["record_commit"] == (
        "d30e3d0b43767397ecc13ef3dd2b1eb05e5e74cb"
    )
    assert consumer_record["sha256"] == (
        "3d591b04ab5dc23212bdd9dc8172d1e534dba8f6e17d666aaf8512b99befe98a"
    )
    assert consumer_record["record_commit_is_qualified_source"] is False
    assert consumer_record["requires_cpu_requalification"] is False
    assert consumer_record["pushed"] is False


def test_rc6_control_cpu_matrix_is_source_installed_equivalent():
    matrix = _record()["cpu_matrix"]
    assert matrix["source_targeted"] == matrix["installed_targeted"] == {
        "passed": 87,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert matrix["source_complete"] == matrix["installed_complete"] == {
        "passed": 339,
        "failed": 0,
        "skipped": 6,
        "xfailed": 0,
    }
    assert matrix["source_and_installed_counts_equal"] is True
    assert matrix["source_and_installed_skip_identities_equal"] is True
    assert matrix["installed_targeted_import_audit_passed"] is True
    assert matrix["installed_complete_import_audit_passed"] is True
    assert matrix["pip_check"] == "PASS"
    assert matrix["source_shadow_import"] is False


def test_rc6_control_cpu_analyzer_is_bound_and_fail_closed():
    analyzer = _record()["analyzer_contract"]
    assert _sha256(ROOT / analyzer["path"]) == analyzer["sha256"]
    assert analyzer["expected_classification"] == (
        "PASS_RC6_PSSOLVER_CONTROL_GUARD_ORDER_INSTALLED_WHEEL_CPU"
    )
    for key in (
        "exact_junit_counts",
        "exact_skip_identities",
        "fail_closed_import_audit",
        "artifact_hash_binding",
        "atomic_output",
        "refuse_overwrite",
    ):
        assert analyzer[key] is True


def test_rc6_control_cpu_external_archive_is_bound():
    evidence = _record()["evidence"]
    assert evidence["analysis_sha256"] == (
        "a5de341eebfc52e69ce27db5d75b450e1378e2e644c2a2e4b60383bd12938228"
    )
    assert evidence["checksums_entries"] == 17
    assert evidence["checksums_passed"] == 17
    assert evidence["checksums_failed"] == 0
    assert evidence["checksums_sha256"] == (
        "2e3d6e058702e193ef6a3ab8876f4420efd5dc58719f9bfb06b02785f186710c"
    )


def test_rc6_control_cpu_record_leaves_only_h100_and_independent_review_pending():
    record = _record()
    assert record["pending_rc6_gates"] == [
        "single_h100_cumulative_qualification",
        "independent_post_release_verification",
    ]
    authorization = record["authorization"]
    assert authorization["provider_identity_impact_review_complete"] is True
    assert authorization["provider_identity_migration_required"] is False
    assert authorization["pssolver_control_installed_wheel_cpu_complete"] is True
    assert authorization["eligible_for_rc6_cumulative_h100_planning"] is True
    assert authorization["rc6_cumulative_h100_qualification_complete"] is False
    assert authorization["eligible_for_push"] is False
    assert authorization["eligible_for_h100_submission"] is False
    assert authorization["h100_submission_count"] == 0
    for key in (
        "automatic_retry",
        "automatic_merge",
        "tag_created",
        "release_created",
        "pypi_published",
        "production_default_changed",
        "PSSolver_modified_by_execution",
        "PSSolver_Control_runtime_modified_by_execution",
        "nematics3d_modified",
    ):
        assert authorization[key] is False
