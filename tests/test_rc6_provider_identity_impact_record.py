"""Evidence contract for the RC6 functional Provider-identity review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_provider_identity_impact.json"
CHECKPOINT = ROOT / "notes/PSSolver_v0_2_0rc6_checkpoint_cpu.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc6_identity_review_binds_the_frozen_installed_wheel():
    record = _record()
    assert record["phase"] == "RC6-11"
    assert record["classification"] == (
        "PASS_RC6_PROVIDER_IDENTITY_UNCHANGED_NO_CONTROL_IDENTITY_MIGRATION_REQUIRED"
    )
    assert record["qualified_source"]["commit"] == (
        "ea801ec08f5b2465661334e1c126133e55b68e96"
    )
    assert record["qualified_source"]["record_commit_is_qualified_source"] is False
    assert _sha256(CHECKPOINT) == record["prerequisite"][
        "checkpoint_record_sha256"
    ]


def test_rc6_identity_review_uses_the_final_rc5_consumer_fixture():
    fixture = _record()["consumer_fixture"]
    assert fixture["qualified_source_commit"] == (
        "c14e6b56c88081e7f9a5642e65ffd722eec7445b"
    )
    assert fixture["archive_sha256"] == (
        "3b66240d6de16956c8958268e75b79e3e250efca61f21ff51ec44609ab93728e"
    )
    assert fixture["historical_identity_evidence_sha256"] == (
        "1198d6b1b870d40a69d4eb379ef83cb2247c718a6fd08ad7fec34822664355ac"
    )


def test_rc6_identity_matrix_is_exactly_unchanged():
    matrix = _record()["identity_matrix"]
    assert matrix["fixture_count"] == 3
    assert matrix["changed_fixture_count"] == 0
    assert matrix["changed_path_count"] == 0
    assert matrix["full_metadata_equal"] is True
    assert matrix["canonical_sha256_equal"] is True
    assert matrix["path_independent"] is True
    assert matrix["historical_rc5_evidence_reproduced_exactly"] is True
    expected = {
        "periodic_canonical": (
            "d85f6540494696fddb4549cd4d431d362fe4a55bc53cb9cfd8158fae7c09530d"
        ),
        "channel_canonical_fixed_iteration": (
            "b95aa0ce22035df610282f1066cce23b47819807f3d942138b0d5038b727a905"
        ),
        "channel_tolerance": (
            "b18000feec39cdcdb8e8d371ad80b5037a10ba844c21db0dcddd7b162d324908"
        ),
    }
    for name, digest in expected.items():
        fixture = matrix["fixtures"][name]
        assert fixture["rc5_canonical_sha256"] == digest
        assert fixture["rc6_canonical_sha256"] == digest
        assert fixture["changed_paths"] == []


def test_rc6_identity_review_does_not_require_a_control_identity_migration():
    record = _record()
    version = record["version_contract"]
    assert version == {
        "functional_api_version": "1.0",
        "functional_api_version_changed_in_rc6": False,
        "provider_identity_changed_in_rc6": False,
        "production_default_changed": False,
    }
    impact = record["consumer_impact"]
    assert impact["provider_identity_migration_required"] is False
    assert impact["identity_pin_update_required"] is False


def test_rc6_identity_review_keeps_guard_order_work_separate():
    impact = _record()["consumer_impact"]
    assert impact["guard_order_work_is_separate"] is True
    assert impact["guard_order_evidence_commit"] == (
        "208edd03e6b26a0e788c07d295fddd7a67237189"
    )
    assert impact["guard_order_cpu_closure_complete"] is True
    assert impact["installed_wheel_cpu_against_frozen_rc6_provider_complete"] is False
    assert impact["PSSolver_Control_modified_by_this_review"] is False


def test_rc6_identity_runner_and_external_evidence_are_bound():
    record = _record()
    runner = record["runner_contract"]
    assert _sha256(ROOT / runner["path"]) == runner["sha256"]
    assert runner["python_isolated_mode"] is True
    assert runner["installed_package_imported_before_fixture"] is True
    assert runner["source_shadow_import"] is False
    assert runner["two_output_roots_per_fixture"] is True
    evidence = record["evidence"]
    assert evidence["checksum_entries"] == 19
    assert evidence["checksum_passed"] == 19
    assert evidence["checksum_failed"] == 0
    assert evidence["checksums_sha256"] == (
        "59340714ebeda6cf9ba90503ebee305d79335aebbd653bc18eba1c9f845071a0"
    )


def test_rc6_identity_review_remains_fail_closed_for_later_gates():
    record = _record()
    assert record["pending_rc6_gates"] == [
        "pssolver_control_guard_order_installed_wheel_cpu_with_frozen_rc6_provider",
        "single_h100_cumulative_qualification",
        "independent_post_release_verification",
    ]
    authorization = record["authorization"]
    assert authorization["provider_identity_impact_review_complete"] is True
    assert authorization["provider_identity_migration_required"] is False
    assert authorization["eligible_for_pssolver_control_installed_wheel_cpu_review"] is True
    assert authorization["pssolver_control_installed_wheel_cpu_complete"] is False
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
