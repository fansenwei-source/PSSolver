"""Final release and post-release acceptance record for v0.2.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver
from pssolver.functional.api import functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/pssolver_v0_2_0.json"
V021_N1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_v020_has_the_final_public_version():
    assert pssolver.__version__ == "0.2.0"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.2.0",
    }


def test_v020_changelog_records_the_rc6_source_boundary():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## 0.2.0 — 2026-10-19" in changelog
    assert "qualified as\n`v0.2.0rc6`" in changelog
    assert "it does not change the\nsolver implementation" in changelog
    assert "functional\nAPI version 1.0" in changelog
    assert "production defaults" in changelog


def test_v020_documents_nfs_checkpoint_limit_and_patch_policy():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "### Known limitation" in changelog
    assert "cannot publish a\n  checkpoint directly" in changelog
    assert "node-local `/tmp`" in changelog
    assert "operational workaround, not a PSSolver public API" in changelog
    assert "### Maintenance policy" in changelog
    assert "unavailable on a supported target cluster" in changelog
    assert "NFS-compatible no-clobber\n  checkpoint publication in v0.2.1" in changelog


def test_v020_readme_uses_the_final_version():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "PSSolver 0.2.0 is" in readme
    assert "qualified executable combinations in 0.2.0 are" in readme
    assert "0.2.0rc6 is" not in readme


def test_v020_machine_record_limits_the_release_delta():
    record = _record()
    assert record["release"] == "0.2.0"
    assert record["status"] == "released_and_independently_verified"
    assert record["classification"] == "PASS_V0_2_0_RELEASE_EQUALS_RC6_PLUS_VERSION"
    source = record["source"]
    assert source["qualified_release_candidate_commit"] == (
        "fd7323d5e9ef87c537ed07476f8bd50b982af443"
    )
    for key in (
        "solver_implementation_changed",
        "numerical_algorithm_changed",
        "checkpoint_schema_changed",
        "functional_api_version_changed",
        "adapter_version_changed",
        "production_default_changed",
    ):
        assert source[key] is False


def test_v020_publication_record_binds_the_immutable_tag_and_assets():
    record = _record()
    scope = record["record_scope"]
    assert scope == {
        "release_source_commit": "6d2934e73b543f6652d112733657775333e0f396",
        "record_commit_is_release_source": False,
        "release_tag_rewritten": False,
        "release_assets_replaced": False,
        "requires_h100_requalification": False,
        "reason": (
            "This post-tag successor records publication and independent read-only "
            "acceptance evidence; it does not change the released source or artifacts."
        ),
    }
    publication = record["release_publication"]
    assert publication["annotated_tag"] == "v0.2.0"
    assert publication["tag_object"] == "5ef13697b618689b34d3353d3968632ed92086f3"
    assert publication["tag_target"] == scope["release_source_commit"]
    assert publication["github_release_id"] == 417912386
    assert publication["draft"] is False
    assert publication["prerelease"] is False
    assert publication["assets"] == {
        "pssolver-0.2.0-py3-none-any.whl": {
            "asset_id": 657941454,
            "size_bytes": 799309,
            "sha256": "ab4beef1a347568e8ad4162291feb62e94e179849853ca1417583634649a1151",
        },
        "pssolver-0.2.0.tar.gz": {
            "asset_id": 657941456,
            "size_bytes": 1385808,
            "sha256": "3cb3f3a9c2d0d95412ee78db4183872ccedb8e4b51562042215db65feeb01c82",
        },
    }
    assert publication["pypi_published"] is False
    assert publication["h100_rerun_performed"] is False


def test_v020_independent_acceptance_is_bound_without_requalifying_h100():
    acceptance = _record()["independent_post_release_acceptance"]
    assert acceptance["classification"] == "PASS_V0_2_0_RELEASE_EQUALS_RC6_PLUS_VERSION"
    assert acceptance["archive_complete_present"] is True
    assert acceptance["checksum_entry_count"] == 4642
    assert acceptance["checksums_sha256"] == (
        "6f666f4a7e5bf65540eedfd65c42fa77f09ff4fa91ad214e75cade856e7e9099"
    )
    assert acceptance["checksum_verification"] == {"passed": 4642, "failed": 0}
    assert all(acceptance["checks"].values())
    assert acceptance["cpu_suite"] == {
        "passed": 4075,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
        "nematics3d_installed": True,
    }
    assert acceptance["provider_identity_sha256"] == {
        "periodic": "d85f6540494696fddb4549cd4d431d362fe4a55bc53cb9cfd8158fae7c09530d",
        "channel_fixed_iteration": "b95aa0ce22035df610282f1066cce23b47819807f3d942138b0d5038b727a905",
        "channel_tolerance": "b18000feec39cdcdb8e8d371ad80b5037a10ba844c21db0dcddd7b162d324908",
    }
    assert acceptance["control_identity_migration_needed"] is False
    assert acceptance["h100_rerun_needed"] is False


def test_v020_p5_successor_classifies_frozen_and_control_path_impact():
    record = _record()
    n1 = record["n1"]
    assert n1["affects_v0_2_frozen"] is True
    assert n1["affects_control_paths"] is True
    assert n1["exercised_by_current_control_research"] is False

    expected = {
        "C1-05": True,
        "C1-10": True,
        "C1-11": True,
        "C2-17": False,
        "C2-18": True,
    }
    assert set(record["deferred"]) == set(expected)
    for issue, affects_control in expected.items():
        item = record["deferred"][issue]
        assert item["target_release"] == "0.2.x/0.3"
        assert item["affects_v0_2_frozen"] is True
        assert item["affects_control_paths"] is affects_control

    successor = record["freeze_tracking_successor"]
    assert successor["source_record_commit"] == (
        "96f93d016e86604eb25e4bd71183c74f0132cb7d"
    )
    assert successor["historical_record_sha256"] == (
        "3a4ea320f2a53aca397008099ae2b332f6b8af74650e4bd5888db1bcb0f735ec"
    )
    for key in (
        "record_commit_is_release_source",
        "release_tag_rewritten",
        "release_assets_replaced",
        "runtime_changed",
        "requires_h100_requalification",
    ):
        assert successor[key] is False

    control_successor = record["control_path_tracking_successor"]
    assert control_successor["source_record_commit"] == (
        "269cdf1c8f8258311fb63f8031611430345353e9"
    )
    assert control_successor["historical_record_sha256"] == (
        "7a0c85ada69f05504989ec243d49fe67bb6b12e32106ce739c10380ea94b7600"
    )
    for key in (
        "record_commit_is_release_source",
        "release_tag_rewritten",
        "release_assets_replaced",
        "runtime_changed",
        "requires_h100_requalification",
    ):
        assert control_successor[key] is False


def test_v020_successor_record_matches_every_registered_source():
    successors = _record()["historical_successor_sources"]
    n1_successors = json.loads(V021_N1_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    assert set(successors) == {
        "CHANGELOG.md",
        "README.md",
        "pssolver/_version.py",
        "tests/test_release_v0_1.py",
        "tests/test_release_v0_2_rc6.py",
        "tests/test_rc5_g1_checkpoint_glibc_compatibility_record.py",
        "tests/test_rc5_g1_historical_audit_portability.py",
        "tests/test_rc5_legacy_functional_checkpoint_record.py",
        "tests/test_rc5_r510_channel_checkpoint_observation_record.py",
        "tests/test_rc5_r511_metadata_strict_record.py",
        "tests/test_rc5_r512_compiled_simulation_integrity_record.py",
        "tests/test_rc5_r513_optional_hygiene_record.py",
        "tests/test_rc5_r52_finite_workflow_record.py",
        "tests/test_rc5_r53_channel_checkpoint_record.py",
        "tests/test_rc5_r55_tf32_state_restore.py",
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_r57_device_normalization_record.py",
        "tests/test_rc5_r58_checkpoint_integrity_record.py",
        "tests/test_rc5_r59_pressure_postcondition_record.py",
        "tests/test_rc5_w1_documentation.py",
        "tests/test_rc6_1_plane_diagnostic_error_record.py",
        "tests/test_rc6_2_pressure_floor_record.py",
        "tests/test_rc6_3_compile_preflight_record.py",
        "tests/test_rc6_4_channel_failure_record.py",
        "tests/test_rc6_5_checkpoint_control_flow_record.py",
        "tests/test_rc6_6_restart_output_atomicity_record.py",
    }
    for relative, identity in successors.items():
        assert identity["historical_sha256"] != identity["current_sha256"]
        expected = identity["current_sha256"]
        registration = n1_successors.get(relative)
        if registration is not None and registration["historical_sha256"] == expected:
            expected = registration["current_sha256"]
        assert _sha256(ROOT / relative) == expected
        assert identity["reason"]


def test_v020_records_completed_remote_release_actions_and_explicit_exclusions():
    authorization = _record()["authorization"]
    assert authorization["local_preparation_authorized"] is True
    for key in (
        "push_authorized",
        "tag_authorized",
        "github_release_authorized",
        "post_release_registration_authorized",
    ):
        assert authorization[key] is True
    for key in (
        "pypi_publication_authorized",
        "h100_rerun_authorized",
    ):
        assert authorization[key] is False
