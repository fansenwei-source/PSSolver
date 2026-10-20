"""Registration contract for the v0.2.0rc6 second-party acceptance."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_second_party_acceptance.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_rc6_second_party_acceptance_binds_release_identity():
    record = _record()
    assert record["classification"] == (
        "PASS_RC6_INDEPENDENT_SECOND_PARTY_POST_RELEASE_ACCEPTANCE"
    )
    assert record["acceptance_complete"] is True
    source = record["qualified_source"]
    assert source == {
        "package_source_commit": "fd7323d5e9ef87c537ed07476f8bd50b982af443",
        "record_commit_is_package_source": False,
        "requires_h100_requalification": False,
    }
    release = record["release_identity"]
    assert release["release_id"] == 417781605
    assert release["tag"] == "v0.2.0rc6"
    assert release["tag_object_sha"] == (
        "533e3fb4254319d1a6a9a9a0e463447339c18b36"
    )
    assert release["tag_target_commit"] == source["package_source_commit"]
    assert release["draft"] is False
    assert release["prerelease"] is True


def test_rc6_second_party_acceptance_binds_external_archive():
    evidence = _record()["external_evidence"]
    assert evidence["control_directory"] == (
        "/home/fansenwei/pssolver_v0_2_0rc6_claude_acceptance_20261019_v1"
    )
    assert evidence["manifest_entries"] == 34198
    assert evidence["manifest_passed"] == 34198
    assert evidence["manifest_sha256"] == (
        "8e3f190aca37113d54414e833ed7e3695b7703040e2a4662e9edde133353ceaa"
    )
    assert evidence["complete_present"] is True
    assert evidence["sealed_before_codex_records_were_read"] is True


def test_rc6_second_party_acceptance_stages_and_cpu_suite_pass():
    record = _record()
    assert set(record["acceptance_stages"].values()) == {"PASS"}
    assert record["source_cpu_suite"] == {
        "passed": 4068,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
    }
    assert record["closeout"]["release_blockers"] == []
    assert record["closeout"]["regressions"] == []


def test_rc6_second_party_acceptance_preserves_h100_evidence():
    h100 = _record()["h100_evidence"]
    assert h100["independently_rehashed"] is True
    assert h100["release_smoke_job_id"] == 10901816
    assert h100["release_smoke_state"] == "COMPLETED"
    assert h100["release_smoke_exit_code"] == "0:0"
    assert h100["control_manifest"] == "716/716 PASS"
    assert h100["scratch_manifest"] == "14/14 PASS"


def test_rc6_second_party_acceptance_records_the_v020_decision():
    decision = _record()["release_decision"]
    assert decision["keep_rc6_prerelease_unchanged"] is True
    assert decision["open_rc7"] is False
    assert decision["v0_2_0_source_is_rc6_plus_version_and_changelog_only"] is True
    assert decision["n1_target_release"] == "v0.2.1"
    assert decision["n1_blocks_v0_2_0"] is False


def test_rc6_second_party_acceptance_does_not_overclaim_scope():
    scope = _record()["scope"]
    for key, value in scope.items():
        assert value is False, key
