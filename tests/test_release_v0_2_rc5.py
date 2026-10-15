"""Source-freeze contracts for PSSolver 0.2.0rc5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver
from pssolver import capability_catalog
from pssolver.functional.api import functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_0_rc5.json"
G1_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc5_version_and_public_provenance_agree():
    assert pssolver.__version__ == "0.2.0rc5"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.2.0rc5",
    }


def test_rc5_documents_and_machine_record_agree():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "notes/pssolver_v0_2_0_rc5.md").read_text(
        encoding="utf-8"
    )
    record = _json(RELEASE_RECORD)

    assert "PSSolver 0.2.0rc5" in readme
    assert "## 0.2.0rc5" in changelog
    assert "PSSolver 0.2.0rc5 release candidate" in release_notes
    assert record["release"] == pssolver.__version__
    assert record["status"] == "source_freeze_preparation"
    assert record["qualification"]["g1_complete"] is False
    assert record["qualification"]["single_h100_cumulative"] == (
        "not_authorized"
    )


def test_rc5_capability_counts_match_live_catalog():
    expected = _json(RELEASE_RECORD)["public_contract"]
    catalog = capability_catalog()

    assert len(catalog.models) == expected["models"]
    assert len(catalog.geometries) == expected["geometries"]
    assert len(catalog.boundary_policies) == expected["boundary_policies"]
    assert len(catalog.qualified_combinations) == expected["qualified_combinations"]
    assert sum(
        len(combination.runtime_paths)
        for combination in catalog.qualified_combinations
    ) == expected["runtime_paths"]


def test_rc5_work_evidence_is_complete_and_content_addressed():
    record = _json(RELEASE_RECORD)
    assert set(record["completed_work"]) == {
        "W0",
        "W1",
        *(f"R5-{index}" for index in range(1, 14)),
    }
    for relative, expected in record["work_evidence"].items():
        assert _sha256(ROOT / relative) == expected


def test_rc5_freeze_keeps_public_compatibility_boundaries_stable():
    record = _json(RELEASE_RECORD)
    contract = record["public_contract"]
    assert contract["functional_api_version"] == "1.0"
    assert contract["functional_batch_sizes"] == [1]
    assert contract["production_default_changed"] is False
    assert contract["compiled_runtime_promoted"] is False
    assert contract["checkpoint_schema_changed_by_freeze"] is False

    boundary = record["release_boundary"]
    for key in (
        "branch_pushed",
        "tag_created",
        "github_release_created",
        "pypi_published",
        "production_default_changed",
        "larger_batch_qualified",
        "lifting_reopened",
        "optimizer_campaign_authorized",
        "scientific_control_result",
        "pssolver_control_modified",
        "nematics3d_modified",
    ):
        assert boundary[key] is False


def test_rc5_g1_plan_is_fail_closed_and_requires_one_authorized_h100_run():
    record = _json(G1_RECORD)
    assert record["phase"] == "G-1.0"
    assert record["classification"] == (
        "READY_FOR_FROZEN_SOURCE_CPU_QUALIFICATION"
    )
    assert record["authorization"] == {
        "source_freeze_prepared": True,
        "g1_complete": False,
        "eligible_for_push": False,
        "eligible_for_h100_submission": False,
        "h100_submission_count": 0,
        "automatic_retry": False,
    }
    assert record["gate_order"] == [
        "frozen_source_full_cpu",
        "frozen_installed_wheel_cpu",
        "released_checkpoint_matrix_rc1_through_rc4",
        "provider_identity_impact_review",
        "pssolver_control_installed_wheel_qualification_if_required",
        "single_h100_cumulative_qualification",
        "post_release_verification",
    ]


def test_rc5_release_documents_are_in_source_manifest():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for relative in (
        "notes/pssolver_v0_2_0_rc5.md",
        "notes/pssolver_v0_2_0_rc5.json",
    ):
        assert (ROOT / relative).is_file()
        assert f"include {relative}" in manifest
