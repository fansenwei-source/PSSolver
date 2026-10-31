"""Bindings for the design-only S-17 R3/R6 proposal."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_s17_r3_r6_design_proposal.json"


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_s17_r3_keeps_the_v04_stable_control_boundary():
    record = _read()
    r3 = record["r3"]
    boundary = r3["frozen_public_boundary"]
    assert boundary == {
        "module": "pssolver.functional.api",
        "api_version": "1.1",
        "must_remain_source_compatible_during_v0_4": True,
        "broad_compatibility_facade": "pssolver.functional",
        "internal_layers": [
            "pssolver.functional.contracts",
            "pssolver.functional.errors",
            "pssolver.functional.validation",
            "pssolver.functional.versioning",
        ],
    }
    assert r3["options"]["A"]["recommended"] is True
    assert r3["options"]["A"]["functional_api_version_effect"] == (
        "none during v0.4"
    )


def test_s17_r3_keeps_research_outside_the_control_main_migration():
    control = _read()["r3"]["control_coordination"]
    assert control["qualified_control_main_source"] == (
        "41d1668a7310cc022343ac5917c76234724eeae2"
    )
    assert control["provider_release_source"] == (
        "47473880534a66f6da940db82122c79cda48a86b"
    )
    assert control["imports_only"] == "pssolver.functional.api"
    research = control["research_branch"]
    assert research["branch"] == "research/3d-warmup"
    assert research["provider_generation"] == "v0.2.0"
    assert research["included_in_current_migration"] is False


def test_s17_r6_recommends_a_combined_audit_without_reopening_lifting():
    r6 = _read()["r6"]
    assert r6["options"]["A"]["recommended"] is True
    assert r6["current_boundaries"] == {
        "robin_public_application_qualified": False,
        "finite_q_public_application_qualified": False,
        "lifting_public_execution_open": False,
        "lifting_deferred_issue": "B8/S2-07",
        "released_checkpoint_read_compatibility_must_be_preserved": True,
    }
    assert "B8/S2-07" in " ".join(r6["options"]["A"]["design"])


def test_s17_proposal_is_design_only_and_requires_four_user_choices():
    record = _read()
    assert record["classification"] == "DESIGN_ONLY_AWAITING_USER_DECISION"
    assert [item["id"] for item in record["questions_for_user"]] == [
        "Q-S17-R3-01",
        "Q-S17-R3-02",
        "Q-S17-R6-01",
        "Q-S17-R6-02",
    ]
    assert all(
        item["recommended_option"] == "A"
        for item in record["questions_for_user"]
    )
    assert record["implementation"] == {
        "started": False,
        "runtime_files_changed": [],
        "public_interfaces_changed": [],
        "numerical_output_changed": False,
        "checkpoint_schema_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
    }
    assert all(value is False for value in record["authorization"].values())
