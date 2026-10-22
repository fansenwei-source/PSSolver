"""Contract checks for the S-05 design before implementation begins."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "notes/PSSolver_v0_3_0_s05_checkpoint_metadata_integrity_design.json"


def _read() -> dict[str, object]:
    return json.loads(DESIGN.read_text(encoding="utf-8"))


def test_s05_design_is_pending_user_confirmation_and_changes_no_runtime() -> None:
    design = _read()
    assert design["classification"] == (
        "DRAFT_V0_3_S05_CHECKPOINT_METADATA_INTEGRITY_DESIGN"
    )
    assert design["status"] == "pending_user_confirmation"
    assert design["plan_step"] == "S-05"
    assert design["issue_ids"] == ["A2-21/S2-12"]
    assert design["governing_decisions"] == ["D-0010", "D-0011"]
    assert design["selected_answers"] == {}
    scope = design["scope"]
    assert scope["design_only"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key != "design_only"
    )


def test_s05_inventory_accounts_for_every_checkpoint_integrity_pattern() -> None:
    inventory = {item["family"]: item for item in _read()["inventory"]}
    assert set(inventory) == {
        "plane_workflow",
        "legacy_channel_workflow",
        "finite_q_anchoring_workflow",
        "experimental_shadow_checkpoint",
        "periodic_workflow_and_functional_carrier",
        "complete_stress_channel_and_functional_channel",
        "reference_and_modal_block_manifests",
    }
    affected = {
        family
        for family, item in inventory.items()
        if item["progress_covered_by_aggregate_digest"] is False
    }
    assert affected == {
        "plane_workflow",
        "legacy_channel_workflow",
        "finite_q_anchoring_workflow",
        "experimental_shadow_checkpoint",
    }
    assert all(
        inventory[family]["proposed_format"] != "unchanged"
        for family in affected
    )


def test_s05_contract_preserves_legacy_reads_and_does_not_claim_authenticity() -> None:
    design = _read()
    assert design["already_confirmed"] == {
        "introduce_digest_protected_writer_in_v0_3": True,
        "retain_released_v0_2_read_compatibility": True,
        "legacy_sources_are_read_only": True,
        "restart_equivalence_required": True,
        "source": "D-0011 Q-02 and Q-05",
    }
    terminology = design["terminology"]
    assert "not a MAC or signature" in terminology["integrity_not_authenticity"]
    legacy = " ".join(design["proposed_contract"]["legacy_reads"])
    assert "never rewritten in place" in legacy
    assert "unsealed_legacy" in legacy
    identity = " ".join(design["proposed_contract"]["identity_boundary"])
    assert "Provider identity" in identity
    assert "remain unchanged" in identity


def test_s05_tamper_gate_is_pre_payload_and_covers_coordinated_progress() -> None:
    gate = _read()["tamper_gate"]
    mutations = set(gate["required_current_format_mutations"])
    assert {
        "completed_steps_only",
        "integrator_step_count_only",
        "integrator_refresh_count_only",
        "coordinated_logically_consistent_progress_fields",
        "remove_metadata_sha256",
        "malformed_metadata_sha256",
    }.issubset(mutations)
    assert "before payload loading" in gate["required_result"]
    assert "target mutation" in gate["required_result"]
    assert "unsealed fixture remains readable" in gate["legacy_control"]


def test_s05_questions_are_explicit_and_recommend_the_complete_contract() -> None:
    design = _read()
    questions = {item["id"]: item for item in design["decision_questions"]}
    assert set(questions) == {f"S05-Q{number:02d}" for number in range(1, 6)}
    assert design["recommended_answers"] == {
        question_id: "A" for question_id in questions
    }
    for question in questions.values():
        assert question["recommended_option"] == "A"
        assert set(question["options"]) == {"A", "B", "C"}
        assert question["recommendation"].startswith("Choose A")
    assert design["implementation_gate"].startswith("Do not edit checkpoint")


def test_s05_design_records_the_d0008_baseline_environment() -> None:
    validation = _read()["design_validation"]
    assert validation["environment_policy"] == (
        "D-0008 baseline-only hard-link clone"
    )
    assert validation["baseline_venv_source"] == (
        "/home/fansenwei/pssolver_v0_2_1_claude_acceptance_20261020_v1/"
        "a2/venv_wheel"
    )
    assert validation["clone_method"] == "cp -al"
    assert validation["python_executable"].startswith(validation["clone_path"])
    assert validation["sys_prefix"] == validation["clone_path"]
    assert validation["sys_path"][-1] == (
        f"{validation['clone_path']}/lib/python3.12/site-packages"
    )
    assert validation["targeted_tests"] == {
        "passed": 6,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert validation["clone_pruned_after_validation"] is True
