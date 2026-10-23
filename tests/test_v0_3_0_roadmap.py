"""Bindings for the confirmed S-02 v0.3.0 completion-contract candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
ROADMAP = ROOT / "notes/PSSolver_v0_3_0_roadmap.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s02_draft_binds_the_registry_and_confirmed_execution_order() -> None:
    draft = _read(ROADMAP)
    registry = _read(REGISTRY)
    assert draft["classification"] == (
        "CONFIRMED_V0_3_0_GOALS_AND_COMPLETION_CONTRACT_CANDIDATE"
    )
    assert draft["status"] == "confirmed_pending_independent_review"
    assert draft["plan_step"] == "S-02"
    assert draft["governing_decision"] == "D-0010"
    assert draft["confirmed_decision"] == "D-0011"
    assert draft["execution_policy"]["ordered_steps"] == [
        "S-03",
        "S-04",
        "S-05",
        "S-06",
        "S-07",
        "S-08",
        "S-09",
        "S-10",
    ]
    scope = draft["known_issue_scope"]
    assert scope["registry_sha256"] == (
        "18a769b757a9c4325d69e8a9e86751c19642e66804141cda678dde408fed9eb3"
    )
    assert _sha256(REGISTRY) == resolve_s03_successor(
        "notes/pssolver_v0_2_x_open_issues.json", scope["registry_sha256"]
    )
    assert scope["registry_issue_count"] == 30
    assert registry["issue_count"] == 39
    assert "V-25" in scope["required_by_step"]["S-04"]


def test_s02_accounts_for_every_registry_issue_exactly_once() -> None:
    draft = _read(ROADMAP)
    registry_ids = {item["id"] for item in _read(REGISTRY)["issues"]}
    scope = draft["known_issue_scope"]
    planned_sequence = [
        issue_id
        for issue_ids in scope["required_by_step"].values()
        for issue_id in issue_ids
    ]
    planned = {
        issue_id for issue_id in planned_sequence
    }
    accounted = (
        planned
        | set(scope["already_verified"])
        | set(scope["user_decision_at_s10"])
    )
    assert accounted | {
        "V-26",
        "V-27",
        "V-28",
        "V-29",
        "V-30",
        "V-31",
        "V-32",
        "V-33",
        "V-34",
    } == registry_ids
    assert len(planned_sequence) == len(planned)
    assert set(scope["already_verified"]).isdisjoint(planned)
    assert set(scope["user_decision_at_s10"]).isdisjoint(planned)


def test_s02_completion_contract_covers_all_required_domains() -> None:
    draft = _read(ROADMAP)
    gates = {gate["id"]: gate for gate in draft["completion_gates"]}
    assert set(gates) == {
        "G-01",
        "G-02",
        "G-03",
        "G-04",
        "G-05",
        "G-06",
        "G-07",
        "G-08",
        "G-09",
    }
    combined = " ".join(
        f"{gate['title']} {gate['criterion']}" for gate in gates.values()
    )
    for required in (
        "Registry",
        "checkpoint",
        "Numerical",
        "installed-wheel",
        "Performance",
        "Control",
        "GPU/H100",
        "release candidate",
    ):
        assert required.lower() in combined.lower()


def test_s02_questions_have_recommended_mutually_named_options() -> None:
    draft = _read(ROADMAP)
    questions = {question["id"]: question for question in draft["decision_questions"]}
    assert set(questions) == {f"Q-{number:02d}" for number in range(1, 8)}
    assert draft["recommended_answers"] == {
        question_id: "A" for question_id in questions
    }
    assert draft["selected_answers"] == draft["recommended_answers"]
    assert "all selected A" in draft["user_confirmation"]
    for question in questions.values():
        assert question["recommended_option"] == "A"
        assert set(question["options"]) == {"A", "B", "C"}
        assert question["question"]
        assert question["recommendation"].startswith("Choose A.")


def test_s02_requires_backward_compatibility_and_control_independence() -> None:
    draft = _read(ROADMAP)
    compatibility = draft["compatibility_contract"]
    assert compatibility["read_released_v0_2_checkpoints"] is True
    assert compatibility["legacy_checkpoint_reads_must_not_rewrite_sources"] is True
    assert compatibility["restart_equivalence_required"] is True
    assert compatibility["provider_identity_changes_require_explicit_migration"] is True
    control = draft["control_sync_contract"]
    assert control["release_gate_recommended"] is True
    assert control["scientific_control_result_required"] is False
    assert control["optimizer_campaign_required"] is False


def test_s02_records_the_d0008_baseline_clone_and_sys_path() -> None:
    environment = _read(ROADMAP)["draft_validation_environment"]
    assert environment["policy"] == "D-0008 baseline-only hard-link clone"
    assert environment["baseline_venv_source"] == (
        "/home/fansenwei/pssolver_v0_2_1_claude_acceptance_20261020_v1/"
        "a2/venv_wheel"
    )
    assert environment["clone_method"] == "cp -al"
    assert environment["sys_prefix"] == environment["clone_path"]
    assert environment["python_executable"].startswith(environment["clone_path"])
    assert environment["sys_path"][-1] == (
        f"{environment['clone_path']}/lib/python3.12/site-packages"
    )
    assert environment["clone_pruned_after_validation"] is True


def test_s02_draft_is_planning_only_and_has_no_external_authority() -> None:
    draft = _read(ROADMAP)
    assert draft["scope"]["planning_only"] is True
    assert all(
        value is False
        for key, value in draft["scope"].items()
        if key != "planning_only"
    )
    assert draft["next_action"] == (
        "Create R-0007 with plan_step S-02 for Claude consistency review before "
        "any push."
    )
