from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "notes/PSSolver_v0_2_0rc4_rc430_lifting_correctness_plan.json"


def _plan() -> dict[str, object]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc430_plan_starts_from_the_rc42_closure_commit_and_evidence():
    baseline = _plan()["baseline"]
    assert baseline["planning_parent_commit"] == (
        "1b15bf856cb6597ae709ee7496cde96d0b2cc7d1"
    )
    assert _sha256(ROOT / baseline["rc4_2_closure_record"]) == baseline[
        "rc4_2_closure_record_sha256"
    ]
    # The plan binds the pre-adjudication matrix. RC4.3.1 legitimately adds
    # its successor status to the live matrix without rewriting this frozen
    # planning record or pretending the old digest still names current bytes.
    assert baseline["known_issues_record_sha256"] == (
        "f23fd7d759ac57f2bc57a449884cb4d6d175232bf749e9d060164a12319f2f38"
    )
    current = json.loads(
        (ROOT / baseline["known_issues_record"]).read_text(encoding="utf-8")
    )
    assert current["progress"]["rc4_3_1_analysis_complete"] is True
    assert _sha256(ROOT / baseline["known_issues_record"]) != baseline[
        "known_issues_record_sha256"
    ]


def test_rc430_plan_handles_exactly_the_three_lifting_residuals():
    issues = _plan()["issues"]
    assert [item["id"] for item in issues] == ["B8", "B20", "N4"]
    assert {item["required_disposition"] for item in issues} == {
        "repair_or_explicit_public_restriction",
        "localized_fresh_conditioning_or_complete_public_documentation",
    }


def test_rc430_adjudication_uses_real_solver_and_independent_references():
    plan = _plan()
    contracts = plan["non_negotiable_contracts"]
    assert contracts["no_metadata_only_fix"] is True
    assert contracts["no_projection_only_oracle"] is True
    assert contracts["end_to_end_solver_must_be_exercised"] is True
    assert contracts["independent_reference_required"] is True
    adjudication = plan["rc4_3_1_adjudication"]
    assert adjudication["kind"] == "analysis_only"
    assert adjudication["source_changes_allowed"] is False
    assert list(adjudication["required_cases"]) == [
        "L01", "L02", "L03", "L04", "L05"
    ]
    assert adjudication["wall_normal_resolutions"] == [16, 32, 64, 128, 256]
    assert set(adjudication["decision_classes"]) == {
        "REPRESENTATION_REPAIR_SUPPORTED",
        "PUBLIC_CAPABILITY_RESTRICTION_REQUIRED",
    }
    assert (
        adjudication["pass_requirements"][
            "minimum_regression_rate_for_claimed_second_order_quantities"
        ]
        == 1.8
    )


def test_rc430_plan_forbids_silent_space_override_and_approximate_restriction():
    plan = _plan()
    contracts = plan["non_negotiable_contracts"]
    assert contracts["requested_effective_space_must_match_or_compile_must_reject"]
    assert contracts["silent_physical_space_override_for_spectral_request_forbidden"]
    restricted = plan["decision_protocol"]["PUBLIC_CAPABILITY_RESTRICTION_REQUIRED"]
    assert any("Compile or lowering" in statement for statement in restricted)
    assert any("No approximate execution" in statement for statement in restricted)


def test_rc430_fresh_conditioning_is_local_explicit_and_not_restart_identity():
    requirements = _plan()["fresh_initial_conditioning_contract"][
        "acceptance_requirements"
    ]
    assert any("canonical metadata" in statement for statement in requirements)
    assert any("byte-identical" in statement for statement in requirements)
    assert any("grid-independent physical semantics" in statement for statement in requirements)
    assert any("never rewrites restored checkpoint state" in statement for statement in requirements)
    assert _plan()["non_negotiable_contracts"][
        "fresh_conditioning_is_not_restart_compatibility_identity"
    ] is True


def test_rc430_stage_order_ends_in_installed_cpu_then_single_h100():
    stages = _plan()["stages"]
    assert [stage["stage"] for stage in stages] == [
        "RC4.3.1",
        "RC4.3.2",
        "RC4.3.3",
        "RC4.3.4",
        "RC4.3.5",
        "RC4.3.6",
        "RC4.3.7",
    ]
    assert stages[-2]["name"] == "installed-wheel CPU closure"
    assert stages[-1]["name"] == "single-H100 lifting closure"


def test_rc430_h100_thresholds_preserve_the_corrected_p845_contract():
    contract = _plan()["h100_contract"]
    assert contract["authorization_required_separately"] is True
    assert contract["single_formal_submission"] is True
    assert contract["automatic_retry"] is False
    assert contract["profile_shapes"] == [[128, 128, 32], [320, 320, 80]]
    assert contract["profile_trials"] == 3
    assert contract["mean_and_median_timestep_ratio_max"] == 1.15
    assert contract["peak_allocated_ratio_max"] == 1.1
    assert contract["peak_active_ratio_max"] == 1.1
    assert contract["peak_reserved_role"] == "diagnostic_only"
    assert contract["same_version_restart_byte_identity_required"] is True


def test_rc430_reviewed_sources_are_content_addressed():
    superseded_by_rc434 = {
        "pssolver/configuration/lifting.py",
        "pssolver/operators/lifting.py",
        "pssolver/planning/lifting.py",
        "pssolver/runtime/static_lifting.py",
        "tests/test_phase8_p842_plane_static_lifting.py",
        "tests/test_phase8_p844_lifting_workflow_restart.py",
    }
    for relative, expected in _plan()["reviewed_source_sha256"].items():
        if relative in superseded_by_rc434:
            continue
        assert _sha256(ROOT / relative) == expected, relative


def test_rc430_scope_keeps_neumann_control_and_defaults_outside_the_batch():
    scope = _plan()["scope"]
    assert scope["field_neutral_lifting_core_preserved"] is True
    assert scope["q_specific_policy_remains_in_composition_layer"] is True
    for key in (
        "nonhomogeneous_neumann_added",
        "finite_robin_anchoring_modified",
        "new_geometry_or_model_added",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "production_default_changed",
    ):
        assert scope[key] is False


def test_rc430_only_completes_planning():
    assert _plan()["authorization"] == {
        "rc4_3_planning_authorized": True,
        "rc4_3_planning_complete": True,
        "rc4_3_1_analysis_authorized": False,
        "rc4_3_implementation_authorized": False,
        "h100_submission_authorized": False,
        "PSSolver_Control_change_authorized": False,
        "nematics3d_change_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "eligible_for_release": False,
    }
