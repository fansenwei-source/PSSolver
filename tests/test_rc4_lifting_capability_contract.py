"""Static RC4.3.2 public lifting capability and identity contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc432_lifting_capability_contract.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc432_binds_the_analysis_and_selects_fail_closed_restriction():
    record = _record()
    baseline = record["baseline"]

    assert record["phase"] == "RC4.3.2"
    assert record["classification"] == (
        "PASS_RC4_3_2_PUBLIC_LIFTING_CAPABILITY_RESTRICTION_CONTRACT"
    )
    assert baseline["contract_parent_commit"] == (
        "7dbdd395b7665782c17d8cf2cb1362a693f06882"
    )
    assert _sha256(ROOT / baseline["rc4_3_plan"]) == baseline[
        "rc4_3_plan_sha256"
    ]
    assert _sha256(ROOT / baseline["rc4_3_1_analysis"]) == baseline[
        "rc4_3_1_analysis_sha256"
    ]
    decision = record["decision"]
    assert decision["mode"] == "public_fail_closed_restriction"
    assert decision["representation_repair_claimed"] is False
    assert decision["approximate_public_execution_allowed"] is False
    assert decision["warning_only_execution_allowed"] is False
    assert decision["silent_runtime_fallback_allowed"] is False


def test_rc432_restricts_only_the_precise_lifted_plane_capability():
    record = _record()
    unsupported = record["unsupported_public_capability"]
    supported = record["supported_public_subset"]

    assert unsupported == {
        "equation_variant": "complete_stress_beris_edwards",
        "geometry_name": "plane_slab",
        "field_name": "Q",
        "components": ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"],
        "boundary_semantics": (
            "static_constant_two_sided_prescribed_dirichlet"
        ),
        "runtime_paths": ["legacy_production", "compiled_v2"],
        "molecular_field_linear_space_requests": ["physical", "spectral"],
        "scope_rule": unsupported["scope_rule"],
        "publicly_qualified": False,
        "internal_primitive_removed": False,
        "historical_runtime_claimed_qualified": False,
    }
    assert "Malformed, partial" in unsupported["scope_rule"]
    assert supported["plane_complete_stress"]["Q_boundary"] == (
        "homogeneous_neumann"
    )
    assert supported["plane_complete_stress"]["remains_qualified"] is True
    assert supported["periodic_complete_stress_unchanged"] is True
    assert supported["channel_complete_stress_unchanged"] is True
    assert supported["legacy_channel_unchanged"] is True
    assert supported["nonhomogeneous_Neumann_added"] is False


def test_rc432_freezes_public_error_and_preconstruction_order():
    rejection = _record()["public_rejection_contract"]

    assert rejection["exception_type"].endswith(
        ".PublicSimulationCompilationError"
    )
    assert rejection["rejection_enum_member"] == (
        "UNSUPPORTED_LIFTED_WALL_MOLECULAR_FIELD"
    )
    assert rejection["rejection_code"] == (
        "unsupported_lifted_wall_molecular_field"
    )
    assert "nonzero wall molecular-field semantics" in rejection[
        "message_token"
    ]
    assert rejection["same_error_for_physical_and_spectral_requests"] is True
    assert rejection["rejection_before_runtime_construction"] is True
    assert rejection["rejection_before_output_creation"] is True
    assert rejection["enforcement_order"][-1].startswith("do not create")
    assert set(rejection["required_context_keys"]) == {
        "equation_variant",
        "geometry_name",
        "field_name",
        "boundary_semantics",
        "requested_runtime_path",
        "requested_molecular_field_linear_space",
        "supported_Q_boundary",
        "analysis_record_sha256",
    }


def test_rc432_catalog_keeps_declarations_but_removes_execution_claim():
    catalog = _record()["capability_catalog_contract"]

    assert catalog["catalog_schema_version"] == 1
    assert catalog["catalog_shape_changes"] is False
    assert catalog["declaration_does_not_imply_execution"] is True
    for name in (
        "prescribed_q",
        "strong_homeotropic_q",
        "strong_planar_q",
    ):
        declaration = catalog["boundary_declarations"][name]
        assert declaration == {
            "declarable": True,
            "executable": False,
            "qualified_applications": [],
        }
    assert catalog["boundary_declarations"]["neumann_q"][
        "plane_complete_stress_remains_qualified"
    ] is True


def test_rc432_forbids_silent_molecular_field_space_override():
    spaces = _record()["molecular_field_space_contract"]

    assert spaces["successful_public_compile_requires_requested_equals_effective"]
    assert spaces["molecular_field_space_fallback_used"] is False
    assert spaces["lifted_physical_request"] == "reject"
    assert spaces["lifted_spectral_request"] == "reject"
    assert spaces["nonlifted_physical_request"] == (
        "execute_physical_and_report_physical"
    )
    assert spaces["nonlifted_spectral_request"] == (
        "execute_spectral_and_report_spectral"
    )
    assert spaces["silent_physical_override_for_lifted_spectral_request"] is False


def test_rc432_fresh_conditioning_is_explicit_local_and_fresh_only():
    policy = _record()["fresh_initial_conditioning_contract"]

    assert policy["public_default"]["kind"] == (
        "strict_no_automatic_conditioning_v1"
    )
    assert policy["public_default"]["mutates_requested_initial_field"] is False
    localized = policy["qualified_opt_in_candidate"]
    assert localized["kind"] == "localized_quintic_wall_transition_v1"
    assert "6*x^5 - 15*x^4 + 10*x^3" in localized["formula"]
    assert localized["transition_width_units"] == (
        "same_physical_length_units_as_geometry"
    )
    assert "byte-identical" in localized["interior_identity"]
    legacy = policy["legacy_policy"]
    assert legacy["kind"] == "legacy_whole_slab_sine_v1"
    assert legacy["automatic_public_default"] is False
    assert policy["fresh_only"] is True
    assert policy["restored_checkpoint_state_may_be_conditioned"] is False
    assert policy["no_lifting_construction_must_remain_byte_identical"] is True


def test_rc432_versions_restriction_separately_from_future_dynamics():
    versions = _record()["identity_and_version_contract"]
    restriction = versions["RC4_3_3_public_restriction"]
    future = versions["future_representation_repair"]

    assert restriction["changes_forward_numerical_dynamics"] is False
    assert restriction["checkpoint_format_bump_required"] is False
    assert restriction["plane_forward_dynamics_version_bump_required"] is False
    assert restriction["new_stable_public_rejection_code_required"] is True
    assert versions["RC4_3_4_fresh_policy"][
        "part_of_restart_compatibility_digest"
    ] is False
    assert future["requires_plane_forward_dynamics_version_bump"] is True
    assert future["requires_lifting_restart_identity_version_bump"] is True
    assert future["requires_new_CPU_and_H100_qualification"] is True
    assert versions["historical_checkpoints"]["rewritten_in_place"] is False


def test_rc432_stages_stop_before_implementation_and_h100():
    record = _record()
    stages = record["implementation_slices"]

    assert [item["stage"] for item in stages] == [
        "RC4.3.2",
        "RC4.3.3",
        "RC4.3.4",
        "RC4.3.5",
        "RC4.3.6",
        "RC4.3.7",
    ]
    assert stages[0]["authorized"] is True
    assert stages[0]["complete"] is True
    assert all(item["authorized"] is False for item in stages[1:])
    assert all(item["complete"] is False for item in stages[1:])
    assert record["authorization"] == {
        "rc4_3_2_authorized_by_user": True,
        "rc4_3_2_complete": True,
        "rc4_3_3_implementation_authorized": False,
        "rc4_3_4_implementation_authorized": False,
        "h100_submission_authorized": False,
        "push_authorized": False,
        "automatic_merge_authorized": False,
        "default_promotion_authorized": False,
        "release_authorized": False,
    }


def test_rc432_reviewed_sources_are_content_addressed():
    s16_successor = json.loads(
        (
            ROOT
            / "notes/PSSolver_v0_3_0_s03_successor_72_s16_r0043_rework.json"
        ).read_text(encoding="utf-8")
    )["historical_successor_sources"]
    superseded_by_implementation = {
        "pssolver/api/capabilities.py",
        "pssolver/configuration/simulation_lowering.py",
        "pssolver/configuration/public_simulation_runner.py",
        "pssolver/models/active_nematics/stokes.py",
        "pssolver/operators/lifting.py",
        "pssolver/planning/lifting.py",
        "pssolver/runtime/static_lifting.py",
        "pssolver/runtime/plane_legacy.py",
        "pssolver/applications/plane_beris_edwards.py",
        "pssolver/workflows/plane_checkpoint.py",
        "tests/test_phase8_p842_plane_static_lifting.py",
    }
    for relative, expected in _record()["reviewed_source_sha256"].items():
        # RC4.3.2 is an immutable planning record. RC4.3.3 and RC4.3.4
        # implement its frozen slices and content-address successor sources in
        # their own implementation records.
        if relative in superseded_by_implementation:
            continue
        if relative in s16_successor:
            identity = s16_successor[relative]
            assert identity["historical_sha256"] == expected, relative
            assert _sha256(ROOT / relative) == identity["current_sha256"], relative
            continue
        assert _sha256(ROOT / relative) == expected, relative


def test_rc432_scope_changes_no_runtime_or_external_project():
    assert _record()["scope"] == {
        "runtime_source_modified": False,
        "public_API_modified": False,
        "checkpoint_reader_or_writer_modified": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "H100_executed": False,
    }
