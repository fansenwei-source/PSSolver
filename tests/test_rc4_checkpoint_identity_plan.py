"""Static contracts for the planning-only RC4.2 identity migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc420_checkpoint_identity_plan.json"
)


def _plan() -> dict[str, object]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha256(commit: str, relative: str) -> str:
    value = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(value).hexdigest()


def test_rc420_freezes_only_the_checkpoint_identity_batch():
    plan = _plan()

    assert plan["schema_version"] == 1
    assert plan["phase"] == "RC4.2.0"
    assert plan["classification"] == (
        "PASS_RC4_2_CHECKPOINT_IDENTITY_PLANNING"
    )
    assert plan["status"] == (
        "RC4_2_PLANNING_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED"
    )
    assert plan["issues"] == ["B10", "B22", "N3", "N5", "N6"]
    assert plan["baseline"]["released_baseline"] == {
        "release": "v0.2.0rc3",
        "commit": "5071d73a00e0d19be62ebd39edf9918818d1267a",
    }
    assert _sha256(ROOT / plan["baseline"]["rc4_1_closure_record"]) == (
        plan["baseline"]["rc4_1_closure_record_sha256"]
    )
    assert _sha256(ROOT / plan["baseline"]["known_issues_record"]) == (
        plan["baseline"]["known_issues_record_sha256"]
    )


def test_rc420_separates_compatibility_from_provenance_and_initialization():
    plan = _plan()
    layers = {item["name"]: item for item in plan["identity_layers"]}

    assert list(layers) == [
        "run_provenance",
        "forward_dynamics",
        "derivative_dynamics",
        "state_layout",
        "backend_restart",
        "materialization_provenance",
    ]
    assert layers["run_provenance"]["checkpoint_compatibility_gate"] is False
    assert layers["materialization_provenance"][
        "checkpoint_compatibility_gate"
    ] is False
    assert layers["forward_dynamics"]["checkpoint_compatibility_gate"] is True
    assert layers["derivative_dynamics"][
        "checkpoint_compatibility_gate"
    ] is True
    assert "requested or allocated device" in layers["forward_dynamics"][
        "excludes"
    ]
    assert "fresh_initial_remainder_conditioning" in layers[
        "forward_dynamics"
    ]["excludes"]
    assert plan["dynamics_components"]["plane"][
        "prescribed_lift_and_linear_correction_remain_dynamics"
    ] is True


def test_rc420_compatibility_matrix_is_complete_and_fail_closed():
    plan = _plan()
    matrix = plan["cross_version_restart_matrix"]

    assert [row["id"] for row in matrix] == [
        f"M{index:02d}" for index in range(1, 17)
    ]
    covered = {issue for row in matrix for issue in row["covers"]}
    assert covered == set(plan["issues"])
    by_id = {row["id"]: row for row in matrix}
    assert by_id["M01"]["expected"] == "accept_exact_restart"
    assert by_id["M02"]["expected"] == "accept_exact_restart"
    assert by_id["M03"]["expected"] == (
        "reject_before_tensor_load_and_target_mutation"
    )
    assert by_id["M04"]["expected"] == (
        "reject_exact_restart_by_forward_dynamics_version"
    )
    assert by_id["M07"]["expected"] == (
        "reject_exact_restart_allow_only_separately_qualified_state_migration"
    )
    assert by_id["M10"]["expected"] == (
        "reject_exact_restart_by_forward_dynamics_version"
    )
    assert by_id["M13"]["expected"] == (
        "reject_functional_exact_restart_or_use_explicit_state_only_migration"
    )
    assert by_id["M15"]["expected"] == (
        "reject_before_tensor_load_and_target_mutation"
    )
    assert by_id["M16"]["expected"] == "retain_existing_rejection"


def test_rc420_legacy_policy_does_not_derive_history_from_live_metadata():
    policy = _plan()["legacy_schema_policy"]
    opaque = _plan()["legacy_opaque_hash_constraint"]

    assert policy["registry_is_independent_of_live_identity_objects"] is True
    assert policy["forbidden_implementation"] == (
        "copy_current_metadata_then_delete_or_rename_fields"
    )
    assert policy["unknown_versions"] == (
        "reject_before_tensor_load_or_target_mutation"
    )
    assert policy["frozen_release_generations"] == {
        "rc1": "1c5237194c5e4f696bb5d3d01d2caa2c4bdd27a1",
        "rc2": "2bbe21e88fec318787fc31dc0a82bca5a171c024",
        "rc3": "5071d73a00e0d19be62ebd39edf9918818d1267a",
    }
    assert opaque["guess_source_device_or_initializer"] is False
    assert "explicit source run specification" in opaque["upgrade_tool_rule"]
    assert "without mutating the source" in opaque["upgrade_tool_rule"]


def test_rc420_rejects_every_mismatch_before_first_target_mutation():
    plan = _plan()
    order = plan["validation_order"]
    first_mutation = order.index("first target mutation")

    for gate in (
        "forward dynamics identity",
        "derivative dynamics identity for functional state",
        "state layout identity",
        "backend restart identity",
        "lifting dynamics identity",
        "tensor manifest filenames and checksums",
        "tensor shape, dtype, and finite values",
        "construct complete copy or migration plan",
    ):
        assert order.index(gate) < first_mutation
    assert order[-1] == "restore progress and rebuild declared derived caches"
    assert all(plan["pre_mutation_contract"].values())


def test_rc420_implementation_is_sliced_and_still_unauthorized():
    plan = _plan()
    slices = plan["implementation_slices"]

    assert [item["id"] for item in slices] == [
        "RC4.2.0",
        "RC4.2.1",
        "RC4.2.2",
        "RC4.2.3",
        "RC4.2.4",
        "RC4.2.5",
        "RC4.2.6",
    ]
    assert slices[0]["authorized"] is True
    assert slices[0]["implemented"] is True
    assert slices[0]["runtime_change"] is False
    assert all(item["authorized"] is False for item in slices[1:])
    assert all(item["implemented"] is False for item in slices[1:])
    assert slices[4]["consumer_requalification_required"] is True
    assert slices[-1]["h100_required"] is True
    assert plan["authorization"] == {
        "rc4_2_planning_authorized": True,
        "rc4_2_planning_complete": True,
        "rc4_2_1_implementation_authorized": False,
        "rc4_2_implementation_authorized": False,
        "legacy_checkpoint_rewrite_authorized": False,
        "PSSolver_Control_change_authorized": False,
        "H100_authorized": False,
        "rc4_3_authorized": False,
        "automatic_merge_authorized": False,
        "default_promotion_authorized": False,
        "release_authorized": False,
        "production_default_changed": False,
    }


def test_rc420_reviewed_sources_are_content_addressed_and_unmodified():
    plan = _plan()

    for relative, expected in plan["reviewed_source_sha256"].items():
        assert _git_blob_sha256(
            plan["baseline"]["planning_parent"],
            relative,
        ) == expected
    assert plan["scope"] == {
        "runtime_source_modified": False,
        "checkpoint_reader_or_writer_modified": False,
        "public_API_modified": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "H100_executed": False,
    }


def test_rc420_qualification_boundary_is_explicit():
    qualification = _plan()["qualification"]

    assert "cross_version_restart_matrix" in qualification[
        "required_provider_CPU_gates"
    ]
    assert "pre_mutation_rejection" in qualification[
        "required_provider_CPU_gates"
    ]
    assert "current_checkpoint_round_trip" in qualification[
        "required_installed_wheel_gates"
    ]
    assert qualification["single_H100_scope"][-1] == (
        "no_profile_or_performance_claim"
    )
    assert qualification["functional_identity_change_triggers_PSSolver_Control"] is True
