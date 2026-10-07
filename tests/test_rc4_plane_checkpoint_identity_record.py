"""Frozen implementation record for RC4.2.2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc422_plane_checkpoint_identity.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc422_record_binds_plan_parent_and_implementation_files():
    record = _record()
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_2_2_PLANE_LAYERED_CHECKPOINT_IDENTITY_AND_LEGACY_UPGRADE"
    )
    assert record["git_identity"]["implementation_parent"] == (
        "1cbb8167dc9fbbaa57cad5b118766618d8ebb3bb"
    )
    assert _sha256(record["git_identity"]["planning_record"]) == (
        record["git_identity"]["planning_record_sha256"]
    )
    for relative, expected in record["implementation_files"].items():
        assert _sha256(relative) == expected


def test_rc422_record_freezes_v2_and_legacy_upgrade_contracts():
    record = _record()
    implementation = record["implementation"]
    legacy = record["legacy_upgrade"]

    assert implementation["current_format_version"] == 2
    assert implementation["legacy_format_version"] == 1
    assert implementation["current_writer_emits_v2_only"] is True
    assert implementation["legacy_v1_direct_restore_allowed"] is False
    assert implementation["device_token_in_compatibility_digest"] is False
    assert implementation["prescribed_lift_in_compatibility_digest"] is True
    assert legacy["registered_generations"] == ["rc1", "rc2", "rc3"]
    assert legacy["opaque_v1_digest_authenticated_before_tensor_load"] is True
    assert legacy["even_periodic_axis_pre_rc4_1_checkpoint_rejected"] is True
    assert legacy["in_place_rewrite"] is False


def test_rc422_record_closes_exactly_m01_through_m06():
    matrix = _record()["cross_version_matrix"]
    assert set(matrix) == {
        "M01_device_only",
        "M02_fresh_initializer_or_conditioning_only",
        "M03_prescribed_lift_or_correction_difference",
        "M04_rc1_lifted_pre_B8_dynamics",
        "M05_rc1_rc2_rc3_even_periodic_nyquist_applicable",
        "M06_rc1_rc2_rc3_odd_periodic_nyquist_inapplicable",
    }
    assert all(value.startswith("PASS_") for value in matrix.values())


def test_rc422_scope_and_next_authorization_remain_narrow():
    record = _record()
    scope = record["scope"]
    assert scope["plane_checkpoint_reader_modified"] is True
    assert scope["plane_checkpoint_writer_modified"] is True
    assert scope["plane_restore_acceptance_behavior_changed"] is True
    for key in (
        "plane_runtime_hot_path_changed",
        "periodic_checkpoint_modified",
        "channel_checkpoint_modified",
        "functional_checkpoint_modified",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "production_default_changed",
        "H100_executed",
    ):
        assert scope[key] is False
    authorization = record["authorization"]
    assert authorization["rc4_2_2_complete"] is True
    assert authorization["eligible_for_rc4_2_3_planning"] is True
    for key, value in authorization.items():
        if key not in {"rc4_2_2_complete", "eligible_for_rc4_2_3_planning"}:
            assert value is False
