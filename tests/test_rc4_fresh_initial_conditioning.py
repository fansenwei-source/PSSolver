"""RC4.3.4 explicit fresh-state conditioning policy regressions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import hashlib
import json
from pathlib import Path

import pytest

from pssolver.operators import (
    FreshInitialConditioningPolicy,
    legacy_whole_slab_sine_conditioning,
    localized_quintic_wall_transition,
    strict_no_automatic_conditioning,
)
from pssolver.runtime.plane_beris_edwards import PlaneRuntimeBuildRequest


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "notes/PSSolver_v0_2_0rc4_rc434_fresh_conditioning.json"
CONTRACT_PATH = (
    ROOT / "notes/PSSolver_v0_2_0rc4_rc432_lifting_capability_contract.json"
)
RC456 = ROOT / "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
RC5_R57 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RC5_R511 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
RC5_R513 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
RC6_R6 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def _rc456_successor(relative: str, historical: str) -> str:
    resolved = historical
    for path in (RC456, RC5_R57, RC5_R511, RC5_R513, RC6_R6):
        registration = json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        if (
            registration is not None
            and registration["historical_sha256"] == resolved
        ):
            resolved = registration["current_sha256"]
    return resolved


def test_policy_declarations_are_immutable_and_match_the_frozen_contract():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    frozen = contract["fresh_initial_conditioning_contract"]
    strict = strict_no_automatic_conditioning()
    localized = localized_quintic_wall_transition(0.5)
    legacy = legacy_whole_slab_sine_conditioning()

    assert strict.kind == frozen["public_default"]["kind"]
    assert localized.kind == frozen["qualified_opt_in_candidate"]["kind"]
    assert localized.formula == frozen["qualified_opt_in_candidate"]["formula"]
    assert legacy.kind == frozen["legacy_policy"]["kind"]
    assert legacy.formula == frozen["legacy_policy"]["formula"]
    with pytest.raises(FrozenInstanceError):
        strict.kind = "mutated"


def test_runtime_request_defaults_to_strict_fresh_only_behavior():
    by_name = {item.name: item for item in fields(PlaneRuntimeBuildRequest)}
    policy = by_name["fresh_initial_conditioning"].default_factory()

    assert isinstance(policy, FreshInitialConditioningPolicy)
    assert policy == strict_no_automatic_conditioning()
    assert by_name["apply_fresh_initial_conditioning"].default is True


def test_restart_and_legacy_edges_make_conditioning_control_explicit():
    application = (
        ROOT / "pssolver/applications/plane_beris_edwards.py"
    ).read_text(encoding="utf-8")
    runtime_edge = (
        ROOT / "pssolver/runtime/plane_beris_edwards.py"
    ).read_text(encoding="utf-8")
    legacy = (ROOT / "pssolver/runtime/plane_legacy.py").read_text(
        encoding="utf-8"
    )

    assert "apply_fresh_initial_conditioning=not is_restart" in application
    assert "request.apply_fresh_initial_conditioning" in runtime_edge
    assert "apply_fresh_conditioning=apply_fresh_initial_conditioning" in legacy


def test_rc434_record_binds_sources_scope_and_qualification_boundary():
    record = _record()

    assert record["classification"] == (
        "PASS_RC4_3_4_EXPLICIT_FRESH_CONDITIONING_CPU_CANDIDATE"
    )
    assert record["implementation"]["default_policy"] == (
        "strict_no_automatic_conditioning_v1"
    )
    assert record["implementation"]["localized_opt_in_policy"] == (
        "localized_quintic_wall_transition_v1"
    )
    assert record["implementation"]["legacy_policy"] == (
        "legacy_whole_slab_sine_v1"
    )
    assert record["restart_contract"] == {
        "fresh_only": True,
        "restored_state_reconditioned": False,
        "part_of_restart_compatibility_digest": False,
        "checkpoint_format_changed": False,
        "forward_dynamics_identity_changed": False,
    }
    assert record["scope"]["public_lifting_restriction_changed"] is False
    assert record["scope"]["production_default_changed"] is False
    assert record["scope"]["PSSolver_Control_modified"] is False
    assert record["scope"]["nematics3d_modified"] is False
    assert record["authorization"]["h100_submission_authorized"] is False
    assert record["authorization"]["push_authorized"] is False
    assert record["eligibility"]["eligible_for_rc4_3_5_integration"] is True
    assert record["eligibility"]["localized_policy_publicly_qualified"] is False

    assert _sha256(CONTRACT_PATH) == record["baseline"][
        "contract_record_sha256"
    ]
    superseded_by_rc435 = {
        "pssolver/runtime/static_lifting.py",
        "pssolver/runtime/plane_legacy.py",
        "pssolver/runtime/plane_beris_edwards.py",
        "tests/test_phase8_p844_lifting_workflow_restart.py",
        "tests/test_rc4_fresh_initial_conditioning.py",
    }
    for relative, expected in record["source_sha256"].items():
        if relative in superseded_by_rc435:
            continue
        assert _sha256(ROOT / relative) == _rc456_successor(
            relative,
            expected,
        ), relative
