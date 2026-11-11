"""RC4.3.4 explicit fresh-state conditioning policy regressions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pssolver.operators import (
    legacy_whole_slab_sine_conditioning,
    localized_quintic_wall_transition,
    strict_no_automatic_conditioning,
)
from tests.v0_3_s03_successor import resolve_s03_successor


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
    return resolve_s03_successor(relative, resolved)


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






def test_rc434_record_binds_sources_scope_and_qualification_boundary():
    from qualification.s27_forward import assert_forward_binding
    assert_forward_binding(Path(__file__).resolve().parents[1], 'tests/test_rc4_fresh_initial_conditioning.py::test_rc434_record_binds_sources_scope_and_qualification_boundary')
