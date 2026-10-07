from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from pssolver.io.checkpoint_identity import (
    LEGACY_IDENTITY_SCHEMAS,
    legacy_registry_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc421_checkpoint_identity_primitives.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


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


def test_rc421_record_binds_the_frozen_plan_and_implementation_files():
    record = _record()
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_2_1_TYPED_IDENTITY_PRIMITIVES_AND_FROZEN_LEGACY_REGISTRY"
    )
    assert record["git_identity"]["implementation_parent"] == (
        "0684e61b30f8a29fc4fe822df8c68602d7b8bf6e"
    )
    plan = ROOT / record["git_identity"]["planning_record"]
    assert _sha256(plan) == record["git_identity"]["planning_record_sha256"]
    fixture = ROOT / record["legacy_registry"]["fixture"]
    assert _git_blob_sha256(
        "1cbb8167dc9fbbaa57cad5b118766618d8ebb3bb",
        record["implementation"]["module"],
    ) == record["implementation"]["module_sha256"]
    assert _sha256(fixture) == record["legacy_registry"]["fixture_sha256"]


def test_rc421_record_freezes_registry_and_layer_contracts():
    record = _record()
    registry = record["legacy_registry"]
    implementation = record["implementation"]

    assert registry["entry_count"] == len(LEGACY_IDENTITY_SCHEMAS) == 21
    assert registry["registry_sha256"] == legacy_registry_sha256()
    assert sum(registry["family_counts"].values()) == 21
    assert registry["unknown_key_policy"] == "reject_fail_closed"
    assert registry["registration_alone_accepts_checkpoint"] is False
    assert implementation["provenance_layers_excluded_from_compatibility_digest"] == [
        "run_provenance",
        "materialization_provenance",
    ]
    assert implementation["functional_identity_requires_derivative_dynamics"] is True
    assert implementation["production_identity_rejects_derivative_dynamics"] is True


def test_rc421_record_preserves_runtime_public_and_external_boundaries():
    record = _record()
    scope = record["scope"]
    assert not any(scope.values())
    assert record["local_verification"] == {
        "primitive_and_plan_tests_passed": 20,
        "closure_tests_passed": 4,
        "known_issues_matrix_tests_passed": 8,
        "targeted_total_passed": 32,
        "full_suite_passed": 2820,
        "full_suite_deselected_cuda_only": 7,
        "full_suite_subtests_passed": 8,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "full_suite_elapsed_seconds": 202.56,
        "git_diff_check": "pass",
    }


def test_rc421_record_authorizes_only_the_next_planning_boundary():
    authorization = _record()["authorization"]
    assert authorization["rc4_2_1_complete"] is True
    assert authorization["eligible_for_rc4_2_2_planning"] is True
    for key, value in authorization.items():
        if key not in {"rc4_2_1_complete", "eligible_for_rc4_2_2_planning"}:
            assert value is False
