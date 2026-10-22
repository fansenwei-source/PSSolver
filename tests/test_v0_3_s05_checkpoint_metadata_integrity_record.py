"""Bindings for the S-05 checkpoint-metadata integrity candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes/PSSolver_v0_3_0_s03_successor_09_s05_metadata_integrity.json"
)
CANDIDATE_RECORD = (
    ROOT / "notes/PSSolver_v0_3_0_s03_successor_10_s05_candidate.json"
)
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
IMPLEMENTATION = "af7c1217b4599f7e870b42a9c0fc92f5b3c90c61"
PARENT = "26216537ee38c2258e06f32ce9b095d17a9dbc40"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s05_record_binds_the_linear_implementation_commit():
    record = _read(RECORD)
    identity = record["git_identity"]
    assert record["stage"] == "S-05"
    assert identity["implementation_commit"] == IMPLEMENTATION
    assert identity["implementation_parent"] == PARENT
    assert identity["record_commit_is_qualified_source"] is False
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, IMPLEMENTATION],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    assert subprocess.run(
        ["git", "rev-parse", f"{IMPLEMENTATION}^"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == PARENT


def test_s05_record_binds_every_historical_successor_source():
    successors = _read(RECORD)["historical_successor_sources"]
    for relative, registration in successors.items():
        historical = subprocess.run(
            ["git", "show", f"{PARENT}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        current = subprocess.run(
            ["git", "show", f"{IMPLEMENTATION}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == registration[
            "historical_sha256"
        ]
        assert hashlib.sha256(current).hexdigest() == registration[
            "current_sha256"
        ]
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, registration["current_sha256"]
        )


def test_s05_record_preserves_the_selected_contract_and_scope():
    record = _read(RECORD)
    contract = record["contract"]
    assert set(contract["selected_answers"].values()) == {"A"}
    assert contract["new_formats"] == {
        "plane_workflow": 3,
        "legacy_channel_workflow": 3,
        "finite_q_anchoring_workflow": 2,
        "experimental_shadow_checkpoint": 2,
    }
    assert "not a MAC or signature" in contract["digest_boundary"]
    scope = record["scope"]
    assert scope["runtime_changed"] is True
    assert scope["checkpoint_schema_changed"] is True
    assert scope["checkpoint_tensor_bytes_changed"] is False
    assert scope["numerical_implementation_changed"] is False
    assert scope["provider_identity_changed"] is False
    assert scope["functional_api_version_changed"] is False
    assert scope["threshold_changed"] is False
    assert scope["skip_or_xfail_added"] is False
    assert scope["production_default_changed"] is False
    assert scope["PSSolver_Control_modified"] is False
    assert scope["nematics3d_modified"] is False


def test_s05_registry_claim_is_pending_independent_review():
    issue = next(
        item
        for item in _read(REGISTRY)["issues"]
        if item["id"] == "A2-21/S2-12"
    )
    assert issue["status"] == "claimed_fixed"
    assert issue["fixed_in"] == (
        "65ef1080cb0215131efe90143863de7014fe4b5b"
    )
    assert "not a MAC or signature" in issue["notes"]
    assert "Independent verification is pending" in issue["notes"]
    assert "verified_by" not in issue


def test_s05_record_does_not_authorize_external_actions():
    authorization = _read(RECORD)["authorization"]
    assert authorization["local_implementation_authorized"] is True
    assert all(
        authorization[key] is False
        for key in (
            "push_authorized",
            "tag_authorized",
            "release_authorized",
            "HPCC_authorized",
            "GPU_authorized",
            "H100_authorized",
        )
    )


def test_s05_candidate_successors_bind_the_current_tree():
    record = _read(CANDIDATE_RECORD)
    assert record["git_identity"]["qualified_source_candidate"] == (
        "65ef1080cb0215131efe90143863de7014fe4b5b"
    )
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_s05_candidate_records_the_complete_local_qualification():
    validation = _read(CANDIDATE_RECORD)["validation"]
    assert validation["archive_checksums_entries"] == 67
    assert validation["archive_checksums_sha256"] == (
        "98574514a422d9d8b9184ee964d0f8b2f8a3c91a85d45861a569918cad95dbde"
    )
    assert validation["archive_complete_present"] is True
    assert validation["installed_wheel"] == {
        "filename": "pssolver-0.3.0.dev0-py3-none-any.whl",
        "sha256": "d8d52ff229600aee45c8d319e6e7ddd78af35378bd342137e3736d17a87c4bd3",
        "source_shadow_import": False,
        "pip_check": "PASS",
        "s05_tests_passed": 53,
        "failed": 0,
        "skipped": 0,
    }
    assert validation["released_v0_2_2_forward_rejection"][
        "accepted_count"
    ] == 0
    assert validation["released_v0_2_2_forward_rejection"][
        "explicit_rejection_count"
    ] == 4
    assert validation["full_cpu_suite"] == {
        "passed": 4430,
        "failed": 0,
        "skipped": 1,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
        "optional_skip": (
            "tests/test_analytic_periodic_defect_gas_2d.py::"
            "test_nematics3d_recovers_requested_defect_count_charges_and_positions"
        ),
    }
    assert validation["qualification_complete"] is True
    assert validation["eligible_for_independent_review"] is True
