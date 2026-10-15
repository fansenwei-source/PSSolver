"""Static CPU-candidate record for R5-3 Channel checkpoint dynamics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
)
R5_5_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
R5_6_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
R5_7_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
R5_8_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
R5_9_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
R5_10_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
R5_11_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
R5_12_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)
R5_13_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
G1_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _latest_successor(relative: str, historical: str) -> str:
    resolved = historical
    for record_path in (
        R5_5_RECORD,
        R5_6_RECORD,
        R5_7_RECORD,
        R5_8_RECORD,
        R5_9_RECORD,
        R5_10_RECORD,
        R5_11_RECORD,
        R5_12_RECORD,
        R5_13_RECORD,
        G1_RECORD,
    ):
        registration = json.loads(record_path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        if (
            registration is not None
            and registration["historical_sha256"] == resolved
        ):
            resolved = registration["current_sha256"]
    return resolved


def test_r53_record_binds_candidate_commits_and_exact_sources():
    record = _record()
    assert record["status"] == "implementation_complete_cpu_verified_h100_deferred"
    assert record["classification"] == (
        "PASS_R5_3_CHANNEL_CHECKPOINT_DYNAMICS_CPU_CANDIDATE"
    )
    assert record["git_identity"] == {
        "branch": "fix/v0.2.0rc5",
        "implementation_commit": "6eee28e54b2068ad5946807b779fdc1444fb1254",
        "fixture_payload_commit": "d0423073db2a8e84a6fe5ced0ec15b60ef1ab666",
        "initial_qualification_support_commit": (
            "125a3de3e839c614a658a1a75c53c4aab53a2b78"
        ),
        "qualification_support_commit": (
            "b7839b72cc21961e6abaccf77d6f9c2dc054e9cf"
        ),
        "candidate_source_commit": "d0423073db2a8e84a6fe5ced0ec15b60ef1ab666",
        "record_commit_is_candidate_source": False,
        "pushed": False,
    }
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == _latest_successor(relative, expected)


def test_r53_released_fixture_and_disposition_matrix_are_exact():
    record = _record()
    fixture = record["released_fixture"]
    assert _sha256(fixture["manifest"]) == fixture["manifest_sha256"]
    assert fixture["case_count"] == 4
    assert fixture["payload_file_count"] == 80
    manifest = json.loads((ROOT / fixture["manifest"]).read_text(encoding="utf-8"))
    assert set(manifest["cases"]) == {"rc1_even", "rc1_odd", "rc2_even", "rc3_even"}
    assert sum(case["file_count"] for case in manifest["cases"].values()) == 80
    matrix = record["released_checkpoint_matrix"]
    assert matrix["p7_legacy_channel_rc1_even"].startswith("reject_")
    assert matrix["p7_legacy_channel_rc1_odd"].startswith("accept_")
    assert matrix["p7_legacy_channel_rc2_even"].startswith("accept_")
    assert matrix["p7_legacy_channel_rc3_even"].startswith("accept_")
    assert matrix["complete_stress_channel_rc1_even"].startswith("reject_")
    assert matrix["complete_stress_channel_rc1_odd"].startswith("reject_")


def test_r53_contract_versions_dynamics_and_fails_closed_before_mutation():
    contract = _record()["contract"]
    assert contract["current_checkpoint_format_version"] == 2
    assert contract["legacy_checkpoint_format_version"] == 1
    assert contract["forward_stokes_nyquist_identity"] == (
        "channel_no_slip.periodic_nyquist_zero.v2"
    )
    assert contract["current_runtime_identity_includes_dynamics_version"] is True
    assert contract["legacy_identity_is_only_used_for_format_v1_adjudication"] is True
    assert contract["new_checkpoint_writes_use_current_identity_only"] is True
    assert contract["rejection_before_target_mutation"] is True
    assert contract["per_timestep_host_synchronization_added"] is False


def test_r53_successor_registry_matches_every_changed_historical_source():
    successors = _record()["historical_successor_sources"]
    assert len(successors) == 26
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(relative) == _latest_successor(
            relative, registration["current_sha256"]
        )
        assert registration["reason"]


def test_r53_cpu_closure_defers_exactly_the_cumulative_h100_matrix():
    record = _record()
    validation = record["validation"]
    assert validation["targeted_cpu"] == "216 passed, 0 failed"
    assert validation["installed_wheel_targeted_cpu"] == "37 passed, 0 failed"
    assert validation["installed_wheel_applicable_full_cpu"] == (
        "3455 passed, 7 skipped, 8 subtests passed, 0 failed"
    )
    assert validation["h100_checkpoint_matrix"] == (
        "deferred_to_single_rc5_cumulative_qualification"
    )
    evidence = record["installed_wheel_evidence"]
    assert evidence["source_shadow_import"] is False
    assert evidence["pip_check"] == "PASS"
    authorization = record["authorization"]
    assert authorization["r5_3_implementation_complete"] is True
    assert authorization["r5_3_cpu_complete"] is True
    assert authorization["r5_3_h100_complete"] is False
    assert authorization["eligible_for_r5_4"] is True
    assert authorization["eligible_for_automatic_merge"] is False
    assert authorization["eligible_for_default_promotion"] is False
    assert authorization["production_default_changed"] is False
