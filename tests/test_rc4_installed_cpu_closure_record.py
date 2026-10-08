"""Frozen qualification record for RC4.2.5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc4_rc425_installed_cpu_closure.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc425_record_binds_the_plan_and_qualification_support():
    record = _record()
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_2_5_INSTALLED_WHEEL_CROSS_VERSION_AND_CONSUMER_CPU_CLOSURE"
    )
    identity = record["git_identity"]
    assert identity["provider_runtime_source_commit"] == (
        "573daaf1e2d4ad7e54c6c084a6fbb04126fd4966"
    )
    assert identity["provider_qualification_support_commit"] == (
        "d8a7fdeb99b810d60d63fe07103704b9110f9fd9"
    )
    assert _sha256(identity["planning_record"]) == identity["planning_record_sha256"]
    assert _sha256(identity["predecessor_record"]) == identity["predecessor_record_sha256"]
    assert _sha256("benchmarks/run_rc425_installed_checkpoint_closure.py") == (
        record["artifact_identity"]["runner_sha256"]
    )


def test_rc425_record_freezes_exact_installed_wheel_results():
    gate = _record()["installed_wheel_gate"]
    assert gate["classification"] == "PASS_RC4_2_5_INSTALLED_WHEEL_CPU_CLOSURE"
    assert gate["provider_tests_passed"] == 35
    assert gate["consumer_tests_passed"] == 60
    assert gate["total_tests_passed"] == 95
    for key in ("failed", "skipped", "xfailed", "collection_errors"):
        assert gate[key] == 0
    assert gate["pip_check"] == "pass"
    assert gate["source_shadow_import"] is False
    assert gate["functional_api_version"] == "1.0"
    assert gate["functional_compatibility_policy_version"] == 3
    assert gate["periodic_functional_bridge_format_version"] == 3
    assert gate["channel_functional_bridge_format_version"] == 3


def test_rc425_record_closes_the_cross_version_and_consumer_contracts():
    record = _record()
    cross_version = record["cross_version_closure"]
    assert cross_version["matrix_M01_through_M16"] == "PASS"
    assert cross_version["current_checkpoint_round_trip"] == "PASS"
    assert cross_version["registered_legacy_acceptance"] == "PASS"
    assert cross_version["upgrade_source_immutability"] == "PASS"
    assert cross_version["continuous_vs_restart_equivalence"] == "PASS"
    assert cross_version["device_token_excluded_from_compatibility_identity"] is True
    assert cross_version["state_only_migration_not_reported_as_exact_restart"] is True
    consumer = record["consumer_closure"]
    for key in (
        "installed_wheel_checkpoint_round_trip",
        "frozen_periodic_identity_oracle",
        "frozen_channel_identity_oracle",
        "checkpointed_vs_full_history_gradient",
        "finite_difference_and_Taylor",
        "legacy_or_identity_negative_gates",
    ):
        assert consumer[key] == "PASS"
    assert consumer["consumer_source_modified_by_RC4_2_5"] is False
    assert consumer["consumer_requalification_executed"] is True


def test_rc425_environment_recovery_does_not_modify_nematics3d():
    record = _record()
    recovery = record["recovery_history"]
    assert recovery["v1_installed_tests_started"] is False
    assert recovery["v1_nematics3d_modified"] is False
    assert recovery["v2_excludes_nematics3d"] is True
    assert recovery["v2_pip_check"] == "pass"
    assert record["environment"]["nematics3d_installed_in_qualification_venv"] is False
    assert record["scope"]["nematics3d_modified"] is False


def test_rc425_scope_authorizes_only_the_next_planning_step():
    record = _record()
    scope = record["scope"]
    assert scope["PSSolver_Control_requalification_executed"] is True
    for key in (
        "runtime_hot_path_changed",
        "checkpoint_acceptance_changed",
        "stable_functional_API_version_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "production_default_changed",
        "H100_executed",
        "GPU_portability_claimed",
    ):
        assert scope[key] is False
    authorization = record["authorization"]
    assert authorization["rc4_2_5_complete"] is True
    assert authorization["eligible_for_rc4_2_6_planning"] is True
    for key, value in authorization.items():
        if key not in {"rc4_2_5_complete", "eligible_for_rc4_2_6_planning"}:
            assert value is False
