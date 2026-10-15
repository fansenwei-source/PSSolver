"""Frozen handoff contract for the rc5 PSSolver-Control identity migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_control_identity_migration_handoff.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_handoff_binds_the_frozen_rc5_provider_evidence():
    provider = _record()["provider"]
    assert provider["qualified_source_commit"] == (
        "674a8121779be72c95aa5dc9e2babca0ce7b93a9"
    )
    assert provider["package_version"] == "0.2.0rc5"
    assert provider["functional_api_version"] == "1.0"
    assert provider["qualified_wheel_sha256"] == (
        "3097b08b08ba7b2d488fd8af348ec48986d69007a640bd806a225ec9963bb65f"
    )
    for key in ("cpu_closure_record", "checkpoint_matrix_record"):
        bound = provider[key]
        assert _sha256(ROOT / bound["path"]) == bound["sha256"]


def test_handoff_binds_the_remote_control_baseline_and_qualified_source():
    consumer = _record()["consumer_baseline"]
    assert consumer["branch"] == "migration/pssolver-v0.2.0rc4-rc457-tf32-identity"
    assert consumer["remote_tip"] == "0e3895d664d1b81925127848b2dfde638f39763d"
    assert consumer["qualified_source_commit"] == (
        "b8539e0ec4c137a6c13746f6832e26e74be24515"
    )
    assert consumer["qualified_wheel_sha256"] == (
        "c7330f3759e01defc824f2492240974b744abc4872c20060d90ca957b2f35ade"
    )


def test_handoff_freezes_the_single_periodic_identity_change():
    change = _record()["identity_contract_change"]
    assert change["component"] == "periodic_activity_batch_one"
    assert change["only_changed_path"] == (
        "provider_identity.execution.functional_runtime.hermitian_state_projection"
    )
    assert change["old_value"] == "self_conjugate_planes_each_step"
    assert change["new_value"] == (
        "self_conjugate_stored_planes_projection_each_step.v2"
    )
    assert change["diagnostic_fixture_example"]["portable_pin"] is False


def test_handoff_forbids_reusing_diagnostic_digests_as_global_pins():
    change = _record()["identity_contract_change"]
    rule = change["canonical_consumer_rule"]
    assert "each existing PSSolver-Control canonical fixture" in rule
    assert "never to the diagnostic digest above" in rule
    assert change["channel"]["changed"] is False
    assert "byte-identical" in change["channel"]["required_result"]


def test_handoff_requires_fail_closed_and_installed_wheel_cpu_gates():
    actions = "\n".join(item["action"] for item in _record()["consumer_work_order"])
    for required in (
        "old Periodic expected identity must reject rc5",
        "newly computed fixture-specific identity must accept rc5",
        "complete Consumer CPU suite",
        "install both with --no-deps",
        "prove imports come only from site-packages",
        "Do not push, submit H100",
    ):
        assert required in actions


def test_handoff_preserves_public_versions_and_out_of_scope_components():
    unchanged = _record()["identity_contract_change"]["unchanged_contracts"]
    assert unchanged == {
        "functional_api_version": "1.0",
        "periodic_adapter_version": 2,
        "channel_adapter_version": 3,
        "checkpoint_schema_changed": False,
        "control_algorithm_changed": False,
        "production_default_changed": False,
        "nematics3d_modified": False,
    }


def test_handoff_is_cpu_only_and_keeps_g1_fail_closed():
    record = _record()
    assert record["acceptance"] == {
        "periodic_identity_migration_passed": False,
        "channel_identity_unchanged_proven": False,
        "source_cpu_qualification_passed": False,
        "installed_wheel_cpu_qualification_passed": False,
        "eligible_for_h100_planning": False,
        "g1_complete": False,
    }
    authorization = record["authorization"]
    assert authorization["PSSolver_Control_source_change_authorized_for_Claude"] is True
    for key, value in authorization.items():
        if key != "PSSolver_Control_source_change_authorized_for_Claude":
            assert value is False
    assert record["next_gate_after_consumer_cpu_pass"] == (
        "review_consumer_evidence_and_freeze_a_single_h100_qualification_contract"
    )
