"""Historical shadow-stage scope; no assertions about future branch status."""

import json
import subprocess
from pathlib import Path

from tests.v0_3_s03_successor import registered_source_sha256, resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_111_r1_shadow_registry.json"


def _record():
    return json.loads(RECORD.read_text())


def test_r1_shadow_record_limits_claim_to_pre_dispatch_stage():
    record = _record()
    assert record["stage"] == "R1-shadow-only"
    assert record["r1_complete"] is False
    assert record["production_dispatch_switched"] is False
    assert record["registry_updated"] is False
    assert record["decision"]["id"] == "D-0018"
    assert record["decision"]["sha256"] == "38cfe71b63efcca6ef47a759d5955bd332266cf4a65cef6e61635c43bfd1a627"
    assert record["nonzero_tolerance_authorized"] is False
    assert not any(record["external_actions_authorized"].values())


def test_r1_shadow_record_binds_introduced_sources_forward():
    for path, digest in _record()["introduced_sources"].items():
        actual = registered_source_sha256(path)
        assert actual == resolve_s03_successor(path, digest)


def test_r1_shadow_prototype_only_added_a_module_and_its_test():
    record = _record()
    parent = record["git_identity"]["batch_parent"]
    prototype = record["git_identity"]["shadow_prototype"]
    changes = subprocess.check_output(
        ["git", "diff", "--name-status", parent, prototype],
        cwd=ROOT, text=True,
    ).splitlines()
    assert changes == [
        "A\tpssolver/configuration/application_registry.py",
        "A\ttests/test_v0_4_s18_r1_shadow_registry.py",
    ]
    assert record["prototype_test_correction"]["failed_tests"] == 4
    assert record["prototype_test_correction"]["runtime_changed"] is False
    assert record["prototype_test_correction"]["history_rewritten"] is False


def test_r1_final_review_still_requires_installed_consumer_and_checkpoint_gates():
    pending = set(_record()["remaining_before_r1_review"])
    assert {"dispatch_switch_with_legacy_negative_control", "two_cpu_suites",
            "blocked_external_paths_suite", "bitwise_states_and_identities",
            "released_checkpoints_read_only", "writer_byte_trees_including_functional",
            "fault_matrix", "clean_wheel", "control_main_installed_cpu_suite"} <= pending
