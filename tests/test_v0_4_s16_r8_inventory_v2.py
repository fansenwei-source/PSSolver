"""Forward-only checks for the execution-based S-16 R8 inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "notes/PSSolver_v0_4_0_s16_r8_inventory.json"
V2 = ROOT / "notes/PSSolver_v0_4_0_s16_r8_inventory_v2.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v1_inventory_is_not_rewritten():
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == (
        "be0b0ca2923f88bd43909bf43e5d86f175271c0b35eadd2a61c7f79239879758"
    )


def test_v2_inventory_binds_the_execution_observed_failure_set():
    record = _read(V2)
    assert record["classification"] == (
        "R8_EXECUTION_INVENTORY_CORRECTED_IMPLEMENTATION_NOT_STARTED"
    )
    assert record["execution_audit"]["audited_commit"] == (
        "7e9871ca1f2ce1b83f296c3a482e99bda78f1bee"
    )
    failures = record["failing_tests"]
    assert len(failures) == 27
    assert sum(len(names) for names in failures.values()) == 35
    assert record["summary"] == {
        "tests_with_live_absolute_dependencies": 27,
        "failing_tests": 35,
        "external_roots": 13,
        "observed_root_bytes_lstat_sum": 496619097,
        "fresh_github_clone_is_self_contained": False,
        "r8_implemented": False,
    }


def test_v2_inventory_adds_every_r0043_missing_file_and_root():
    record = _read(V2)
    missing_files = {
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_r57_device_normalization_record.py",
        "tests/test_v0_3_s09_cpu_6x60_preregistration.py",
        "tests/test_v0_3_s09_cpu_6x60_stop_record.py",
        "tests/test_v0_3_s09_cpu_diagnosis_record.py",
        "tests/test_v0_3_s09_failure_analysis_record.py",
        "tests/test_v0_3_s09_r0029_candidate.py",
        "tests/test_v0_3_s09_r0030_candidate.py",
        "tests/test_v0_3_s09_transient_telemetry_record.py",
    }
    assert missing_files <= set(record["failing_tests"])
    missing_roots = {
        "/home/fansenwei/pssolver_rc5_r56_installed_cpu_cf572bc_20261014_v1",
        "/home/fansenwei/pssolver_rc5_r57_installed_cpu_2d67ba2_20261014_v1",
        "/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0029_local_b031421_20261026_v2",
        "/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0030_local_658baca_20261026_v1",
        "/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_failure_analysis_87f6bfe_20261023_v4",
        "/home/fansenwei/pssolver_v0_3_s09_cpu_transient_telemetry_d568039_20261023_v5",
    }
    assert missing_roots <= set(record["external_roots"])


def test_v2_inventory_remains_analysis_only():
    scope = _read(V2)["scope"]
    assert scope == {
        "inventory_only": True,
        "r8_implementation_changed": False,
        "historical_record_rewritten": False,
        "skip_or_xfail_added": False,
        "external_action_authorized": False,
    }
