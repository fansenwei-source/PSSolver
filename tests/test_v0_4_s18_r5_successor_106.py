from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_s03_successor_106_s18_r5_local_qualification.json"
)


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_106_binds_the_r5_qualified_source_and_repository_inputs() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["successor"] == 106
    assert record["git_identity"]["qualified_source"] == (
        "64d2f81b5c0775667672e764af4ae675d3219fdd"
    )
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["forward_resolution"] == {
        "record": "notes/PSSolver_v0_4_0_s18_r5_channel_pcg_control.json",
        "record_sha256": "22408b217abdc60b6cbb8ac00d0fe35ea1d68ae00c8571790531ba6b9356ca3a",
        "field": "qualification_support.formal_clean_clone_qualification_pending",
        "historical_value": True,
        "resolved_value": False,
        "historical_record_rewritten": False,
    }
    assert _sha256(record["forward_resolution"]["record"]) == (
        record["forward_resolution"]["record_sha256"]
    )
    for relative, expected in record["repository_bindings"].items():
        assert _sha256(relative) == resolve_s03_successor(relative, expected)


def test_successor_106_records_zero_tolerance_cpu_qualification() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    qualification = record["qualification"]
    assert qualification["pcg_case_count"] == 4
    assert qualification["pcg_path_count"] == 4
    assert qualification["pcg_parent_bitwise_equal"] is True
    assert qualification["pcg_v0_3_0_bitwise_equal"] is True
    assert qualification["eight_family_twenty_step_parent_bitwise_equal"] is True
    assert qualification["eight_family_twenty_step_v0_3_0_bitwise_equal"] is True
    assert qualification["silent_divergence_runs"] == 2160
    assert qualification["silent_divergence_fail_closed"] == 0
    assert qualification["silent_acceptable_above_1e_3"] == 0
    assert qualification["floor_crossing_runs"] == 360
    assert qualification["floor_crossing_fail_closed"] == 0
    assert qualification["nonzero_comparison_tolerance"] is None
    assert len(record["evidence_bindings"]) == 11


def test_successor_106_grants_no_remote_or_gpu_authority() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_qualification": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
    assert record["triggers"]["real_nfs_retest_required"] is False
    assert record["triggers"]["control_first_migration_required"] is False
    assert record["triggers"]["gpu_and_h100_owner"] == "S-19"
