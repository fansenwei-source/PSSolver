"""Contract for importing the independent R-0038 rc1 acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_62_r0038_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0038_identity_and_independent_archive_are_exact() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_R0038_V0_3_0RC1_INDEPENDENT_CANDIDATE_ACCEPTANCE"
    )
    assert record["git_identity"] == {
        "candidate_target": "d612c9a4df4bbb8c724b6865b90fca9a9286dbc2",
        "release_source": "c031cdae646edc5b202b05238db1eb513b255805",
        "development_base": "0e96ee5d05193b0ddae189f6171ba387a8a96b6b",
        "record_commit_is_release_source": False,
        "pushed": False,
    }
    assert record["sync_result"] == {
        "id": "R-0038",
        "path": "PSSolver_sync/results/R-0038.result.json",
        "sha256": "5522be286445d49c2935d60d5e5dca3486edd9e817968f4990cfc26ae6f763d4",
        "sync_commit": "47c81b885383350703b7bcb1c2b0e3a7e921e36b",
        "classification": "PASS",
    }
    assert record["independent_archive"] == {
        "path": "/home/fansenwei/pssolver_v0_3_0rc1_claude_acceptance_20261028_v1",
        "checksums_entries": 76,
        "checksums_sha256": "f930ef864dbffdbe1db4de509fc33ff42210ab56c28615f55c8e5a0ed4e50f95",
        "complete_present": True,
        "final_report_sha256": "66d42f423fc30dd79b8c1e06dd25447b5beaba36703d4e4fb386d83e1a41f3d1",
        "summary_sha256": "ccf1f841f9bd5f4281283f11d7b3b9aa62c2b70c870e4fc68d155784b42f3f2e",
    }


def test_r0038_imports_v44_without_rewriting_sealed_evidence() -> None:
    imported = _read(RECORD)["registry_import"]["V-44"]
    assert imported == {
        "status": "verified",
        "fixed_in": "4baa26c8c67d5e01c7615fa1dae29679f450645b",
        "verified_by": "R-0038",
        "sealed_archive_rewritten": False,
    }
    v44 = next(item for item in _read(REGISTRY)["issues"] if item["id"] == "V-44")
    assert v44["status"] == "verified"
    assert v44["fixed_in"] == imported["fixed_in"]
    assert v44["verified_by"] == "R-0038"


def test_r0038_records_only_the_new_gate_boundary() -> None:
    assert _read(RECORD)["completion_gates"] == {
        "G-01": "PASS",
        "G-05": "PASS",
        "G-09": (
            "rc1 independent acceptance PASS; online asset and post-release "
            "verification pending"
        ),
    }


def test_r0038_leaves_the_frozen_source_records_unchanged() -> None:
    for relative, expected in _read(RECORD)["immutable_source_records"].items():
        assert _sha256(ROOT / relative) == expected


def test_successor_62_binds_every_changed_historical_source() -> None:
    for relative, identity in _read(RECORD)["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == identity["current_sha256"]
        assert identity["historical_sha256"] != identity["current_sha256"]


def test_r0038_import_does_not_authorize_external_actions() -> None:
    record = _read(RECORD)
    assert record["scope"] == {
        "records_and_tests_only": True,
        "runtime_changed": False,
        "packaging_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "tag_or_release_modified": False,
    }
    assert record["authorization"] == {
        "local_import": True,
        "push": False,
        "tag": False,
        "github_release": False,
        "pypi": False,
        "hpcc_or_slurm": False,
    }
    assert record["next_plan_step"] == "S-12"
