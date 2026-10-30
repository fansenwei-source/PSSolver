"""Forward-only binding for the D-0016 S-16 harness-sharing implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_75_s16_d0016_harness_sharing.json"
INVENTORY = ROOT / "notes/PSSolver_v0_4_0_s16_r8_benchmark_harness_sharing.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_75_binds_d0016_and_all_forward_sources():
    record = _read(RECORD)
    assert record["successor"] == 75
    assert record["decision"] == {
        "commit": "98d29d611ad048b3f5004e97e7ee0b8a4a76aa6c",
        "id": "D-0016",
        "selection": "option b: share only live harness support",
        "sha256": "fe2ddb0ee5ba87bc159888f95e59ea45e7ded1b293f122ee5f6d44c651e45b24",
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
        assert identity["historical_sha256"] != identity["current_sha256"]
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_successor_75_records_the_no_growth_ratchet():
    record = _read(RECORD)
    inventory = _read(INVENTORY)
    assert record["harness_sharing"] == {
        "shared_module": "benchmarks/_artifact_io.py",
        "standard_library_only": True,
        "baseline_local_helper_files": 118,
        "frozen_hash_bound_files": 87,
        "documented_exceptions": 29,
        "migrated_active_files": 2,
        "allowlist_can_only_shrink": True,
        "new_local_helpers_forbidden": True,
        "historical_frozen_scripts_changed": 0,
    }
    assert inventory["baseline"]["conservative_ast_audit"] == {
        "hash_bound_files": 87,
        "local_helper_files": 118,
        "migrated_files": 2,
        "note": (
            "The conservative AST audit also counts provenance, multi-artifact, "
            "and binary atomic helpers, so it is intentionally broader than the "
            "R-0044 audit."
        ),
        "retained_exceptions": 29,
    }


def test_successor_75_changes_no_runtime_or_gate():
    record = _read(RECORD)
    assert record["scope"] == {
        "runtime_changed": False,
        "public_interface_changed": False,
        "numerical_algorithm_changed": False,
        "gpu_specific_path_changed": False,
        "checkpoint_schema_changed": False,
        "provider_identity_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "historical_record_rewritten": False,
    }
    assert record["authorization"] == {
        "local_commits": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
