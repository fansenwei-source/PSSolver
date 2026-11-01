from __future__ import annotations

import hashlib
import json
from pathlib import Path

from v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_95_s18_r2_checkpoint_integrity.json"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_95_binds_every_r2_source_forward() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["successor"] == 95
    assert record["git_identity"] == {
        "implementation": "f83539e76cf027c37010205f03b7b4e4f599002e",
        "implementation_parent": "eb6a191f83ec7bc59ed41dde0b6d67dab57acc10",
        "record_commit_is_qualified_source": False,
    }
    for relative, binding in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative,
            binding["historical_sha256"],
        ) == binding["current_sha256"]
        assert _sha256(relative) == binding["current_sha256"]
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(relative) == expected


def test_successor_95_does_not_claim_remote_or_nfs_authority() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["publication_trigger"] == {
        "real_nfs_required": False,
        "publication_call_graph_changed": False,
    }
    assert record["scope"] == {
        "public_api_changed": False,
        "numerical_algorithm_changed": False,
        "gpu_specific_path_changed": False,
        "checkpoint_schema_changed": False,
        "provider_identity_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "historical_record_rewritten": False,
    }
    assert record["authorization"]["push"] is False
    assert record["authorization"]["hpcc"] is False
    assert record["authorization"]["slurm_or_h100"] is False
