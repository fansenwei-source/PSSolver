from __future__ import annotations

import hashlib
import json
from pathlib import Path

from v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_s03_successor_104_s18_r5_channel_pcg.json"
)


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_successor_104_binds_every_r5_source_forward() -> None:
    record = _record()
    assert record["successor"] == 104
    assert record["git_identity"] == {
        "implementation": "318e634eb09de27cb683c31c4860ab34761e8e44",
        "implementation_parent": (
            "d84e883df58af92268dc55f0b02a78b3b5cffd57"
        ),
        "qualification_support": (
            "0238deabc280193e79aeaeb5c0074886bb7b6e1b"
        ),
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


def test_successor_104_claims_no_external_authority() -> None:
    record = _record()
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
