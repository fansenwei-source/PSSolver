"""Forward-only binding for the S-17 R3/R6 implementation candidate."""

from __future__ import annotations

import json
from pathlib import Path

from tests.v0_3_s03_successor import registered_source_sha256, resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_3_0_s03_successor_87_s17_r3_r6_implementation.json"
)
IMPLEMENTATION = ROOT / "notes/PSSolver_v0_4_0_s17_r3_r6_implementation.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return registered_source_sha256(path.relative_to(ROOT).as_posix())


def test_successor_87_advances_every_changed_historical_source() -> None:
    record = _read(RECORD)
    assert record["successor"] == 87
    assert record["classification"] == (
        "S17_R3_R6_LOCAL_CPU_CANDIDATE_AWAITING_R0050_REVIEW"
    )
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative,
            identity["historical_sha256"],
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative,
            expected,
        )


def test_successor_87_binds_the_confirmed_r3_r6_boundary() -> None:
    record = _read(RECORD)
    implementation = _read(IMPLEMENTATION)
    assert record["governance"] == {
        "decision": "D-0017",
        "implementation_commit": (
            "667735ed79154ecdaae25a5861eb2fd33391c90e"
        ),
        "implementation_record_sha256": _sha256(IMPLEMENTATION),
    }
    assert implementation["r3"]["functional_api_version"] == "1.1"
    assert implementation["r3"]["stable_names_or_semantics_changed"] is False
    assert implementation["r6"]["public_simulation_execution_qualified"] is False
    assert implementation["r6"]["lifting_execution_open"] is False
    assert implementation["r6"]["checkpoint_reader_changed"] is False


def test_successor_87_authorizes_no_external_or_gpu_action() -> None:
    record = _read(RECORD)
    assert record["scope"] == {
        "functional_api_changed": False,
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
        "local_implementation": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
