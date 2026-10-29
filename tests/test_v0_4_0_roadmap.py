"""Bindings for the confirmed v0.4.0 target and completion contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver
from pssolver.functional.api import FUNCTIONAL_API_VERSION


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_roadmap.json"
SUCCESSOR = ROOT / (
    "notes/PSSolver_v0_3_0_s03_successor_69_s16_v04_contract.json"
)


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v04_development_identity_and_governance_are_exact() -> None:
    record = _read()
    assert pssolver.__version__ == "0.4.0.dev0"
    assert FUNCTIONAL_API_VERSION == "1.1"
    assert record["classification"] == (
        "V0_4_0_DEVELOPMENT_TARGET_AND_COMPLETION_CONTRACT"
    )
    assert record["status"] == "confirmed_by_D_0014"
    assert record["plan_step"] == "S-16"
    assert record["governance"] == {
        "decision": "D-0014",
        "decision_sha256": (
            "b01e70f81a0e8dcfff851203aa2a56cf459f42f2ef4431d606c2a7358480bcd2"
        ),
        "inherits": "D-0011",
        "inherits_sha256": (
            "cc9e71723afb00eda4df2602e0c38e30b96f2a2fdc445dd46402decba38abff1"
        ),
    }
    assert record["source"] == {
        "branch": "dev/0.4",
        "source_parent": "3ea68b5e288e670fe0a6af1e690cbcdf784d8a17",
        "v0_3_0_release_source": "47473880534a66f6da940db82122c79cda48a86b",
        "comparison_baseline": "v0.3.0",
        "version": "0.4.0.dev0",
        "functional_api_version_at_start": "1.1",
    }


def test_v04_completion_gates_are_complete_and_ordered() -> None:
    gates = _read()["completion_gates"]
    assert [gate["id"] for gate in gates] == [
        f"G-{index:02d}" for index in range(1, 10)
    ]
    assert all(gate["title"] and gate["criterion"] for gate in gates)
    by_id = {gate["id"]: gate for gate in gates}
    assert "fresh GitHub clone" in by_id["G-03"]["criterion"]
    assert "v0.2.x and v0.3.0" in by_id["G-05"]["criterion"]
    assert "byte-identical" in by_id["G-06"]["criterion"]
    assert "without the D-0013 memory exception" in by_id["G-08"]["criterion"]


def test_v04_execution_policy_matches_d0014() -> None:
    record = _read()
    policy = record["execution_policy"]
    assert "byte-identical to v0.3.0" in policy["s16_s17_numerical_rule"]
    assert "do not run H100" in policy["s16_s17_numerical_rule"]
    assert "stop" in policy["gpu_path_rule"]
    assert "S-19" in policy["gpu_path_rule"]
    assert "pre-registered tolerance" in policy["s18_numerical_rule"]
    assert "One pre-registered H100" in policy["s19_budget"]
    assert "does not extend" in policy["d0013_exception"]
    compatibility = record["compatibility_contract"]
    assert compatibility["read_all_released_v0_2_x_checkpoints"] is True
    assert compatibility["read_released_v0_3_0_checkpoints"] is True
    assert compatibility["published_checkpoint_sources_are_read_only"] is True
    assert compatibility["restart_equivalence_required"] is True


def test_v04_contract_changes_only_the_development_version() -> None:
    scope = _read()["scope"]
    assert scope["version_changed_to_development_line"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key != "version_changed_to_development_line"
    )
    assert _read()["authorization"] == {
        "local_s16_work": True,
        "push": False,
        "tag": False,
        "github_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }


def test_v04_contract_successor_binds_new_sources_and_forward_changes() -> None:
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    assert successor["classification"] == (
        "V0_4_0_DEVELOPMENT_VERSION_AND_COMPLETION_CONTRACT"
    )
    for relative, expected in successor["new_sources"].items():
        assert _sha256(ROOT / relative) == expected
    for relative, registration in successor[
        "historical_successor_sources"
    ].items():
        assert _sha256(ROOT / relative) == registration["current_sha256"]
        assert registration["historical_sha256"] in registration[
            "accepted_predecessor_sha256"
        ]
